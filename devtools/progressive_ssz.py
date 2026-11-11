"""Experimental tests-zkevm@v0.8.4 input decoder and SSZ commitment.

This module does not execute the EVM. It prototypes the schema boundary for
upstream issue #30 independently of EELS and remerkleable. The Sail decoder
and both C backends still implement the v0.6.2 schema.

Run this module with a binary v0.8.4 input to print its request root and chain
identifier. Run ``python -m unittest devtools.tests.test_progressive_ssz`` for
the local checks. Set EELS_V084 to an execution-specs checkout at revision
8cd0018e5ee88eead0fb10520b6b16f936ff08a3 with its dependencies in .venv to
enable differential checks. Set V084_FIXTURE to a released JSON fixture to
check its embedded commitments. These tests do not check EVM execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

ZERO = bytes(32)
SCHEMA_ID = b"\x15\x01"


def pair(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(left + right).digest()


def merkleize(chunks: tuple[bytes, ...], limit: int) -> bytes:
    """Hash a bounded tree without allocating its absent leaves."""
    if len(chunks) > limit or limit < 1:
        raise ValueError("invalid Merkle capacity")
    depth = (limit - 1).bit_length()
    zeros = [ZERO]
    for _ in range(depth):
        zeros.append(pair(zeros[-1], zeros[-1]))

    def subtree(start: int, level: int) -> bytes:
        if start >= len(chunks):
            return zeros[level]
        if level == 0:
            return chunks[start]
        return pair(subtree(start, level - 1), subtree(start + (1 << (level - 1)), level - 1))

    return subtree(0, depth)


def merkleize_progressive(chunks: tuple[bytes, ...]) -> bytes:
    """Hash progressive subtrees with capacities 1, 4, 16, 64, ... ."""

    def subtree(start: int, capacity: int) -> bytes:
        if start >= len(chunks):
            return ZERO
        left = merkleize(chunks[start : start + capacity], capacity)
        return pair(left, subtree(start + capacity, capacity * 4))

    return subtree(0, 1)


def chunks_of(data: bytes) -> tuple[bytes, ...]:
    return tuple(data[start : start + 32].ljust(32, b"\0") for start in range(0, len(data), 32))


class Schema(Protocol):
    @property
    def fixed_size(self) -> int | None: ...

    def hash_tree_root(self, data: bytes) -> bytes: ...


@dataclass(frozen=True)
class FixedBytes:
    fixed_size: int

    def hash_tree_root(self, data: bytes) -> bytes:
        if len(data) != self.fixed_size:
            raise ValueError(f"expected {self.fixed_size} bytes, received {len(data)}")
        return merkleize(chunks_of(data), max(1, (self.fixed_size + 31) // 32))


@dataclass(frozen=True)
class ByteList:
    limit: int | None = None

    @property
    def fixed_size(self) -> None:
        return None

    def hash_tree_root(self, data: bytes) -> bytes:
        chunks = chunks_of(data)
        if self.limit is None:
            root = merkleize_progressive(chunks)
        else:
            if len(data) > self.limit:
                raise ValueError("byte list exceeds schema limit")
            root = merkleize(chunks, max(1, (self.limit + 31) // 32))
        return pair(root, len(data).to_bytes(32, "little"))


def variable_spans(data: bytes, fixed_length: int, offsets: tuple[int, ...]) -> tuple[bytes, ...]:
    """Check canonical SSZ offsets and return their element spans."""
    if not offsets or offsets[0] != fixed_length:
        raise ValueError("first offset does not equal fixed length")
    ends = (*offsets[1:], len(data))
    if any(start > end or end > len(data) for start, end in zip(offsets, ends, strict=True)):
        raise ValueError("offsets decrease or exceed input")
    return tuple(data[start:end] for start, end in zip(offsets, ends, strict=True))


@dataclass(frozen=True)
class SszList:
    element: Schema
    limit: int | None = None

    @property
    def fixed_size(self) -> None:
        return None

    def elements(self, data: bytes) -> tuple[bytes, ...]:
        size = self.element.fixed_size
        if size is not None:
            if len(data) % size:
                raise ValueError("fixed list contains a partial element")
            count = len(data) // size
            if self.limit is not None and count > self.limit:
                raise ValueError("list exceeds schema limit")
            return tuple(data[start : start + size] for start in range(0, len(data), size))
        if not data:
            return ()
        if len(data) < 4:
            raise ValueError("list offset is truncated")
        first = int.from_bytes(data[:4], "little")
        if first == 0 or first % 4 or first > len(data):
            raise ValueError("invalid list offset table")
        if self.limit is not None and first // 4 > self.limit:
            raise ValueError("list exceeds schema limit")
        offsets = tuple(int.from_bytes(data[i : i + 4], "little") for i in range(0, first, 4))
        return variable_spans(data, first, offsets)

    def hash_tree_root(self, data: bytes) -> bytes:
        elements = self.elements(data)
        # All list elements in this schema are composite SSZ types.
        roots = tuple(self.element.hash_tree_root(item) for item in elements)
        root = merkleize_progressive(roots) if self.limit is None else merkleize(roots, self.limit)
        return pair(root, len(elements).to_bytes(32, "little"))


@dataclass(frozen=True)
class Container:
    fields: tuple[tuple[str, Schema], ...]
    progressive: bool = False

    @property
    def fixed_size(self) -> int | None:
        sizes = tuple(schema.fixed_size for _, schema in self.fields)
        return None if None in sizes else sum(size for size in sizes if size is not None)

    def split(self, data: bytes) -> dict[str, bytes]:
        fixed_length = sum(schema.fixed_size or 4 for _, schema in self.fields)
        if len(data) < fixed_length:
            raise ValueError("container fixed region is truncated")
        result = {}
        variable = []
        offsets = []
        position = 0
        for name, schema in self.fields:
            size = schema.fixed_size
            if size is None:
                variable.append(name)
                offsets.append(int.from_bytes(data[position : position + 4], "little"))
                position += 4
            else:
                result[name] = data[position : position + size]
                position += size
        if variable:
            spans = variable_spans(data, fixed_length, tuple(offsets))
            result.update(zip(variable, spans, strict=True))
        elif len(data) != fixed_length:
            raise ValueError("fixed container has trailing bytes")
        return result

    def hash_tree_root(self, data: bytes) -> bytes:
        values = self.split(data)
        roots = tuple(schema.hash_tree_root(values[name]) for name, schema in self.fields)
        if self.progressive:
            # v0.8.4 activates every field in declaration order.
            active = ((1 << len(self.fields)) - 1).to_bytes(32, "little")
            return pair(merkleize_progressive(roots), active)
        return merkleize(roots, len(roots))


U64 = FixedBytes(8)
B20 = FixedBytes(20)
B32 = FixedBytes(32)
B48 = FixedBytes(48)
B96 = FixedBytes(96)
WITHDRAWAL = Container(
    (("index", U64), ("validator_index", U64), ("address", B20), ("amount", U64))
)
DEPOSIT = Container(
    (
        ("pubkey", B48),
        ("withdrawal_credentials", B32),
        ("amount", U64),
        ("signature", B96),
        ("index", U64),
    )
)
WITHDRAWAL_REQUEST = Container(
    (("source_address", B20), ("validator_pubkey", B48), ("amount", U64))
)
CONSOLIDATION = Container((("source_address", B20), ("source_pubkey", B48), ("target_pubkey", B48)))
BUILDER_DEPOSIT = Container(DEPOSIT.fields[:-1])
BUILDER_EXIT = Container((("source_address", B20), ("pubkey", B48)))
REQUESTS = Container(
    (
        ("deposits", SszList(DEPOSIT)),
        ("withdrawals", SszList(WITHDRAWAL_REQUEST)),
        ("consolidations", SszList(CONSOLIDATION)),
        ("builder_deposits", SszList(BUILDER_DEPOSIT)),
        ("builder_exits", SszList(BUILDER_EXIT)),
    ),
    progressive=True,
)
PAYLOAD = Container(
    (
        ("parent_hash", B32),
        ("fee_recipient", B20),
        ("state_root", B32),
        ("receipts_root", B32),
        ("logs_bloom", FixedBytes(256)),
        ("prev_randao", B32),
        ("block_number", U64),
        ("gas_limit", U64),
        ("gas_used", U64),
        ("timestamp", U64),
        ("extra_data", ByteList(32)),
        ("base_fee_per_gas", B32),
        ("block_hash", B32),
        ("transactions", SszList(ByteList())),
        ("withdrawals", SszList(WITHDRAWAL)),
        ("blob_gas_used", U64),
        ("excess_blob_gas", U64),
        ("block_access_list", ByteList()),
        ("slot_number", U64),
    ),
    progressive=True,
)
NEW_PAYLOAD_REQUEST = Container(
    (
        ("execution_payload", PAYLOAD),
        ("versioned_hashes", SszList(B32)),
        ("parent_beacon_block_root", B32),
        ("execution_requests", REQUESTS),
    )
)
WITNESS = Container(
    (
        ("state", SszList(ByteList(1024))),
        ("codes", SszList(ByteList(65536))),
        ("headers", SszList(ByteList(1024), 256)),
    )
)
PUBLIC_KEYS = SszList(FixedBytes(65))
STATELESS_INPUT = Container(
    (
        ("new_payload_request", NEW_PAYLOAD_REQUEST),
        ("witness", WITNESS),
        ("chain_id", U64),
        ("public_keys", PUBLIC_KEYS),
    )
)


@dataclass(frozen=True)
class DecodedInput:
    request_root: bytes
    chain_id: int
    public_keys: tuple[bytes, ...]


def decode_stateless_input(data: bytes) -> DecodedInput:
    """Decode the exact Amsterdam v0.8.4 schema and compute its request root."""
    if data[:2] != SCHEMA_ID:
        raise ValueError("unsupported schema id")
    fields = STATELESS_INPUT.split(data[2:])
    root = NEW_PAYLOAD_REQUEST.hash_tree_root(fields["new_payload_request"])
    # Validate all witness elements, including material unused by execution.
    WITNESS.hash_tree_root(fields["witness"])
    return DecodedInput(
        root,
        int.from_bytes(fields["chain_id"], "little"),
        PUBLIC_KEYS.elements(fields["public_keys"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="binary stateless input from tests-zkevm@v0.8.4")
    args = parser.parse_args()
    result = decode_stateless_input(args.input.read_bytes())
    print(json.dumps({"request_root": result.request_root.hex(), "chain_id": result.chain_id}))


if __name__ == "__main__":
    main()
