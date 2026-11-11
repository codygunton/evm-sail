"""Check the prototype against the pinned EELS schema when EELS_V084 is set."""

from __future__ import annotations

import json
import os
import subprocess
import unittest
from pathlib import Path

from devtools import progressive_ssz as ssz

EELS_REVISION = "8cd0018e5ee88eead0fb10520b6b16f936ff08a3"

# This program runs in the reference environment. The prototype does not import
# EELS or remerkleable. Use the actual EELS dataclass schema for every vector.
REFERENCE = r"""
import json
import sys
from ethereum.forks.amsterdam.stateless import StatelessInput
from ethereum.utils.ssz import _container_type

Model = _container_type(StatelessInput)

def vector(count=0, byte_length=0, key_count=0):
    view = Model()
    view.chain_id = 0x123456789abcdef0
    payload = view.new_payload_request.execution_payload
    for name, value in (('parent_hash', 1), ('state_root', 2), ('receipts_root', 3),
                        ('prev_randao', 4), ('block_hash', 5)):
        setattr(payload, name, bytes([value]) * 32)
    payload.fee_recipient = b'\x06' * 20
    payload.logs_bloom = b'\x07' * 256
    payload.block_number = 42
    payload.gas_limit = 30_000_000
    payload.gas_used = 1234
    payload.timestamp = 5678
    payload.base_fee_per_gas = 2**200 + 9
    payload.blob_gas_used = 131072
    payload.excess_blob_gas = 262144
    payload.slot_number = 33
    payload.extra_data = b'\x12' * 32
    payload.transactions = [bytes([i % 256]) * byte_length for i in range(count)]
    Withdrawal = payload.fields()['withdrawals'].element_cls()
    payload.withdrawals = [Withdrawal(index=i, validator_index=i + 1,
                                      address=bytes([i % 256]) * 20, amount=i + 2)
                           for i in range(count)]
    payload.block_access_list = b'\xab' * byte_length
    view.new_payload_request.versioned_hashes = [bytes([i % 256]) * 32 for i in range(count)]
    requests = view.new_payload_request.execution_requests
    for name, typ in requests.fields().items():
        Item = typ.element_cls()
        items = []
        for i in range(count):
            item = Item()
            for j, (field, field_type) in enumerate(item.fields().items()):
                value = i + j + 1
                setattr(item, field, value if field in ('amount', 'index')
                        else bytes([value % 256]) * field_type.type_byte_length())
            items.append(item)
        requests.__setattr__(name, items)
    view.new_payload_request.parent_beacon_block_root = b'\x08' * 32
    view.witness.state = [b'\x01' * min(byte_length, 1024)] * count
    view.witness.codes = [b'\x02' * byte_length] * count
    view.public_keys = [b'\x04' * 65] * key_count
    raw = view.encode_bytes()
    # Check the public dataclass API too, rather than only its generated view.
    decoded = StatelessInput.decode_bytes(raw)
    return dict(input=(b'\x15\x01' + raw).hex(),
                request_root=decoded.new_payload_request.hash_tree_root().hex(),
                payload_root=decoded.new_payload_request.execution_payload.hash_tree_root().hex(),
                requests_root=decoded.new_payload_request.execution_requests.hash_tree_root().hex(),
                chain_id=int(decoded.chain_id), public_keys=len(decoded.public_keys))

command = json.load(sys.stdin)
if command['operation'] == 'vectors':
    result = [vector(**case) for case in command['cases']]
else:
    result = []
    for encoded in command['inputs']:
        try:
            StatelessInput.decode_bytes(bytes.fromhex(encoded)[2:])
            result.append(True)
        except Exception:
            result.append(False)
json.dump(result, sys.stdout)
"""


def encode_fields(schema: ssz.Container, fields: dict[str, bytes]) -> bytes:
    """Rebuild container offsets after a test changes one field."""
    fixed_length = sum(element.fixed_size or 4 for _, element in schema.fields)
    fixed = bytearray()
    variable = bytearray()
    for name, element in schema.fields:
        data = fields[name]
        if element.fixed_size is None:
            fixed.extend((fixed_length + len(variable)).to_bytes(4, "little"))
            variable.extend(data)
        else:
            fixed.extend(data)
    return bytes(fixed + variable)


def replace_field(schema: ssz.Container, data: bytes, path: tuple[str, ...], value: bytes) -> bytes:
    fields = schema.split(data)
    name = path[0]
    if len(path) == 1:
        fields[name] = value
    else:
        child = dict(schema.fields)[name]
        assert isinstance(child, ssz.Container)
        fields[name] = replace_field(child, fields[name], path[1:], value)
    return encode_fields(schema, fields)


class ProgressiveSszTests(unittest.TestCase):
    def test_empty_progressive_tree_is_one_zero_chunk(self) -> None:
        self.assertEqual(ssz.merkleize_progressive(()), bytes(32))
        self.assertEqual(ssz.ByteList().hash_tree_root(b""), ssz.pair(bytes(32), bytes(32)))

    def test_progressive_tree_retains_its_right_zero_child(self) -> None:
        leaf = bytes(range(32))
        self.assertEqual(ssz.merkleize_progressive((leaf,)), ssz.pair(leaf, bytes(32)))

    def test_progressive_lists_do_not_inherit_legacy_count_limits(self) -> None:
        self.assertEqual(len(ssz.SszList(ssz.WITHDRAWAL).elements(bytes(44 * 17))), 17)
        self.assertEqual(len(ssz.PUBLIC_KEYS.elements(bytes(65 * 32769))), 32769)
        with self.assertRaises(ValueError):
            ssz.SszList(ssz.WITHDRAWAL, 16).elements(bytes(44 * 17))

    def test_schema_limits_still_apply(self) -> None:
        for schema, data in (
            (ssz.ByteList(32), bytes(33)),
            (ssz.ByteList(1024), bytes(1025)),
            (ssz.ByteList(65536), bytes(65537)),
            (ssz.SszList(ssz.ByteList(1024), 256), (1028).to_bytes(4, "little") * 257),
            (ssz.PUBLIC_KEYS, bytes(64)),
        ):
            with self.subTest(schema=schema), self.assertRaises(ValueError):
                schema.hash_tree_root(data)

    def test_list_rejects_malformed_offsets(self) -> None:
        for data in (
            b"\0",
            bytes(4),
            (5).to_bytes(4, "little") + bytes(1),
            (8).to_bytes(4, "little"),
            (8).to_bytes(4, "little") + (7).to_bytes(4, "little"),
            (8).to_bytes(4, "little") + (12).to_bytes(4, "little"),
        ):
            with self.subTest(data=data.hex()), self.assertRaises(ValueError):
                ssz.SszList(ssz.ByteList()).hash_tree_root(data)

    def test_container_rejects_gaps_truncation_and_trailing_bytes(self) -> None:
        for data in (bytes(19), (21).to_bytes(4, "little") * 5 + b"\0"):
            with self.assertRaises(ValueError):
                ssz.REQUESTS.hash_tree_root(data)
        with self.assertRaises(ValueError):
            ssz.WITHDRAWAL.hash_tree_root(bytes(45))


@unittest.skipUnless(
    os.environ.get("V084_FIXTURE"), "set V084_FIXTURE to a released v0.8.4 JSON fixture"
)
class ReleasedFixtureTests(unittest.TestCase):
    def test_released_fixture_commitments(self) -> None:
        fixture = json.loads(Path(os.environ["V084_FIXTURE"]).read_text())
        count = 0

        def check(value: object) -> None:
            nonlocal count
            if isinstance(value, dict):
                if "statelessInputBytes" in value:
                    raw = bytes.fromhex(value["statelessInputBytes"].removeprefix("0x"))
                    expected = bytes.fromhex(value["statelessOutputBytes"].removeprefix("0x"))
                    result = ssz.decode_stateless_input(raw)
                    self.assertEqual(len(expected), 43)
                    self.assertEqual(result.request_root, expected[:32])
                    self.assertEqual(result.chain_id, int.from_bytes(expected[33:41], "little"))
                    self.assertEqual(int.from_bytes(expected[41:], "little"), 0x1501)
                    count += 1
                for child in value.values():
                    check(child)
            elif isinstance(value, list):
                for child in value:
                    check(child)

        check(fixture)
        self.assertGreater(count, 0, "fixture has no embedded stateless input/output pairs")


@unittest.skipUnless(os.environ.get("EELS_V084"), "set EELS_V084 to the pinned EELS checkout")
class EelsProgressiveSszTests(unittest.TestCase):
    root: Path
    python: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(os.environ["EELS_V084"])
        cls.python = cls.root / ".venv/bin/python"
        revision = subprocess.check_output(
            ["git", "-C", str(cls.root), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != EELS_REVISION:
            raise RuntimeError(f"EELS revision {revision} differs from {EELS_REVISION}")

    def reference(self, operation: str, **arguments: object) -> list[object]:
        completed = subprocess.run(
            [str(self.python), "-c", REFERENCE],
            input=json.dumps({"operation": operation, **arguments}),
            text=True,
            capture_output=True,
            check=False,
            cwd=self.root,
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr)
        result: list[object] = json.loads(completed.stdout)
        return result

    def test_roots_at_progressive_boundaries(self) -> None:
        counts = (0, 1, 2, 4, 5, 6, 16, 17, 20, 21, 22, 64, 65, 84, 85, 86)
        lengths = (0, 1, 31, 32, 33, 159, 160, 161, 671, 672, 673, 2719, 2720, 2721)
        cases = [{"count": count, "byte_length": 33, "key_count": count} for count in counts]
        cases += [{"count": 2, "byte_length": length} for length in lengths]
        cases.append({"key_count": 32769})
        vectors = self.reference("vectors", cases=cases)
        for case, vector in zip(cases, vectors, strict=True):
            assert isinstance(vector, dict)
            with self.subTest(case=case):
                raw = bytes.fromhex(vector["input"])
                result = ssz.decode_stateless_input(raw)
                self.assertEqual(result.request_root.hex(), vector["request_root"])
                self.assertEqual(result.chain_id, vector["chain_id"])
                self.assertEqual(len(result.public_keys), vector["public_keys"])
                body = ssz.STATELESS_INPUT.split(raw[2:])
                request = ssz.NEW_PAYLOAD_REQUEST.split(body["new_payload_request"])
                self.assertEqual(
                    ssz.PAYLOAD.hash_tree_root(request["execution_payload"]).hex(),
                    vector["payload_root"],
                )
                self.assertEqual(
                    ssz.REQUESTS.hash_tree_root(request["execution_requests"]).hex(),
                    vector["requests_root"],
                )

    def test_reference_and_prototype_reject_invalid_inputs(self) -> None:
        vector = self.reference("vectors", cases=[{}])[0]
        assert isinstance(vector, dict)
        base = bytes.fromhex(vector["input"])
        payload_path = ("new_payload_request", "execution_payload")
        changes = (
            ((*payload_path, "extra_data"), bytes(33)),
            ((*payload_path, "transactions"), bytes(4)),
            ((*payload_path, "withdrawals"), bytes(43)),
            (("new_payload_request", "versioned_hashes"), bytes(31)),
            (("public_keys",), bytes(64)),
            (("witness", "state"), (4).to_bytes(4, "little") + bytes(1025)),
            (("witness", "codes"), (4).to_bytes(4, "little") + bytes(65537)),
            (("witness", "headers"), (4).to_bytes(4, "little") + bytes(1025)),
            (("witness", "headers"), (1028).to_bytes(4, "little") * 257),
        )
        invalid = [
            ssz.SCHEMA_ID + replace_field(ssz.STATELESS_INPUT, base[2:], p, v) for p, v in changes
        ]
        # The body starts with two offsets, then the fixed chain_id, then an offset.
        gap = bytearray(base)
        gap[2:6] = (21).to_bytes(4, "little")
        invalid.extend((bytes(gap), base[:21], base[:-1]))
        accepted = self.reference("decode", inputs=[raw.hex() for raw in invalid])
        self.assertEqual(accepted, [False] * len(invalid))
        for index, raw in enumerate(invalid):
            with self.subTest(index=index), self.assertRaises(ValueError):
                ssz.decode_stateless_input(raw)


if __name__ == "__main__":
    unittest.main()
