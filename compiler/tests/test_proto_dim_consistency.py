"""Proto dim consistency test — verify compiled dylib proto weight dimensions
match the actual lowered function argument shapes.

Contract hardening Fix 3: prevents regression where proto dims are assigned
to wrong weight names due to ordering mismatch between pre-lowering and
lowered MLIR.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def _load_proto(sfa_abi_path: str):
    """Parse sfa_abi.c hex bytes into SfaAbiHeader protobuf."""
    from gen.proto.python import sfa_abi_pb2
    with open(sfa_abi_path) as f:
        hex_bytes = re.findall(r'0x[0-9a-fA-F]{2}', f.read())
    raw = bytes(int(h, 16) for h in hex_bytes)
    hdr = sfa_abi_pb2.SfaAbiHeader()
    hdr.ParseFromString(raw)
    return hdr


def _parse_lowered_arg_types(lowered_path: str) -> dict[str, list[tuple[int, list[int]]]]:
    """Parse lowered MLIR function argument tensor types."""
    from compiler.sfa_abi import parse_lowered_argument_types
    return parse_lowered_argument_types(lowered_path)


def _parse_lowered_weight_names(lowered_path: str) -> dict[str, list[str]]:
    """Parse sf.weight_names from lowered MLIR."""
    from compiler.sfa_abi import parse_lowered_weight_names
    return parse_lowered_weight_names(lowered_path)


class TestProtoDimConsistency:
    """Verify proto weight dims match lowered function argument dims."""

    def test_main0_proto_dims_match_lowered_args(self):
        """main_0 weight entries in proto must match lowered arg types at
        corresponding positions.  Uses the current compiled dylib."""
        compiled = Path("outputs/compiled/opt_125m_p0")
        sfa_abi = compiled / "sfa_abi.c"
        lowered = compiled / "model.lowered.mlir"
        if not sfa_abi.exists() or not lowered.exists():
            pytest.skip("opt_125m_p0 not compiled")

        hdr = _load_proto(str(sfa_abi))
        lt_all = _parse_lowered_arg_types(str(lowered))
        lwn_all = _parse_lowered_weight_names(str(lowered))

        func0 = hdr.funcs[0]
        lt = lt_all.get("main_0", [])
        lwn = lwn_all.get("main_0", [])

        if not lwn:
            pytest.fail("main_0 has no weight names in lowered MLIR")

        proto_weights = {
            f.weight_name: (f.rank, list(f.dims))
            for f in func0.input_fields if f.weight_name
        }

        weight_start = len(lt) - len(lwn)
        errors = []

        for wi, wname in enumerate(lwn):
            lt_pos = weight_start + wi
            if lt_pos >= len(lt):
                errors.append(f"lwn[{wi}]='{wname}' → lt[{lt_pos}] out of bounds")
                continue
            expected_rank, expected_dims = lt[lt_pos]

            if wname not in proto_weights:
                errors.append(f"'{wname}' not found in proto weight entries")
                continue

            proto_rank, proto_dims = proto_weights[wname]
            if proto_rank != expected_rank:
                errors.append(f"'{wname}' rank: proto={proto_rank} lowered={expected_rank}")
            elif proto_dims != expected_dims:
                errors.append(f"'{wname}' dims: proto={proto_dims} lowered={expected_dims}")

        if errors:
            pytest.fail("\n  ".join([f"Proto dim errors ({len(errors)}):"] + errors[:20]))
