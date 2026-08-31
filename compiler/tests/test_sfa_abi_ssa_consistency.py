"""RED→GREEN test: SfaAbiHeader SSA routing must match ExecutionPlan proto."""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from gen.proto.python import sfa_abi_pb2  # noqa: E402


def _execution_plan_ssa_map(plan: sfa_abi_pb2.ExecutionPlan):
    edges: list[tuple[int, int, int, int]] = []
    for fi, step in enumerate(plan.steps):
        ssa_order = 0
        for edge in step.inputs:
            if edge.source == sfa_abi_pb2.STEP_OUTPUT:
                edges.append((fi, ssa_order, edge.source_index, edge.producer_step))
                ssa_order += 1
    return edges


def _sfa_abi_ssa_edges(abi: sfa_abi_pb2.SfaAbiHeader):
    edges: list[tuple[int, int, int, int]] = []
    for fi, func_meta in enumerate(abi.funcs):
        ssa_order = 0
        for field in func_meta.input_fields:
            if field.kind == sfa_abi_pb2.SFA_INPUT_SSA:
                edges.append(
                    (fi, ssa_order, field.ssa.producer_func, field.ssa.producer_out)
                )
                ssa_order += 1
    return edges


@pytest.mark.unit
@pytest.mark.timeout(10)
class TestSfaAbiSsaConsistency:

    def test_execution_plan_maps_to_sfa_abi_fields(self) -> None:
        """merge_with_semantics produces correct SSA fields from ExecutionPlan."""
        compiled_dir = ROOT / "outputs" / "compiled" / "opt_125m_fresh"
        metadata_path = compiled_dir / "metadata.json"
        if not metadata_path.exists():
            pytest.skip(f"Compiled model not found at {compiled_dir}")

        meta = json.loads(metadata_path.read_text())
        ep_b64 = meta.get("exec_plan_proto", "")
        if not ep_b64:
            pytest.skip("No exec_plan_proto in metadata.json")

        plan = sfa_abi_pb2.ExecutionPlan()
        plan.ParseFromString(base64.b64decode(ep_b64))

        pre_lowering = {
            "functions": [
                {"name": step.func_name, "inputs": [], "outputs": []}
                for step in plan.steps
            ]
        }
        sigs = {
            f"_mlir_ciface_{step.func_name}": (len(step.inputs), 3)
            for step in plan.steps
        }

        from compiler.sfa_abi import merge_with_semantics, serialize_abi

        func_metas = merge_with_semantics(
            sigs,
            pre_lowering,
            lowered_arg_types=None,
            lowered_output_types=None,
            lowered_weight_names={},
            execution_plan_bytes=base64.b64decode(ep_b64),
        )

        assert len(func_metas) == len(plan.steps), (
            f"Expected {len(plan.steps)} func_metas, got {len(func_metas)}"
        )

        abi_bytes = serialize_abi(func_metas)
        abi = sfa_abi_pb2.SfaAbiHeader()
        abi.ParseFromString(abi_bytes)

        ep_edges = _execution_plan_ssa_map(plan)
        abi_edges = _sfa_abi_ssa_edges(abi)

        assert len(abi_edges) == len(ep_edges), (
            f"SSA edge count mismatch: ExecutionPlan={len(ep_edges)}, "
            f"SfaAbiHeader={len(abi_edges)}"
        )

        errors: list[str] = []
        for i, (ep, abi_e) in enumerate(zip(ep_edges, abi_edges, strict=True)):
            if ep != abi_e:
                errors.append(
                    f"Edge {i}: ExecutionPlan={ep} SfaAbiHeader={abi_e}"
                )

        if errors:
            pytest.fail(
                f"SSA routing mismatch ({len(errors)}/{len(ep_edges)}):\n"
                + "\n".join(errors[:10])
            )

        print(f"  All {len(ep_edges)} SSA edges match")
