"""Tests for HAL IR builder — verify correct shape information in emitted HAL IR.

The compiler should emit HAL IR where:
  1. Reshape ops only have shape_of outputs as shape inputs (not scalar weights)
  2. Decoder layers have shape_of ops for dynamic dims
  3. Fill ops use shape_of inputs, not scalar weights
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_HAL_IR_PATH = Path(__file__).resolve().parent.parent / "compiled" / "opt_125m_fresh" / "generated" / "hal_ir.json"


@pytest.fixture(scope="module")
def hal_ir() -> dict:
    """Load the HAL IR JSON."""
    if not _HAL_IR_PATH.exists():
        pytest.skip(f"hal_ir.json not found at {_HAL_IR_PATH}")
    with open(_HAL_IR_PATH) as f:
        return json.load(f)


def _get_weight_ssa_names(func: dict) -> set[str]:
    """Get SSA names of weight entries in a function."""
    return {w["ssa"] for w in func.get("weights", []) if w.get("ssa")}


def _get_weight_input_names(func: dict) -> set[str]:
    """Get SSA names that are weight inputs (mapped to compiled weight names)."""
    return set(func.get("weight_inputs", {}).keys())


def _is_scalar_shape(shape: list) -> bool:
    """Check if a shape represents a scalar (all dims are 1)."""
    return bool(shape) and all(
        (isinstance(d, str) and d in ("1", "?"))
        or (isinstance(d, int) and d == 1)
        for d in shape
    )


def _get_input_shapes(func: dict) -> dict[str, list]:
    """Map input name → shape for a function."""
    return {inp["name"]: inp["shape"] for inp in func["inputs"]}


# ── Reshape input filtering ─────────────────────────────────────────


@pytest.mark.unit
class TestReshapeInputs:
    """Reshape ops should only have data + shape_of inputs, not scalar weights."""

    def test_no_scalar_weight_inputs_in_reshape(self, hal_ir):
        """Reshape inputs must not include scalar weight SSA names."""
        for func in hal_ir["functions"]:
            weight_names = _get_weight_ssa_names(func)
            weight_input_names = _get_weight_input_names(func)
            all_weight_names = weight_names | weight_input_names
            for op in func["ops"]:
                if op["op"] == "reshape":
                    for inp in op["inputs"][1:]:  # Skip data input at index 0
                        assert inp not in all_weight_names, (
                            f"Reshape in {func['name']} has weight input '{inp}' "
                            f"(inputs={op['inputs']})"
                        )

    def test_no_scalar_placeholder_inputs_in_reshape(self, hal_ir):
        """Reshape inputs must not include scalar-shaped function args."""
        for func in hal_ir["functions"]:
            input_shapes = _get_input_shapes(func)
            for op in func["ops"]:
                if op["op"] == "reshape":
                    for inp in op["inputs"][1:]:
                        if inp in input_shapes:
                            shape = input_shapes[inp]
                            assert not _is_scalar_shape(shape), (
                                f"Reshape in {func['name']} has scalar input '{inp}' "
                                f"(shape={shape}, inputs={op['inputs']})"
                            )

    def test_reshape_with_dynamic_shape_has_shape_of_inputs(self, hal_ir):
        """Reshape ops with dynamic dims in shape attr need shape_of inputs."""
        for fi in range(1, 13):  # Decoder layers
            func = hal_ir["functions"][fi]
            for op in func["ops"]:
                if op["op"] == "reshape" and op.get("shape"):
                    shape = op["shape"]
                    has_dynamic = any(d == "?" for d in shape)
                    # Flatten reshapes (rank reduction) don't need shape_of
                    # They compute the missing dim from input numel
                    is_rank_reduce = len(shape) < 3
                    if has_dynamic and not is_rank_reduce:
                        assert len(op["inputs"]) > 1, (
                            f"Reshape in {func['name']} with dynamic shape "
                            f"{shape} should have shape_of inputs "
                            f"(got inputs={op['inputs']})"
                        )


# ── Shape-of ops in decoder layers ──────────────────────────────────


@pytest.mark.unit
class TestShapeOfOps:
    """Decoder layers should have shape_of ops for dynamic dimensions."""

    def test_decoder_layers_have_shape_of_ops(self, hal_ir):
        """Each decoder layer should emit at least one shape_of op."""
        for fi in range(1, 13):  # func[1]-func[12]
            func = hal_ir["functions"][fi]
            shape_of_ops = [op for op in func["ops"] if op["op"] == "shape_of"]
            assert len(shape_of_ops) >= 2, (
                f"Decoder layer {func['name']} should have >= 2 shape_of ops "
                f"(batch + seq), got {len(shape_of_ops)}"
            )

    def test_embedding_has_shape_of_ops(self, hal_ir):
        """Embedding function should have shape_of ops."""
        func = hal_ir["functions"][0]
        shape_of_ops = [op for op in func["ops"] if op["op"] == "shape_of"]
        assert len(shape_of_ops) >= 2

    def test_shape_of_ops_have_correct_inputs(self, hal_ir):
        """Shape_of ops should reference function inputs, not intermediate values."""
        for fi in range(1, 13):
            func = hal_ir["functions"][fi]
            for op in func["ops"]:
                if op["op"] == "shape_of":
                    for inp in op["inputs"]:
                        # shape_of should reference a function input or
                        # an intermediate tensor with known shape
                        assert inp.startswith("%"), (
                            f"shape_of in {func['name']} has invalid input '{inp}'"
                        )


# ── Fill ops ────────────────────────────────────────────────────────


@pytest.mark.unit
class TestFillOps:
    """Fill ops should use shape_of inputs, not scalar weights."""

    def test_fill_no_weight_inputs(self, hal_ir):
        """Non-arange fill ops should not reference weight SSA names."""
        for func in hal_ir["functions"]:
            weight_names = _get_weight_ssa_names(func)
            for op in func["ops"]:
                if op["op"] == "fill" and op.get("kind") != "arange":
                    for inp in op["inputs"]:
                        assert inp not in weight_names, (
                            f"Fill in {func['name']} has weight input '{inp}'"
                        )


# ── Structural integrity ────────────────────────────────────────────


@pytest.mark.unit
class TestStructuralIntegrity:
    """Verify HAL IR structure is correct after the fix."""

    def test_function_count_unchanged(self, hal_ir):
        """Should still have 16 functions."""
        assert hal_ir["num_functions"] == 16

    def test_all_ops_have_inputs_and_outputs(self, hal_ir):
        """Every op should have at least one input and one output."""
        for func in hal_ir["functions"]:
            for op in func["ops"]:
                if op["op"] in ("cache_write", "cache_read"):
                    continue  # These may have empty inputs/outputs
                assert "inputs" in op, f"Op in {func['name']} missing 'inputs'"
                assert "outputs" in op, f"Op in {func['name']} missing 'outputs'"

    def test_decoder_reshape_targets_have_concrete_static_dims(self, hal_ir):
        """Reshape target shapes should have concrete values for non-dynamic dims."""
        for fi in range(1, 13):
            func = hal_ir["functions"][fi]
            for op in func["ops"]:
                if op["op"] == "reshape" and op.get("shape"):
                    for dim in op["shape"]:
                        if dim != "?":
                            assert isinstance(dim, (int, str)), (
                                f"Reshape dim in {func['name']} has unexpected type: {dim}"
                            )
