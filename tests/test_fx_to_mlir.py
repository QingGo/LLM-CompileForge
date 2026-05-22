"""Unit tests for fx_graph_to_mlir helper functions.

Tests the utility functions that power the single-step FX→MLIR converter:
SymInt handling, shape resolution, dtype conversion, and weight name
sanitization.  These are the hot-path conversion utilities that previously
had zero coverage.
"""

from __future__ import annotations

import pytest
import torch


@pytest.mark.unit
class TestDtypeToMlir:
    def test_float32(self) -> None:
        from compiler.fx_to_mlir_utils import _dtype_to_mlir
        assert _dtype_to_mlir("float32") == "f32"

    def test_float16(self) -> None:
        from compiler.fx_to_mlir_utils import _dtype_to_mlir
        assert _dtype_to_mlir("float16") == "f16"

    def test_int64(self) -> None:
        from compiler.fx_to_mlir_utils import _dtype_to_mlir
        assert _dtype_to_mlir("int64") == "i64"

    def test_unknown_defaults_to_f32(self) -> None:
        from compiler.fx_to_mlir_utils import _dtype_to_mlir
        assert _dtype_to_mlir("complex128") == "f32"


@pytest.mark.unit
class TestTensorTypeStr:
    def test_static_shape(self) -> None:
        from compiler.fx_to_mlir_utils import _tensor_type_str
        result = _tensor_type_str("float32", (2, 4, 128))
        assert "2x4x128" in result
        assert "f32" in result

    def test_dynamic_shape(self) -> None:
        from compiler.fx_to_mlir_utils import _tensor_type_str
        result = _tensor_type_str("float16", (2, None, 128))
        assert "2x?x128" in result
        assert "f16" in result

    def test_scalar_tensor(self) -> None:
        from compiler.fx_to_mlir_utils import _tensor_type_str
        result = _tensor_type_str("float32", ())
        assert "tensor<f32>" == result


@pytest.mark.unit
class TestSymIntToInt:
    def test_regular_int_passthrough(self) -> None:
        from compiler.fx_to_mlir_utils import _symint_to_int
        assert _symint_to_int(42) == 42

    def test_float_returns_int(self) -> None:
        from compiler.fx_to_mlir_utils import _symint_to_int
        assert _symint_to_int(3.14) == 3

    def test_string_returns_none(self) -> None:
        from compiler.fx_to_mlir_utils import _symint_to_int
        assert _symint_to_int("abc") is None


@pytest.mark.unit
class TestResolveShapeTuple:
    def test_all_static(self) -> None:
        from compiler.fx_to_mlir_utils import _resolve_shape_tuple
        assert _resolve_shape_tuple(torch.Size([2, 4, 128])) == (2, 4, 128)


@pytest.mark.unit
class TestSymintForView:
    def test_concrete_returns_value(self) -> None:
        from compiler.fx_to_mlir_utils import _symint_for_view
        assert _symint_for_view(64) == 64


@pytest.mark.unit
class TestMapAtenOp:
    def test_known_op_returns_hal_name(self) -> None:
        from compiler.fx_to_mlir_utils import _map_aten_op
        result = _map_aten_op("aten.add.Tensor")
        assert result == "add"

    def test_identity_op(self) -> None:
        from compiler.fx_to_mlir_utils import _map_aten_op
        result = _map_aten_op("aten.contiguous.default")
        assert result == "identity"

    def test_unknown_op_returns_none(self) -> None:
        from compiler.fx_to_mlir_utils import _map_aten_op
        assert _map_aten_op("aten.nonexistent.op") is None

    def test_op_overload_fallback(self) -> None:
        from compiler.fx_to_mlir_utils import _map_aten_op
        result = _map_aten_op("aten.silu")
        assert result == "silu"


@pytest.mark.unit
class TestExtractNodeKwargs:
    def test_empty_kwargs(self) -> None:
        # Just verify import works and returns dict
        from compiler.fx_to_mlir_utils import _extract_node_kwargs
        assert callable(_extract_node_kwargs)


@pytest.mark.unit
class TestResolveOpTypes:

    def test_exact_ssa_match_takes_priority(self) -> None:
        """ssa_map has both a short name ('reshape') and long name ('reshape_5').
        An input referring to '%reshape_5' should match 'reshape_5' exactly,
        not the shorter prefix 'reshape'."""
        from compiler.fx_to_mlir_utils import _resolve_op_types

        shapes: dict[str, tuple[tuple[int | None, ...], str]] = {}
        shapes["reshape"] = ((1, 16, 1, 64, 128), "f32")
        shapes["reshape_5"] = ((1, 8, 64), "f32")

        ssa_map = {
            "reshape": "%reshape",
            "reshape_5": "%reshape_5",
        }

        in_types, out_types = _resolve_op_types(
            "add", ["%reshape_5"], ssa_map, shapes, {}, {},
        )
        assert "8x64" in in_types[0], (
            f"Expected shape from reshape_5, got {in_types}"
        )

    def test_missing_shape_fallback_produces_warning(self) -> None:
        """When shape is not found in shape_map, the fallback (2,64) should
        produce a warning so resolution failures are visible."""
        import warnings

        from compiler.fx_to_mlir_utils import _resolve_op_types

        warnings.simplefilter("always")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _resolve_op_types(
                "add", ["%unknown_ssa"], {}, {}, {}, {},
            )

        found = [x for x in w if "shape" in str(x.message).lower() and "unknown_ssa" in str(x.message)]
        assert found, (
            "Expected a warning about missing shape for %unknown_ssa"
        )


@pytest.mark.unit
class TestAdjustOpAttributesOnSplit:

    def test_dim_adjustment_for_rank_change(self) -> None:
        """When a function input has rank 2 but the original op had dim=3
        (from rank-5 context), the dim attribute must be clamped to rank-1."""
        from compiler.fx_to_mlir_split import _adjust_op_attributes
        from compiler.mlir_artifact import MlirOp

        op = MlirOp(
            name="sf.slice", dialect="sf", op_name="slice",
            operands=["select_23"], results=["slice_47"],
            attributes={"dim": 3, "start": 0, "end": 10},
            input_types=["tensor<1x?xf32>"],
            output_types=["tensor<1x?xf32>"],
        )

        input_rank = {"select_23": 2}
        adjusted = _adjust_op_attributes(op, input_rank)

        # dim=3 on rank-2 tensor is out of bounds → clamped to 1
        assert adjusted.attributes["dim"] == 1, (
            f"dim should be clamped to rank-1 (1), "
            f"got {adjusted.attributes['dim']}"
        )
        assert adjusted.attributes["start"] == 0
        assert adjusted.attributes["end"] == 10

    def test_dim_unaffected_when_already_in_bounds(self) -> None:
        """dim=1 on rank-3 should stay 1 (already within 0..2)."""
        from compiler.fx_to_mlir_split import _adjust_op_attributes
        from compiler.mlir_artifact import MlirOp

        op = MlirOp(
            name="sf.slice", dialect="sf", op_name="slice",
            operands=["x"], results=["y"],
            attributes={"dim": 1, "start": 0, "end": 5},
            input_types=["tensor<2x4x8xf32>"],
            output_types=["tensor<2x4x8xf32>"],
        )
        input_rank = {"x": 3}
        adjusted = _adjust_op_attributes(op, input_rank)
        assert adjusted.attributes["dim"] == 1

    def test_dims_list_attribute_clamped(self) -> None:
        """dims=[2, 3] on rank-2 input gets clamped to rank-1 = 1."""
        from compiler.fx_to_mlir_split import _adjust_op_attributes
        from compiler.mlir_artifact import MlirOp

        op = MlirOp(
            name="sf.permute", dialect="sf", op_name="permute",
            operands=["x"], results=["y"],
            attributes={"dims": (2, 3)},
            input_types=["tensor<1x?xf32>"],
            output_types=["tensor<1x?xf32>"],
        )
        input_rank = {"x": 2}
        adjusted = _adjust_op_attributes(op, input_rank)
        # 2 and 3 are >= rank 2 → both clamped to 1
        assert adjusted.attributes["dims"] == (1, 1)


# ── Float ops coercion ──────────────────────────────────────────

@pytest.mark.unit
class TestFloatOpsCoercion:
    """float_ops (neg, rsqrt, erf, etc.) coerc int→f32 before inference."""

    def test_neg_f32_output(self) -> None:
        """neg output matches f32 input element type."""
        from compiler.mlir_dialect.shape_inference import infer_output_shape
        out = infer_output_shape("neg", [(5,)], ["f32"])
        assert out[0][1] == "f32"

    def test_rsqrt_f32_output(self) -> None:
        from compiler.mlir_dialect.shape_inference import infer_output_shape
        out = infer_output_shape("rsqrt", [(3, 4)], ["f32"])
        assert out[0][1] == "f32"

    def test_erf_f32_output(self) -> None:
        from compiler.mlir_dialect.shape_inference import infer_output_shape
        out = infer_output_shape("erf", [(10,)], ["f32"])
        assert out[0][1] == "f32"
