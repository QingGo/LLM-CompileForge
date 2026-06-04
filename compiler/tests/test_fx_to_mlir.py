"""Seam tests for FX→MLIR op resolution through public compiler API.

Tests ``_map_aten_op()`` and ``_shape_to_mlir_type()`` from
``compiler.fx_to_mlir_utils`` — pure string-based operations that
require NO MLIR context, NO compiled dylib, NO torch tensors in test code.
"""

from compiler.fx_to_mlir_utils import (
    _map_aten_op,
    _parse_mlir_type_to_shape,
    _shape_to_mlir_type,
)

# ── String targets for _map_aten_op (no torch.fx.Node needed) ────────


class TestMapAtenOp:
    """Verify aten operator name → HAL operator name resolution."""

    def test_matmul_overloads_map(self) -> None:
        """Multiple aten.matmul variants all map to 'matmul'."""
        assert _map_aten_op("aten.matmul") == "matmul"
        assert _map_aten_op("aten.matmul.default") == "matmul"
        assert _map_aten_op("aten.mm") == "matmul"
        assert _map_aten_op("aten.bmm") == "matmul"

    def test_add_overloads_map(self) -> None:
        """aten.add variants map to 'add'."""
        assert _map_aten_op("aten.add.Tensor") == "add"
        assert _map_aten_op("aten.add.Scalar") == "add"

    def test_activation_overloads_map(self) -> None:
        """Activation function overloads are stripped correctly."""
        assert _map_aten_op("aten.relu") == "relu"
        assert _map_aten_op("aten.relu.default") == "relu"
        assert _map_aten_op("aten.gelu") == "gelu"
        assert _map_aten_op("aten.silu") == "silu"
        assert _map_aten_op("aten.sigmoid.default") == "sigmoid"

    def test_softmax_overloads_map(self) -> None:
        """softmax variants all map to 'softmax'."""
        assert _map_aten_op("aten._softmax") == "softmax"
        assert _map_aten_op("aten._softmax.default") == "softmax"
        assert _map_aten_op("aten.softmax.int") == "softmax"

    def test_view_overloads_map(self) -> None:
        """view/reshape map to 'view'; unknown overloads drop to base name."""
        assert _map_aten_op("aten.view") == "view"
        assert _map_aten_op("aten.view.default") == "view"
        assert _map_aten_op("aten.reshape") == "view"

    def test_unmapped_op_returns_none(self) -> None:
        """An op not in the catalog returns None."""
        assert _map_aten_op("aten.unknown_op") is None
        assert _map_aten_op("some.random.target") is None

    def test_compare_ops_map(self) -> None:
        """Comparison operators map correctly."""
        assert _map_aten_op("aten.gt.Tensor") == "gt"
        assert _map_aten_op("aten.lt.Tensor") == "lt"
        assert _map_aten_op("aten.eq.Tensor") == "eq"
        assert _map_aten_op("aten.ne.Scalar") == "ne"


class TestShapeToMlirType:
    """Verify shape + element type → MLIR type string conversion."""

    def test_2d_f32(self) -> None:
        result = _shape_to_mlir_type((2, 64), "f32")
        assert result == "tensor<2x64xf32>"

    def test_3d_f16(self) -> None:
        result = _shape_to_mlir_type((1, 8, 128), "f16")
        assert result == "tensor<1x8x128xf16>"

    def test_dynamic_dims(self) -> None:
        """None and -1 are rendered as '?' (dynamic dims)."""
        result = _shape_to_mlir_type((None, 64), "f32")
        assert result == "tensor<?x64xf32>"
        result = _shape_to_mlir_type((-1, 32), "f32")
        assert result == "tensor<?x32xf32>"

    def test_scalar_tensor(self) -> None:
        """Empty shape produces scalar tensor type."""
        result = _shape_to_mlir_type((), "f32")
        assert result == "tensor<f32>"

    def test_int64_element_type(self) -> None:
        result = _shape_to_mlir_type((4,), "i64")
        assert result == "tensor<4xi64>"

    def test_bool_element_type(self) -> None:
        result = _shape_to_mlir_type((2, 3), "i1")
        assert result == "tensor<2x3xi1>"


class TestParseMlirTypeToShape:
    """Verify MLIR type string → (shape, element_type) parsing."""

    def test_2d_f32_roundtrip(self) -> None:
        shape, elt = _parse_mlir_type_to_shape("tensor<2x64xf32>")
        assert shape == (2, 64)
        assert elt == "f32"

    def test_dynamic_dims_parsed(self) -> None:
        shape, elt = _parse_mlir_type_to_shape("tensor<?x128xf16>")
        assert shape == (None, 128)
        assert elt == "f16"

    def test_scalar_tensor_parsed(self) -> None:
        shape, elt = _parse_mlir_type_to_shape("tensor<f32>")
        assert shape == ()
        assert elt == "f32"

    def test_non_tensor_fallback(self) -> None:
        """Non-tensor type string falls back to ((1,), 'f32')."""
        shape, elt = _parse_mlir_type_to_shape("i64")
        assert shape == (1,)
        assert elt == "f32"
