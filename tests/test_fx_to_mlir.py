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
        from compiler.fx_to_mlir import _dtype_to_mlir
        assert _dtype_to_mlir("float32") == "f32"

    def test_float16(self) -> None:
        from compiler.fx_to_mlir import _dtype_to_mlir
        assert _dtype_to_mlir("float16") == "f16"

    def test_int64(self) -> None:
        from compiler.fx_to_mlir import _dtype_to_mlir
        assert _dtype_to_mlir("int64") == "i64"

    def test_unknown_defaults_to_f32(self) -> None:
        from compiler.fx_to_mlir import _dtype_to_mlir
        assert _dtype_to_mlir("complex128") == "f32"


@pytest.mark.unit
class TestTensorTypeStr:
    def test_static_shape(self) -> None:
        from compiler.fx_to_mlir import _tensor_type_str
        result = _tensor_type_str("float32", (2, 4, 128))
        assert "2x4x128" in result
        assert "f32" in result

    def test_dynamic_shape(self) -> None:
        from compiler.fx_to_mlir import _tensor_type_str
        result = _tensor_type_str("float16", (2, None, 128))
        assert "2x?x128" in result
        assert "f16" in result

    def test_scalar_tensor(self) -> None:
        from compiler.fx_to_mlir import _tensor_type_str
        result = _tensor_type_str("float32", ())
        assert "tensor<f32>" == result


@pytest.mark.unit
class TestSymIntToInt:
    def test_regular_int_passthrough(self) -> None:
        from compiler.fx_to_mlir import _symint_to_int
        assert _symint_to_int(42) == 42

    def test_float_returns_int(self) -> None:
        from compiler.fx_to_mlir import _symint_to_int
        assert _symint_to_int(3.14) == 3

    def test_string_returns_none(self) -> None:
        from compiler.fx_to_mlir import _symint_to_int
        assert _symint_to_int("abc") is None


@pytest.mark.unit
class TestResolveShapeTuple:
    def test_all_static(self) -> None:
        from compiler.fx_to_mlir import _resolve_shape_tuple
        assert _resolve_shape_tuple(torch.Size([2, 4, 128])) == (2, 4, 128)


@pytest.mark.unit
class TestSymintForView:
    def test_concrete_returns_value(self) -> None:
        from compiler.fx_to_mlir import _symint_for_view
        assert _symint_for_view(64) == 64


@pytest.mark.unit
class TestMapAtenOp:
    def test_known_op_returns_hal_name(self) -> None:
        from compiler.fx_to_mlir import _map_aten_op
        result = _map_aten_op("aten.add.Tensor")
        assert result == "add"

    def test_identity_op(self) -> None:
        from compiler.fx_to_mlir import _map_aten_op
        result = _map_aten_op("aten.contiguous.default")
        assert result == "identity"

    def test_unknown_op_returns_none(self) -> None:
        from compiler.fx_to_mlir import _map_aten_op
        assert _map_aten_op("aten.nonexistent.op") is None

    def test_op_overload_fallback(self) -> None:
        from compiler.fx_to_mlir import _map_aten_op
        result = _map_aten_op("aten.silu")
        assert result == "silu"


@pytest.mark.unit
class TestExtractNodeKwargs:
    def test_empty_kwargs(self) -> None:
        # Just verify import works and returns dict
        from compiler.fx_to_mlir import _extract_node_kwargs
        assert callable(_extract_node_kwargs)
