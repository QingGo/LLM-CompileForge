"""Seam tests for FX→MLIR op mapping through public compiler API.

Tests ``_ATEN_TO_HAL`` and ``_map_aten_op()`` from
``compiler.fx_to_mlir_utils`` — the aten→HAL mapping is a pure data
structure that requires NO MLIR context, NO compiled artifacts, and
NO torch tensors.
"""

from compiler.fx.utils import _ATEN_TO_HAL, _map_aten_op


class TestAtenToHalMapping:
    """Verify aten op name → HAL op name mappings are correct."""

    def test_matmul_mapping(self) -> None:
        assert _ATEN_TO_HAL["aten.matmul"] == "matmul"
        assert _ATEN_TO_HAL["aten.matmul.default"] == "matmul"
        assert _ATEN_TO_HAL["aten.mm"] == "matmul"
        assert _ATEN_TO_HAL["aten.bmm"] == "matmul"

    def test_add_mapping(self) -> None:
        assert _ATEN_TO_HAL["aten.add.Tensor"] == "add"
        assert _ATEN_TO_HAL["aten.add.Scalar"] == "add"

    def test_relu_mapping(self) -> None:
        assert _ATEN_TO_HAL["aten.relu"] == "relu"
        assert _ATEN_TO_HAL["aten.relu.default"] == "relu"

    def test_softmax_mapping(self) -> None:
        assert _ATEN_TO_HAL["aten._softmax"] == "softmax"
        assert _ATEN_TO_HAL["aten._softmax.default"] == "softmax"
        assert _ATEN_TO_HAL["aten.softmax.int"] == "softmax"

    def test_view_mapping(self) -> None:
        assert _ATEN_TO_HAL["aten.view"] == "view"
        assert _ATEN_TO_HAL["aten.view.default"] == "view"
        assert _ATEN_TO_HAL["aten.reshape"] == "view"

    def test_gelu_silu_sigmoid(self) -> None:
        assert _ATEN_TO_HAL["aten.gelu"] == "gelu"
        assert _ATEN_TO_HAL["aten.silu"] == "silu"
        assert _ATEN_TO_HAL["aten.sigmoid.default"] == "sigmoid"

    def test_linear_mapping(self) -> None:
        assert _ATEN_TO_HAL["aten.linear"] == "linear"
        assert _ATEN_TO_HAL["aten.linear.default"] == "linear"

    def test_layer_norm(self) -> None:
        assert _ATEN_TO_HAL["aten.layer_norm.default"] == "layer_norm"
        assert _ATEN_TO_HAL["aten.native_layer_norm"] == "layer_norm"

    def test_cmp_ops(self) -> None:
        assert _ATEN_TO_HAL["aten.gt.Tensor"] == "gt"
        assert _ATEN_TO_HAL["aten.lt.Tensor"] == "lt"
        assert _ATEN_TO_HAL["aten.eq.Tensor"] == "eq"
        assert _ATEN_TO_HAL["aten.ne.Scalar"] == "ne"
        assert _ATEN_TO_HAL["aten.le.Tensor"] == "le"

    def test_embedding_mapping(self) -> None:
        assert _ATEN_TO_HAL["aten.embedding"] == "embedding"
        assert _ATEN_TO_HAL["aten.embedding.default"] == "embedding"


class TestMapAtenOp:
    """Test _map_aten_op() which strips dart overloads back to base name."""

    def test_map_aten_matmul_default(self) -> None:
        assert _map_aten_op("aten.matmul.default") == "matmul"

    def test_map_aten_add_tensor(self) -> None:
        assert _map_aten_op("aten.add.Tensor") == "add"

    def test_map_aten_relu(self) -> None:
        assert _map_aten_op("aten.relu") == "relu"

    def test_map_with_overload_stripped(self) -> None:
        """Overload suffixes like .int, .default, .Tensor are stripped."""
        # aten.unsqueeze.default → aten.unsqueeze → unsqueeze
        assert _map_aten_op("aten.unsqueeze.default") == "unsqueeze"
        # aten.rsub.Scalar → aten.rsub → sub
        assert _map_aten_op("aten.rsub.Scalar") == "sub"

    def test_map_direct_hal_name(self) -> None:
        """Some ops map directly (no aten prefix)."""
        assert _map_aten_op("add") == "add"
        assert _map_aten_op("neg") == "neg"

    def test_map_unknown_op_returns_none(self) -> None:
        assert _map_aten_op("aten.nonexistent_op") is None
        assert _map_aten_op("custom.my_op") is None

    def test_map_with_double_colon(self) -> None:
        """PyTorch sometimes uses :: instead of ."""
        assert _map_aten_op("aten::matmul") == "matmul"
        assert _map_aten_op("aten::add.Tensor") == "add"


class TestMappingCompleteness:
    """Ensure key aten ops are in the table."""

    REQUIRED_ATEN_OPS = {
        "aten.matmul", "aten.add.Tensor", "aten.relu", "aten.softmax.int",
        "aten.view", "aten.gelu", "aten.silu", "aten.sigmoid",
        "aten.linear", "aten.layer_norm", "aten.rms_norm",
        "aten.embedding", "aten.scaled_dot_product_attention",
        "aten.cat", "aten.transpose.int", "aten.unsqueeze.default", "aten.slice.Tensor",
        "aten.clone", "aten.contiguous", "aten.dropout",
    }

    def test_all_required_atens_present(self) -> None:
        missing = self.REQUIRED_ATEN_OPS - set(_ATEN_TO_HAL.keys())
        assert not missing, f"Missing aten→HAL mappings: {missing}"


class TestNoCrossBoundaryImports:
    """Guard: this file must not import from restricted modules."""

    def test_no_forbidden_imports(self) -> None:
        import inspect

        import compiler.fx.utils as m
        for name in dir(m):
            obj = getattr(m, name)
            if inspect.ismodule(obj):
                mod_name = getattr(obj, "__name__", "")
                if any(p in mod_name for p in ("hal", "engine", "server", "runtime", "mlir_sf")):
                    raise AssertionError(
                        f"fx_to_mlir_utils exports forbidden module: {name} → {mod_name}"
                    )
