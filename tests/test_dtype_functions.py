"""Pure Python unit tests for the dtype function registry and per-operator
dtype functions.

Registry tests (always passing after T1):
  - ``test_registry_dispatch_unregistered``
  - ``test_registry_decorator``
  - ``test_apply_dtype_hook_element_type_replacement``

Per-op tests (RED until T3/T4/T5):
  - ones_like: no override vs. explicit override
  - arange: float scalar vs. int scalar
  - cumsum: int promotion vs. float passthrough vs. explicit override
"""

from __future__ import annotations

from typing import Any

from compiler.fx_to_mlir_utils import (
    DTYPE_REGISTRY,
    _apply_dtype_hook,
    _replace_element_type,
    dtype_rule,
    resolve_dtype,
)

# ── Registry base tests (T1: dtype function registry) ──────────────


def test_registry_dispatch_unregistered() -> None:
    """resolve_dtype returns None for ops with no registered dtype function."""
    assert resolve_dtype("unknown_op", ["f32"], {}) is None


def test_registry_decorator() -> None:
    """@dtype_rule registers a function; resolve_dtype dispatches to it."""
    @dtype_rule("_test_mock_op")
    def _mock_fn(input_elts: list[str], kwargs: dict[str, Any]) -> str | None:
        return "i64"

    try:
        result = resolve_dtype("_test_mock_op", ["f32"], {})
        assert result == "i64"
    finally:
        DTYPE_REGISTRY.pop("_test_mock_op", None)


def test_apply_dtype_hook_element_type_replacement() -> None:
    """_apply_dtype_hook replaces the element type in out_type_strs[0]."""
    @dtype_rule("_test_hook_op")
    def _mock_hook(input_elts: list[str], kwargs: dict[str, Any]) -> str | None:
        return "i64"

    try:
        out = ["tensor<4x768xf32>"]
        result = _apply_dtype_hook("_test_hook_op", ["f32"], {}, out)
        assert result == ["tensor<4x768xi64>"]
        # In-place modification means `out` is also updated
        assert out[0] == "tensor<4x768xi64>"
    finally:
        DTYPE_REGISTRY.pop("_test_hook_op", None)


def test_replace_element_type_rank0() -> None:
    """_replace_element_type handles rank-0 tensors correctly."""
    assert _replace_element_type("tensor<f32>", "i64") == "tensor<i64>"


def test_replace_element_type_dynamic() -> None:
    """_replace_element_type handles dynamic dimensions."""
    assert _replace_element_type("tensor<?x?xf32>", "i64") == "tensor<?x?xi64>"


def test_replace_element_type_noop_non_tensor() -> None:
    """_replace_element_type returns the input unchanged for non-tensor types."""
    assert _replace_element_type("f32", "i64") == "f32"


def test_ones_like_dtype_no_override() -> None:
    """ones_like without dtype override inherits the input element type."""
    assert resolve_dtype("ones_like", ["f32"], {}) == "f32"


def test_ones_like_dtype_with_override() -> None:
    """ones_like with explicit dtype override uses the override."""
    # dtype=3  ->  torch.int64  ->  "i64"
    assert resolve_dtype("ones_like", ["f32"], {"dtype": 3}) == "i64"


def test_arange_dtype_float_scalar() -> None:
    """arange always produces i64 (ODS constraint: Sf_Int64Tensor)."""
    assert resolve_dtype("arange", ["f32"], {}) == "i64"


def test_arange_dtype_int_scalar() -> None:
    """arange with an integer scalar argument produces i64."""
    assert resolve_dtype("arange", ["i64"], {}) == "i64"


def test_cumsum_dtype_int_input() -> None:
    """cumsum with integer input promotes to i64."""
    assert resolve_dtype("cumsum", ["i32"], {"dim": 0}) == "i64"


def test_cumsum_dtype_float_input() -> None:
    """cumsum with float input keeps the same dtype."""
    assert resolve_dtype("cumsum", ["f32"], {"dim": 0}) == "f32"


def test_cumsum_dtype_with_override() -> None:
    """cumsum with explicit dtype override uses the override."""
    # dtype=6  ->  torch.float32  ->  "f32"
    assert resolve_dtype("cumsum", ["i32"], {"dim": 0, "dtype": 6}) == "f32"
