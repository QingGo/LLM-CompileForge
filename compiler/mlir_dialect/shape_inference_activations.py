"""MLIR-level shape inference for element-wise, activation, and basic ops.

All functions operate on ``mlir.ir.Type`` (``RankedTensorType``) and return
``list[ir.Type]``.  Ops requiring complex shape logic delegate to the
corresponding ``_infer_*_pure`` from ``shape_inference_pure``.
"""

from __future__ import annotations

from typing import Any

import mlir.ir as ir

from compiler.mlir_dialect.shape_inference_pure import (
    _infer_cat_pure,
    _infer_expand_pure,
    _infer_linear_pure,
    _infer_matmul_pure,
    _infer_select_pure,
    _infer_slice_pure,
    _infer_transpose_pure,
    _infer_unsqueeze_pure,
    _infer_view_pure,
)
from compiler.mlir_dialect.shape_inference_utils import (
    _broadcast_types,
    _elt_type_str,
    _infer_ir_via_pure,
    _make_ranked_type,
    _ranked_shape,
)

# ── Shape inference functions ────────────────────────────────


def _infer_broadcast(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    """Element-wise binary op with full broadcasting support."""
    if not input_types:
        return []
    et = _elt_type_str(input_types[0])
    b = _broadcast_types(*input_types)
    return [_make_ranked_type(b, et)]


infer_add = _infer_broadcast
infer_mul = _infer_broadcast
infer_sub = _infer_broadcast
infer_div = _infer_broadcast
infer_neg = _infer_broadcast
infer_pow = _infer_broadcast
infer_max = _infer_broadcast


# ── Activations: shape-preserving ────────────────────────────


def _infer_elementwise(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    """Element-wise op: broadcast all inputs, use first input's element type."""
    if not input_types:
        return []
    b = _broadcast_types(*input_types)
    et = _elt_type_str(input_types[0])
    return [_make_ranked_type(b, et)]


infer_relu = _infer_elementwise
infer_gelu = _infer_elementwise
infer_silu = _infer_elementwise
infer_sigmoid = _infer_elementwise
infer_softplus = _infer_elementwise
infer_exp = _infer_elementwise
infer_tanh = _infer_elementwise
infer_sqrt = _infer_elementwise
infer_clamp_min = _infer_elementwise
infer_rsqrt = _infer_elementwise
infer_cos = _infer_elementwise
infer_sin = _infer_elementwise


def infer_softmax(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    return _infer_elementwise(input_types)


def infer_layer_norm(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    """input, weight, bias → output (same shape as broadcasted input)."""
    if not input_types:
        return []
    b = _broadcast_types(input_types[0])
    et = _elt_type_str(input_types[0])
    return [_make_ranked_type(b, et)]


def infer_rms_norm(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    if not input_types:
        return []
    b = _broadcast_types(input_types[0])
    et = _elt_type_str(input_types[0])
    return [_make_ranked_type(b, et)]


# ── Matmul / Linear ──────────────────────────────────────────


def infer_matmul(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    """Matmul: (a: [M,K], b: [K,N]) → [M,N]; batched: [B,M,K]×[B,K,N] → [B,M,N]."""
    return _infer_ir_via_pure(_infer_matmul_pure, input_types, **kwargs)


def infer_linear(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    """Linear: (input: [...,K], weight: [N,K]) → [...,N]."""
    return _infer_ir_via_pure(_infer_linear_pure, input_types, **kwargs)


# ── Shape manipulation ───────────────────────────────────────


def infer_view(input_types: list[ir.Type], shape: tuple[int, ...] | None = None, **kwargs: Any) -> list[ir.Type]:
    """View/reshape: input → output with new shape (product of dims must match)."""
    return _infer_ir_via_pure(_infer_view_pure, input_types, shape=shape, **kwargs)


def infer_unsqueeze(input_types: list[ir.Type], dim: int = 0, **kwargs: Any) -> list[ir.Type]:
    """Unsqueeze: insert a dim-1 at position ``dim``."""
    return _infer_ir_via_pure(_infer_unsqueeze_pure, input_types, dim=dim, **kwargs)


def infer_squeeze(input_types: list[ir.Type], dim: int | None = None, **kwargs: Any) -> list[ir.Type]:
    """Squeeze: remove all size-1 dims, or the specific ``dim`` if given."""
    if not input_types:
        return []
    inp = input_types[0]
    s = _ranked_shape(inp)
    et = _elt_type_str(inp)
    if s is None:
        return [_make_ranked_type((None,), et)]
    if dim is None:
        parts = [d for d in s if d != 1]
    else:
        parts = list(s)
        if 0 <= dim < len(parts) and parts[dim] == 1:
            parts.pop(dim)
    return [_make_ranked_type(tuple(parts), et)]


def infer_expand(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    """Expand: broadcast input to a larger shape with new leading dims."""
    return _infer_ir_via_pure(_infer_expand_pure, input_types, **kwargs)


def infer_permute(input_types: list[ir.Type], dims: tuple[int, ...] | None = None, **kwargs: Any) -> list[ir.Type]:
    if not input_types:
        return []
    inp = input_types[0]
    s = _ranked_shape(inp)
    et = _elt_type_str(inp)
    if "dims" in kwargs:
        dims = kwargs["dims"]
    if s and dims:
        new_shape = tuple(s[d] for d in dims)
        return [_make_ranked_type(new_shape, et)]
    return [inp]


def infer_transpose(input_types: list[ir.Type], dim0: int = 0, dim1: int = 1, **kwargs: Any) -> list[ir.Type]:
    """Transpose: swap dimensions ``dim0`` and ``dim1``."""
    return _infer_ir_via_pure(_infer_transpose_pure, input_types, dim0=dim0, dim1=dim1, **kwargs)


def infer_slice(
    input_types: list[ir.Type],
    dim: int = 0, start: int = 0, end: int = -1, step: int = 1,
    **kwargs: Any,
) -> list[ir.Type]:
    """Slice: extract ``[start:end:step]`` along ``dim``."""
    return _infer_ir_via_pure(_infer_slice_pure, input_types, dim=dim, start=start, end=end, step=step, **kwargs)


def infer_select(
    input_types: list[ir.Type],
    dim: int = 0, index: int = 0,
    **kwargs: Any,
) -> list[ir.Type]:
    """Select: remove ``dim`` by indexing with a scalar ``index``."""
    return _infer_ir_via_pure(_infer_select_pure, input_types, dim=dim, index=index, **kwargs)


def infer_cat(input_types: list[ir.Type], dim: int = 0, **kwargs: Any) -> list[ir.Type]:
    """Concat: join multiple tensors along ``dim``."""
    return _infer_ir_via_pure(_infer_cat_pure, input_types, dim=dim, **kwargs)


def infer_pad(input_types: list[ir.Type], pad: tuple[int, ...] | None = None, **kwargs: Any) -> list[ir.Type]:
    if not input_types:
        return []
    inp = input_types[0]
    s = _ranked_shape(inp)
    et = _elt_type_str(inp)
    if s is None or pad is None:
        return [inp]
    if "pad" in kwargs:
        pad = kwargs["pad"]
    if isinstance(pad, (list, tuple)) and len(pad) % 2 == 0:
        n_pairs = len(pad) // 2
        parts = list(s)
        for i in range(min(n_pairs, len(parts))):
            idx = -(i + 1)
            lo = int(pad[2 * i]) if isinstance(pad, (list, tuple)) else 0
            hi = int(pad[2 * i + 1]) if isinstance(pad, (list, tuple)) else 0
            if parts[idx] is not None:
                parts[idx] = parts[idx] + lo + hi  # type: ignore[operator]
        return [_make_ranked_type(tuple(parts), et)]
    return [inp]


# ── Reduction ────────────────────────────────────────────────


def _infer_reduce(
    input_types: list[ir.Type], dim: int = -1, keepdim: bool = False, **kwargs: Any
) -> list[ir.Type]:
    if not input_types:
        return []
    inp = input_types[0]
    s = _ranked_shape(inp)
    et = _elt_type_str(inp)
    if s is None:
        return [_make_ranked_type((None,), et)]
    dim_v = int(dim) if "dim" in kwargs else dim
    keepdim_v = bool(kwargs.get("keepdim", keepdim))
    parts = list(s)
    if 0 <= dim_v < len(parts):
        if keepdim_v:
            parts[dim_v] = 1
        else:
            parts.pop(dim_v)
    return [_make_ranked_type(tuple(parts), et)]


def infer_mean(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    return _infer_reduce(input_types, **kwargs)


def infer_sum(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    return _infer_reduce(input_types, **kwargs)


def infer_cumsum(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    if input_types:
        return [input_types[0]]
    return []


def infer_var(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    return _infer_reduce(input_types, **kwargs)


def infer_linalg_norm(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    return _infer_reduce(input_types, **kwargs)


# ── Comparison (output is f32, consistent with C++ lowering) ──
# C++ sf-lower-to-linalg always produces f32 for compare ops
# (to avoid i1→f32 unrealized_conversion_cast blocking bufferization).


def _infer_compare(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    """Compare op: broadcast all inputs, output f32.

    Note: the C++ lowering pass always produces f32 output for compare
    ops (LeOp, LogicalAndOp, etc.) to avoid i1→f32 conversion casts
    that block bufferization.  The Python type inference must match.
    """
    if not input_types:
        return [_make_ranked_type((None,), "f32")]
    b = _broadcast_types(*input_types)
    return [_make_ranked_type(b, "f32")]


infer_gt = _infer_compare
infer_lt = _infer_compare
infer_eq = _infer_compare
infer_ne = _infer_compare
infer_le = _infer_compare
infer_logical_and = _infer_compare


# ── Element-wise utility ─────────────────────────────────────


def infer_triu(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    return _infer_elementwise(input_types)


def infer_tril(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    return _infer_elementwise(input_types)
