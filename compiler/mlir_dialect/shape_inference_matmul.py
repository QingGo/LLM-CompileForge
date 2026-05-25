"""MLIR-level shape inference for tensor creation, utility, attention, and fused ops.

Also houses the dispatch table (``_INFERENCE_TABLE``) and the public entry
point ``infer_output_type``.
"""

from __future__ import annotations

from typing import Any

import mlir.ir as ir

from compiler.mlir_dialect.shape_inference_activations import (  # noqa: F401
    infer_add,
    infer_cat,
    infer_clamp_min,
    infer_cos,
    infer_cumsum,
    infer_div,
    infer_eq,
    infer_exp,
    infer_expand,
    infer_gelu,
    infer_gt,
    infer_layer_norm,
    infer_le,
    infer_linalg_norm,
    infer_linear,
    infer_logical_and,
    infer_lt,
    infer_matmul,
    infer_max,
    infer_mean,
    infer_mul,
    infer_ne,
    infer_neg,
    infer_pad,
    infer_permute,
    infer_pow,
    infer_relu,
    infer_rms_norm,
    infer_rsqrt,
    infer_select,
    infer_sigmoid,
    infer_silu,
    infer_sin,
    infer_slice,
    infer_softmax,
    infer_softplus,
    infer_sqrt,
    infer_squeeze,
    infer_sub,
    infer_sum,
    infer_tanh,
    infer_transpose,
    infer_tril,
    infer_triu,
    infer_unsqueeze,
    infer_var,
    infer_view,
)
from compiler.mlir_dialect.shape_inference_pure import (
    _infer_embedding_pure,
)
from compiler.mlir_dialect.shape_inference_utils import (
    _broadcast_shapes,
    _elt_type_str,
    _infer_ir_via_pure,
    _make_ranked_type,
    _ranked_shape,
)

# ── Tensor creation / utility ────────────────────────────────


def infer_embedding(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    """Embedding: (weight, indices) → output, keep indice dims and use weight dims."""
    return _infer_ir_via_pure(_infer_embedding_pure, input_types, **kwargs)


def infer_masked_fill(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    if input_types:
        return [input_types[0]]
    return []


def infer_copy_(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    if input_types:
        return [input_types[0]]
    return []


def infer_type_as(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    if len(input_types) >= 2:
        inp = input_types[0]
        ref = input_types[1]
        s = _ranked_shape(inp)
        et = _elt_type_str(ref)
        if s:
            return [_make_ranked_type(s, et)]
    return input_types[:1]


def infer_identity(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    if input_types:
        return [input_types[0]]
    return []


def infer_conv1d(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    if not input_types:
        return []
    return [input_types[0]]


def infer_arange(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    if "n" in kwargs:
        return [_make_ranked_type((int(kwargs["n"]),), "i64")]
    return [_make_ranked_type((None,), "i64")]


def infer_ones_like(input_types: list[ir.Type], shape: tuple[int, ...] | None = None, **kwargs: Any) -> list[ir.Type]:
    if "shape" in kwargs:
        shape = kwargs["shape"]
    if shape:
        return [_make_ranked_type(tuple(shape), "f32")]
    if input_types:
        return [input_types[0]]
    return [_make_ranked_type((1, 1), "f32")]


def infer_full_like(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    return infer_ones_like(input_types, **kwargs)


def infer_zeros(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    return infer_ones_like(input_types, **kwargs)


def infer_zeros_like(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    if input_types:
        return [input_types[0]]
    return []


def infer_new_ones(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    if input_types:
        return [input_types[0]]
    return [_make_ranked_type((1,), "f32")]


def infer_eye(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    n = int(kwargs.get("n", 1))
    m = int(kwargs.get("m", n))
    return [_make_ranked_type((n, m), "f32")]


def infer_diff(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    if input_types:
        return [input_types[0]]
    return []


def infer_sym_size(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    return [_make_ranked_type((1,), "i64")]


def infer_index(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    if len(input_types) < 2:
        return input_types if input_types else []
    data_type = input_types[0]
    idx_types = input_types[1:]
    data_shape = _ranked_shape(data_type)
    idx_shapes = [_ranked_shape(t) for t in idx_types]
    # If any shape is unknown, return fully dynamic result
    if data_shape is None or any(s is None for s in idx_shapes):
        return [_make_ranked_type((None,) * 1, _elt_type_str(data_type))]
    # All shapes known at this point — broadcast index dims then append trailing data dims
    valid_shapes: list[tuple[int | None, ...]] = [s for s in idx_shapes if s is not None]
    broadcast_shape = _broadcast_shapes(*valid_shapes)
    num_indices = len(idx_types)
    trailing = data_shape[num_indices:] if num_indices < len(data_shape) else ()
    result_shape = broadcast_shape + trailing
    return [_make_ranked_type(result_shape, _elt_type_str(data_type))]


def infer_einsum(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    if input_types:
        return [_make_ranked_type((None, None), _elt_type_str(input_types[0]))]
    return [_make_ranked_type((None,), "f32")]


def infer_stack(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    if not input_types:
        return []
    inp = input_types[0]
    s = _ranked_shape(inp)
    et = _elt_type_str(inp)
    if s:
        return [_make_ranked_type(tuple([len(input_types)] + list(s)), et)]
    return [_make_ranked_type((None,), et)]


def infer_view_as(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    if len(input_types) >= 2:
        return [input_types[1]]
    if input_types:
        return [input_types[0]]
    return []


def infer_expand_as(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    if len(input_types) >= 2:
        return [input_types[1]]
    if input_types:
        return [input_types[0]]
    return []


# ── Attention ────────────────────────────────────────────────


def infer_scaled_dot_product_attention(
    input_types: list[ir.Type], **kwargs: Any
) -> list[ir.Type]:
    if len(input_types) >= 1:
        return [input_types[0]]
    return []


# ── Fused ops ────────────────────────────────────────────────


def infer_fused_silu_mul(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    if input_types:
        return [input_types[0]]
    return []


def infer_fused_rms_norm_matmul(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    if len(input_types) >= 3:
        return [_make_ranked_type((None, None), _elt_type_str(input_types[0]))]
    return []


def infer_fused_qkv(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    if len(input_types) >= 2:
        et = _elt_type_str(input_types[0])
        return [
            _make_ranked_type((None, None), et),
            _make_ranked_type((None, None), et),
            _make_ranked_type((None, None), et),
        ]
    return []


def infer_fused_attention_output(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    if len(input_types) >= 1:
        return [input_types[0]]
    return []


def infer_fused_attention_block(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    if len(input_types) >= 1:
        return [input_types[0]]
    return []


def infer_weight(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    return [_make_ranked_type((None,), "f32")]


def infer_constant(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    return [_make_ranked_type((1,), "f32")]


def infer_split(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    if input_types:
        return [input_types[0]]
    return []


def infer_chunk(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    return infer_split(input_types)


# ── Dispatch table ───────────────────────────────────────────


_INFERENCE_TABLE: dict[str, Any] = {}


def _build_inference_table() -> dict[str, Any]:
    """Build op_name → infer_* function dispatch table from module globals.

    ``infer_*`` functions defined in this module or imported from sibling
    modules (``shape_inference_activations``) are automatically registered,
    so adding a new op just requires defining ``def infer_<op>(...)``.
    """
    table: dict[str, Any] = {}
    for name, obj in list(globals().items()):
        if name.startswith("infer_") and callable(obj):
            op_key = name[len("infer_"):]
            table[op_key] = obj
    return table


_INFERENCE_TABLE = _build_inference_table()


def infer_output_type(op_name: str, input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    """Compute output MLIR tensor types for an sf operation.

    Args:
        op_name: sf op name without dialect prefix (e.g. "add", "matmul")
        input_types: list of mlir.ir.Type (tensor types)
        **kwargs: operation attributes (dim, shape, keepdim, etc.)

    Returns:
        list of mlir.ir.Type for operation results
    """
    fn: Any = _INFERENCE_TABLE.get(op_name)
    if fn is not None:
        return list(fn(input_types, **kwargs))
    if input_types:
        return [input_types[0]]
    return []
