"""Pure-Python shape inference — no MLIR context required.

All ``_infer_*_pure`` functions operate on plain Python tuples
``(shape, element_type_str)``.  The dispatch function
``infer_output_shape`` is the entry point used by ``fx_to_mlir.py``.
"""

from __future__ import annotations

from typing import Any

from compiler.mlir_dialect.shape_inference_utils import _broadcast_shapes


def _infer_elementwise_pure(
    shapes: list[tuple[int | None, ...]],
    elts: list[str],
    **kwargs: Any,
) -> list[tuple[tuple[int | None, ...], str]]:
    """Element-wise op: broadcast all input shapes, use first input's element type."""
    if not shapes:
        return [((1,), "f32")]
    b = _broadcast_shapes(*shapes)
    return [(b, elts[0])]


def _infer_index_pure(
    shapes: list[tuple[int | None, ...]],
    elts: list[str],
    **kwargs: Any,
) -> list[tuple[tuple[int | None, ...], str]]:
    """sf.index: output = broadcast(index_shapes) + data_trailing_dims."""
    if len(shapes) < 2:
        return [(shapes[0] if shapes else (1,), elts[0] if elts else "f32")]
    data_shape = shapes[0]
    index_shapes = shapes[1:]
    num_indices = len(index_shapes)
    # If all index tensors are 1-D (scalar indices per dim),
    # each contributes its size as a separate output dimension.
    # Otherwise broadcast all index shapes together.
    if all(len(s) == 1 for s in index_shapes):
        index_out = tuple(s[0] for s in index_shapes)
    else:
        index_out = _broadcast_shapes(*index_shapes)
    # Append trailing data dims beyond num_indices
    trailing = data_shape[num_indices:] if num_indices < len(data_shape) else ()
    return [(index_out + trailing, elts[0])]


def _infer_arange_pure(
    shapes: list[tuple[int | None, ...]],
    elts: list[str],
    **kwargs: Any,
) -> list[tuple[tuple[int | None, ...], str]]:
    """sf.arange: output size depends on input VALUE (runtime), not input shape.
    The input is a scalar (shape=[1]) whose VALUE gives the output length.
    Return dynamic shape (None,) so downstream ops see the tensor as dynamic-sized."""
    return [((None,), "i64")]


def _infer_new_ones_pure(
    shapes: list[tuple[int | None, ...]],
    elts: list[str],
    **kwargs: Any,
) -> list[tuple[tuple[int | None, ...], str]]:
    """new_ones: create tensor of ones with given shape, defaulting to f32.
    
    PyTorch's torch.ones() defaults to float32. The input is the shape
    tensor (from sym_size → i64) — we use its shape but NOT its element type.
    """
    if not shapes:
        return [((1,), "f32")]
    return [(shapes[0], "f32")]


def _infer_scalar_pure(
    shapes: list[tuple[int | None, ...]],
    elts: list[str],
    **kwargs: Any,
) -> list[tuple[tuple[int | None, ...], str]]:
    """SymSize: return 1-element tensor (shape (1,)) with f32 element type."""
    return [((1,), "f32")]


def _infer_matmul_pure(
    shapes: list[tuple[int | None, ...]],
    elts: list[str],
    **kwargs: Any,
) -> list[tuple[tuple[int | None, ...], str]]:
    if len(shapes) < 2:
        return [(shapes[0], elts[0]) if shapes else ((1,), "f32")]
    a, b = shapes[0], shapes[1]
    if len(a) == 2 and len(b) == 2:
        return [((a[0], b[1]), elts[0])]
    if len(a) == 3 and len(b) == 2:
        return [((a[0], a[1], b[1]), elts[0])]
    return [(a, elts[0])]


def _infer_linear_pure(
    shapes: list[tuple[int | None, ...]],
    elts: list[str],
    **kwargs: Any,
) -> list[tuple[tuple[int | None, ...], str]]:
    if len(shapes) < 2:
        return [(shapes[0], elts[0]) if shapes else ((1,), "f32")]
    a, w = shapes[0], shapes[1]
    if len(a) >= 2 and len(w) == 2:
        out = tuple(list(a[:-1]) + [w[0]])
        return [(out, elts[0])]
    return [((None, None), elts[0])]


def _infer_view_pure(
    shapes: list[tuple[int | None, ...]],
    elts: list[str],
    shape: tuple[int, ...] | None = None,
    **kwargs: Any,
) -> list[tuple[tuple[int | None, ...], str]]:
    if "shape" in kwargs:
        shape = kwargs["shape"]
    if shape:
        return [(tuple(shape), elts[0])]
    if shapes:
        return [(shapes[0], elts[0])]
    return [((1,), "f32")]


def _infer_unsqueeze_pure(
    shapes: list[tuple[int | None, ...]],
    elts: list[str],
    dim: int = 0,
    **kwargs: Any,
) -> list[tuple[tuple[int | None, ...], str]]:
    if not shapes:
        return [((1,), elts[0] if elts else "f32")]
    s = list(shapes[0])
    dim = int(dim)
    if dim < 0:
        dim = len(s) + 1 + dim
    s.insert(dim, 1)
    return [(tuple(s), elts[0])]


def _infer_transpose_pure(
    shapes: list[tuple[int | None, ...]],
    elts: list[str],
    dim0: int = 0,
    dim1: int = 1,
    **kwargs: Any,
) -> list[tuple[tuple[int | None, ...], str]]:
    if not shapes:
        return [((1,), elts[0] if elts else "f32")]
    s = list(shapes[0])
    d0, d1 = int(dim0), int(dim1)
    if d0 < len(s) and d1 < len(s):
        s[d0], s[d1] = s[d1], s[d0]
    return [(tuple(s), elts[0])]


def _infer_slice_pure(
    shapes: list[tuple[int | None, ...]],
    elts: list[str],
    dim: int = 0,
    start: int = 0,
    end: int = -1,
    step: int = 1,
    **kwargs: Any,
) -> list[tuple[tuple[int | None, ...], str]]:
    if not shapes:
        return [((1,), elts[0] if elts else "f32")]
    s = list(shapes[0])
    dim = int(dim)
    if dim < len(s):
        dim_val = s[dim]
        if dim_val is not None:
            st = int(start) if start is not None else 0
            # 9223372036854775807 (INT64_MAX) is PyTorch's sys.maxsize
            # sentinel for aten.slice, meaning "to the end". When end is
            # the sentinel, the output dim is unknown at compile time
            # (depends on the input's runtime dim), so use None (dynamic).
            _int64_max_sentinel = 9223372036854775807
            if end is not None and end >= 0 and end != _int64_max_sentinel:
                en = int(end)
                s[dim] = (en - st + int(step) - 1) // int(step)
            else:
                # end == INT64_MAX sentinel or end < 0: runtime-dependent size
                s[dim] = None
    return [(tuple(s), elts[0])]


def _infer_select_pure(
    shapes: list[tuple[int | None, ...]],
    elts: list[str],
    dim: int = 0,
    index: int = 0,
    **kwargs: Any,
) -> list[tuple[tuple[int | None, ...], str]]:
    if not shapes:
        return [((1,), elts[0] if elts else "f32")]
    s = list(shapes[0])
    dim = int(dim)
    if dim < len(s):
        s.pop(dim)
    return [(tuple(s), elts[0])]


def _infer_cat_pure(
    shapes: list[tuple[int | None, ...]],
    elts: list[str],
    dim: int = 0,
    **kwargs: Any,
) -> list[tuple[tuple[int | None, ...], str]]:
    if not shapes:
        return [((1,), elts[0] if elts else "f32")]
    dim = int(dim)
    parts = list(shapes[0])
    total = 0
    for s in shapes:
        if dim < len(s) and s[dim] is not None:
            total += s[dim]  # type: ignore[operator]
    if dim < len(parts):
        parts[dim] = total if total > 0 else None
    return [(tuple(parts), elts[0])]


def _infer_reduce_pure(
    shapes: list[tuple[int | None, ...]],
    elts: list[str],
    dim: int = -1,
    keepdim: bool = False,
    **kwargs: Any,
) -> list[tuple[tuple[int | None, ...], str]]:
    if not shapes:
        return [((1,), elts[0] if elts else "f32")]
    dim_v = int(kwargs.get("dim", dim))
    keepdim_v = bool(kwargs.get("keepdim", keepdim))
    s = list(shapes[0])
    if 0 <= dim_v < len(s):
        if keepdim_v:
            s[dim_v] = 1
        else:
            s.pop(dim_v)
    return [(tuple(s), elts[0])]


def _infer_expand_pure(
    shapes: list[tuple[int | None, ...]],
    elts: list[str],
    **kwargs: Any,
) -> list[tuple[tuple[int | None, ...], str]]:
    """Expand: broadcast input to the target shape.

    shape attr is a tuple with ints (explicit or -1 for keep) and
    strings (SSA references, meaning dynamic).
    """
    if not shapes:
        return [((1,), elts[0] if elts else "f32")]
    inp = shapes[0]
    shape = kwargs.get("shape")
    if shape:
        out: list[int | None] = []
        in_idx = len(shape) - len(inp)  # leading dims are new
        for entry in shape:
            if isinstance(entry, int):
                if entry == -1:
                    if in_idx < len(inp):
                        out.append(inp[in_idx])
                    else:
                        out.append(None)
                    in_idx += 1
                else:
                    out.append(entry)
            else:
                # String (SSA ref) → dynamic
                out.append(None)
                in_idx += 1
        return [(tuple(out), elts[0])]
    return [(inp, elts[0])]


def _infer_embedding_pure(
    shapes: list[tuple[int | None, ...]],
    elts: list[str],
    **kwargs: Any,
) -> list[tuple[tuple[int | None, ...], str]]:
    if len(shapes) < 2:
        return [(shapes[0], elts[0]) if shapes else ((1,), "f32")]
    w_s, inp_s = shapes[0], shapes[1]  # weight=[vocab, embed], indices=[batch, seq, ...]
    if len(w_s) >= 2:
        # Output: indices_shape + [embed_dim]
        return [(tuple(list(inp_s) + [w_s[1]]), elts[0] if elts else "f32")]
    return [(shapes[0], elts[0] if elts else "f32")]


def _infer_ones_like_pure(
    shapes: list[tuple[int | None, ...]],
    elts: list[str],
    **kwargs: Any,
) -> list[tuple[tuple[int | None, ...], str]]:
    """OnesLike: shape comes from kwargs (not from broadcasting inputs).

    When `shape` kwarg contains strings (SSA references), those dimensions
    are dynamic (None).  Static ints in the shape kwarg are kept as-is.
    When no `shape` kwarg, fall back to copying the first input's shape.
    """
    shape_kwarg = kwargs.get("shape")
    if shape_kwarg:
        # shape entries can be str (SSA ref → dynamic/None) or int (static)
        result_shape: tuple[int | None, ...] = tuple(
            d if isinstance(d, int) else None
            for d in shape_kwarg
        )
        return [(result_shape, elts[0] if elts else "f32")]
    if shapes:
        return [(shapes[0], elts[0] if elts else "f32")]
    return [((), "f32")]


def _infer_compare_pure(
    shapes: list[tuple[int | None, ...]],
    elts: list[str],
    **kwargs: Any,
) -> list[tuple[tuple[int | None, ...], str]]:
    """Compare op: broadcast all inputs, output f32 to match C++ lowering."""
    if not shapes:
        return [((1,), "f32")]
    b = _broadcast_shapes(*shapes)
    return [(b, "f32")]


# ── Pure Python inference table (no MLIR context needed) ───

_PURE_TABLE: dict[str, Any] = {
    "add": _infer_elementwise_pure,
    "mul": _infer_elementwise_pure,
    "sub": _infer_elementwise_pure,
    "div": _infer_elementwise_pure,
    "neg": _infer_elementwise_pure,
    "pow": _infer_elementwise_pure,
    "max": _infer_elementwise_pure,
    "relu": _infer_elementwise_pure,
    "gelu": _infer_elementwise_pure,
    "silu": _infer_elementwise_pure,
    "sigmoid": _infer_elementwise_pure,
    "softplus": _infer_elementwise_pure,
    "exp": _infer_elementwise_pure,
    "tanh": _infer_elementwise_pure,
    "sqrt": _infer_elementwise_pure,
    "clamp_min": _infer_elementwise_pure,
    "rsqrt": _infer_elementwise_pure,
    "cos": _infer_elementwise_pure,
    "sin": _infer_elementwise_pure,
    "softmax": _infer_elementwise_pure,
    "layer_norm": _infer_elementwise_pure,
    "rms_norm": _infer_elementwise_pure,
    "triu": _infer_elementwise_pure,
    "tril": _infer_elementwise_pure,
    "copy_": _infer_elementwise_pure,
    "type_as": _infer_elementwise_pure,
    "identity": _infer_elementwise_pure,
    "conv1d": _infer_elementwise_pure,
    "expand": _infer_expand_pure,
    "zeros_like": _infer_elementwise_pure,
    "new_ones": _infer_new_ones_pure,
    "diff": _infer_elementwise_pure,
    "index": _infer_index_pure,
    "scaled_dot_product_attention": _infer_elementwise_pure,
    "fused_silu_mul": _infer_elementwise_pure,
    "fused_attention_output": _infer_elementwise_pure,
    "fused_attention_block": _infer_elementwise_pure,
    "matmul": _infer_matmul_pure,
    "linear": _infer_linear_pure,
    "view": _infer_view_pure,
    "unsqueeze": _infer_unsqueeze_pure,
    "transpose": _infer_transpose_pure,
    "slice": _infer_slice_pure,
    "select": _infer_select_pure,
    "cat": _infer_cat_pure,
    "mean": _infer_reduce_pure,
    "sum": _infer_reduce_pure,
    "var": _infer_reduce_pure,
    "linalg_norm": _infer_reduce_pure,
    "embedding": _infer_embedding_pure,
    "gt": _infer_compare_pure,
    "lt": _infer_compare_pure,
    "eq": _infer_compare_pure,
    "ne": _infer_compare_pure,
    "le": _infer_compare_pure,
    "logical_and": _infer_compare_pure,
    "cumsum": _infer_elementwise_pure,
    "masked_fill": _infer_elementwise_pure,
    "arange": _infer_arange_pure,
    "ones_like": _infer_ones_like_pure,
    "full_like": _infer_ones_like_pure,
    "zeros": _infer_elementwise_pure,
    "eye": _infer_elementwise_pure,
    "pad": _infer_elementwise_pure,
    "sym_size": _infer_scalar_pure,
    "einsum": _infer_elementwise_pure,
    "stack": _infer_elementwise_pure,
    "view_as": _infer_elementwise_pure,
    "expand_as": _infer_elementwise_pure,
    "fused_rms_norm_matmul": _infer_elementwise_pure,
    "fused_qkv": _infer_elementwise_pure,
    "weight": _infer_elementwise_pure,
    "constant": _infer_elementwise_pure,
    "split": _infer_elementwise_pure,
    "chunk": _infer_elementwise_pure,
}


def infer_output_shape(
    op_name: str,
    input_shapes: list[tuple[int | None, ...]],
    input_elts: list[str],
    **kwargs: Any,
) -> list[tuple[tuple[int | None, ...], str]]:
    """Compute output shape and element type as Python tuples.

    Pure-Python implementation — does NOT require MLIR context.
    Usable from fx_to_mlir.py before MLIR objects exist.
    """
    fn = _PURE_TABLE.get(op_name)
    if fn is not None:
        return list(fn(input_shapes, input_elts, **kwargs))
    if input_shapes:
        return [(input_shapes[0], input_elts[0] if input_elts else "f32")]
    return [((1,), "f32")]
