"""Pure-Python shape inference — no MLIR context required.

All ``_infer_*_pure`` functions operate on plain Python tuples
``(shape, element_type_str)``.  The dispatch function
``infer_output_shape`` is the entry point used by ``fx_to_mlir.py``.
"""

from __future__ import annotations

from typing import Any

from compiler.shape.shape_inference_utils import _broadcast_shapes


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


def _infer_identity_pure(
    shapes: list[tuple[int | None, ...]],
    elts: list[str],
    **kwargs: Any,
) -> list[tuple[tuple[int | None, ...], str]]:
    if not shapes:
        return [((1,), "f32")]
    elt = elts[0]
    dtype_raw = kwargs.get("dtype")
    if dtype_raw is not None:
        if hasattr(dtype_raw, "replace"):
            elt = dtype_raw.replace("torch.", "")
        else:
            import torch
            for dt, name in {
                torch.float32: "f32", torch.float: "f32",
                torch.float16: "f16", torch.half: "f16",
                torch.bfloat16: "bf16", torch.float64: "f64", torch.double: "f64",
                torch.int32: "i32", torch.int64: "i64", torch.long: "i64",
                torch.int8: "i8", torch.uint8: "ui8", torch.bool: "i1",
            }.items():
                if dtype_raw == dt:
                    elt = name
                    break
    return [(shapes[0], elt)]


def _infer_type_as_pure(
    shapes: list[tuple[int | None, ...]],
    elts: list[str],
    **kwargs: Any,
) -> list[tuple[tuple[int | None, ...], str]]:
    """type_as: take shape/values from operand 0, dtype from operand 1."""
    if not shapes:
        return [((1,), "f32")]
    if len(shapes) >= 2 and len(elts) >= 2:
        return [(shapes[0], elts[1])]
    return [(shapes[0], elts[0])]


def _infer_copy__pure(
    shapes: list[tuple[int | None, ...]],
    elts: list[str],
    **kwargs: Any,
) -> list[tuple[tuple[int | None, ...], str]]:
    """copy_: shape and dtype come from the first operand."""
    if not shapes:
        return [((1,), "f32")]
    return [(shapes[0], elts[0])]


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
    """sf.arange: the first operand is the START value, dyn_shape[0] is the
    SIZE (= end - start, normalized by converter._collect_arange_args).
    The size is a runtime VALUE, not derivable from input shapes, so the
    output shape is dynamic: (None,)."""
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
    if len(a) >= 3 and len(b) == 2:
        return [(tuple(a[:-1]) + (b[1],), elts[0])]
    if len(a) == 2 and len(b) >= 3:
        return [(tuple(b[:-2]) + (a[0], b[-1]), elts[0])]
    if len(a) >= 3 and len(a) == len(b):
        batch = _broadcast_shapes(a[:-2], b[:-2])
        return [(batch + (a[-2], b[-1]), elts[0])]
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


def _conv1d_scalar(value: Any, default: int = 1) -> int:
    """Extract the first element from PyTorch conv1d's list-like attrs."""
    if isinstance(value, (list, tuple)):
        if value:
            return int(value[0])
        return default
    if value is None:
        return default
    return int(value)


def _infer_zeros_pure(
    shapes: list[tuple[int | None, ...]],
    elts: list[str],
    **kwargs: Any,
) -> list[tuple[tuple[int | None, ...], str]]:
    """torch.zeros: use the shape attribute when present."""
    shape = kwargs.get("shape")
    if isinstance(shape, (list, tuple)):
        return [(tuple(int(d) if d is not None else None for d in shape), "f32")]
    if shapes:
        return [(shapes[0], "f32")]
    return [((1,), "f32")]


def _infer_zeros_like_pure(
    shapes: list[tuple[int | None, ...]],
    elts: list[str],
    **kwargs: Any,
) -> list[tuple[tuple[int | None, ...], str]]:
    if shapes:
        return [(shapes[0], elts[0] if elts else "f32")]
    return [((1,), "f32")]


def _infer_eye_pure(
    shapes: list[tuple[int | None, ...]],
    elts: list[str],
    **kwargs: Any,
) -> list[tuple[tuple[int | None, ...], str]]:
    n = int(kwargs.get("n", 1))
    m = int(kwargs.get("m", n or 1))
    return [((n, m), elts[0] if elts else "f32")]


def _infer_pad_pure(
    shapes: list[tuple[int | None, ...]],
    elts: list[str],
    **kwargs: Any,
) -> list[tuple[tuple[int | None, ...], str]]:
    """Infer the output shape for PyTorch ``aten.pad.default``.

    The pad list is in PyTorch order: (left, right, top, bottom, ...) from
    the last dimension backwards.
    """
    if not shapes:
        return [((1,), "f32")]
    shape = shapes[0]
    elt = elts[0] if elts else "f32"
    pad = kwargs.get("pad")
    if not isinstance(pad, (list, tuple)) or len(pad) % 2 != 0:
        return [(shape, elt)]
    n_pairs = len(pad) // 2
    parts = list(shape)
    for i in range(min(n_pairs, len(parts))):
        dim = len(parts) - 1 - i
        lo = int(pad[2 * i])
        hi = int(pad[2 * i + 1])
        if parts[dim] is not None:
            parts[dim] = parts[dim] + lo + hi  # type: ignore[operator]
    return [(tuple(parts), elt)]


def _infer_conv1d_pure(
    shapes: list[tuple[int | None, ...]],
    elts: list[str],
    **kwargs: Any,
) -> list[tuple[tuple[int | None, ...], str]]:
    """Infer Conv1d output shape: [batch, out_channels, out_length]."""
    if len(shapes) < 2:
        if shapes:
            return [(shapes[0], elts[0] if elts else "f32")]
        return [((1,), "f32")]
    input_shape = shapes[0]
    weight_shape = shapes[1]
    if len(input_shape) < 3 or len(weight_shape) < 3:
        # Fall back to broadcast-like behavior for malformed graphs.
        return [(input_shape, elts[0] if elts else "f32")]

    batch = input_shape[0]
    out_channels = weight_shape[0]
    kernel = weight_shape[2]
    in_len = input_shape[2]
    stride = _conv1d_scalar(kwargs.get("stride"), 1)
    padding = _conv1d_scalar(kwargs.get("padding"), 0)
    dilation = _conv1d_scalar(kwargs.get("dilation"), 1)

    out_len: int | None
    if in_len is None:
        out_len = None
    else:
        in_len_int = int(in_len)
        numerator = in_len_int + 2 * padding - dilation * (kernel - 1) - 1  # type: ignore[operator]
        out_len = numerator // stride + 1 if stride > 0 else in_len_int
    return [((batch, out_channels, out_len), elts[0] if elts else "f32")]


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
    has_unknown = False
    for s in shapes:
        if dim < len(s):
            dim_size = s[dim]
            if dim_size is not None:
                total += dim_size
            else:
                has_unknown = True
        else:
            has_unknown = True
    if dim < len(parts):
        parts[dim] = None if has_unknown else (total if total > 0 else None)
    return [(tuple(parts), elts[0])]


def _normalize_reduce_dims(
    dims: Any, rank: int
) -> list[int]:
    """Normalize a PyTorch/ATen reduction dim argument to a list of non-negative dims.

    ``dim`` may be an int or a list/tuple of ints (``aten.sum.dim_IntList``
    passes a one-element list even for a single dimension).
    """
    if dims is None:
        return [rank - 1] if rank > 0 else []
    if isinstance(dims, (list, tuple)):
        raw = list(dims)
    else:
        raw = [dims]
    out: list[int] = []
    for d in raw:
        try:
            dv = int(d)
        except (TypeError, ValueError):
            continue
        if dv < 0:
            dv += rank
        if 0 <= dv < rank and dv not in out:
            out.append(dv)
    return out


def _infer_reduce_pure(
    shapes: list[tuple[int | None, ...]],
    elts: list[str],
    dim: int = -1,
    keepdim: bool = False,
    **kwargs: Any,
) -> list[tuple[tuple[int | None, ...], str]]:
    if not shapes:
        return [((1,), elts[0] if elts else "f32")]
    keepdim_v = bool(kwargs.get("keepdim", keepdim))
    s = list(shapes[0])
    reduced = set(_normalize_reduce_dims(kwargs.get("dim", dim), len(s)))
    out: list[int | None] = []
    for i, d in enumerate(s):
        if i in reduced:
            if keepdim_v:
                out.append(1)
        else:
            out.append(d)
    return [(tuple(out), elts[0])]


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
        lead = len(shape) - len(inp)
        for i, entry in enumerate(shape):
            if isinstance(entry, str):
                # String (SSA ref) → dynamic
                out.append(None)
            elif isinstance(entry, int) and entry == -1:
                # Torch broadcasts input to the target rank by right
                # alignment; -1 copies the corresponding input dim.
                in_idx = i - lead
                out.append(inp[in_idx] if 0 <= in_idx < len(inp) else None)
            else:
                out.append(entry)
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
        result_shape: tuple[int | None, ...] = tuple(d if isinstance(d, int) else None for d in shape_kwarg)
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


def _build_pure_table() -> dict[str, Any]:
    """Build op_name → _infer_*_pure function dispatch table.

    ``_infer_*_pure`` functions are auto-registered: ``_infer_matmul_pure``
    maps to ``"matmul"``, ``_infer_view_pure`` to ``"view"``, etc.
    Explicit overrides handle ops where the naming convention diverges
    (e.g. ``"sym_size"`` → ``_infer_scalar_pure``).
    """
    table: dict[str, Any] = {}

    # 1. Auto-map _infer_<op>_pure → op
    for name, obj in list(globals().items()):
        if name.startswith("_infer_") and name.endswith("_pure") and callable(obj):
            op_key = name[len("_infer_") : -len("_pure")]
            table[op_key] = obj

    # 2. Explicit overrides for ops not following _infer_<op>_pure convention
    table["sym_size"] = _infer_scalar_pure
    table["full_like"] = _infer_ones_like_pure
    for op in ("mean", "sum", "var", "linalg_norm"):
        table[op] = _infer_reduce_pure
    for op in ("gt", "lt", "eq", "ne", "le", "logical_and"):
        table[op] = _infer_compare_pure

    # 3. Elementwise ops use _infer_elementwise_pure as the shared function
    _elementwise_pure_ops: set[str] = {
        "add",
        "mul",
        "sub",
        "div",
        "neg",
        "pow",
        "max",
        "relu",
        "gelu",
        "silu",
        "sigmoid",
        "softplus",
        "exp",
        "tanh",
        "sqrt",
        "clamp_min",
        "rsqrt",
        "cos",
        "sin",
        "softmax",
        "layer_norm",
        "rms_norm",
        "triu",
        "tril",
        "conv1d",
        "zeros_like",
        "diff",
        "scaled_dot_product_attention",
        "fused_silu_mul",
        "fused_attention_output",
        "fused_attention_block",
        "cumsum",
        "masked_fill",
        "zeros",
        "eye",
        "pad",
        "einsum",
        "stack",
        "view_as",
        "expand_as",
        "fused_rms_norm_matmul",
        "fused_qkv",
        "weight",
        "constant",
        "split",
        "chunk",
    }
    for op in _elementwise_pure_ops:
        if op not in table:
            table[op] = _infer_elementwise_pure

    return table


_PURE_TABLE = _build_pure_table()


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
