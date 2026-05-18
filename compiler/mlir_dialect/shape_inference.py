"""Shape inference for sf dialect operations.

Each inference function takes the operation name, input types, and attributes
and returns the output tensor type(s).

Supports both RankedTensorType (from official MLIR) and Python tuples
(shape, element_type_string) for use in fx_to_mlir.py before MLIR objects exist.
"""

from __future__ import annotations

from typing import Any

import mlir.ir as ir

# ── Type helpers ─────────────────────────────────────────────


def _elt_type_str(tp: ir.Type) -> str:
    """Extract element type string from MLIR tensor type."""
    if isinstance(tp, (ir.RankedTensorType, ir.UnrankedTensorType)):
        et = str(tp.element_type)
        for s in ["f32", "f64", "f16", "bf16", "i32", "i64", "i8", "i1", "ui8"]:
            if s in et:
                return s
    return "f32"


def _ranked_shape(tp: ir.Type) -> tuple[int | None, ...] | None:
    """Extract shape tuple from MLIR tensor type. None for unranked."""
    if isinstance(tp, ir.RankedTensorType):
        return tuple(int(d) if d >= 0 else None for d in tp.shape)
    return None


def _elt_from_str(s: str) -> ir.Type:
    m: dict[str, ir.Type] = {
        "f32": ir.F32Type.get(), "f64": ir.F64Type.get(),
        "f16": ir.F16Type.get(), "bf16": ir.BF16Type.get(),
        "i32": ir.IntegerType.get_signless(32),
        "i64": ir.IntegerType.get_signless(64),
        "i8": ir.IntegerType.get_signless(8),
        "bool": ir.IntegerType.get_signless(1),
    }
    return m.get(s, ir.F32Type.get())


def _make_ranked_type(shape: tuple[int | None, ...], elt: str) -> ir.RankedTensorType:
    return ir.RankedTensorType.get(
        [-1 if d is None else d for d in shape],
        _elt_from_str(elt),
    )


# ── Broadcasting helpers ─────────────────────────────────────


def _broadcast_shapes(*shapes: tuple[int | None, ...]) -> tuple[int | None, ...]:
    """NumPy/MLIR-style broadcasting: align right, size-1 broadcasts to any.

    Each dim must be 1, equal, or one can be None (dynamic, which is compatible
    with anything).  Returns the broadcasted shape.
    """
    if not shapes:
        return ()
    max_rank = max(len(s) for s in shapes)
    result: list[int | None] = []
    for i in range(max_rank):
        dims = []
        for s in shapes:
            idx = len(s) - max_rank + i
            dims.append(s[idx] if idx >= 0 else 1)
        non_one = [d for d in dims if d != 1 and d is not None]
        if not non_one:
            result.append(dims[0])
        elif len(non_one) == 1:
            result.append(non_one[0])
        else:
            if len(set(non_one)) > 1:
                raise ValueError(
                    f"Incompatible dims for broadcast: {dims}"
                )
            result.append(non_one[0])
    return tuple(result)


def _broadcast_types(*types: ir.Type) -> tuple[int | None, ...]:
    """Broadcast shapes from MLIR tensor types, returning a shape tuple."""
    shapes: list[tuple[int | None, ...]] = []
    for t in types:
        s = _ranked_shape(t)
        if s is None:
            return (None,) * max(len(shapes) + 1, 1)
        shapes.append(s)
    return _broadcast_shapes(*shapes)


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
    if len(input_types) < 2:
        return input_types
    a = input_types[0]
    b = input_types[1]
    a_s = _ranked_shape(a)
    b_s = _ranked_shape(b)
    et = _elt_type_str(a)
    if a_s is None or b_s is None:
        return [_make_ranked_type((None, None), et)]
    if len(a_s) == 2 and len(b_s) == 2:
        return [_make_ranked_type((a_s[0], b_s[1]), et)]
    if len(a_s) == 3 and len(b_s) == 2:
        return [_make_ranked_type((a_s[0], a_s[1], b_s[1]), et)]
    if len(a_s) == 3 and len(b_s) == 3:
        return [_make_ranked_type((a_s[0], a_s[1], b_s[2]), et)]
    return [input_types[0]]


def infer_linear(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    if len(input_types) < 2:
        return input_types
    a = input_types[0]
    w = input_types[1]
    a_s = _ranked_shape(a)
    w_s = _ranked_shape(w)
    et = _elt_type_str(a)
    if a_s is None or w_s is None:
        return [_make_ranked_type((None, None), et)]
    # input: [..., K], weight: [N, K] → output: [..., N]
    if len(a_s) >= 2 and len(w_s) == 2:
        out_shape = list(a_s[:-1]) + [w_s[0]]
        return [_make_ranked_type(tuple(out_shape), et)]
    return [_make_ranked_type((None, None), et)]


# ── Shape manipulation ───────────────────────────────────────


def infer_view(input_types: list[ir.Type], shape: tuple[int, ...] | None = None, **kwargs: Any) -> list[ir.Type]:
    if not input_types:
        return []
    et = _elt_type_str(input_types[0])
    if "shape" in kwargs:
        shape = kwargs["shape"]
    if shape:
        return [_make_ranked_type(tuple(shape), et)]
    return [input_types[0]]


def infer_unsqueeze(input_types: list[ir.Type], dim: int = 0, **kwargs: Any) -> list[ir.Type]:
    if not input_types:
        return []
    inp = input_types[0]
    s = _ranked_shape(inp)
    et = _elt_type_str(inp)
    if s is None:
        return [_make_ranked_type((None,), et)]
    parts = list(s)
    dim = int(dim)
    if dim < 0:
        dim = len(parts) + 1 + dim
    parts.insert(dim, 1)
    return [_make_ranked_type(tuple(parts), et)]


def infer_squeeze(input_types: list[ir.Type], dim: int | None = None, **kwargs: Any) -> list[ir.Type]:
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
    """Expand (broadcast) input to a larger shape.

    The output type is determined by the shape attribute and dyn_shape operands.
    New leading dims are inserted, -1 means "keep from input", explicit values
    set the dim size.  SSA-referenced dims are dynamic (?).
    """
    if not input_types:
        return []
    inp = input_types[0]
    s = _ranked_shape(inp)
    et = _elt_type_str(inp)
    if s is None:
        return [_make_ranked_type((None,), et)]
    # shape attr from kwargs contains the target shape with -1 for "keep"
    shape = kwargs.get("shape")
    if shape:
        out_dims: list[int | None] = []
        in_idx = len(shape) - len(s)  # leading dims are new
        for dim_entry in shape:
            if isinstance(dim_entry, int):
                if dim_entry == -1:
                    # Keep from input (right-aligned)
                    if in_idx < len(s):
                        val = s[in_idx]
                        out_dims.append(val)
                    else:
                        out_dims.append(None)
                    in_idx += 1
                else:
                    out_dims.append(dim_entry)
            else:
                # SSA reference → dynamic
                out_dims.append(None)
                in_idx += 1
        return [_make_ranked_type(tuple(out_dims), et)]
    return [inp]


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
    if not input_types:
        return []
    inp = input_types[0]
    s = _ranked_shape(inp)
    et = _elt_type_str(inp)
    if s and len(s) > max(int(dim0), int(dim1)):
        parts = list(s)
        parts[int(dim0)], parts[int(dim1)] = parts[int(dim1)], parts[int(dim0)]
        return [_make_ranked_type(tuple(parts), et)]
    return [inp]


def infer_slice(
    input_types: list[ir.Type],
    dim: int = 0, start: int = 0, end: int = -1, step: int = 1,
    **kwargs: Any,
) -> list[ir.Type]:
    if not input_types:
        return []
    inp = input_types[0]
    s = _ranked_shape(inp)
    et = _elt_type_str(inp)
    if s is None or dim is None:
        return [_make_ranked_type((None,), et)]
    dim = int(dim)
    if dim < len(s):
        parts = list(s)
        orig = parts[dim] if parts[dim] is not None else None
        if orig is not None:
            st = int(start) if start is not None else 0
            en = int(end) if end is not None and end >= 0 else orig
            parts[dim] = (en - st + step - 1) // step
        else:
            parts[dim] = None
        return [_make_ranked_type(tuple(parts), et)]
    return [inp]


def infer_select(
    input_types: list[ir.Type],
    dim: int = 0, index: int = 0,
    **kwargs: Any,
) -> list[ir.Type]:
    if not input_types:
        return []
    inp = input_types[0]
    s = _ranked_shape(inp)
    et = _elt_type_str(inp)
    if s is None:
        return [_make_ranked_type((None,), et)]
    dim = int(dim)
    if dim < len(s):
        parts = list(s)
        parts.pop(dim)
        return [_make_ranked_type(tuple(parts), et)]
    return [inp]


def infer_cat(input_types: list[ir.Type], dim: int = 0, **kwargs: Any) -> list[ir.Type]:
    if not input_types:
        return []
    et = _elt_type_str(input_types[0])
    shapes = [_ranked_shape(t) for t in input_types]
    if None in shapes or any(s is None for s in shapes):
        return [_make_ranked_type((None,), et)]
    dim = int(dim)
    parts = list(shapes[0])  # type: ignore[arg-type]
    total = 0
    for s in shapes:
        if s is not None and dim < len(s):
            d = s[dim]
            total += d if d is not None else 0
    if dim < len(parts):
        parts[dim] = total if total > 0 else None
    return [_make_ranked_type(tuple(parts), et)]


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


# ── Tensor creation / utility ────────────────────────────────


def infer_embedding(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    if len(input_types) < 2:
        return input_types
    w = input_types[0]   # weight tensor: [vocab, embed_dim]
    inp = input_types[1]  # indices tensor: [batch, seq, ...]
    w_s = _ranked_shape(w)
    inp_s = _ranked_shape(inp)
    et = _elt_type_str(w)
    if w_s and inp_s and len(w_s) >= 2:
        # Output: indices_shape + [embed_dim]
        return [_make_ranked_type(tuple(list(inp_s) + [w_s[1]]), et)]
    return [_make_ranked_type((None, None), et)]


def infer_triu(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    return _infer_elementwise(input_types)


def infer_tril(input_types: list[ir.Type], **kwargs: Any) -> list[ir.Type]:
    return _infer_elementwise(input_types)


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
    if input_types:
        return [input_types[0]]
    return []


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


def _infer_scalar_pure(
    shapes: list[tuple[int | None, ...]],
    elts: list[str],
    **kwargs: Any,
) -> list[tuple[tuple[int | None, ...], str]]:
    """SymSize: return scalar tensor (empty shape tuple) with f32 element type."""
    return [((), "f32")]


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
            en = int(end) if end is not None and end >= 0 else dim_val
            s[dim] = (en - st + int(step) - 1) // int(step)
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
    "new_ones": _infer_elementwise_pure,
    "diff": _infer_elementwise_pure,
    "index": _infer_elementwise_pure,
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
    "arange": _infer_elementwise_pure,
    "ones_like": _infer_elementwise_pure,
    "full_like": _infer_elementwise_pure,
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
