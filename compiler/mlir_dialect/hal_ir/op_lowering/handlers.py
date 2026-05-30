"""Per-op handler functions for lowering ``sf.*`` ops to HAL IR.

Each handler takes ``(op, op_name, input_names, output_names, *context)``
where *context unpacks to ``(ssa, weights, constants, weight_index,
param_names, const_names)`` — matching the prelude-computed values from
``lower_op()`` in ``core.py``.
"""

from __future__ import annotations

from collections.abc import Callable

from compiler.mlir_dialect.hal_ir.op_lowering.core import (
    _BINARY_ARITH_MAP,
    _COMPARE_MAP,
    _UNARY_ARITH_MAP,
    infer_dtype_from_type,
    parse_attr_shape,
    parse_mlir_int_attr,
    shape_from_type,
    strip_mlir_quotes,
)

# ── Ops that produce no HAL entry ─────────────────────────────────────


def _handle_weight(op, op_name, input_names, output_names, *context):
    """Record weight — no hal op. Mutates *context* lists/dicts."""
    ssa, weights, constants, weight_index, param_names, const_names = context  # noqa: F841
    name_attr = op.attributes.get("name", "")
    weight_name = strip_mlir_quotes(str(name_attr))
    results = list(op.results) if hasattr(op, "results") else []
    if results:
        result_type = str(results[0].type) if hasattr(results[0], "type") else ""
        shape = shape_from_type(results[0].type) if hasattr(results[0], "type") else []
        dtype = infer_dtype_from_type(result_type)
        idx = len(weights)
        weight_index[weight_name] = idx
        ssa_name = output_names[0] if output_names else ""
        weights.append({
            "name": weight_name,
            "shape": shape,
            "dtype": dtype,
            "hal_name": f"w{idx}",
            "ssa": ssa_name,
        })
    return None


def _handle_constant(op, op_name, input_names, output_names, *context):
    """Inline constant — no hal op."""
    return None


def _handle_identity(op, op_name, input_names, output_names, *context):
    """Skip — map result to same name as input operand."""
    ssa = context[0]
    operands = list(op.operands) if hasattr(op, "operands") else []
    results = list(op.results) if hasattr(op, "results") else []
    if operands and results:
        input_name = ssa.lookup(operands[0])
        for r in results:
            result_key = ssa._val_key(r)
            ssa._val_to_name[result_key] = input_name
    return None


# ── Shape / metadata ops ────────────────────────────────────────────


def _handle_sym_size(op, op_name, input_names, output_names, *context):
    return {"op": "shape_of", "inputs": input_names, "outputs": output_names}


def _is_shape_of_output(value: object) -> bool:
    """Check if an MLIR Value is produced by ``sf.sym_size`` (shape_of).

    Uses ``.owner`` (the MLIR Python API property) rather than
    ``get_defining_op()`` which does not exist on OpResult in this
    MLIR version.  Without this fix, ALL operands silently fail the
    check, causing ``_handle_view_expand`` to drop shape_of inputs
    from the reshape op — making the causal mask [1,1,1,1] instead
    of [1,1,seq,seq].
    """
    try:
        owner = value.owner  # type: ignore[attr-defined]
        if owner is None:
            return False
        return str(owner.operation.name) == "sf.sym_size"  # type: ignore[attr-defined]
    except Exception:
        return False


def _filter_shape_inputs(op: object, input_names: list[str]) -> list[str]:
    """Keep only shape_of outputs from op operands (drop scalar weights)."""
    operands = list(op.operands) if hasattr(op, "operands") else []  # type: ignore[union-attr]
    filtered = []
    for i, operand in enumerate(operands):
        if i < len(input_names) and _is_shape_of_output(operand):
            filtered.append(input_names[i])
    return filtered


def _handle_view_expand(op, op_name, input_names, output_names, *context):
    shape_attr = op.attributes.get("shape")
    shape_val = parse_attr_shape(str(shape_attr)) if shape_attr is not None else []

    filtered_inputs = [input_names[0]]
    operands = list(op.operands) if hasattr(op, "operands") else []
    for i in range(1, len(operands)):
        if i < len(input_names) and _is_shape_of_output(operands[i]):
            filtered_inputs.append(input_names[i])

    entry = {"op": "reshape", "inputs": filtered_inputs, "outputs": output_names}
    if shape_val:
        entry["shape"] = shape_val
    return entry


def _handle_unsqueeze(op, op_name, input_names, output_names, *context):
    dim = (
        parse_mlir_int_attr(
            str(op.attributes.get("dim")) if op.attributes.get("dim") is not None else None
        )
        or 0
    )
    return {"op": "unsqueeze", "inputs": input_names, "outputs": output_names, "dim": dim}


def _handle_transpose(op, op_name, input_names, output_names, *context):
    dim0 = (
        parse_mlir_int_attr(
            str(op.attributes.get("dim0")) if op.attributes.get("dim0") is not None else None
        )
        or 0
    )
    dim1 = (
        parse_mlir_int_attr(
            str(op.attributes.get("dim1")) if op.attributes.get("dim1") is not None else None
        )
        or 1
    )
    return {"op": "transpose", "inputs": input_names, "outputs": output_names, "dims": [dim0, dim1]}


def _handle_slice(op, op_name, input_names, output_names, *context):
    dim = (
        parse_mlir_int_attr(
            str(op.attributes.get("dim")) if op.attributes.get("dim") is not None else None
        )
        or 0
    )
    start = (
        parse_mlir_int_attr(
            str(op.attributes.get("start")) if op.attributes.get("start") is not None else None
        )
        or 0
    )
    end_raw_attr = op.attributes.get("end")
    end_raw = str(end_raw_attr) if end_raw_attr is not None else "MAX"
    parsed_end = parse_mlir_int_attr(end_raw)
    if parsed_end is not None and abs(parsed_end) > 1_000_000_000:
        end_raw = "MAX"
    return {
        "op": "slice",
        "inputs": input_names,
        "outputs": output_names,
        "dim": dim,
        "start": start,
        "end": end_raw,
    }


def _handle_cat(op, op_name, input_names, output_names, *context):
    dim = (
        parse_mlir_int_attr(
            str(op.attributes.get("dim")) if op.attributes.get("dim") is not None else None
        )
        or 0
    )
    return {"op": "concat", "inputs": input_names, "outputs": output_names, "dim": dim}


def _handle_mean(op, op_name, input_names, output_names, *context):
    return {"op": "reduce", "inputs": input_names, "outputs": output_names, "kind": "mean"}


def _handle_sum(op, op_name, input_names, output_names, *context):
    return {"op": "reduce", "inputs": input_names, "outputs": output_names, "kind": "sum"}


# ── Named ops ────────────────────────────────────────────────────────


def _handle_embedding(op, op_name, input_names, output_names, *context):
    # Look up the weight name from the first input's SSA.
    # This makes the weight reference explicit in the HAL IR so the
    # runtime can verify the correct weight table is used for each gather.
    _, weights, _, _, _, _ = context
    weight_name = ""
    if input_names:
        weight_ssa = input_names[0]
        weight_entry = next((w for w in weights if w.get("ssa") == weight_ssa), None)
        if weight_entry is not None:
            weight_name = weight_entry["name"]
    return {
        "op": "gather", "inputs": input_names, "outputs": output_names,
        "weight_name": weight_name,
    }


def _handle_index(op, op_name, input_names, output_names, *context):
    return {"op": "gather", "inputs": input_names, "outputs": output_names, "mode": "indexed"}


def _handle_matmul(op, op_name, input_names, output_names, *context):
    return {"op": "matmul", "inputs": input_names, "outputs": output_names}


def _handle_softmax(op, op_name, input_names, output_names, *context):
    return {"op": "softmax", "inputs": input_names, "outputs": output_names}


def _handle_ones_like(op, op_name, input_names, output_names, *context):
    dtype_attr = op.attributes.get("dtype")
    raw_dtype = str(dtype_attr) if dtype_attr is not None else "f32"
    dtype_str = strip_mlir_quotes(raw_dtype)
    filtered_inputs = _filter_shape_inputs(op, input_names)
    return {
        "op": "fill", "inputs": filtered_inputs, "outputs": output_names,
        "value": 1.0, "dtype": dtype_str,
    }


def _handle_arange(op, op_name, input_names, output_names, *context):
    return {"op": "fill", "inputs": input_names, "outputs": output_names, "kind": "arange"}


def _handle_rms_norm(op, op_name, input_names, output_names, *context):
    return {"op": "rms_norm", "inputs": input_names, "outputs": output_names}


def _handle_layer_norm(op, op_name, input_names, output_names, *context):
    return {"op": "layer_norm", "inputs": input_names, "outputs": output_names}


def _handle_cumsum(op, op_name, input_names, output_names, *context):
    return {"op": "scan", "inputs": input_names, "outputs": output_names, "kind": "cumsum"}


# ── Dispatch table builder ──────────────────────────────────────────


def register_handlers(dispatch: dict[str, Callable]) -> None:
    """Populate *dispatch* with op-name → handler mappings."""
    # Ops that produce no HAL entry
    dispatch["sf.weight"] = _handle_weight
    dispatch["sf.constant"] = _handle_constant
    dispatch["sf.identity"] = _handle_identity

    # Shape / metadata ops
    dispatch["sf.sym_size"] = _handle_sym_size
    dispatch["sf.view"] = _handle_view_expand
    dispatch["sf.expand"] = _handle_view_expand
    dispatch["sf.unsqueeze"] = _handle_unsqueeze
    dispatch["sf.transpose"] = _handle_transpose
    dispatch["sf.slice"] = _handle_slice
    dispatch["sf.cat"] = _handle_cat
    dispatch["sf.mean"] = _handle_mean
    dispatch["sf.sum"] = _handle_sum

    # Arithmetic / compare ops (via maps)
    for sf_name, kind in _BINARY_ARITH_MAP.items():
        dispatch[sf_name] = (
            lambda op, on, ins, outs, *ctx, kind=kind: {
                "op": "element_wise", "inputs": ins, "outputs": outs, "kind": kind,
            }
        )
    for sf_name, kind in _UNARY_ARITH_MAP.items():
        dispatch[sf_name] = (
            lambda op, on, ins, outs, *ctx, kind=kind: {
                "op": "element_wise", "inputs": ins, "outputs": outs, "kind": kind,
            }
        )
    for sf_name, kind in _COMPARE_MAP.items():
        dispatch[sf_name] = (
            lambda op, on, ins, outs, *ctx, kind=kind: {
                "op": "compare", "inputs": ins, "outputs": outs, "kind": kind,
            }
        )

    # Named ops
    dispatch["sf.embedding"] = _handle_embedding
    dispatch["sf.index"] = _handle_index
    dispatch["sf.matmul"] = _handle_matmul
    dispatch["sf.softmax"] = _handle_softmax
    dispatch["sf.ones_like"] = _handle_ones_like
    dispatch["sf.new_ones"] = _handle_ones_like
    dispatch["sf.arange"] = _handle_arange
    dispatch["sf.rms_norm"] = _handle_rms_norm
    dispatch["sf.layer_norm"] = _handle_layer_norm
    dispatch["sf.cumsum"] = _handle_cumsum
