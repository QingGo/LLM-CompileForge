"""Per-op lowering dispatch — map ``sf.*`` ops to HAL IR entries.

Each ``sf.*`` operation is mapped to a ``hal.execute`` entry with the
appropriate op name, inputs, outputs, and attributes.

Lowering is stateless (pure dispatch) — all state (SSA tracker,
weights/constants accumulators) is passed explicitly.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from compiler.mlir_dialect.hal_ir.ssa_tracker import SSATracker

_log = logging.getLogger(__name__)


# ── Op mapping tables ───────────────────────────────────────────────

_UNARY_ARITH_MAP: dict[str, str] = {
    "sf.relu": "relu",
    "sf.gelu": "gelu",
    "sf.silu": "silu",
    "sf.sigmoid": "sigmoid",
    "sf.tanh": "tanh",
    "sf.exp": "exp",
    "sf.neg": "neg",
    "sf.softplus": "softplus",
    "sf.sqrt": "sqrt",
    "sf.rsqrt": "rsqrt",
    "sf.cos": "cos",
    "sf.sin": "sin",
}

_BINARY_ARITH_MAP: dict[str, str] = {
    "sf.add": "add",
    "sf.sub": "sub",
    "sf.mul": "mul",
    "sf.div": "div",
    "sf.pow": "pow",
    "sf.max": "max",
}

_COMPARE_MAP: dict[str, str] = {
    "sf.eq": "eq",
    "sf.ne": "ne",
    "sf.gt": "gt",
    "sf.lt": "lt",
    "sf.le": "le",
    "sf.ge": "ge",
    "sf.logical_and": "logical_and",
}


# ── Helper functions ────────────────────────────────────────────────


def strip_mlir_quotes(s: str) -> str:
    """Strip MLIR string attribute quotes from a value.

    MLIR ``StringAttr`` values print as ``"foo"`` (with quotes).
    """
    return s.strip().strip('"')


def parse_sf_op_name(raw: str) -> str:
    """Normalize ``sf.op_name`` regardless of quoting style.

    In the normalized IR, some ops appear as ``sf.add`` (bare) and others
    as ``"sf.add"`` (quoted string).  Both refer to the same op.
    """
    return strip_mlir_quotes(raw)


def parse_mlir_int_attr(attr_str: str | None) -> int | None:
    """Parse an MLIR integer attribute to a Python int.

    Handles formats like ``"0"``, ``"0 : i64"``, ``"1 : i64"``.
    """
    if attr_str is None:
        return None
    s = str(attr_str).strip()
    # Strip type suffix like " : i64"
    if " : " in s:
        s = s.split(" : ")[0]
    try:
        return int(s)
    except ValueError:
        return None


def infer_dtype_from_type(t: Any) -> str:
    """Infer a short dtype string from an MLIR type."""
    s = str(t)
    if "f32" in s or "float" in s:
        return "f32"
    if "f16" in s or "bfloat" in s:
        return "f16"
    if "i64" in s:
        return "i64"
    if "i32" in s:
        return "i32"
    if "i8" in s or "i1" in s or "bool" in s:
        return "i8"
    return "f32"


def shape_from_type(t: Any) -> list[int | str]:
    """Extract shape as a list of ints or '?' for dynamic dims from an MLIR type."""
    s = str(t)
    # Extract shape from tensor<...>
    m = re.search(r"tensor<(.+?)x", s)
    if not m:
        return []
    shape_str = s[s.index("<") + 1: s.rindex("x")]
    parts = shape_str.split("x")
    shape: list[int | str] = []
    for p in parts:
        p = p.strip()
        if p == "?":
            shape.append("?")
        else:
            try:
                shape.append(int(p))
            except ValueError:
                shape.append("?")
    return shape


def parse_attr_shape(shape_str: str) -> list[int | str]:
    """Parse an MLIR shape attribute like ``[-1, -1, 12, 64]``."""
    shape_str = shape_str.strip()
    if shape_str.startswith("["):
        shape_str = shape_str[1:]
    if shape_str.endswith("]"):
        shape_str = shape_str[:-1]
    parts = shape_str.split(",")
    shape: list[int | str] = []
    for p in parts:
        p = p.strip()
        if p in ("-1", "?"):
            shape.append("?")
        else:
            try:
                shape.append(int(p))
            except ValueError:
                shape.append("?")
    return shape


# ── Main lowering dispatch ──────────────────────────────────────────


def lower_op(
    op: Any,
    op_name: str,
    ssa: SSATracker,
    weights: list[dict[str, Any]],
    constants: list[dict[str, Any]],
    weight_index: dict[str, int],
    param_names: list[str],
    const_names: list[str],
) -> dict[str, Any] | None:
    """Lower a single sf.* op to a HAL IR entry.

    Returns ``None`` for ops that should be skipped (identity, constant, weight).

    Parameters
    ----------
    op:
        The MLIR operation (``OpView``).
    op_name:
        Parsed SF op name (e.g. ``"sf.matmul"``).
    ssa:
        SSA tracker for ``%name`` assignment.
    weights:
        Accumulator list of weight entries (mutated in-place).
    constants:
        Accumulator list of constant entries (mutated in-place).
    weight_index:
        Map from weight name → index into *weights*.
    param_names, const_names:
        Lists from weight classification metadata.
    """
    operands = list(op.operands) if hasattr(op, "operands") else []
    results = list(op.results) if hasattr(op, "results") else []

    # Get input %names
    input_names = [ssa.lookup(o) for o in operands]

    # Register results and get output %names
    output_names = [ssa.register_result(r) for r in results]

    # ── Ops that produce no HAL entry ────────────────────────────

    if op_name == "sf.weight":
        # Record weight — no hal op.  Result already registered above.
        name_attr = op.attributes.get("name", "")
        weight_name = strip_mlir_quotes(str(name_attr))
        if results:
            result_type = str(results[0].type) if hasattr(results[0], "type") else ""
            shape = shape_from_type(results[0].type) if hasattr(results[0], "type") else []
            dtype = infer_dtype_from_type(result_type)
            idx = len(weights)
            weight_index[weight_name] = idx
            weights.append({
                "name": weight_name,
                "shape": shape,
                "dtype": dtype,
                "hal_name": f"w{idx}",
            })
        return None

    if op_name == "sf.constant":
        # Inline constant value — result already registered above.
        return None

    if op_name == "sf.identity":
        # Skip — map result to same name as input operand
        if operands and results:
            input_name = ssa.lookup(operands[0])
            for r in results:
                result_key = ssa._val_key(r)
                ssa._val_to_name[result_key] = input_name
        return None

    # ── Shape / metadata ops ─────────────────────────────────────

    if op_name == "sf.sym_size":
        return {
            "op": "shape_of",
            "inputs": input_names,
            "outputs": output_names,
        }

    if op_name == "sf.view" or op_name == "sf.expand":
        shape_attr = op.attributes.get("shape")
        shape_val: list[int | str] = []
        if shape_attr is not None:
            shape_str = str(shape_attr)
            shape_val = parse_attr_shape(shape_str)
        entry: dict[str, Any] = {
            "op": "reshape",
            "inputs": input_names,
            "outputs": output_names,
        }
        if shape_val:
            entry["shape"] = shape_val
        return entry

    if op_name == "sf.unsqueeze":
        dim = parse_mlir_int_attr(
            str(op.attributes.get("dim")) if op.attributes.get("dim") is not None else None
        ) or 0
        return {
            "op": "unsqueeze",
            "inputs": input_names,
            "outputs": output_names,
            "dim": dim,
        }

    if op_name == "sf.transpose":
        dim0 = parse_mlir_int_attr(
            str(op.attributes.get("dim0")) if op.attributes.get("dim0") is not None else None
        ) or 0
        dim1 = parse_mlir_int_attr(
            str(op.attributes.get("dim1")) if op.attributes.get("dim1") is not None else None
        ) or 1
        return {
            "op": "transpose",
            "inputs": input_names,
            "outputs": output_names,
            "dims": [dim0, dim1],
        }

    if op_name == "sf.slice":
        dim = parse_mlir_int_attr(
            str(op.attributes.get("dim")) if op.attributes.get("dim") is not None else None
        ) or 0
        start = parse_mlir_int_attr(
            str(op.attributes.get("start")) if op.attributes.get("start") is not None else None
        ) or 0
        end_raw_attr = op.attributes.get("end")
        end_raw = str(end_raw_attr) if end_raw_attr is not None else "MAX"
        # MLIR uses MAX_INT for "to end"
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

    if op_name == "sf.cat":
        dim = parse_mlir_int_attr(
            str(op.attributes.get("dim")) if op.attributes.get("dim") is not None else None
        ) or 0
        return {
            "op": "concat",
            "inputs": input_names,
            "outputs": output_names,
            "dim": dim,
        }

    if op_name == "sf.mean":
        return {
            "op": "reduce",
            "inputs": input_names,
            "outputs": output_names,
            "kind": "mean",
        }

    if op_name == "sf.sum":
        return {
            "op": "reduce",
            "inputs": input_names,
            "outputs": output_names,
            "kind": "sum",
        }

    # ── Arithmetic / compare ops (via maps) ──────────────────────

    if op_name in _BINARY_ARITH_MAP:
        return {
            "op": "element_wise",
            "inputs": input_names,
            "outputs": output_names,
            "kind": _BINARY_ARITH_MAP[op_name],
        }

    if op_name in _UNARY_ARITH_MAP:
        return {
            "op": "element_wise",
            "inputs": input_names,
            "outputs": output_names,
            "kind": _UNARY_ARITH_MAP[op_name],
        }

    if op_name in _COMPARE_MAP:
        return {
            "op": "compare",
            "inputs": input_names,
            "outputs": output_names,
            "kind": _COMPARE_MAP[op_name],
        }

    # ── Named ops (one-per-type) ─────────────────────────────────

    if op_name == "sf.embedding":
        return {
            "op": "gather",
            "inputs": input_names,
            "outputs": output_names,
        }

    if op_name == "sf.index":
        return {
            "op": "gather",
            "inputs": input_names,
            "outputs": output_names,
            "mode": "indexed",
        }

    if op_name == "sf.matmul":
        return {
            "op": "matmul",
            "inputs": input_names,
            "outputs": output_names,
        }

    if op_name == "sf.softmax":
        return {
            "op": "softmax",
            "inputs": input_names,
            "outputs": output_names,
        }

    if op_name == "sf.ones_like" or op_name == "sf.new_ones":
        dtype_attr = op.attributes.get("dtype")
        raw_dtype = str(dtype_attr) if dtype_attr is not None else "f32"
        dtype_str = strip_mlir_quotes(raw_dtype)
        return {
            "op": "fill",
            "inputs": input_names,
            "outputs": output_names,
            "value": 1.0,
            "dtype": dtype_str,
        }

    if op_name == "sf.arange":
        return {
            "op": "fill",
            "inputs": input_names,
            "outputs": output_names,
            "kind": "arange",
        }

    if op_name == "sf.rms_norm":
        return {
            "op": "rms_norm",
            "inputs": input_names,
            "outputs": output_names,
        }

    if op_name == "sf.layer_norm":
        return {
            "op": "layer_norm",
            "inputs": input_names,
            "outputs": output_names,
        }

    if op_name == "sf.silu":
        return {
            "op": "element_wise",
            "inputs": input_names,
            "outputs": output_names,
            "kind": "silu",
        }

    if op_name == "sf.relu":
        return {
            "op": "element_wise",
            "inputs": input_names,
            "outputs": output_names,
            "kind": "relu",
        }

    if op_name == "sf.cumsum":
        return {
            "op": "scan",
            "inputs": input_names,
            "outputs": output_names,
            "kind": "cumsum",
        }

    if op_name == "sf.gelu":
        return {
            "op": "element_wise",
            "inputs": input_names,
            "outputs": output_names,
            "kind": "gelu",
        }

    if op_name == "sf.sigmoid":
        return {
            "op": "element_wise",
            "inputs": input_names,
            "outputs": output_names,
            "kind": "sigmoid",
        }

    # Unknown op — skip non-SF ops silently, warn for unrecognized SF ops
    if op_name in ("func.return", "return"):
        return None
    _log.warning("Unknown SF op %s in function, passing through", op_name)
    return {
        "op": op_name.removeprefix("sf."),
        "inputs": input_names,
        "outputs": output_names,
    }
