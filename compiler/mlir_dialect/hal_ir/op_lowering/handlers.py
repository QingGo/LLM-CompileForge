"""Per-op handler functions for lowering ``sf.*`` ops to HAL IR.

Each handler takes ``(op, op_name, input_names, output_names, *context)``
where *context unpacks to ``(ssa, weights, constants, weight_index,
param_names, const_names)`` — matching the prelude-computed values from
``lower_op()`` in ``core.py``.

Handlers return :class:`MlirOp` or ``None`` (for ops that should be skipped).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from compiler.mlir_artifact import MlirOp  # type: ignore[attr-defined]
from compiler.mlir_dialect.hal_ir.op_lowering.core import (
    _COMPARE_MAP,
    _ELEMENT_WISE_MAP,
    _get_int_attr,
    infer_dtype_from_type,
    parse_attr_shape,
    shape_from_type,
    strip_mlir_quotes,
)

# ── Handler factory ──────────────────────────────────────────────────


def _make_mapped_handler(hal_name: str, **fixed_attrs: Any) -> Callable[..., Any]:
    """Factory for simple MlirOp-returning handler functions."""

    def handler(
        op: Any, op_name: str, input_names: list[str], output_names: list[str],
        *context: Any,
    ) -> MlirOp:
        return MlirOp(
            name=f"hal.{hal_name}", dialect="hal", op_name=hal_name,
            operands=input_names, results=output_names,
            attributes=dict(fixed_attrs) if fixed_attrs else {},
        )

    return handler


# ── Ops that produce no HAL entry ─────────────────────────────────────


def _handle_weight(op: Any, op_name: str, input_names: list[str], output_names: list[str], *context: Any) -> MlirOp | None:  # noqa: E501
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


def _handle_constant(op: Any, op_name: str, input_names: list[str], output_names: list[str], *context: Any) -> MlirOp | None:  # noqa: E501
    """Inline constant — no hal op."""
    return None


def _handle_identity(op: Any, op_name: str, input_names: list[str], output_names: list[str], *context: Any) -> MlirOp | None:  # noqa: E501
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


# ── Shape-of filtering helpers ───────────────────────────────────────


def _is_shape_of_output(value: object) -> bool:
    """Check if an MLIR Value is produced by ``sf.sym_size`` (shape_of)."""
    try:
        owner = value.owner  # type: ignore[attr-defined]
        if owner is None:
            return False
        return str(owner.operation.name) == "sf.sym_size"
    except Exception:
        return False


def _filter_shape_inputs(op: object, input_names: list[str]) -> list[str]:
    """Keep only shape_of outputs from op operands (drop scalar weights)."""
    operands = list(op.operands) if hasattr(op, "operands") else []
    filtered = []
    for i, operand in enumerate(operands):
        if i < len(input_names) and _is_shape_of_output(operand):
            filtered.append(input_names[i])
    return filtered


# ── Shape / metadata ops ────────────────────────────────────────────

_handle_sym_size = _make_mapped_handler("shape_of")


def _handle_view_expand(op: Any, op_name: str, input_names: list[str], output_names: list[str], *context: Any) -> MlirOp | None:  # noqa: E501
    shape_attr = op.attributes.get("shape")
    shape_val = parse_attr_shape(str(shape_attr)) if shape_attr is not None else []

    filtered_inputs = [input_names[0]]
    operands = list(op.operands) if hasattr(op, "operands") else []
    for i in range(1, len(operands)):
        if i < len(input_names) and _is_shape_of_output(operands[i]):
            filtered_inputs.append(input_names[i])

    attrs = {}
    if shape_val:
        attrs["shape"] = shape_val
    return MlirOp(
        name="hal.reshape", dialect="hal", op_name="reshape",
        operands=filtered_inputs, results=output_names,
        attributes=attrs,
    )


def _handle_unsqueeze(op: Any, op_name: str, input_names: list[str], output_names: list[str], *context: Any) -> MlirOp | None:  # noqa: E501
    dim = _get_int_attr(op, "dim")
    return MlirOp(
        name="hal.unsqueeze", dialect="hal", op_name="unsqueeze",
        operands=input_names, results=output_names,
        attributes={"dim": dim},
    )


def _handle_transpose(op: Any, op_name: str, input_names: list[str], output_names: list[str], *context: Any) -> MlirOp | None:  # noqa: E501
    dim0 = _get_int_attr(op, "dim0")
    dim1 = _get_int_attr(op, "dim1", default=1)
    return MlirOp(
        name="hal.transpose", dialect="hal", op_name="transpose",
        operands=input_names, results=output_names,
        attributes={"dims": [dim0, dim1]},
    )


def _handle_slice(op: Any, op_name: str, input_names: list[str], output_names: list[str], *context: Any) -> MlirOp | None:  # noqa: E501
    dim = _get_int_attr(op, "dim")
    start = _get_int_attr(op, "start")
    end_raw_attr = op.attributes.get("end")
    end_raw = str(end_raw_attr) if end_raw_attr is not None else "MAX"
    parsed_end = _get_int_attr(op, "end")
    if parsed_end is not None and abs(parsed_end) > 1_000_000_000:
        end_raw = "MAX"
    return MlirOp(
        name="hal.slice", dialect="hal", op_name="slice",
        operands=input_names, results=output_names,
        attributes={"dim": dim, "start": start, "end": end_raw},
    )


def _handle_cat(op: Any, op_name: str, input_names: list[str], output_names: list[str], *context: Any) -> MlirOp | None:
    dim = _get_int_attr(op, "dim")
    return MlirOp(
        name="hal.concat", dialect="hal", op_name="concat",
        operands=input_names, results=output_names,
        attributes={"dim": dim},
    )


_handle_mean = _make_mapped_handler("reduce", kind="mean")
_handle_sum = _make_mapped_handler("reduce", kind="sum")


# ── Named ops ────────────────────────────────────────────────────────


def _handle_embedding(op: Any, op_name: str, input_names: list[str], output_names: list[str], *context: Any) -> MlirOp | None:  # noqa: E501
    _, weights, _, _, _, _ = context
    weight_name = ""
    if input_names:
        weight_ssa = input_names[0]
        weight_entry = next((w for w in weights if w.get("ssa") == weight_ssa), None)
        if weight_entry is not None:
            weight_name = weight_entry["name"]
    return MlirOp(
        name="hal.gather", dialect="hal", op_name="gather",
        operands=input_names, results=output_names,
        attributes={"weight_name": weight_name},
    )


_handle_index = _make_mapped_handler("gather", mode="indexed")
_handle_matmul = _make_mapped_handler("matmul")
_handle_softmax = _make_mapped_handler("softmax")


def _handle_ones_like(op: Any, op_name: str, input_names: list[str], output_names: list[str], *context: Any) -> MlirOp | None:  # noqa: E501
    dtype_attr = op.attributes.get("dtype")
    raw_dtype = str(dtype_attr) if dtype_attr is not None else "f32"
    dtype_str = strip_mlir_quotes(raw_dtype)
    filtered_inputs = _filter_shape_inputs(op, input_names)
    return MlirOp(
        name="hal.fill", dialect="hal", op_name="fill",
        operands=filtered_inputs, results=output_names,
        attributes={"value": 1.0, "dtype": dtype_str},
    )


_handle_arange = _make_mapped_handler("fill", kind="arange")
_handle_rms_norm = _make_mapped_handler("rms_norm")
_handle_layer_norm = _make_mapped_handler("layer_norm")
_handle_cumsum = _make_mapped_handler("scan", kind="cumsum")


# ── Dispatch table builder ──────────────────────────────────────────


def register_handlers(dispatch: dict[str, Callable[..., Any]]) -> None:
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

    # Arithmetic / compare ops (single combined loop)
    _op_table = {
        **{k: ("element_wise", v) for k, v in _ELEMENT_WISE_MAP.items()},
        **{k: ("compare", v) for k, v in _COMPARE_MAP.items()},
    }
    for sf_name, (hal_op, kind) in _op_table.items():
        dispatch[sf_name] = _make_mapped_handler(hal_op, kind=kind)

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
