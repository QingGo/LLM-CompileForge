"""MLIR IR building — construct ir.Module from MlirModule.

Section D: IR building functions from the original mlir_artifact.py.

These functions create a valid ``ir.Module`` that can be passed directly to
PassManager-based passes, avoiding the MLIR text round-trip entirely.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import mlir.ir as ir

from compiler.mlir_dialect.mlir_op_types import (
    MlirFunction,
    MlirModule,
    MlirOp,
)

_log = logging.getLogger(__name__)


# ── Internal parsing ──────────────────────────────────────────

_ELT_MAP: dict[str, ir.Type] = {}
_DYNAMIC_DIM: int | None = None


def _get_dynamic_dim() -> int:
    global _DYNAMIC_DIM
    if _DYNAMIC_DIM is None:
        import mlir.ir as _ir
        _DYNAMIC_DIM = _ir.ShapedType.get_dynamic_size()
    return _DYNAMIC_DIM


def _init_elt_map() -> dict[str, ir.Type]:
    global _ELT_MAP
    if _ELT_MAP:
        return _ELT_MAP
    import mlir.ir as _ir
    _ELT_MAP = {
        "f32": _ir.F32Type.get(), "f64": _ir.F64Type.get(),
        "f16": _ir.F16Type.get(), "bf16": _ir.BF16Type.get(),
        "i1": _ir.IntegerType.get_signless(1),
        "i8": _ir.IntegerType.get_signless(8),
        "i32": _ir.IntegerType.get_signless(32),
        "i64": _ir.IntegerType.get_signless(64),
        "ui8": _ir.IntegerType.get_unsigned(8),
    }
    return _ELT_MAP


def _type_str_to_ir_type(type_str: str) -> ir.Type:
    """Convert MLIR type string to ir.Type object.

    tensor<1x64xf32>  → RankedTensorType([1, 64], f32)
    tensor<f32>        → RankedTensorType([], f32)  (scalar tensor)
    tensor<?x64xf32>   → RankedTensorType([dynamic], f32)
    """
    import re as _re

    import mlir.ir as _ir

    elt_map = _init_elt_map()

    m = _re.match(r"tensor<\s*(.+?)\s*>$", type_str.strip())
    if not m:
        elt = elt_map.get(type_str.strip())
        if elt is not None:
            return _ir.UnrankedTensorType.get(elt)
        return _ir.UnrankedTensorType.get(_ir.F32Type.get())

    inner = m.group(1).strip()
    parts = inner.split("x")
    dtype_str = parts[-1].strip()
    elt_type: ir.Type = elt_map.get(dtype_str, _ir.F32Type.get())
    dim_strs = parts[:-1]

    for d in dim_strs:
        d = d.strip()
        if d == "*":
            return _ir.UnrankedTensorType.get(elt_type)

    dyn_dim = _get_dynamic_dim()
    dims: list[int] = []
    for d in dim_strs:
        d = d.strip()
        try:
            dims.append(int(d))
        except ValueError:
            dims.append(dyn_dim)

    if not dims:
        return _ir.RankedTensorType.get([], elt_type)
    try:
        return _ir.RankedTensorType.get(dims, elt_type)
    except Exception as e:
        _log.warning(
            "Failed to create RankedTensorType for '%s': dims=%s, elt=%s",
            type_str, dims, elt_type, exc_info=True,
        )
        raise RuntimeError(
            f"Failed to create RankedTensorType for '{type_str}': dims={dims}, elt={elt_type}"
        ) from e


def _build_mlir_function(func: MlirFunction, ir_mod: Any, ctx: Any) -> tuple[Any, Any, list[Any]]:
    """Create a func.func op and its entry block.

    Returns (func_op, body_blk, arg_values).
    """
    import mlir.ir as _ir

    arg_types: list[ir.Type] = []
    for _, tp in func.inputs:
        arg_types.append(_type_str_to_ir_type(tp))

    func_type = _ir.FunctionType.get(arg_types, [])
    func_op = _ir.Operation.create(
        "func.func",
        attributes={
            "function_type": _ir.TypeAttr.get(func_type),
            "sym_name": _ir.StringAttr.get(func.name),
        },
        regions=1,
    )
    ir_mod.body.append(func_op.operation)

    body_region = func_op.operation.regions[0]
    body_blk = _ir.Block.create_at_start(body_region, arg_types)
    arg_values: list[ir.Value] = list(body_blk.arguments)
    return func_op, body_blk, arg_values


def _build_ssa_map(func: MlirFunction, arg_values: list[ir.Value]) -> dict[str, ir.Value]:
    """Build SSA name → ir.Value mapping from function inputs."""

    ssa_map: dict[str, ir.Value] = {}
    for i, (name, _) in enumerate(func.inputs):
        ssa_map[name] = arg_values[i]
        ssa_map[name.lstrip("%")] = arg_values[i]
    return ssa_map


def _emit_weight_op(op: MlirOp, ctx: Any, ssa_map: dict[str, ir.Value], body_blk: Any,
                     weights: dict | None = None) -> None:
    """Emit a weight/constant op into the IR builder.

    For scalar constants (_const_*) with known values, emits ``arith.constant``
    instead of ``sf.weight``, baking the value into the IR so the MLIR pipeline
    cannot DCE scalar additions (Issue #45 root cause fix).
    """
    import mlir.ir as _ir

    wname = op.attributes.get("name")
    if wname and wname.startswith("_const_") and weights and wname in weights:
        wt = weights[wname]
        if wt.numel() == 1:
            result_type = _type_str_to_ir_type(op.output_types[0])
            elt_type = result_type.element_type
            with _ir.InsertionPoint(body_blk):
                if isinstance(wt.item(), float):
                    elt_attr = _ir.FloatAttr.get(elt_type, float(wt.item()))
                else:
                    elt_attr = _ir.IntegerAttr.get(elt_type, int(wt.item()))
                attr = _ir.DenseElementsAttr.get_splat(result_type, elt_attr)
                ir_op = _ir.Operation.create("arith.constant",
                    results=[result_type],
                    attributes={"value": attr})
            for i, rname in enumerate(op.results):
                if i < len(ir_op.operation.results):
                    v = ir_op.operation.results[i]
                    ssa_map[rname] = v
                    ssa_map[rname.lstrip("%")] = v
            if wname:
                ssa_map[wname] = ir_op.operation.results[0]
            return

    w_attrs: dict[str, ir.Attribute] = {}
    for k, v in op.attributes.items():
        if k == "source_node":
            continue
        w_attrs[k] = _python_to_attr_ir(v)

    if op.output_types:
        w_result_types = [_type_str_to_ir_type(t) for t in op.output_types]
    else:
        w_result_types = [_ir.UnrankedTensorType.get(_ir.F32Type.get(ctx))]

    with _ir.InsertionPoint(body_blk):
        loc = _ir.Location.unknown(ctx)
        if "dump_layer" in op.attributes:
            loc = _ir.Location.name(str(op.attributes["dump_layer"]), loc)
        ir_op = _ir.Operation.create(
            op.name,
            results=w_result_types,
            attributes=w_attrs if w_attrs else {},
            loc=loc,
        )
    for i, rname in enumerate(op.results):
        if i < len(ir_op.operation.results):
            val = ir_op.operation.results[i]
            ssa_map[rname] = val
            ssa_map[rname.lstrip("%")] = val
    wname = op.attributes.get("name")
    if wname:
        ssa_map[wname] = ir_op.operation.results[0]


def _resolve_operands(op: MlirOp, ssa_map: dict[str, ir.Value]) -> list[ir.Value]:
    """Resolve SSA operands for a compute op."""
    operands: list[ir.Value] = []
    for o in op.operands:
        key = o
        if key in ssa_map:
            operands.append(ssa_map[key])
        elif key.lstrip("%") in ssa_map:
            operands.append(ssa_map[key.lstrip("%")])
        else:
            raise KeyError(
                f"ssa_map missing operand '{key}' for op '{op.name}'. "
                f"Known: {list(ssa_map.keys())[:10]}"
            )
    return operands


def _infer_result_types(op: MlirOp, operands: list[ir.Value], ctx: Any) -> list[ir.Type]:
    """Infer IR result types for a compute op."""
    import mlir.ir as _ir

    if op.output_types:
        return [_type_str_to_ir_type(t) for t in op.output_types]
    if op.op_name == "sym_size":
        return [_ir.RankedTensorType.get([], _ir.F32Type.get(ctx))]
    if operands:
        opnd_type = operands[0].type
        try:
            _rank = len(opnd_type.shape)
            return [opnd_type]
        except Exception:
            _log.warning("Shape inference failed for operand type, trying element type", exc_info=True)
            try:
                elt = opnd_type.element_type
            except Exception:
                _log.warning("Element type inference failed, defaulting to F32", exc_info=True)
                elt = _ir.F32Type.get(ctx)
            return [_ir.RankedTensorType.get([1], elt)]
    return [_ir.RankedTensorType.get([1], _ir.F32Type.get(ctx))]


def _build_mlir_attrs(op: MlirOp) -> dict[str, ir.Attribute]:
    """Build MLIR attributes dict from an MlirOp."""

    mlir_attrs: dict[str, ir.Attribute] = {}
    for k, v in op.attributes.items():
        if k == "source_node":
            continue
        mlir_attrs[k] = _python_to_attr_ir(v)
    return mlir_attrs


def _emit_compute_op(op: MlirOp, operands: list[ir.Value], body_blk: Any, ctx: Any) -> Any:
    """Emit a compute op into the IR builder. Returns the ir.Operation."""
    import mlir.ir as _ir

    mlir_attrs = _build_mlir_attrs(op)
    result_types = _infer_result_types(op, operands, ctx)
    loc = _ir.Location.unknown(ctx)
    if "dump_layer" in op.attributes:
        loc = _ir.Location.name(str(op.attributes["dump_layer"]), loc)

    try:
        with _ir.InsertionPoint(body_blk):
            ir_op = _ir.Operation.create(
                op.name,
                operands=operands,
                results=result_types,
                attributes=mlir_attrs if mlir_attrs else {},
                loc=loc,
            )
    except Exception as e:
        _log.warning("Failed to build op '%s': %s", op.name, e, exc_info=True)
        raise RuntimeError(
            f"Failed to build op '{op.name}' (result '{op.results[0] if op.results else '?'}'): "
            f"output_types={op.output_types}, operands_count={len(op.operands)}, "
            f"attr_keys={list(op.attributes.keys())[:5]}, error={e}"
        ) from e
    return ir_op


def _map_op_results(op: MlirOp, ir_op: Any, ssa_map: dict[str, ir.Value]) -> None:
    """Map IR op results into the SSA map."""
    for i, rname in enumerate(op.results):
        if i < len(ir_op.operation.results):
            val = ir_op.operation.results[i]
            ssa_map[rname] = val
            ssa_map[rname.lstrip("%")] = val


def _resolve_output_values(func: MlirFunction, ssa_map: dict[str, ir.Value]) -> list[ir.Value]:
    """Resolve function output values from SSA map."""
    output_values: list[ir.Value] = []
    for out_name, _, _ in func.outputs:
        if out_name in ssa_map:
            output_values.append(ssa_map[out_name])
        elif out_name.lstrip("%") in ssa_map:
            output_values.append(ssa_map[out_name.lstrip("%")])

    if not output_values and func.ops:
        last_op = func.ops[-1]
        if last_op.results:
            rname = last_op.results[-1]
            if rname in ssa_map:
                output_values.append(ssa_map[rname])
    return output_values


def _build_return_op(output_values: list[ir.Value], body_blk: Any) -> None:
    """Emit a func.return op."""
    import mlir.ir as _ir

    with _ir.InsertionPoint(body_blk):
        _ir.Operation.create("func.return", operands=output_values)


def _update_function_type(func_op: Any, arg_values: list[ir.Value], output_values: list[ir.Value]) -> None:
    """Update function signature with actual return types."""
    import mlir.ir as _ir

    ret_types = [v.type for v in output_values]
    arg_types_list = [a.type for a in arg_values]
    new_func_type = _ir.FunctionType.get(arg_types_list, ret_types)
    func_op.operation.attributes["function_type"] = _ir.TypeAttr.get(new_func_type)


def mlir_module_to_ir_module(module: MlirModule, ctx: Any = None) -> Any:
    """Build an ir.Module from an MlirModule using MLIR Python API.

    This bypasses the MLIR text round-trip entirely, creating a valid
    ir.Module that can be passed directly to PassManager-based passes.

    Decomposed into helper functions for testability and clarity.
    """
    import mlir.ir as _ir

    if ctx is None:
        ctx = _ir.Context()
    ctx.allow_unregistered_dialects = True

    with ctx, _ir.Location.unknown(ctx):
        ir_mod = _ir.Module.create()

        for func in module.functions:
            # Build function entry block
            func_op, body_blk, arg_values = _build_mlir_function(func, ir_mod, ctx)

            # Build SSA map
            ssa_map = _build_ssa_map(func, arg_values)

            # Build all ops
            for op in func.ops:
                if op.op_name in ("weight", "constant"):
                    _emit_weight_op(op, ctx, ssa_map, body_blk, func.weights)
                    continue

                operands = _resolve_operands(op, ssa_map)
                ir_op = _emit_compute_op(op, operands, body_blk, ctx)
                _map_op_results(op, ir_op, ssa_map)

            # Resolve outputs and finalize function
            output_values = _resolve_output_values(func, ssa_map)
            _build_return_op(output_values, body_blk)
            _update_function_type(func_op, arg_values, output_values)

        return ir_mod


def _python_to_attr_ir(value: Any) -> ir.Attribute:
    """Convert Python value to ir.Attribute for ir.Operation.create()."""
    import mlir.ir as _ir

    if isinstance(value, bool):
        return _ir.BoolAttr.get(value)
    if isinstance(value, int):
        return _ir.IntegerAttr.get(_ir.IntegerType.get_signless(64), value)
    if isinstance(value, float):
        return _ir.FloatAttr.get(_ir.F64Type.get(), value)
    if isinstance(value, str):
        return _ir.StringAttr.get(value)
    if isinstance(value, (list, tuple)):
        items = [_python_to_attr_ir(v) for v in value]
        return _ir.ArrayAttr.get(items)
    if value is None:
        return _ir.UnitAttr.get()
    return _ir.StringAttr.get(str(value))
