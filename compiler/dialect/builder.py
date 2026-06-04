"""SfModule — MLIR module builder for sf dialect.

Builds a properly-typed MLIR module using ir.Module.create() and
ir.Block.create_at_start() — no template parse hack.
"""

from __future__ import annotations

from typing import Any

import mlir.ir as ir

from compiler.shape.shape_inference import infer_output_type  # type: ignore[attr-defined]


def _make_ranked(shape: tuple[int, ...], elt_str: str) -> ir.RankedTensorType:
    elt_map = {
        "f32": ir.F32Type.get(), "f64": ir.F64Type.get(),
        "f16": ir.F16Type.get(), "bf16": ir.BF16Type.get(),
        "i32": ir.IntegerType.get_signless(32),
        "i64": ir.IntegerType.get_signless(64),
        "i8": ir.IntegerType.get_signless(8),
        "bool": ir.IntegerType.get_signless(1),
    }
    return ir.RankedTensorType.get(list(shape), elt_map.get(elt_str, ir.F32Type.get()))


class SfModule:
    """Builder for an MLIR module containing typed sf dialect operations."""

    def __init__(
        self,
        function_name: str = "main",
        input_types: list[ir.Type] | None = None,
    ) -> None:
        self._function_name = function_name
        in_types = input_types or []

        # Build func.func operation
        func_type = ir.FunctionType.get(in_types, [])
        self._func_op = ir.Operation.create(
            "func.func",
            attributes={
                "function_type": ir.TypeAttr.get(func_type),
                "sym_name": ir.StringAttr.get(function_name),
            },
            regions=1,
        )

        # Create body block with arguments matching input types
        body_region = self._func_op.operation.regions[0]
        self._body_blk = ir.Block.create_at_start(body_region, in_types)
        self._input_values: list[ir.Value] = list(self._body_blk.arguments)

        # Build module and add the function to it
        self._module = ir.Module.create()
        self._module.body.append(self._func_op.operation)

        # Insertion point at end of body block — new ops appended before terminator
        self._has_terminator = False

    @property
    def inputs(self) -> list[ir.Value]:
        return self._input_values

    def add_op(
        self,
        op_name: str,
        operands: list[Any],
        attributes: dict[str, Any] | None = None,
    ) -> Any:
        attrs = dict(attributes or {})

        input_types: list[ir.Type] = []
        for opnd in operands:
            if hasattr(opnd, "type"):
                input_types.append(opnd.type)
            else:
                input_types.append(_make_ranked((1,), "f32"))

        output_types = infer_output_type(op_name, input_types, **attrs)

        mlir_attrs: dict[str, ir.Attribute] = {}
        for k, v in attrs.items():
            if k == "source_node":
                continue
            mlir_attrs[k] = _python_to_attr(v)

        with ir.InsertionPoint(self._body_blk):
            op = ir.Operation.create(
                f"sf.{op_name}",
                operands=operands,
                results=output_types,
                attributes=mlir_attrs,
            )
        return op.result

    def add_weight_op(self, name: str) -> Any:
        with ir.InsertionPoint(self._body_blk):
            op = ir.Operation.create(
                "sf.weight",
                results=[_make_ranked((1,), "f32")],
                attributes={"name": ir.StringAttr.get(name)},
            )
        return op.result

    def set_outputs(self, values: list[Any]) -> None:
        with ir.InsertionPoint(self._body_blk):
            ir.Operation.create("func.return", operands=values)
        self._has_terminator = True
        # Update function signature to include return types
        ret_types: list[ir.Type] = [v.type for v in values]
        arg_types = [arg.type for arg in self._body_blk.arguments]
        func_type = ir.FunctionType.get(arg_types, ret_types)
        self._func_op.operation.attributes["function_type"] = ir.TypeAttr.get(
            func_type
        )

    def to_string(self) -> str:
        return str(self._module)

    def __str__(self) -> str:
        return self.to_string()


def _python_to_attr(value: Any) -> ir.Attribute:
    if isinstance(value, bool):
        return ir.BoolAttr.get(value)
    if isinstance(value, int):
        return ir.IntegerAttr.get(ir.IntegerType.get_signless(64), value)
    if isinstance(value, float):
        return ir.FloatAttr.get(ir.F64Type.get(), value)
    if isinstance(value, str):
        return ir.StringAttr.get(value)
    if isinstance(value, (list, tuple)):
        items = [_python_to_attr(v) for v in value]
        return ir.ArrayAttr.get(items)
    if value is None:
        return ir.UnitAttr.get()
    return ir.StringAttr.get(str(value))
