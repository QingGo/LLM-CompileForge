"""SiLU + Mul fusion pass (SwiGLU).

Detects the pattern:
    SiLU(x) * y

and replaces it with a single fused.silu_mul operation.
This is the core activation in Llama-style FFN gating (SwiGLU).
"""

from __future__ import annotations

from compiler.ir import IrFunction, IrModule, IrOp
from compiler.passes.base import Pass


class FuseSiLU(Pass):
    """Fuse SiLU → Mul chains into a single fused operation."""

    def apply(self, module: IrModule) -> IrModule:
        for func in module.functions:
            self._fuse_function(func)
        return module

    @staticmethod
    def _fuse_function(func: IrFunction) -> None:
        producer_map: dict[str, IrOp] = {}
        for op in func.ops:
            for out in op.outputs:
                producer_map[out] = op

        # Identify silu outputs that are consumed by mul ops
        silu_by_mul: set[str] = set()  # silu outputs consumed by a mul
        mul_to_other: dict[str, tuple[list[str], list[str]]] = {}  # mul output -> (silu_inputs, other_inputs)

        for op in func.ops:
            if op.name == "mul" and len(op.inputs) >= 2 and op.outputs:
                for inp in op.inputs:
                    if inp in producer_map and producer_map[inp].name == "silu":
                        silu_op = producer_map[inp]
                        silu_by_mul.add(silu_op.outputs[0])
                        other = [i for i in op.inputs if i != inp]
                        mul_to_other[op.outputs[0]] = (silu_op.inputs, other)
                        break

        # Rebuild ops list: skip consumed silu, replace mul with fused
        new_ops: list[IrOp] = []
        for op in func.ops:
            if op.name == "silu" and op.outputs and op.outputs[0] in silu_by_mul:
                continue
            if op.name == "mul" and op.outputs and op.outputs[0] in mul_to_other:
                silu_inputs, other_inputs = mul_to_other[op.outputs[0]]
                new_ops.append(IrOp(
                    name="fused_silu_mul",
                    inputs=silu_inputs + other_inputs,
                    outputs=op.outputs,
                    attributes=op.attributes,
                ))
                continue
            new_ops.append(op)

        func.ops = new_ops

    @staticmethod
    def _find_consumer(output_name: str, op_name: str, producer_map: dict[str, IrOp]) -> IrOp | None:
        for op in producer_map.values():
            if output_name in op.inputs and op.name == op_name:
                return op
        return None

    @staticmethod
    def _find_producer(output_name: str, op_name: str, producer_map: dict[str, IrOp]) -> IrOp | None:
        if output_name in producer_map:
            op = producer_map[output_name]
            if op.name == op_name:
                return op
        return None
