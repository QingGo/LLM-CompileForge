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

        fused_ops: list[IrOp] = []
        removed_outputs: set[str] = set()

        for op in func.ops:
            if any(out in removed_outputs for out in op.outputs):
                continue

            # Pattern: silu produces value A; mul consumes A and B → A * B
            if op.name == "silu" and len(op.outputs) >= 1:
                silu_out = op.outputs[0]
                mul_op = FuseSiLU._find_consumer(silu_out, "mul", producer_map)
                if mul_op:
                    # Determine which mul input is the silu output, which is the gate
                    other_inputs = [inp for inp in mul_op.inputs if inp != silu_out]
                    fused = IrOp(
                        name="fused_silu_mul",
                        inputs=op.inputs + other_inputs,
                        outputs=mul_op.outputs,
                        attributes=mul_op.attributes,
                    )
                    fused_ops.append(fused)
                    removed_outputs.update(op.outputs)
                    removed_outputs.update(mul_op.outputs)
                    continue

            fused_ops.append(op)

        # Remap inputs
        fused_outputs: set[str] = {out for o in fused_ops for out in o.outputs}
        for op in fused_ops:
            new_inputs: list[str] = []
            for inp in op.inputs:
                if inp in removed_outputs and inp not in fused_outputs:
                    found = False
                    for fo in fused_ops:
                        if inp in fo.outputs:
                            new_inputs.append(fo.outputs[0])
                            found = True
                            break
                    if found:
                        continue
                new_inputs.append(inp)
            op.inputs = new_inputs

        func.ops = fused_ops

    @staticmethod
    def _find_consumer(output_name: str, op_name: str, producer_map: dict[str, IrOp]) -> IrOp | None:
        for op in producer_map.values():
            if output_name in op.inputs and op.name == op_name:
                return op
        return None
