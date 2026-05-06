"""Dead Code Elimination (DCE) pass.

Eliminates operations whose outputs are never consumed by any
subsequent operation or by the function's output list.
"""

from __future__ import annotations

from compiler.ir import IrFunction, IrModule, IrOp
from compiler.passes.base import Pass


class DeadCodeElimination(Pass):
    """Eliminates operations with no live consumers."""

    def apply(self, module: IrModule) -> IrModule:
        for func in module.functions:
            self._dce_function(func)
        return module

    @staticmethod
    def _dce_function(func: IrFunction) -> None:
        # Collect all SSA names that are live (consumed as inputs)
        live: set[str] = {out_name for out_name, _ in func.outputs}

        # Build consumer sets
        consumers: dict[str, list[IrOp]] = {}
        for op in func.ops:
            for inp in op.inputs:
                consumers.setdefault(inp, []).append(op)

        # Mark live outputs and propagate backwards
        worklist: list[str] = list(live)
        while worklist:
            name = worklist.pop()
            # Find the producer of this live name
            for op in func.ops:
                if name in op.outputs:
                    for inp in op.inputs:
                        if inp not in live:
                            live.add(inp)
                            worklist.append(inp)
                    break

        # Filter ops: keep iff any output is live
        new_ops: list[IrOp] = []
        removed_outputs: set[str] = set()
        for op in func.ops:
            if any(out in live for out in op.outputs):
                new_ops.append(op)
            else:
                for out in op.outputs:
                    removed_outputs.add(out)

        # Remap inputs of surviving ops to drop dead references
        for op in new_ops:
            op.inputs = [inp for inp in op.inputs if inp not in removed_outputs]

        func.ops = new_ops
