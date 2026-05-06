"""Common Subexpression Elimination (CSE) pass.

Eliminates duplicate operations that produce identical results.
Two IrOps are considered duplicates if they have the same name,
identical inputs, and identical attributes.
"""

from __future__ import annotations

from typing import Any

from compiler.ir import IrFunction, IrModule, IrOp
from compiler.passes.base import Pass


def _op_signature(op: IrOp) -> tuple[Any, ...]:
    """Generate a hashable signature for an IrOp."""
    return (
        op.name,
        tuple(op.inputs),
        tuple(sorted(op.attributes.items())),
    )


class CommonSubexpressionElimination(Pass):
    """Eliminates duplicate IrOps within each function."""

    def apply(self, module: IrModule) -> IrModule:
        for func in module.functions:
            self._cse_function(func)
        return module

    @staticmethod
    def _cse_function(func: IrFunction) -> None:
        seen: dict[tuple[Any, ...], IrOp] = {}
        # Map from old output names to the canonical output names
        replacements: dict[str, str] = {}
        new_ops: list[IrOp] = []

        for op in func.ops:
            sig = _op_signature(op)
            if sig in seen:
                # This is a duplicate — remap outputs
                canonical = seen[sig]
                for old_out, new_out in zip(op.outputs, canonical.outputs, strict=False):
                    replacements[old_out] = new_out
            else:
                seen[sig] = op
                new_ops.append(op)

        # Apply replacements to remaining ops
        for op in new_ops:
            op.inputs = [replacements.get(inp, inp) for inp in op.inputs]

        # Update function outputs
        new_outputs = []
        for name, tp in func.outputs:
            new_outputs.append((replacements.get(name, name), tp))

        func.ops = new_ops
        func.outputs = new_outputs
