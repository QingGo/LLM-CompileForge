"""IR validation pass — checks SSA consistency, preventing silent data-flow bugs.

Catches before execution:
  1. Undefined inputs: SSA names consumed but never produced.
  2. Duplicate output names: two ops producing the same SSA value.
  3. Missing outputs: function output names with no producer.
"""

from __future__ import annotations

from compiler.ir import IrFunction, IrModule
from compiler.passes.base import Pass


class IRValidationError(Exception):
    """Raised when the IR fails structural consistency checks."""


class ValidateIR(Pass):
    """Stateless pass that validates SSA consistency of an IrModule."""

    def apply(self, module: IrModule) -> IrModule:
        for func in module.functions:
            self._validate_function(func, module)
        return module

    @staticmethod
    def _validate_function(func: IrFunction, module: IrModule) -> None:
        produced: set[str] = set()
        produced_counts: dict[str, int] = {}

        # Collect all produced names
        for op in func.ops:
            for out_name in op.outputs:
                produced.add(out_name)
                produced_counts[out_name] = produced_counts.get(out_name, 0) + 1

        # Check 1: no duplicate output names
        dupes = {k: v for k, v in produced_counts.items() if v > 1}
        if dupes:
            examples = sorted(dupes.items(), key=lambda x: -x[1])[:5]
            raise IRValidationError(
                f"Duplicate SSA output names in '{func.name}': "
                + ", ".join(f"'{n}' (x{c})" for n, c in examples)
            )

        # Collect all consumed names
        consumed: set[str] = set()
        for op in func.ops:
            for inp in op.inputs:
                consumed.add(inp)

        # Check 2: all consumed names have producers
        # Producers can be: ops, function inputs, or weights
        input_names = {name for name, _ in func.inputs}
        weight_names = set(func.weights.keys())
        allowed = produced | input_names | weight_names

        missing = consumed - allowed
        if missing:
            example_names = sorted(missing)[:10]
            raise IRValidationError(
                f"Undefined SSA inputs in '{func.name}': {example_names}"
                + (f" (+ {len(missing) - 10} more)" if len(missing) > 10 else "")
            )

        # Check 3: function outputs have producers
        for out_name, _ in func.outputs:
            if out_name not in produced | input_names:
                raise IRValidationError(
                    f"Function output '{out_name}' in '{func.name}' has no producer"
                )

        # Check 4: note unreferenced produced names (dead ops)
        unreferenced = produced - consumed - {name for name, _ in func.outputs}
        if unreferenced:
            count = len(unreferenced)
            # Logged as metadata warning — DCE handles this, but useful for debugging
            if "ir_warnings" not in module.metadata:
                module.metadata["ir_warnings"] = []
            module.metadata["ir_warnings"].append(
                f"Unreferenced SSA values in '{func.name}': "
                f"{count} names (will be removed by DCE)"
            )
