"""SSA value tracker for HAL IR building.

Tracks SSA values within a function block, assigning ``%local`` names.

Usage::

    ssa = SSATracker()
    ssa.register_arg(arg_val, "%arg0")
    name = ssa.register_result(op_result)  # → "%0", "%1", ...
    lookup = ssa.lookup(some_val)
"""

from __future__ import annotations

from typing import Any


class SSATracker:
    """Tracks SSA values within a function block, assigning %local names.

    Uses the MLIR value's string representation (``str(val)``) as the
    stable identity key — this accounts for nanobind wrapper objects that
    may differ across ``id()`` calls.

    Example::

        args:  %arg0 → "%arg0"
        ops:   %result → "%0", "%1", ...
    """

    def __init__(self) -> None:
        self._val_to_name: dict[str, str] = {}
        self._name_to_val: dict[str, str] = {}
        self._counter = 0

    @staticmethod
    def _val_key(val: Any) -> str:
        """Stable key for an MLIR value — its string representation."""
        return str(val)

    def register_arg(self, arg_val: Any, name: str) -> None:
        """Register a function argument."""
        key = self._val_key(arg_val)
        self._val_to_name[key] = name
        self._name_to_val[name] = key

    def register_result(self, op_result: Any) -> str:
        """Register an op result, returning a fresh ``%N`` name."""
        name = f"%{self._counter}"
        self._counter += 1
        key = self._val_key(op_result)
        self._val_to_name[key] = name
        self._name_to_val[name] = key
        return name

    def lookup(self, val: Any) -> str:
        """Return the %name for an SSA value (operand or block arg)."""
        key = self._val_key(val)
        if key in self._val_to_name:
            return self._val_to_name[key]
        # Fallback for values not explicitly tracked (e.g. nested block results)
        return f"%x{self._counter}"
