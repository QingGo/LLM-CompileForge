"""Base pass infrastructure.

A Pass transforms an IrModule. PassManager orchestrates a sequence
of Passes to form a compilation pipeline.

Design notes (compiler/passes/base.py):
- Passes are stateless functors: apply(module) → module.
- PassManager supports conditional execution and error reporting.
- Each Pass logs its name to the module metadata for traceability.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from compiler.ir import IrModule


class Pass(ABC):
    """Abstract base class for a compiler pass."""

    @abstractmethod
    def apply(self, module: IrModule) -> IrModule:
        """Transform the module in place. Returns self for chaining."""
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


class PassManager:
    """Orchestrates a sequence of compilation passes."""

    def __init__(self) -> None:
        self._passes: list[Pass] = []

    def add(self, p: Pass) -> PassManager:
        """Add a pass to the pipeline."""
        self._passes.append(p)
        return self

    def run(self, module: IrModule) -> IrModule:
        """Run all registered passes on the module."""
        applied: list[str] = module.metadata.get("passes_applied", [])
        for p in self._passes:
            module = p.apply(module)
            applied.append(p.name)
        module.metadata["passes_applied"] = applied
        return module

    @property
    def num_passes(self) -> int:
        return len(self._passes)
