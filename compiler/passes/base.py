"""Base pass infrastructure.

A Pass transforms an IrModule. PassManager orchestrates a sequence
of Passes to form a compilation pipeline.

Design notes (compiler/passes/base.py):
- Passes are stateless functors: apply(module) → module.
- PassManager preserves the caller's original module: a structural
  copy is made before passes run.
- Each Pass logs its name to the module metadata for traceability.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from compiler.ir import IrFunction, IrModule, IrOp


class Pass(ABC):
    """Abstract base class for a compiler pass."""

    @abstractmethod
    def apply(self, module: IrModule) -> IrModule:
        """Transform the module. Must not mutate the input; return a new or modified copy."""
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


class PassManager:
    """Orchestrates a sequence of compilation passes.

    Makes a structural copy of the input module before applying
    passes, so the caller's original is never mutated.
    """

    def __init__(self) -> None:
        self._passes: list[Pass] = []

    def add(self, p: Pass) -> PassManager:
        """Add a pass to the pipeline."""
        self._passes.append(p)
        return self

    def run(self, module: IrModule) -> IrModule:
        """Run all registered passes on the module.

        A structural copy is made first, so the original module is
        preserved.  Individual IrOp objects may be shared between
        the copy and the original; passes that mutate IrOp attributes
        must create new IrOp instances.
        """
        module = _structural_copy(module)
        applied: list[str] = list(module.metadata.get("passes_applied", []))
        for p in self._passes:
            module = p.apply(module)
            applied.append(p.name)
        module.metadata["passes_applied"] = applied
        return module

    @property
    def num_passes(self) -> int:
        return len(self._passes)


def _structural_copy(module: IrModule) -> IrModule:
    """Return a shallow structural copy of *module*.

    Function and op *lists* are new, so list-level mutations
    (e.g. ``func.ops = new_ops``) are safe.  Weights and IrOp
    objects are shared — passes that mutate IrOp fields in-place
    must create new IrOp instances instead.
    """
    new_funcs: list[IrFunction] = []
    for func in module.functions:
        new_ops: list[IrOp] = []
        for op in func.ops:
            new_ops.append(IrOp(
                name=op.name,
                inputs=list(op.inputs),
                outputs=list(op.outputs),
                attributes=dict(op.attributes),
                in_place=op.in_place,
            ))
        new_func = IrFunction(
            name=func.name,
            inputs=list(func.inputs),
            outputs=list(func.outputs),
            ops=new_ops,
            weights=func.weights,  # shared — weights are read-only
        )
        new_funcs.append(new_func)
    return IrModule(functions=new_funcs, metadata=dict(module.metadata))
