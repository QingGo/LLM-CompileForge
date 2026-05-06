"""Intermediate Representation (IR) for the MLIR compiler pipeline.

Phase 1 MVP: custom Python IR structurally aligned with MLIR
(IrModule → IrFunction → IrOp). When real MLIR bindings are available,
these can be lowered to mlir.ir.Module / mlir.ir.FuncOp / mlir.ir.Operation.

Design rationale (compiler/ir.py):
- Shape / dtype stored on values, not ops — matching MLIR SSA semantics.
- Weight tensors are separated from graph ops to keep serialization clean
  (weights.bin vs model.ir vs metadata.json).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import torch

# ── Types ───────────────────────────────────────────────────


@dataclass(frozen=True)
class IrType:
    """Represents a tensor type in the IR.

    Attributes:
        dtype: torch dtype string (e.g. "float32", "float16", "int64").
        shape: symbolic shape — None dimensions mean dynamic.
    """

    dtype: str
    shape: tuple[int | None, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"dtype": self.dtype, "shape": list(self.shape)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> IrType:
        shape = tuple(s if s is not None else None for s in d.get("shape", []))
        return cls(dtype=d["dtype"], shape=shape)

    def __str__(self) -> str:
        dims = "×".join(str(d) if d is not None else "?" for d in self.shape)
        return f"tensor<{dims}×{self.dtype}>"


# ── Operations ──────────────────────────────────────────────


@dataclass
class IrOp:
    """A single operation node in the computation graph.

    Inputs/outputs are identified by SSA value names (strings).
    Attributes carry op-specific parameters (e.g. dim, eps, is_causal).
    """

    name: str  # e.g. "matmul", "rms_norm", "sdpa"
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> IrOp:
        return cls(
            name=d["name"],
            inputs=d.get("inputs", []),
            outputs=d.get("outputs", []),
            attributes=d.get("attributes", {}),
        )


# ── Functions ───────────────────────────────────────────────


@dataclass
class IrFunction:
    """A named function (sub-graph) containing a list of IrOps.

    Weights are stored separately from ops — each weight has a name
    that can be referenced as a constant input in IrOp.inputs.
    """

    name: str
    inputs: list[tuple[str, IrType]] = field(default_factory=list)
    outputs: list[tuple[str, IrType]] = field(default_factory=list)
    ops: list[IrOp] = field(default_factory=list)
    weights: dict[str, torch.Tensor] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "inputs": [(name, tp.to_dict()) for name, tp in self.inputs],
            "outputs": [(name, tp.to_dict()) for name, tp in self.outputs],
            "ops": [op.to_dict() for op in self.ops],
            "weight_names": list(self.weights.keys()),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], weights: dict[str, torch.Tensor] | None = None) -> IrFunction:
        func = cls(
            name=d["name"],
            inputs=[(name, IrType.from_dict(tp)) for name, tp in d.get("inputs", [])],
            outputs=[(name, IrType.from_dict(tp)) for name, tp in d.get("outputs", [])],
            ops=[IrOp.from_dict(op) for op in d.get("ops", [])],
            weights=weights or {},
        )
        return func

    def find_op_by_output(self, output_name: str) -> IrOp | None:
        """Find the operation that produces a given output value."""
        for op in self.ops:
            if output_name in op.outputs:
                return op
        return None


# ── Module ──────────────────────────────────────────────────


@dataclass
class IrModule:
    """Top-level IR module — the compiler's output artifact.

    Contains one or more IrFunctions and associated metadata.
    This is the contract between compiler and engine.
    """

    functions: list[IrFunction] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def main(self) -> IrFunction:
        """Convenience accessor for the primary function."""
        if not self.functions:
            raise ValueError("IrModule has no functions")
        return self.functions[0]

    def add_function(self, func: IrFunction) -> None:
        self.functions.append(func)

    def to_dict(self) -> dict[str, Any]:
        return {
            "functions": [f.to_dict() for f in self.functions],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], all_weights: dict[str, dict[str, torch.Tensor]] | None = None
    ) -> IrModule:
        weights_map = all_weights or {}
        functions = []
        for fd in d.get("functions", []):
            func_weights = weights_map.get(fd["name"], {})
            functions.append(IrFunction.from_dict(fd, weights=func_weights))
        return cls(functions=functions, metadata=d.get("metadata", {}))


# ── Serialization helpers (used by compiler/serialize.py) ───


def module_to_json(module: IrModule) -> str:
    """Serialize IrModule structure to JSON string (weights excluded)."""
    return json.dumps(module.to_dict(), indent=2)


def module_from_json(json_str: str, weights: dict[str, dict[str, torch.Tensor]] | None = None) -> IrModule:
    """Deserialize IrModule from JSON string."""
    d = json.loads(json_str)
    return IrModule.from_dict(d, all_weights=weights)


def pack_weights(module: IrModule) -> dict[str, dict[str, torch.Tensor]]:
    """Extract all weights from an IrModule into a flat dict.

    Returns: {function_name: {weight_name: tensor}}
    """
    result: dict[str, dict[str, torch.Tensor]] = {}
    for func in module.functions:
        if func.weights:
            result[func.name] = dict(func.weights)
    return result
