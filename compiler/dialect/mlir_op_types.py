"""Core MLIR artifact data types — operation, function, and module definitions.

These dataclasses are the primary in-memory representation of a compiled MLIR
model.  They are serialization-agnostic: the same types are used for text
MLIR, binary SFCF, and in-memory IR module conversion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MlirOp:
    """A single MLIR operation parsed from model.mlir."""

    name: str  # full qualified name: "sf.linear", "sf.weight", etc.
    dialect: str  # "sf", "arith", etc.
    op_name: str  # "linear", "matmul", "weight", etc.
    operands: list[str]  # SSA names of inputs
    results: list[str]  # SSA names of outputs
    attributes: dict[str, Any] = field(default_factory=dict)
    input_types: list[str] = field(default_factory=list)  # MLIR type strings for each operand
    output_types: list[str] = field(default_factory=list)  # MLIR type strings for each result


@dataclass
class MlirFunction:
    """A parsed MLIR function (func.func)."""

    name: str
    inputs: list[tuple[str, str]]  # (ssa_name, mlir_type_string)
    outputs: list[tuple[str, str, bool]]  # (ssa_name, mlir_type_string, is_consumed_internally)
    ops: list[MlirOp] = field(default_factory=list)
    weights: dict[str, Any] = field(default_factory=dict)
    param_weight_names: set[str] = field(default_factory=set)
    const_weight_names: set[str] = field(default_factory=set)
    weight_names: list[str] = field(default_factory=list)  # Ordered weight arg names for this function


@dataclass
class MlirModule:
    """A parsed MLIR module containing functions and weights."""

    functions: list[MlirFunction] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    chain_order: list[str] = field(default_factory=list)
    exec_plan_data: list[int] = field(default_factory=list)
    exec_plan_proto: bytes = field(default_factory=bytes)

    @property
    def main(self) -> MlirFunction:
        """Return the main (first) function."""
        if not self.functions:
            raise ValueError("MlirModule has no functions")
        return self.functions[0]


def ssa(name: str) -> str:
    """Format an SSA name with a single ``%`` prefix.

    MLIR SSA values use ``%<name>`` syntax.  If *name* already starts with
    ``%``, it is returned as-is (to avoid double-prefixing like ``%%x``).
    """
    if not name:
        return name
    if name.startswith("%"):
        return name
    return f"%{name}"
