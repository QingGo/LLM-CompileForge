"""Operator correctness test registry — single source of truth for op coverage.

Each ``OpCase`` entry declares an sf dialect op to test, its PyTorch
reference implementation, input shapes, and tolerance.  The ``OP_TABLE``
list drives the parametrized pytest suite in ``test_op_correctness.py``.

Adding a new op to the test suite requires only one new entry here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class OpCase:
    """Declarative test case for a single sf dialect op.

    Fields:
        sf_op_name: MLIR op name, e.g. ``"sf.add"``.
        torch_fn: PyTorch reference function (e.g. ``torch.add``).
        input_shapes: List of input tensor shapes, e.g. ``[(4, 768), (4, 768)]``.
        rtol: Minimum required cosine similarity ``1 - rtol`` (default 1e-5).
        name: Display name (auto-filled from ``sf_op_name``).
        kwargs: Extra attributes to attach to the sf op, e.g. ``dim0``/``dim1``
            for ``sf.transpose``.
        output_shapes: Output tensor shapes.  ``None`` (default) means the
            output shape matches ``input_shapes[0]``.
    """

    sf_op_name: str
    torch_fn: Callable[..., Any]
    input_shapes: list[tuple[int, ...]]
    rtol: float = 1e-5
    name: str = ""
    kwargs: dict[str, Any] | None = field(default_factory=dict)
    output_shapes: list[tuple[int, ...]] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            # Derive name from op name: "sf.add" → "add"
            self.name = self.sf_op_name.split(".")[-1]
        if self.kwargs is None:
            self.kwargs = {}
        if self.output_shapes is None:
            self.output_shapes = [self.input_shapes[0]]

    def torch_reference(self) -> Any:
        """Compute the reference PyTorch output for this op case.

        Allocates random ``torch.float32`` input tensors and calls
        ``torch_fn`` on them.
        """
        rng = torch.Generator().manual_seed(42)
        tensors = [torch.randn(*shape, generator=rng, dtype=torch.float32) for shape in self.input_shapes]
        return self.torch_fn(*tensors)


# ── OP_TABLE — the canonical list of ops to test ──────────────────────
#
# Only ops with confirmed C++ lowering patterns are included.
# Ops without lowering (e.g. sqrt, rsqrt, gt, lt, eq, permute) are
# listed as comments — they need lowering implementations first.
#
# Coverage goals (from plan):
#   Unary:   exp, sqrt*, tanh, neg, rsqrt*, relu       (* = no lowering yet)
#   Binary:  add, mul, sub, div, pow, max
#   Compare: le, gt*, lt*, eq*                          (* = no lowering yet)
#   Shape:   identity, transpose, permute*              (* = no lowering yet)

OP_TABLE: list[OpCase] = [
    # ── Unary activation — lowering confirmed ───────────────────────
    OpCase("sf.exp", torch.exp, [(4, 768)], 1e-5, "exp"),
    OpCase("sf.tanh", torch.tanh, [(4, 768)], 1e-5, "tanh"),
    OpCase("sf.neg", torch.neg, [(4, 768)], 1e-5, "neg"),
    OpCase("sf.relu", torch.relu, [(4, 768)], 1e-5, "relu"),
    OpCase("sf.sigmoid", torch.sigmoid, [(4, 768)], 1e-5, "sigmoid"),
    OpCase("sf.gelu", lambda x: torch.nn.functional.gelu(x), [(4, 768)], 1e-5, "gelu"),
    # ── Binary arithmetic — lowering confirmed ──────────────────────
    OpCase("sf.add", torch.add, [(4, 768), (4, 768)], 1e-5, "add"),
    OpCase("sf.mul", torch.mul, [(4, 768), (4, 768)], 1e-5, "mul"),
    OpCase("sf.sub", torch.sub, [(4, 768), (4, 768)], 1e-5, "sub"),
    OpCase("sf.div", torch.div, [(4, 768), (4, 768)], 1e-5, "div"),
    # pow uses absolute values to avoid NaN from negative base ^ non-integer exponent
    OpCase("sf.pow", torch.pow, [(4, 768), (4, 768)], 1e-5, "pow",
           kwargs={"positive_inputs": True}),
    OpCase("sf.max", torch.maximum, [(4, 768), (4, 768)], 1e-5, "max"),
    # ── Comparison — lowering confirmed for le ──────────────────────
    # sf.le outputs f32 0.0/1.0, matching the lowering implementation
    OpCase("sf.le", lambda a, b: torch.le(a, b).float(), [(4, 768), (4, 768)], 1e-5, "le"),
    # ── Shape ops — lowering confirmed ──────────────────────────────
    # sf.identity is a pass-through (same type)
    OpCase("sf.identity", lambda x: x, [(4, 768)], 1e-5, "identity"),
    # sf.transpose with dim0=0, dim1=1; output shape is swapped
    OpCase("sf.transpose", lambda x: torch.transpose(x, 0, 1),
           [(4, 768)], 1e-5, "transpose",
           kwargs={"dim0": 0, "dim1": 1},
           output_shapes=[(768, 4)]),

    # ── Slice — lowering confirmed ──────────────────────────────────
    OpCase("sf.slice", lambda x: x[0:2, :], [(4, 768)], 1e-5, "slice",
           kwargs={"dim": 0, "start": 0, "end": 2},
           output_shapes=[(2, 768)]),
    # Rank-1 ops deferred: LLVM lowering sometimes adds a dimension
    # (tensor<768xf32> → memref<768x1xf32>). Revisit after bufferization fix.
]

# ── Ops missing lowering patterns (documented, not tested) ───────────
# These ops are defined in the sf dialect but lack C++ lowering:
#
#   sf.sqrt   — SfSqrtOp defined, no lowering pattern
#   sf.rsqrt  — SfRsqrtOp defined, no lowering pattern
#   sf.gt     — SfGtOp defined, no lowering pattern
#   sf.lt     — SfLtOp defined, no lowering pattern
#   sf.eq     — SfEqOp defined, no lowering pattern
#   sf.permute — SfPermuteOp defined, no lowering pattern
#   sf.sum    — SfSumOp defined, lowering broken (identityMaps with reduction
#               doesn't handle different input/output shapes; needs linalg.reduce)
#
# Once lowering patterns are added (in sf-dialect/lib/Sf/), add entries:
#   OpCase("sf.sqrt", torch.sqrt, [(4, 768)], 1e-5, "sqrt"),
#   OpCase("sf.rsqrt", torch.rsqrt, [(4, 768)], 1e-5, "rsqrt"),
#   OpCase("sf.gt", lambda a, b: torch.gt(a, b).float(), ...),
#   OpCase("sf.lt", lambda a, b: torch.lt(a, b).float(), ...),
#   OpCase("sf.eq", lambda a, b: torch.eq(a, b).float(), ...),
#   OpCase("sf.permute", lambda x: x.permute(0, 2, 1), ..., kwargs={"dims": [0, 2, 1]}),
