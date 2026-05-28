"""Per-op lowering dispatch — map ``sf.*`` ops to HAL IR entries.

Each ``sf.*`` operation is mapped to a ``hal.execute`` entry with the
appropriate op name, inputs, outputs, and attributes.

Lowering is stateless (pure dispatch) — all state (SSA tracker,
weights/constants accumulators) is passed explicitly.

The actual per-op handler functions live in ``handlers.py`` and are
registered into the dispatch table at module load time.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

from compiler.mlir_dialect.hal_ir.ssa_tracker import SSATracker

_log = logging.getLogger(__name__)


# ── Op mapping tables ───────────────────────────────────────────────

_UNARY_ARITH_MAP: dict[str, str] = {
    "sf.relu": "relu",
    "sf.gelu": "gelu",
    "sf.silu": "silu",
    "sf.sigmoid": "sigmoid",
    "sf.tanh": "tanh",
    "sf.exp": "exp",
    "sf.neg": "neg",
    "sf.softplus": "softplus",
    "sf.sqrt": "sqrt",
    "sf.rsqrt": "rsqrt",
    "sf.cos": "cos",
    "sf.sin": "sin",
}

_BINARY_ARITH_MAP: dict[str, str] = {
    "sf.add": "add",
    "sf.sub": "sub",
    "sf.mul": "mul",
    "sf.div": "div",
    "sf.pow": "pow",
    "sf.max": "max",
}

_COMPARE_MAP: dict[str, str] = {
    "sf.eq": "eq",
    "sf.ne": "ne",
    "sf.gt": "gt",
    "sf.lt": "lt",
    "sf.le": "le",
    "sf.ge": "ge",
    "sf.logical_and": "logical_and",
}


# ── Helper functions ────────────────────────────────────────────────


def strip_mlir_quotes(s: str) -> str:
    """Strip MLIR string attribute quotes from a value.

    MLIR ``StringAttr`` values print as ``"foo"`` (with quotes).
    """
    return s.strip().strip('"')


def parse_sf_op_name(raw: str) -> str:
    """Normalize ``sf.op_name`` regardless of quoting style.

    In the normalized IR, some ops appear as ``sf.add`` (bare) and others
    as ``"sf.add"`` (quoted string).  Both refer to the same op.
    """
    return strip_mlir_quotes(raw)


def parse_mlir_int_attr(attr_str: str | None) -> int | None:
    """Parse an MLIR integer attribute to a Python int.

    Handles formats like ``"0"``, ``"0 : i64"``, ``"1 : i64"``.
    """
    if attr_str is None:
        return None
    s = str(attr_str).strip()
    # Strip type suffix like " : i64"
    if " : " in s:
        s = s.split(" : ")[0]
    try:
        return int(s)
    except ValueError:
        return None


def infer_dtype_from_type(t: Any) -> str:
    """Infer a short dtype string from an MLIR type."""
    s = str(t)
    if "f32" in s or "float" in s:
        return "f32"
    if "f16" in s or "bfloat" in s:
        return "f16"
    if "i64" in s:
        return "i64"
    if "i32" in s:
        return "i32"
    if "i8" in s or "i1" in s or "bool" in s:
        return "i8"
    return "f32"


def shape_from_type(t: Any) -> list[int | str]:
    """Extract shape as a list of ints or '?' for dynamic dims from an MLIR type."""
    s = str(t)
    # Extract shape from tensor<...>
    m = re.search(r"tensor<(.+?)x", s)
    if not m:
        return []
    shape_str = s[s.index("<") + 1 : s.rindex("x")]
    parts = shape_str.split("x")
    shape: list[int | str] = []
    for p in parts:
        p = p.strip()
        if p == "?":
            shape.append("?")
        else:
            try:
                shape.append(int(p))
            except ValueError:
                shape.append("?")
    return shape


def parse_attr_shape(shape_str: str) -> list[int | str]:
    """Parse an MLIR shape attribute like ``[-1, -1, 12, 64]``."""
    shape_str = shape_str.strip()
    if shape_str.startswith("["):
        shape_str = shape_str[1:]
    if shape_str.endswith("]"):
        shape_str = shape_str[:-1]
    parts = shape_str.split(",")
    shape: list[int | str] = []
    for p in parts:
        p = p.strip()
        if p in ("-1", "?"):
            shape.append("?")
        else:
            try:
                shape.append(int(p))
            except ValueError:
                shape.append("?")
    return shape


# ── Dispatch table ──────────────────────────────────────────────────

_OP_HANDLERS: dict[str, Callable] = {}


# ── Main lowering dispatch ──────────────────────────────────────────


def lower_op(
    op: Any,
    op_name: str,
    ssa: SSATracker,
    weights: list[dict[str, Any]],
    constants: list[dict[str, Any]],
    weight_index: dict[str, int],
    param_names: list[str],
    const_names: list[str],
) -> dict[str, Any] | None:
    """Lower a single sf.* op to a HAL IR entry.

    Returns ``None`` for ops that should be skipped (identity, constant, weight).

    Parameters
    ----------
    op:
        The MLIR operation (``OpView``).
    op_name:
        Parsed SF op name (e.g. ``"sf.matmul"``).
    ssa:
        SSA tracker for ``%name`` assignment.
    weights:
        Accumulator list of weight entries (mutated in-place).
    constants:
        Accumulator list of constant entries (mutated in-place).
    weight_index:
        Map from weight name → index into *weights*.
    param_names, const_names:
        Lists from weight classification metadata.
    """
    operands = list(op.operands) if hasattr(op, "operands") else []
    results = list(op.results) if hasattr(op, "results") else []

    # Get input %names
    input_names = [ssa.lookup(o) for o in operands]

    # Register results and get output %names
    output_names = [ssa.register_result(r) for r in results]

    # Look up per-op handler
    handler = _OP_HANDLERS.get(op_name)
    if handler is not None:
        return handler(
            op, op_name, input_names, output_names,
            ssa, weights, constants, weight_index, param_names, const_names,
        )

    # Unknown op — skip non-SF ops silently, warn for unrecognized SF ops
    if op_name in ("func.return", "return"):
        return None
    _log.warning("Unknown SF op %s in function, passing through", op_name)
    return {
        "op": op_name.removeprefix("sf."),
        "inputs": input_names,
        "outputs": output_names,
    }


# ── Late import: populate dispatch table ────────────────────────────
# Circular import safety: handlers.py imports helpers/maps from this
# module, which are already defined above by the time this import runs.

from compiler.mlir_dialect.hal_ir.op_lowering.handlers import (  # noqa: E402
    register_handlers,
)

register_handlers(_OP_HANDLERS)
