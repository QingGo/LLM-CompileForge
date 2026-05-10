"""RWKV MLIR Dialect definition — Phase 2 Module E.

Defines the custom MLIR operations for RWKV models:
  - rwkv.time_mix  — Time mixing (WKV) operation
  - rwkv.channel_mix — Channel mixing operation
  - rwkv.state_evolve — State evolution operation

Reference: design-phase2.md §2.5.2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RwkvTimeMixOp:
    """WKV time mixing: yk = (r * k * v + w * prev_state) / (r * k + w + u)

    Attributes:
        r: Receptance vector [batch, hidden].
        k: Key vector [batch, hidden].
        v: Value vector [batch, hidden].
        w: Decay weight [hidden].
        u: First-term bonus [hidden].
        state: Previous state [batch, hidden].
    """

    name: str = "rwkv.time_mix"
    operands: list[str] = field(default_factory=list)
    results: list[str] = field(default_factory=list)


@dataclass
class RwkvChannelMixOp:
    """Channel mixing: cm_out = sigmoid(r) * gate(key * cur + value * prev)

    Similar to Transformer FFN but with recurrent state.
    """

    name: str = "rwkv.channel_mix"
    operands: list[str] = field(default_factory=list)
    results: list[str] = field(default_factory=list)


@dataclass
class RwkvStateEvolveOp:
    """State evolution: new_state = f(old_state, new_info, decay)

    Updates the persistent RWKV state matrix.
    """

    name: str = "rwkv.state_evolve"
    operands: list[str] = field(default_factory=list)
    results: list[str] = field(default_factory=list)


RWKV_DIALECT_OPS: dict[str, type] = {
    "rwkv.time_mix": RwkvTimeMixOp,
    "rwkv.channel_mix": RwkvChannelMixOp,
    "rwkv.state_evolve": RwkvStateEvolveOp,
}


def emit_rwkv_op(op_name: str, operands: list[str], result: str, **attrs: Any) -> str:
    """Emit an RWKV MLIR operation as a text line.

    Args:
        op_name: RWKV operation name (with dialect prefix).
        operands: SSA operand names.
        result: SSA result name.
        **attrs: Operation attributes.

    Returns:
        MLIR text line.
    """
    op_str = ", ".join(operands)
    attr_parts = []
    for k, v in attrs.items():
        if isinstance(v, str):
            attr_parts.append(f'{k} = "{v}"')
        elif isinstance(v, bool):
            attr_parts.append(f"{k} = {str(v).lower()}")
        else:
            attr_parts.append(f"{k} = {v}")
    attr_str = ", ".join(attr_parts)
    if attr_str:
        return f'    {result} = "{op_name}"({op_str}) {{{attr_str}}} : () -> tensor<*xf32>'
    return f'    {result} = "{op_name}"({op_str}) : () -> tensor<*xf32>'


def is_rwkv_op(op_name: str) -> bool:
    """Check if an operation name belongs to the RWKV dialect."""
    return op_name.startswith("rwkv.")
