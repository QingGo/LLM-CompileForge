"""Tensor Parallelism — Phase 2 Module D.

Megatron-LM style column/row-parallel linear layers with HAL
Communicator integration for hardware-agnostic distributed inference.

Reference: design-phase2.md §2.4
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ColumnParallelLinear",
    "RowParallelLinear",
    "VocabParallelEmbedding",
    "search_tp_strategy",
    "count_parameters",
]

_LAZY_ATTRS = frozenset(__all__)


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        import compiler.tp.linear as _linear
        import compiler.tp.strategy as _strategy

        _globals = {
            "ColumnParallelLinear": _linear.ColumnParallelLinear,
            "RowParallelLinear": _linear.RowParallelLinear,
            "VocabParallelEmbedding": _linear.VocabParallelEmbedding,
            "search_tp_strategy": _strategy.search_tp_strategy,
            "count_parameters": _strategy.count_parameters,
        }
        if name in _globals:
            value = _globals[name]
            globals()[name] = value
            return value
    raise AttributeError(f"module 'compiler.tp' has no attribute '{name}'")
