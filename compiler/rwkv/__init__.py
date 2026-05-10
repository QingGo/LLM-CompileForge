"""RWKV compiler support — Phase 2 Module E.

Provides RWKV MLIR dialect definitions, operation fusion passes,
and state management for non-Transformer architectures.

Reference: design-phase2.md §2.5
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "emit_rwkv_op",
    "is_rwkv_op",
    "apply_rwkv_fusion_passes",
]

_LAZY_ATTRS = frozenset(__all__)


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        import compiler.rwkv.dialect as _dialect
        import compiler.rwkv.fusion as _fusion

        _globals = {
            "emit_rwkv_op": _dialect.emit_rwkv_op,
            "is_rwkv_op": _dialect.is_rwkv_op,
            "apply_rwkv_fusion_passes": _fusion.apply_rwkv_fusion_passes,
        }
        if name in _globals:
            value = _globals[name]
            globals()[name] = value
            return value
    raise AttributeError(f"module 'compiler.rwkv' has no attribute '{name}'")
