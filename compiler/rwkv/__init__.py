"""RWKV compiler support — Phase 2 Module E.

Provides RWKV MLIR dialect definitions, operation fusion passes,
and state management for non-Transformer architectures.

Reference: design-phase2.md §2.5
"""

from compiler._lazy_imports import lazy_imports

lazy_imports(__name__, globals(), {
    "emit_rwkv_op": ("compiler.rwkv.dialect", "emit_rwkv_op"),
    "is_rwkv_op": ("compiler.rwkv.dialect", "is_rwkv_op"),
})
