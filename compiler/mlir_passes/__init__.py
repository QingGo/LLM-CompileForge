"""MLIR-level compiler passes using official mlir Python bindings.

Uses mlir.ir for IR traversal and mlir.passmanager for running
standard MLIR passes (CSE, canonicalize, etc.).

Usage:
    from compiler.mlir_passes import mlir_count_ops, mlir_run_cse
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "mlir_count_ops",
    "mlir_count_ops_in_module",
    "mlir_run_canonicalize",
    "mlir_run_cse",
    "mlir_verify_structure",
]

_LAZY_ATTRS = frozenset(__all__)


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        import compiler.mlir_passes._ops as _ops

        _globals = {
            "mlir_count_ops": _ops.mlir_count_ops,
            "mlir_count_ops_in_module": _ops.mlir_count_ops_in_module,
            "mlir_run_canonicalize": _ops.mlir_run_canonicalize,
            "mlir_run_cse": _ops.mlir_run_cse,
            "mlir_verify_structure": _ops.mlir_verify_structure,
        }
        if name in _globals:
            value = _globals[name]
            globals()[name] = value
            return value
    raise AttributeError(f"module 'compiler.mlir_passes' has no attribute '{name}'")
