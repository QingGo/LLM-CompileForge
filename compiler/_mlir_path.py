"""MLIR binding path setup for LLM-CompileForge.

The official MLIR Python bindings are installed as a pip package
(mlir-core, from https://github.com/QingGo/llvm-project/releases).
No local build path manipulation is needed — ``import mlir.ir``
resolves via site-packages.

This module exists for backward-compatible ``has_mlir_bindings()``
query and may be extended for multi-MLIR-version testing.
"""

from __future__ import annotations

_HAS_MLIR_BINDINGS: bool | None = None


def has_mlir_bindings() -> bool:
    """Return True if official MLIR Python bindings are importable."""
    global _HAS_MLIR_BINDINGS
    if _HAS_MLIR_BINDINGS is not None:
        return _HAS_MLIR_BINDINGS
    try:
        import mlir.ir  # noqa: F401
        _HAS_MLIR_BINDINGS = True
    except ImportError:
        _HAS_MLIR_BINDINGS = False
    return _HAS_MLIR_BINDINGS
