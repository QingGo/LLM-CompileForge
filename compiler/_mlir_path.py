"""MLIR binding path setup for LLM-CompileForge.

Adds the official MLIR Python bindings to sys.path so that
``import mlir.ir`` works.  Two search strategies are tried:

1. ``mlir_binding/mlir_package/`` — the local pre-built bindings
   (LLVM 20.1.8, ~105 MB).
2. Site-packages ``mlir_core`` / ``mlir`` — a pip-installed package.

The empty ``mlir_package/__init__.py`` file is required for the
directory to be recognized as a valid Python package.
"""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
_mlir_pkg = _project_root / "mlir_binding" / "mlir_package"

if _mlir_pkg.is_dir() and str(_mlir_pkg) not in sys.path:
    sys.path.insert(0, str(_mlir_pkg))

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
