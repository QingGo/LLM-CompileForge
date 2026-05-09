"""MLIR binding path setup for LLM-CompileForge.

Adds the official MLIR Python bindings to sys.path so that
`import mlir.ir` works from within the project.

The bindings are built from llvm-project source via:
  mlir_binding/build -> cmake/minja -> mlir_binding/mlir_package/
"""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
_mlir_pkg = _project_root / "mlir_binding" / "mlir_package"

if _mlir_pkg.is_dir() and str(_mlir_pkg) not in sys.path:
    sys.path.insert(0, str(_mlir_pkg))
