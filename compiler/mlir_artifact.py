"""DEPRECATED: compiler.mlir_artifact is now a sub-package.

Import from ``compiler.mlir_artifact`` directly — it will resolve
to the ``compiler/mlir_artifact/__init__.py`` package entry point.
All existing ``from compiler.mlir_artifact import ...`` imports
continue to work unchanged.
"""

from __future__ import annotations

# Re-export through the package. When the ``compiler/mlir_artifact/``
# directory exists, Python imports the package (__init__.py), not this
# file.  This shim exists as a fallback for environments that might not
# find the package directory.
raise ImportError(
    "compiler.mlir_artifact is now a package. "
    "Use 'from compiler.mlir_artifact import ...' — "
    "it still works identically."
)
