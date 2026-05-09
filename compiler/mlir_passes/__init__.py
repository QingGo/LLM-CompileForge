"""MLIR-level compiler passes using official mlir Python bindings.

Uses mlir.ir for IR traversal and mlir.passmanager for running
standard MLIR passes (CSE, canonicalize, etc.).

Since LLVM 20.1.8 bindings don't export PassManager.add() for Python
callbacks, Python-side analysis passes use direct IR tree walks, while
optimization passes delegate to the C++ PassManager.

Usage:
    from compiler.mlir_passes import (
        mlir_count_ops, mlir_run_cse, mlir_verify_structure
    )
"""

from compiler.mlir_passes._ops import (
    mlir_count_ops,
    mlir_count_ops_in_module,
    mlir_run_canonicalize,
    mlir_run_cse,
    mlir_verify_structure,
)

__all__ = [
    "mlir_count_ops",
    "mlir_count_ops_in_module",
    "mlir_run_canonicalize",
    "mlir_run_cse",
    "mlir_verify_structure",
]
