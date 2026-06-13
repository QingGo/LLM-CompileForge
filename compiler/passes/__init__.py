"""MLIR-level compiler passes using official mlir Python bindings.

Uses mlir.ir for IR traversal and mlir.passmanager for running
standard MLIR passes (CSE, canonicalize, etc.).

Usage:
    from compiler.passes import mlir_count_ops, mlir_run_cse
"""

from compiler._lazy_imports import lazy_imports

lazy_imports(
    __name__,
    globals(),
    {
        "mlir_count_ops": ("compiler.passes._ops", "mlir_count_ops"),
        "mlir_count_ops_in_module": ("compiler.passes._ops", "mlir_count_ops_in_module"),
        "mlir_run_canonicalize": ("compiler.passes._ops", "mlir_run_canonicalize"),
        "mlir_run_cse": ("compiler.passes._ops", "mlir_run_cse"),
        "mlir_verify_structure": ("compiler.passes._ops", "mlir_verify_structure"),
    },
)
