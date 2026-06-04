"""Backward-compat shim — re-exports from compiler.mlir_dialect.lowering.llvm_backend."""
import compiler.mlir_dialect.lowering.llvm_backend as _real  # noqa: E402

for _name in dir(_real):
    globals()[_name] = getattr(_real, _name)
