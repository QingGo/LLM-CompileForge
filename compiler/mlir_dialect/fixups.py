"""Backward-compat shim — re-exports from compiler.mlir_dialect.lowering.fixups."""
import compiler.mlir_dialect.lowering.fixups as _real  # noqa: E402

for _name in dir(_real):
    globals()[_name] = getattr(_real, _name)
