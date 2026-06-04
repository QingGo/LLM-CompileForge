"""Backward-compat shim — re-exports from compiler.mlir_dialect.shape.shape_inference_pure."""
import compiler.mlir_dialect.shape.shape_inference_pure as _real  # noqa: E402

for _name in dir(_real):
    globals()[_name] = getattr(_real, _name)
