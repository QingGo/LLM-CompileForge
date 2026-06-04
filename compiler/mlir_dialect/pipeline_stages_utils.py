"""Backward-compat shim — re-exports from compiler.mlir_dialect.pipeline.pipeline_stages_utils."""
import compiler.mlir_dialect.pipeline.pipeline_stages_utils as _real  # noqa: E402

for _name in dir(_real):
    globals()[_name] = getattr(_real, _name)
