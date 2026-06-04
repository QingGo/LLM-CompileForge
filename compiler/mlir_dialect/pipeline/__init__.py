# ruff: noqa: F401 — re-exports for backward compatibility
"""pipeline sub-package — lowering stage execution engine."""

from compiler.mlir_dialect.pipeline.pipeline_actions import (
    tile_matmuls_action,
)
from compiler.mlir_dialect.pipeline.pipeline_stages import (  # type: ignore[attr-defined]
    BUILTIN_STAGES,
    _make_verify_stage,
    _save_ir_snapshot,
    run_stages,
)
from compiler.mlir_dialect.pipeline.pipeline_stages_utils import (
    Stage,
    StageResult,
    _save_ir_stats,
    _verify_stage_output,
)
