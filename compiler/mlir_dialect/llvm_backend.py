"""linalg → LLVM IR lowering pipeline — orchestration layer.

Delegates to ``pipeline_stages`` for stage execution (pass pipeline +
custom actions), ``fixups`` for text-format cast elimination, and
``compile_utils`` for external tool orchestration (mlir-translate, llc,
cc).  This file re-exports all names from those modules to preserve
backward compatibility for existing importers.
"""

# ruff: noqa: E501, F401 — this file re-exports names for backward compatibility

from __future__ import annotations

import logging
from typing import Any

from compiler.exceptions import MissingBindingsError
from compiler.mlir_dialect.compile_utils import (
    _compile_embedded_data,
    _compile_mlir_to_dylib_with_constants,
    _find_cc,
    _find_llc,
    _find_mlir_tool,
    _find_mlir_translate,
    _generate_ciface_wrappers,
    _has_bindings,
    _setup_mlir_path,
    compile_ciface_wrappers,
    compile_mlir_to_dylib,
    compile_module_to_dylib,
    emit_llvm_ir_to_file,
    jit_compile_and_run,
    link_dylib,
    llc_compile,
    mlir_module_to_llvm_ir,
)
from compiler.mlir_dialect.fixups import (
    _fixup_mlir_for_translate,
    _fixup_unrealized_casts,
    _fixup_vector_arith_constant,
    _replace_dense,
    _strided_to_struct,
)
from compiler.mlir_dialect.pipeline_stages import (
    BUILTIN_STAGES,
    BUILTIN_STAGES_NO_FMA,
    Stage,
    StageResult,
    _save_ir_snapshot,
    ensure_filled_matmul_outputs_action,
    fuse_fma_action,
    get_stages,
    run_stages,
    tile_matmuls_action,
)

_ensure_filled_matmul_outputs = ensure_filled_matmul_outputs_action
_fuse_fma_in_module = fuse_fma_action
_tile_matmuls_per_func = tile_matmuls_action

_log = logging.getLogger(__name__)


def lower_linalg_to_llvm_ir(
    ir_module: Any,
    skip_first_canonicalize: bool = False,
) -> str:
    """Run full linalg→LLVM lowering pipeline on an ir.Module.

    All ops must already be lowered to linalg/arith/math/tensor dialect.
    Any remaining sf.* or other unregistered dialect ops will cause
    bufferization failures.

    Args:
        ir_module: The MLIR module to lower.
        skip_first_canonicalize: If True, skip the first ``canonicalize,cse``
            stage (BUILTIN_STAGES[0]). Used with ``--no-verify`` to work
            around benign canonicalization failures on shape-mismatched IR.
            Only the first stage is skipped; the second ``canonicalize,cse-2``
            at stage 14 still runs.

    Returns LLVM IR text.
    """
    if not _has_bindings():
        raise MissingBindingsError()

    import mlir.ir as ir

    ctx = ir_module.operation.context
    ctx.allow_unregistered_dialects = True

    # Register all dialects including bufferization interface extensions
    try:
        from mlir._mlir_libs import _mlirRegisterEverything
        reg = ir.DialectRegistry()
        _mlirRegisterEverything.register_dialects(reg)
        ctx.append_dialect_registry(reg)
    except (ImportError, AttributeError) as e:
        _log.warning(
            "Dialect registry registration failed: %s (may affect bufferization)", e
        )

    with ir.Location.unknown(ctx):
        stages = BUILTIN_STAGES
        if skip_first_canonicalize:
            _log.info(
                "[no-verify] Skipping BUILTIN_STAGES stage 1 canonicalize,cse"
            )
            stages = BUILTIN_STAGES[1:]
        run_stages(ir_module, ctx, stages)
        return str(ir_module)


def lower_linalg_to_llvm_ir_text(mlir_text: str) -> str:
    """Parse MLIR text and run linalg→LLVM lowering."""
    if not _has_bindings():
        raise MissingBindingsError()

    import mlir.ir as ir

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True

    with ir.Location.unknown(ctx):
        module = ir.Module.parse(mlir_text, ctx)
        return lower_linalg_to_llvm_ir(module)


# Legacy: _vectorize_via_transform was a no-op (disabled)
def _vectorize_via_transform(ir_module: Any) -> None:
    """Vectorize via transform dialect — currently disabled (see docstring)."""
    pass
