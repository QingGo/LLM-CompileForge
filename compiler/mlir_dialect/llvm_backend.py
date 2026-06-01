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
import os
import sys
from pathlib import Path
from typing import Any

from compiler.exceptions import MissingBindingsError
from compiler.mlir_dialect.compile_utils import (
    _compile_embedded_data,
    _compile_mlir_to_dylib_with_constants,
    _find_cc,
    _find_llc,
    _find_mlir_tool,
    _find_mlir_translate,
    _has_bindings,
    _setup_mlir_path,
    compile_mlir_to_dylib,
    compile_module_to_dylib,
    emit_llvm_ir_to_file,
    jit_compile_and_run,
    link_dylib,
    llc_compile,
    mlir_module_to_llvm_ir,
)
from compiler.mlir_dialect.pipeline_stages import (  # type: ignore[attr-defined]
    BUILTIN_STAGES,
    Stage,
    StageResult,
    _save_ir_snapshot,
    run_stages,
)

_log = logging.getLogger(__name__)


def _register_sf_passes() -> None:
    """Register sf-dialect C++ passes with the MLIR pass manager.

    The ``sf-strip-gep-nuw`` pass (and any future sf-dialect passes) must be
    registered before the pass manager can resolve them by name.  This function
    is idempotent — calling it multiple times is safe.
    """
    # Ensure sf-dialect Python bindings are on sys.path
    _sf_base = Path(__file__).resolve().parent.parent.parent / "sf-dialect"
    for _sf_candidate in [_sf_base / "python_packages" / "sf", _sf_base / "build" / "python_packages" / "sf"]:
        if _sf_candidate.is_dir() and str(_sf_candidate) not in sys.path:
            sys.path.insert(0, str(_sf_candidate))
            break
    try:
        from mlir_sf._mlir_libs._sfDialectsNanobind import sf
        # Importing the module side-effects to register passes via
        # registerSfPasses() in the nanobind extension init.
    except ImportError:
        _log.debug(
            "sf-dialect Python bindings not available — "
            "sf-strip-gep-nuw pass will not be registered. "
            "IR must not contain nuw flags on GEP ops."
        )


def lower_linalg_to_llvm_ir(
    ir_module: Any,
    skip_first_canonicalize: bool = True,
) -> str:
    """Run full linalg→LLVM lowering pipeline on an ir.Module.

    All ops must already be lowered to linalg/arith/math/tensor dialect.
    Any remaining sf.* or other unregistered dialect ops will cause
    bufferization failures.

    The first BUILTIN_STAGES ``canonicalize,cse`` stage is always skipped
    because the lowered IR may contain shape specializations (from index_op
    dim mappings) that the canonicalize pass cannot verify before bufferization.
    The second ``canonicalize,cse-2`` at stage 14 handles any remaining cleanup.

    Args:
        ir_module: The MLIR module to lower.
        skip_first_canonicalize: If True, skip the first ``canonicalize,cse``
            stage (BUILTIN_STAGES[0]). Always True by default to match
            original behavior (BUGILTIN_STAGES[1:]). Set to False to include
            all stages (e.g. for full verification passes).

    Returns LLVM IR text.
    """
    if not _has_bindings():
        raise MissingBindingsError()

    # Register sf-dialect passes so the pass manager can resolve them
    _register_sf_passes()

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
        if skip_first_canonicalize:
            stages = BUILTIN_STAGES[1:]
        else:
            stages = BUILTIN_STAGES
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
