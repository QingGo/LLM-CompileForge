# ruff: noqa: E501 — long lines in MLIR transform script f-strings

"""Pipeline stage execution engine.

See ``pipeline_stages_utils.py`` for the ``Stage`` / ``StageResult`` types
and IR utility functions.  See ``pipeline_actions.py`` for custom stage
action callables (tiling, FMA fusion, etc.).

This module contains the standard pipeline stage definitions
(``BUILTIN_STAGES``) and the main ``run_stages`` runner.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Sub-module imports ────────────────────────────────────────────────
from compiler.mlir_dialect.pipeline_actions import (
    ensure_filled_matmul_outputs_action,
    fuse_fma_action,
    tile_matmuls_action,
)
from compiler.mlir_dialect.pipeline_stages_utils import (
    Stage,
    StageResult,
    _count_module_ops,
    _DeadStageTracker,
    _save_ir_snapshot,
    _verify_function_signatures,
)

_log = logging.getLogger(__name__)


# ── Standard pipeline stages ──────────────────────────────────────────


def _make_tile_stage() -> Stage:
    """Build a Stage for K=64,N=64 tiling with post-canonicalize."""
    def _tile_action(m: Any) -> None:
        import mlir.passmanager as pm
        tile_matmuls_action(m, tile_k=64)
        ctx = m.operation.context
        pm.PassManager.parse("builtin.module(canonicalize,cse)", ctx).run(m.operation)

        # ── Verify tiling actually happened ──
        txt = str(m)
        scf_count = txt.count("scf.for")
        if scf_count == 0:
            _log.error("  ⚠️ TILING VERIFICATION FAILED: zero scf.for produced after tile_matmuls_action")
            _log.error("  ⚠️ Check transform-interpreter setup — matmuls remain untiled")
        else:
            _log.info("  ✓ Tiling verification: %d scf.for loops produced", scf_count)
    return Stage(
        name="tile_matmuls (K,N=64)",
        action=_tile_action,
        timeout=60.0,
        warn_only=False,
    )


def _make_fma_stage() -> Stage:
    """Build a Stage for FMA fusion."""
    def _fma_action(m: Any) -> None:
        n = fuse_fma_action(m)
        _log.info("    %d / %d fmuladd fusions",
                  n, _count_potential_fma(m))
    return Stage(
        name="fma-fusion",
        action=_fma_action,
        timeout=30.0,
        warn_only=False,
    )


def _count_potential_fma(module: Any) -> int:
    """Count approximate number of FMA opportunities for logging."""
    txt = str(module)
    return txt.count("llvm.fmul")


def _make_verify_stage() -> Stage:
    """Build a warn-only Stage that validates module structure after lowering.

    Checks all ``func.func`` ops have non-empty bodies and the module has at
    least one function.  Never blocks compilation — only logs warnings.
    """
    from compiler.mlir_passes._ops import mlir_verify_structure

    def _verify_action(m: Any) -> None:
        ctx = m.operation.context
        issues = mlir_verify_structure(m, ctx)
        if issues:
            for issue in issues:
                _log.warning("  Module verification: %s", issue)
        _log.info("  Module verification: %d issues found — %s",
                  len(issues), "PASS" if not issues else "FAIL")

    return Stage(
        name="module-verify",
        action=_verify_action,
        timeout=10.0,
        warn_only=True,
    )


# ── Main run function ─────────────────────────────────────────────────


def _fixup_arith_tensor_constants_action(module: Any) -> int:
    """Fix arith.constant ops with scalar value + tensor result type.

    Uses MLIR Python bindings to walk the module and convert scalar-valued
    ``arith.constant`` ops with tensor result types to use ``DenseElementsAttr``
    (splat).  Replaces the regex-based fixup with proper MLIR API usage.
    """
    import mlir.ir as ir
    if not isinstance(module, ir.Module):
        return 0
    from compiler.mlir_dialect.fixups import _walk_and_fix_tensor_constants
    return _walk_and_fix_tensor_constants(module)


def _emit_c_interface_action(module: Any) -> None:
    """Add ``llvm.emit_c_interface`` to each func.func."""
    import mlir.ir as ir
    main_mod = module.operation.regions[0].blocks[0]
    ctx = module.operation.context
    for op in list(main_mod):
        if str(op.operation.name) == "func.func":
            op.operation.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get(context=ctx)


def _strip_sf_attrs_action(module: Any) -> None:
    """Remove ``sf.*`` attributes from func.func ops.

    These attributes (e.g. ``sf.weight_names``) are used by the Python
    executor for weight binding but block the canonicalizer (which
    cannot handle unregistered dialect attributes at the function level).
    Stripping them before ``canonicalize,cse`` + ``bufferize`` allows
    the canonicalizer's ``InferStaticShapeOfOperands`` pattern to fix
    kDynamic dims in linalg.generic output types.
    """
    main_mod = module.operation.regions[0].blocks[0]
    keys_to_strip: list[str] = []
    for op in list(main_mod):
        if str(op.operation.name) != "func.func":
            continue
        keys_to_strip.clear()
        for attr_name in op.operation.attributes:
            if attr_name.startswith("sf."):
                keys_to_strip.append(attr_name)
        for k in keys_to_strip:
            del op.operation.attributes[k]


def _strip_sf_attrs_canon_action(module: Any) -> None:
    """Strip ``sf.*`` and ``llvm.emit_c_interface`` attrs from func.func ops.

    The ``sf.*`` function attributes (e.g. ``sf.weight_names``) block the
    canonicalizer which cannot handle unregistered dialect attributes.

    DOES NOT run canonicalize — with explicit linalg.broadcast + identity-map
    linalg.generic (S5 rewrite), canonicalize's InferStaticShapeOfOperands
    can fold broadcasts into generics incorrectly, creating shape mismatches
    that break bufferize.  (torch-mlir also skips canonicalize before bufferize.)

    ``llvm.emit_c_interface`` is stripped because it is only needed at the
    LLVM level (after ``convert-func-to-llvm``).
    """
    main_mod = module.operation.regions[0].blocks[0]
    stripped_count = 0
    for op in list(main_mod):
        if str(op.operation.name) != "func.func":
            continue
        keys = [k for k in op.operation.attributes if k.startswith("sf.")]
        if keys:
            for k in keys:
                del op.operation.attributes[k]
            stripped_count += len(keys)
    if stripped_count:
        _log.info("  Stripped %d sf.*/llvm.emit_c_interface attrs", stripped_count)


# The flattened equivalent of the LLVM lowering stages below is
# available as LINALG_TO_LLVM_PIPELINE in compile_utils.py.
# Keep the two definitions in sync.
BUILTIN_STAGES: list[Stage] = [
    Stage("canonicalize,cse", "canonicalize,cse"),
    Stage("fuse+canonicalize", "linalg-fuse-elementwise-ops,canonicalize,cse", warn_only=False),
    _make_tile_stage(),
    Stage("emit_c_interface", action=_emit_c_interface_action, timeout=5.0, warn_only=False),
    Stage("ensure-filled-outputs", action=ensure_filled_matmul_outputs_action, timeout=30.0, warn_only=False),
    # Strip sf.weight_names before bufferize: unregistered dialect attributes
    # on func.func block downstream canonicalize/bufferize passes.  After
    # stripping, canonicalize reconciles linalg.generic output types with
    # their operands (which may have been reshaped by the lowering).
    Stage("strip-sf-attrs+canon", action=_strip_sf_attrs_canon_action, timeout=30.0, warn_only=False),
    Stage("bufferize", (
        "one-shot-bufferize{bufferize-function-boundaries allow-unknown-ops"
        " function-boundary-type-conversion=identity-layout-map},"
        "canonicalize,cse,convert-bufferization-to-memref"
    ), timeout=60.0),
    Stage("linalg→loops", "convert-linalg-to-loops"),
    Stage("lower-affine", "lower-affine"),
    Stage("scf→cf", "convert-scf-to-cf"),
    Stage("expand-strided", "expand-strided-metadata"),
    Stage("lower-affine-2", "lower-affine"),
    Stage("lower-vec-mask", "func.func(lower-vector-mask)"),
    Stage("canonicalize,cse-2", "canonicalize,cse"),
    Stage("convert-cf→llvm", "convert-cf-to-llvm"),
    Stage("finalize-memref", "finalize-memref-to-llvm{use-generic-functions=false}"),
    Stage("math→llvm", "convert-math-to-llvm"),
    Stage("vector→llvm", "convert-vector-to-llvm", timeout=120.0),
    Stage("arith→llvm", "convert-arith-to-llvm"),
    Stage("func→llvm", "convert-func-to-llvm"),
    # convert-bufferization-to-memref may introduce new memref ops after
    # func→llvm changes function signatures.  Second finalize-memref-to-llvm
    # catches these residuals (torch-mlir pattern — avoids unrealized casts).
    Stage("bufferization→memref", "convert-bufferization-to-memref"),
    Stage("finalize-memref-2", "finalize-memref-to-llvm{use-generic-functions=false}", timeout=60.0),
    Stage("reconcile-casts-1", "reconcile-unrealized-casts"),
    Stage("reconcile-casts-2", "reconcile-unrealized-casts"),
    _make_fma_stage(),
    Stage("strip-gep-nuw", "sf-strip-gep-nuw", timeout=10.0, warn_only=False),
    _make_verify_stage(),
]


BUILTIN_STAGES_NO_FMA: list[Stage] = [s for s in BUILTIN_STAGES if s.name != "fma-fusion"]
"""BUILTIN_STAGES without FMA fusion stage — for testing if FMA causes cos degradation."""


def get_stages(enable_fma: bool = True) -> list[Stage]:
    """Return BUILTIN_STAGES, optionally without FMA fusion.

    Args:
        enable_fma: If True (default), returns BUILTIN_STAGES as-is.
                    If False, returns BUILTIN_STAGES with the "fma-fusion" stage removed.
    """
    if enable_fma:
        return BUILTIN_STAGES
    return [s for s in BUILTIN_STAGES if s.name != "fma-fusion"]


def get_pipeline_specs() -> list[tuple[str, str, float, bool]]:
    """Return BUILTIN_STAGES as (name, pipeline, timeout, warn_only) tuples.

    Convenience for scripts that need the raw pipeline strings for
    custom execution (debugging, timing, etc.) without depending on
    the Stage/StageResult types.
    """
    return [
        (s.name, s.pipeline, s.timeout, s.warn_only)
        for s in BUILTIN_STAGES
        if s.pipeline  # skip action-only stages (tiling, FMA, etc.)
    ]


def run_stages(
    module: Any,
    ctx: Any,
    stages: list[Stage],
    log_dir: str = "",
) -> list[StageResult]:
    """Run a sequence of pipeline stages.

    Each stage is executed in order.  If a non-warn_only stage fails,
    execution stops and subsequent stages are not run.

    Tracks IR line count across stages and warns if growth exceeds 5x
    (indicating possible IR explosion).

    When ``LLM_SERVEFORGE_LOG=DEBUG``, full IR snapshots are saved to
    ``logs/pipeline/stages/`` after each stage.

    When ``LLM_SERVEFORGE_LOG_FORMAT=json``, structured events with
    ``event_type="pipeline_stage"`` and ``event_type="pipeline_complete"``
    are emitted via the JSON log formatter.

    Returns a list of StageResult objects, one per stage.
    """
    results: list[StageResult] = []
    prev_line_count: int = len(str(module).splitlines())

    # ── Structured event / DEBUG snapshot setup ──
    _emit_json = os.environ.get("LLM_SERVEFORGE_LOG_FORMAT", "text").lower() == "json"
    _is_debug = logging.getLogger().isEnabledFor(logging.DEBUG)

    # Clean stages directory before starting
    stages_dir = Path("logs") / "pipeline" / "stages"
    shutil.rmtree(stages_dir, ignore_errors=True)
    stages_dir.mkdir(parents=True, exist_ok=True)

    # Capture initial state for summary
    initial_module_str = str(module)
    initial_line_count = prev_line_count
    initial_op_count, _ = _count_module_ops(initial_module_str)

    for stage in stages:
        result = stage.run(module, ctx, log_dir)
        results.append(result)

        # ── Per-stage analysis ──
        pre_counts = result.context.get("dialect_counts_pre", {})
        post_counts = result.context.get("dialect_counts_post", {})
        op_count_pre = sum(pre_counts.values()) if pre_counts else 0
        op_count_post = sum(post_counts.values()) if post_counts else 0
        all_dialects = set(pre_counts.keys()) | set(post_counts.keys())
        dialect_deltas = {
            d: post_counts.get(d, 0) - pre_counts.get(d, 0)
            for d in sorted(all_dialects)
        }

        # ── Dead stage detection (persistent across runs) ──
        if pre_counts and post_counts:
            no_change = pre_counts == post_counts
            is_dead = _DeadStageTracker.record(stage.name, dialect_changed=not no_change)
            if no_change and is_dead:
                _log.warning(
                    "  ⚠️ Stage '%s' marked DEAD after %d consecutive runs with "
                    "no dialect change — pre=%s post=%s",
                    stage.name, _DeadStageTracker.CONSECUTIVE_THRESHOLD,
                    dict(sorted(pre_counts.items())),
                    dict(sorted(post_counts.items())),
                )

        # DEBUG: save per-stage snapshot
        if _is_debug:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", stage.name)
            snapshot_path = stages_dir / f"{safe_name}_{timestamp}.mlir"
            snapshot_path.write_text(str(module))
            _log.debug("  Stage '%s' IR saved to %s", stage.name, snapshot_path)

        # JSON structured event: per-stage
        if _emit_json:
            _log.info("", extra={"event_type": "pipeline_stage", "event_data": {
                "stage_name": stage.name,
                "elapsed_secs": round(result.elapsed, 3),
                "ir_lines": result.ir_lines,
                "op_count_pre": op_count_pre,
                "op_count_post": op_count_post,
                "dialect_deltas": dialect_deltas,
                "warn_only": stage.warn_only,
                "success": result.success,
            }})

        # ── Growth check (existing) ──
        if result.ir_lines > 0 and prev_line_count > 0:
            growth_ratio = result.ir_lines / prev_line_count
            if growth_ratio > 5.0:
                _log.warning(
                    "  ⚠️ Stage '%s' caused %dx IR growth (%d → %d lines) — possible explosion",
                    stage.name, int(growth_ratio), prev_line_count, result.ir_lines,
                )
                if result.ir_snapshot_path:
                    _log.warning(
                        "  ⚠️ IR snapshot saved to %s (growth may cause downstream hangs)",
                        result.ir_snapshot_path,
                    )
        if result.ir_lines > 0:
            prev_line_count = result.ir_lines

        if not result.success and not stage.warn_only:
            raise RuntimeError(
                f"Pipeline stage '{stage.name}' failed: {result.error}. "
                f"IR snapshot saved to {result.ir_snapshot_path}"
            )

    # ── Pipeline summary ──
    _log.info("=" * 60)
    total_elapsed = sum(r.elapsed for r in results)
    final_line_count = results[-1].ir_lines if results else initial_line_count
    _log.info("  Pipeline summary: %d stages in %.2f s", len(results), total_elapsed)
    _log.info("  Initial IR: %d lines → Final: %d lines (%+.1f%%)",
              initial_line_count, final_line_count,
              ((final_line_count - initial_line_count) / max(initial_line_count, 1)) * 100)

    # ── Warn-only failure report ──
    warn_failures = []
    for stage, result in zip(stages, results, strict=True):
        if stage.warn_only and not result.success:
            warn_failures.append(stage.name)
    if warn_failures:
        _log.warning(
            "  ⚠️ %d warn_only stage(s) failed: %s — outputs may be degraded",
            len(warn_failures), warn_failures,
        )

    _log.info("=" * 60)

    # ── Post-pipeline: function signature verification ──
    final_text = str(module)
    sig_errors = _verify_function_signatures(final_text)
    if sig_errors:
        for err in sig_errors:
            _log.warning("Post-pipeline function signature mismatch: %s", err)
        sig_snapshot_path = _save_ir_snapshot(module, "signature_mismatch")
        _log.warning(
            "  %d function(s) with signature/return mismatch — IR saved to %s "
            "(this may cause uninitialized output buffers at runtime, see Issue #45)",
            len(sig_errors), sig_snapshot_path,
        )

    # JSON structured event: pipeline complete
    if _emit_json and results:
        final_module_str = str(module)
        final_op_count, _ = _count_module_ops(final_module_str)
        _log.info("", extra={"event_type": "pipeline_complete", "event_data": {
            "num_stages": len(results),
            "total_elapsed_secs": round(total_elapsed, 3),
            "initial_ir_lines": initial_line_count,
            "final_ir_lines": final_line_count,
            "initial_op_count": initial_op_count,
            "final_op_count": final_op_count,
            "all_success": all(r.success for r in results),
            "stages_completed": sum(1 for r in results if r.success),
        }})

    return results
