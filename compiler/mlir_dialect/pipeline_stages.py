# ruff: noqa: E501 — long lines in MLIR transform script f-strings

"""Pipeline stage execution engine.

Defines ``Stage`` and ``StageResult`` types for orchestrating the MLIR
lowering pipeline with per-stage timeout, IR snapshots on failure, and
structured timing output.  Custom actions (tiling, FMA fusion) are
represented via the ``action`` callable parameter.

All MLIR Python bindings (``mlir.ir``, ``mlir.passmanager``) are imported
locally inside functions to allow module-level import without bindings.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import time as _time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


def _save_ir_snapshot(ir_module: Any, stage_name: str) -> str:
    """Save IR module snapshot to logs/pipeline/ for debugging.

    Returns the path to the saved file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    log_dir = Path("logs") / "pipeline"
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', stage_name)
    path = log_dir / f"snapshot_{safe_name}_{timestamp}.mlir"
    path.write_text(str(ir_module))
    _log.info("  IR snapshot saved to %s", path)
    return str(path)


def _count_module_ops(module_str: str) -> tuple[int, dict[str, int]]:
    """Count MLIR operations by dialect from a module string.

    Returns ``(total_op_count, dialect_counts)``.

    Used for summary logging and as a fallback when ``StageResult.context``
    does not contain ``mlir_count_ops`` data (e.g. on failed stages).
    """
    dialect_counts: dict[str, int] = {}
    for line in module_str.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//") or stripped.startswith("#"):
            continue
        if stripped.startswith("module ") or stripped == "}":
            continue
        m = re.search(r'\b([a-zA-Z_]\w*)\.', stripped)
        if m:
            dialect_counts[m.group(1)] = dialect_counts.get(m.group(1), 0) + 1
    return sum(dialect_counts.values()), dialect_counts


@dataclass
class StageResult:
    """Structured result from running a pipeline stage."""
    success: bool
    elapsed: float
    ir_lines: int
    ir_snapshot_path: str | None = None
    error: str | None = None
    context: dict = field(default_factory=dict)


@dataclass
class Stage:
    """A single stage in the lowering pipeline.

    Can be either a PassManager pipeline string or a custom action
    callable.  If ``action`` is provided, it is called with the module
    as the sole argument.  Otherwise, ``pipeline`` is parsed and run
    via ``PassManager``.

    Each stage has a per-stage timeout; on failure or timeout, an IR
    snapshot is saved to ``logs/pipeline/``.
    """
    name: str
    pipeline: str = ""
    action: Callable[[Any], Any] | None = None
    timeout: float = 30.0
    warn_only: bool = False
    save_snapshot: bool = True

    def run(
        self,
        module: Any,
        ctx: Any,
        log_dir: str = "",
    ) -> StageResult:
        """Execute this stage with timeout and IR snapshot on failure."""
        import mlir.passmanager as pm

        from compiler.mlir_passes._ops import mlir_count_ops

        t0 = _time.perf_counter()
        pre_stats = mlir_count_ops(module, ctx)
        snapshot_path: str | None = None

        try:
            if self.action is not None:
                self.action(module)
            else:
                pm_instance = pm.PassManager.parse(
                    f"builtin.module({self.pipeline})", ctx
                )
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(pm_instance.run, module.operation)
                    try:
                        future.result(timeout=self.timeout)
                    except FutureTimeoutError:
                        elapsed = _time.perf_counter() - t0
                        if self.save_snapshot:
                            snapshot_path = _save_ir_snapshot(module, self.name)
                        _log.error(
                            "  STAGE %s TIMED OUT after %.0fs — IR saved to %s",
                            self.name, self.timeout, snapshot_path or "(not saved)",
                        )
                        return StageResult(
                            success=False, elapsed=elapsed,
                            ir_lines=len(str(module).splitlines()),
                            ir_snapshot_path=snapshot_path,
                            error=f"Timed out after {self.timeout}s",
                        )

            elapsed = _time.perf_counter() - t0
            n_lines = len(str(module).splitlines())
            _log.info("  %6.2fs  %-40s %5d lines", elapsed, self.name, n_lines)

            # ── IR dialect op count tracking ──
            post_stats = mlir_count_ops(module, ctx)
            all_keys = set(pre_stats) | set(post_stats)
            deltas = {
                k: post_stats.get(k, 0) - pre_stats.get(k, 0)
                for k in all_keys
            }
            delta_parts = [
                f"{k}:{d:+d}" for k, d in sorted(deltas.items()) if d != 0
            ]
            total_delta = sum(deltas.values())
            delta_parts.append(f"total:{total_delta:+d}")
            _log.info("  Stage '%s' IR stats: %s", self.name, ", ".join(delta_parts))

            return StageResult(
                success=True, elapsed=elapsed,
                ir_lines=n_lines,
                context={
                    "dialect_counts_pre": pre_stats,
                    "dialect_counts_post": post_stats,
                },
            )

        except Exception as e:
            elapsed = _time.perf_counter() - t0
            if self.save_snapshot:
                snapshot_path = _save_ir_snapshot(module, self.name)
            error_msg = str(e).split("\n")[0] if "\n" in str(e) else str(e)

            if self.warn_only:
                _log.warning("  %6.2fs  %-40s FAILED (warn_only=%s)", elapsed,
                             self.name, error_msg)
                if self.save_snapshot:
                    _log.warning("  Snapshot saved: %s", snapshot_path)
                return StageResult(
                    success=False, elapsed=elapsed,
                    ir_lines=len(str(module).splitlines()),
                    ir_snapshot_path=snapshot_path,
                    error=error_msg,
                )
            else:
                _log.error("  STAGE %s FAILED after %.2fs — IR saved to %s",
                           self.name, elapsed, snapshot_path or "(not saved)")
                return StageResult(
                    success=False, elapsed=elapsed,
                    ir_lines=len(str(module).splitlines()),
                    ir_snapshot_path=snapshot_path,
                    error=error_msg,
                )


# ── Custom stage actions ──────────────────────────────────────────────


def tile_matmuls_action(module: Any, tile_k: int = 64) -> None:
    """Tile ``linalg.matmul`` and ``linalg.batch_matmul`` K dim by tile_k.

    Applies the transform dialect ONCE per func.func (to avoid the
    ``tile_using_for`` multi-handle limitation).  Each func is wrapped in
    a temporary module, tiled, and the result is cloned back.
    """
    import mlir.ir as ir
    import mlir.passmanager as pm

    ctx = module.operation.context
    ctx.load_all_available_dialects()

    script = (
        'module attributes {transform.with_named_sequence} {\n'
        '  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {\n'
        '    %mats = transform.structured.match ops{["linalg.matmul"]} in %arg0\n'
        '      : (!transform.any_op) -> !transform.any_op\n'
        '    transform.structured.tile_using_for %mats\n'
        '      tile_sizes [0, ' + str(tile_k) + ', ' + str(tile_k) + ']\n'
        '      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)\n'
        '    %batch_mats = transform.structured.match ops{["linalg.batch_matmul"]} in %arg0\n'
        '      : (!transform.any_op) -> !transform.any_op\n'
        '    transform.structured.tile_using_for %batch_mats\n'
        '      tile_sizes [0, 0, ' + str(tile_k) + ', ' + str(tile_k) + ']\n'
        '      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)\n'
        '    transform.yield\n'
        '  }\n'
        '}\n'
    )

    block = module.operation.regions[0].blocks[0]
    for func in list(block):
        if str(func.operation.name) != "func.func":
            continue
        ftxt = str(func)
        if "linalg.matmul" not in ftxt and "linalg.batch_matmul" not in ftxt:
            continue

        combined = ir.Module.parse(script + "\n" + ftxt, ctx)
        try:
            pm.PassManager.parse("builtin.module(transform-interpreter)", ctx).run(
                combined.operation
            )
        except Exception as e:
            _log.warning("  tile_matmuls: func %s skipped (%s)",
                         str(func.operation.name) if hasattr(func, "operation") else "?",
                         str(e).split("\n")[0] if "\n" in str(e) else str(e),
                         exc_info=True)
            continue

        for op in list(combined.operation.regions[0].blocks[0]):
            name = str(op.operation.name)
            src = None
            if name == "func.func":
                src = op
            elif name == "builtin.module":
                for inner in op.operation.regions[0].blocks[0]:
                    if str(inner.operation.name) == "func.func":
                        src = inner
                        break
            if src is not None:
                cloned = src.operation.clone()
                func.operation.erase()
                block.append(cloned)
                break


def _op_name(op):
    try:
        return op.operation.name
    except Exception as e:
        _log.warning("  _op_name failed: %s", e, exc_info=True)
        return ""


def _def_op(value):
    try:
        own = value.owner
        if own is not None and hasattr(own, 'operation'):
            _op_name(own)
            return own
    except Exception as e:
        _log.warning("  _def_op failed: %s", e, exc_info=True)
    return None


def fuse_fma_action(module: Any) -> int:
    """Replace ``llvm.fmul + llvm.{fadd,fsub}`` with ``llvm.intr.fmuladd``.

    Handles three patterns on LLVM dialect ops:
      (1) ``%s = fmul a, b ; %r = fadd %s, c``   →  ``%r = fmuladd(a, b, c)``
      (2) ``%s = fmul a, b ; %r = fsub c, %s``   →  ``%r = fmuladd(a, fneg(b), c)``
      (3) ``%s = fmul a, b ; %r = fsub %s, c``   →  ``%r = fmuladd(a, b, fneg(c))``

    Returns number of fusions.
    """
    import mlir.ir as ir

    candidates: list = []
    for region in [module.operation.regions[0]]:
        for block in region.blocks:
            for func_op in block:
                if _op_name(func_op) not in ("func.func", "llvm.func"):
                    continue
                func_region = func_op.operation.regions[0]
                for body_block in func_region.blocks:
                    for op in list(body_block):
                        name = _op_name(op)
                        if name not in ("llvm.fadd", "llvm.fsub"):
                            continue
                        is_fsub = (name == "llvm.fsub")
                        for idx in range(2):
                            src = op.operation.operands[idx]
                            if src is None:
                                continue
                            d = _def_op(src)
                            if d is None or _op_name(d) != "llvm.fmul":
                                continue
                            candidates.append((d, op, idx, is_fsub))
                            break

    if not candidates:
        return 0

    f32_type = ir.F32Type.get(context=module.operation.context)
    n_fused = 0

    for fmul_op, tgt_op, fmul_pos, is_fsub in candidates:
        a = fmul_op.operation.operands[0]
        b = fmul_op.operation.operands[1]

        with ir.InsertionPoint(tgt_op):
            try:
                if is_fsub and fmul_pos == 1:
                    c_src = tgt_op.operation.operands[0]
                    neg = ir.Operation.create(
                        "llvm.fneg", operands=[b], results=[f32_type],
                        ip=ir.InsertionPoint(tgt_op),
                    )
                    a_fma, b_fma = a, neg.results[0]
                elif is_fsub and fmul_pos == 0:
                    c_src = tgt_op.operation.operands[1]
                    neg = ir.Operation.create(
                        "llvm.fneg", operands=[c_src], results=[f32_type],
                        ip=ir.InsertionPoint(tgt_op),
                    )
                    a_fma, b_fma = a, b
                    c_src = neg.results[0]
                else:
                    c_src = tgt_op.operation.operands[1 - fmul_pos]
                    a_fma, b_fma = a, b

                new_op = ir.Operation.create(
                    "llvm.intr.fmuladd",
                    operands=[a_fma, b_fma, c_src],
                    results=[f32_type],
                    ip=ir.InsertionPoint(tgt_op),
                )

                tgt_op.operation.result.replace_all_uses_with(new_op.results[0])
                tgt_op.erase()
                try:
                    fmul_op.erase()
                except Exception as e:
                    _log.debug("  FMA: could not erase fmul (multi-use) — %s", e, exc_info=True)
                n_fused += 1
            except Exception as e:
                _log.debug("  FMA: fusion failed for a candidate — %s", e, exc_info=True)

    return n_fused


def ensure_filled_matmul_outputs_action(module: Any) -> int:
    """Insert ``linalg.fill(0.0)`` before matmuls using ``tensor.empty`` output.

    Operates on the MLIR **text** (``str(module)``) to insert fills between
    ``tensor.empty`` and ``linalg.{matmul,batch_matmul}`` that lack them.
    After text modification, re-parses the module in-place.

    Returns the number of fills inserted.
    """
    import mlir.ir as ir

    txt = str(module)
    modified = False

    empty_pattern = re.compile(
        r'%\w+\s*=\s*tensor\.empty\(\)\s*:\s*tensor<([^>]+)>'
    )

    for m in list(empty_pattern.finditer(txt)):
        empty_name = m.group(0).split('=')[0].strip()
        empty_inner_type = m.group(1)

        pos = m.end()
        chunk = txt[pos:pos + 4000]

        use_line = None
        for line_idx, line in enumerate(chunk.split('\n')):
            stripped = line.strip()
            if ('linalg.matmul' in stripped or 'linalg.batch_matmul' in stripped):
                if empty_name in stripped.replace('%__TMP_', ''):
                    use_line = line_idx
                    break

        if use_line is None:
            continue

        has_fill = any(
            'linalg.fill' in chunk.split('\n')[i]
            for i in range(min(use_line, len(chunk.split('\n'))))
        )
        if has_fill:
            continue

        outs_match = re.search(
            rf'outs\(\s*{re.escape(empty_name)}\s*:',
            chunk.split('\n')[use_line],
        )
        if not outs_match:
            continue

        indent = '    '
        fill_lines = [
            f'{indent}%__FILL_{empty_name.lstrip("%")}__ = arith.constant 0.000000e+00 : f32',
            f'{indent}%__FILLED_{empty_name.lstrip("%")}__ = linalg.fill '
            f'ins(%__FILL_{empty_name.lstrip("%")}__ : f32) '
            f'outs({empty_name} : tensor<{empty_inner_type}>) '
            f'-> tensor<{empty_inner_type}>',
        ]

        actual_line_pos = pos
        for _ in range(use_line):
            actual_line_pos = txt.index('\n', actual_line_pos) + 1

        insert_pos = actual_line_pos
        txt = txt[:insert_pos] + '\n'.join(fill_lines) + '\n' + txt[insert_pos:]
        modified = True

    if not modified:
        return 0

    ctx = module.operation.context
    new_mod = ir.Module.parse(txt, ctx)
    old_block = module.operation.regions[0].blocks[0]
    new_block = new_mod.operation.regions[0].blocks[0]
    for op in list(old_block):
        op.erase()
    for op in list(new_block):
        cloned = op.operation.clone()
        old_block.append(cloned)

    return 1


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
        warn_only=True,
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
        warn_only=True,
    )


def _count_potential_fma(module: Any) -> int:
    """Count approximate number of FMA opportunities for logging."""
    txt = str(module)
    return txt.count("llvm.fmul")


# ── Main run function ─────────────────────────────────────────────────


def _emit_c_interface_action(module: Any) -> None:
    """Add ``llvm.emit_c_interface`` to each func.func."""
    import mlir.ir as ir
    main_mod = module.operation.regions[0].blocks[0]
    ctx = module.operation.context
    for op in list(main_mod):
        if str(op.operation.name) == "func.func":
            op.operation.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get(context=ctx)


BUILTIN_STAGES: list[Stage] = [
    Stage("canonicalize,cse", "canonicalize,cse"),
    Stage("fuse+canonicalize", "linalg-fuse-elementwise-ops,canonicalize,cse", warn_only=True),
    _make_tile_stage(),
    Stage("emit_c_interface", action=_emit_c_interface_action, timeout=5.0),
    Stage("ensure-filled-outputs", action=ensure_filled_matmul_outputs_action, timeout=30.0, warn_only=True),
    Stage("bufferize", (
        "one-shot-bufferize{bufferize-function-boundaries},"
        "canonicalize,cse,convert-bufferization-to-memref"
    ), timeout=60.0),
    Stage("linalg→loops", "convert-linalg-to-loops"),
    Stage("lower-affine", "lower-affine"),
    Stage("scf→cf", "convert-scf-to-cf"),
    Stage("expand-strided", "expand-strided-metadata"),
    Stage("lower-affine-2", "lower-affine"),
    Stage("lower-vec-mask", "func.func(lower-vector-mask)"),
    Stage("vec→scf", "func.func(convert-vector-to-scf)"),
    Stage("canonicalize,cse-2", "canonicalize,cse"),
    Stage("scf→cf-2", "convert-scf-to-cf"),
    Stage("lower-affine-3", "lower-affine"),
    Stage("convert-cf→llvm", "convert-cf-to-llvm"),
    Stage("finalize-memref", "finalize-memref-to-llvm{use-generic-functions=false}"),
    Stage("convert-cf→llvm-2", "convert-cf-to-llvm"),
    Stage("math→llvm", "convert-math-to-llvm"),
    Stage("vector→llvm", "convert-vector-to-llvm", timeout=120.0),
    Stage("arith→llvm", "convert-arith-to-llvm"),
    Stage("ub→llvm", "convert-ub-to-llvm"),
    Stage("func→llvm", "convert-func-to-llvm"),
    Stage("reconcile-casts", "reconcile-unrealized-casts"),
    _make_fma_stage(),
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

        # ── Dead stage detection ──
        if pre_counts and post_counts and pre_counts == post_counts and not stage.warn_only:
            _log.warning(
                "  ⚠️ Stage '%s' produced no dialect change (possible dead stage) — "
                "pre=%s post=%s",
                stage.name,
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
            _log.error("  Pipeline stopped at stage '%s'", stage.name)
            break

    # ── Pipeline summary ──
    _log.info("=" * 60)
    total_elapsed = sum(r.elapsed for r in results)
    final_line_count = results[-1].ir_lines if results else initial_line_count
    _log.info("  Pipeline summary: %d stages in %.2f s", len(results), total_elapsed)
    _log.info("  Initial IR: %d lines → Final: %d lines (%+.1f%%)",
              initial_line_count, final_line_count,
              ((final_line_count - initial_line_count) / max(initial_line_count, 1)) * 100)
    _log.info("=" * 60)

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
