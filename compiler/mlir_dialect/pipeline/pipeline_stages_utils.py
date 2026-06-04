# ruff: noqa: E501 — long lines in MLIR transform script f-strings

"""Pipeline stage utilities and types.

Contains the ``Stage`` and ``StageResult`` types plus all IR utility
functions (stats, snapshots, verification).  Split from
``pipeline_stages.py`` to keep each file under 500 lines.
"""

from __future__ import annotations

import json
import logging
import re
import time as _time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


class _DeadStageTracker:
    """Persistent dead stage detection across pipeline runs.

    Records consecutive zero-dialect-change runs for each stage in a JSON
    file (``outputs/logs/pipeline/dead_stages.json``).  When a stage reaches 3
    consecutive such runs it is marked as "dead" and a warning is emitted.
    """

    CONSECUTIVE_THRESHOLD = 3
    STATE_FILE = Path("logs") / "pipeline" / "dead_stages.json"

    @classmethod
    def _load_state(cls) -> dict[str, int]:
        if cls.STATE_FILE.exists():
            try:
                return json.loads(cls.STATE_FILE.read_text())  # type: ignore[no-any-return]
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    @classmethod
    def _save_state(cls, state: dict[str, int]) -> None:
        cls.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        cls.STATE_FILE.write_text(json.dumps(state, indent=2) + "\n")

    @classmethod
    def record(cls, stage_name: str, dialect_changed: bool) -> bool:
        """Record a pipeline run for *stage_name* and return True if dead.

        Returns True when *CONSECUTIVE_THRESHOLD* consecutive runs produce
        zero dialect change (the stage is considered dead).
        """
        state = cls._load_state()
        current = state.get(stage_name, 0)
        if dialect_changed:
            # Reset counter on any dialect change
            if current > 0:
                state.pop(stage_name, None)
                cls._save_state(state)
            return False

        # No dialect change — increment
        current += 1
        state[stage_name] = current
        cls._save_state(state)
        return current >= cls.CONSECUTIVE_THRESHOLD

    @classmethod
    def known_dead_stages(cls) -> list[str]:
        state = cls._load_state()
        return [name for name, count in state.items()
                if count >= cls.CONSECUTIVE_THRESHOLD]

    @classmethod
    def reset(cls, stage_name: str | None = None) -> None:
        state = cls._load_state()
        if stage_name:
            state.pop(stage_name, None)
        else:
            state.clear()
        cls._save_state(state)


# ── IR utility functions ──────────────────────────────────────────────


def _save_ir_stats(ir_module: Any, stage_name: str, timestamp: str = "") -> dict[str, int]:
    """Walk ops by full name, save top-10 counts to outputs/logs/pipeline/stats_{name}.txt."""
    op_counts: dict[str, int] = {}
    total_ops = 0

    import mlir.ir as ir

    def _walk(op: ir.Operation) -> None:
        nonlocal total_ops
        name = str(op.name)
        op_counts[name] = op_counts.get(name, 0) + 1
        total_ops += 1
        for region in op.regions:
            for block in region.blocks:
                for child in block.operations:
                    _walk(child)

    ctx = ir_module.operation.context
    with ctx:
        for region in ir_module.operation.regions:
            for block in region.blocks:
                for op in block.operations:
                    _walk(op)

    if not timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    log_dir = Path("logs") / "pipeline"
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', stage_name)
    path = log_dir / f"stats_{safe_name}_{timestamp}.txt"

    sorted_ops = sorted(op_counts.items(), key=lambda x: -x[1])
    top10 = sorted_ops[:10]

    lines = [
        f"IR Statistics for stage: {stage_name}",
        f"Total ops: {total_ops}",
        f"Unique op types: {len(op_counts)}",
        "",
        "Top 10 op names:",
    ]
    for op_name, count in top10:
        lines.append(f"  {op_name}: {count}")
    lines.append("")
    lines.append("All op name counts:")
    for op_name, count in sorted_ops:
        lines.append(f"  {op_name}: {count}")

    path.write_text("\n".join(lines))
    _log.info("  IR stats saved to %s (top op: %s, count=%d)",
              path, top10[0][0] if top10 else "none", top10[0][1] if top10 else 0)

    return op_counts


def _save_ir_snapshot(ir_module: Any, stage_name: str) -> str:
    """Save IR module snapshot to outputs/logs/pipeline/ for debugging.

    Also saves a companion IR stats file (via ``_save_ir_stats``) containing
    per-op-name counts and the top 10 most frequent operations.

    Returns the path to the saved file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    log_dir = Path("logs") / "pipeline"
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', stage_name)
    path = log_dir / f"snapshot_{safe_name}_{timestamp}.mlir"
    path.write_text(str(ir_module))
    _log.info("  IR snapshot saved to %s", path)

    try:
        _save_ir_stats(ir_module, f"{stage_name}_snapshot", timestamp)
    except Exception as exc:
        _log.warning("  Failed to save IR stats for '%s': %s", stage_name, exc)

    return str(path)


def _verify_stage_output(
    ir_text_before: str, ir_text_after: str, stage_name: str
) -> list[str]:
    """Check func.func / func.return counts unchanged across a stage."""
    warnings: list[str] = []

    func_before = len(re.findall(r"func\.func\s+@", ir_text_before))
    func_after = len(re.findall(r"func\.func\s+@", ir_text_after))
    if func_before != func_after:
        warnings.append(
            f"func.func count changed: {func_before} -> {func_after} ({stage_name})"
        )

    ret_before = len(re.findall(r"func\.return", ir_text_before))
    ret_after = len(re.findall(r"func\.return", ir_text_after))
    if ret_before != ret_after:
        warnings.append(
            f"func.return count changed: {ret_before} -> {ret_after} ({stage_name})"
        )

    for w in warnings:
        _log.warning("  Stage invariant [%s]: %s", stage_name, w)

    return warnings


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


def _count_values(s: str) -> int:
    """Count comma-separated values at the top level (not inside brackets).

    Handles MLIR types like ``tensor<1x2xf32>`` where commas inside ``<>``
    should not be treated as separators, as well as plain SSA value lists
    like ``%0, %1, %2``.
    """
    s = s.strip()
    if not s:
        return 0
    depth = 0
    count = 1
    for ch in s:
        if ch in "<({":
            depth += 1
        elif ch in ">)}":
            depth -= 1
        elif ch == "," and depth == 0:
            count += 1
    return count


def _verify_function_signatures(
    ir_text: str, module_name: str = "unknown"
) -> list[str]:
    """Verify each ``func.func`` return count matches its declared signature.

    After bufferization, a function declared as ::

        (tensor<...>) -> (tensor<A>, tensor<B>, tensor<C>)

    must return **exactly 3 values**.  If it returns fewer, the missing
    outputs read uninitialized memory — exactly the Issue #45 bug.

    Handles both standard MLIR text format and generic (quoted) op format.
    If no ``func.func`` ops remain (e.g. already lowered to ``llvm.func``),
    the verification is a no-op.

    Returns list of error messages (empty = all good).
    """
    errors: list[str] = []
    lines = ir_text.split("\n")

    # ── Pass 1: collect function declarations ──
    func_decls: dict[str, int] = {}

    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.search(r"func\.func\s+@(\w+)\(", line)
        if m:
            func_name = m.group(1)

            # Join signature lines until we find the opening brace
            sig_lines = [line]
            j = i + 1
            while j < len(lines) and "{" not in sig_lines[-1]:
                sig_lines.append(lines[j])
                j += 1
            sig_text = " ".join(sig_lines)

            # Find the closing paren of the argument list
            paren_depth = 0
            start_after_args = 0
            for k, ch in enumerate(sig_text):
                if ch == "(":
                    paren_depth += 1
                elif ch == ")":
                    paren_depth -= 1
                    if paren_depth == 0:
                        start_after_args = k + 1
                        break

            # Look for -> after the argument list
            arrow_pos = sig_text.find("->", start_after_args)
            if arrow_pos >= 0:
                return_part = sig_text[arrow_pos + 2 :].strip()

                # Find where return declaration ends
                end_pos = len(return_part)
                for term in ("{", "attributes"):
                    p = return_part.find(term)
                    if 0 <= p < end_pos:
                        end_pos = p
                return_part = return_part[:end_pos].strip()

                if return_part.startswith("("):
                    inner = return_part[1:-1].strip()
                    declared = _count_values(inner)
                elif return_part:
                    declared = 1
                else:
                    declared = 0
                func_decls[func_name] = declared

            i = j
        else:
            i += 1

    # ── Pass 2: collect func.return operand counts ──
    func_returns: dict[str, list[int]] = {}
    current_func: str | None = None

    for line in lines:
        # Track current function
        m = re.search(r"func\.func\s+@(\w+)\(", line)
        if m:
            current_func = m.group(1)
            continue

        stripped = line.strip()

        # Standard format:  func.return %0, %1 : types
        m = re.match(r"func\.return\s+(.*?)\s*:", stripped)
        if m:
            op_str = m.group(1).strip()
            op_count = _count_values(op_str)
            if current_func is not None:
                func_returns.setdefault(current_func, []).append(op_count)
            continue

        # Generic format:  "func.return"(%0, %1) : types
        m = re.match(r'"func\.return"\(([^)]*)\)', stripped)
        if m:
            op_str = m.group(1).strip()
            op_count = _count_values(op_str)
            if current_func is not None:
                func_returns.setdefault(current_func, []).append(op_count)
            continue

        # Void return:  func.return  (no values)
        if re.match(r"func\.return\s*(?::.*)?$", stripped):
            if current_func is not None:
                func_returns.setdefault(current_func, []).append(0)

    # ── Pass 3: compare declared vs actual ──
    for func_name, declared in func_decls.items():
        ret_counts = func_returns.get(func_name, [])
        if not ret_counts:
            if declared > 0:
                errors.append(
                    f"  '{func_name}': declares {declared} outputs "
                    f"but has no func.return"
                )
        else:
            for idx, rc in enumerate(ret_counts):
                if rc != declared:
                    errors.append(
                        f"  '{func_name}': declares {declared} outputs "
                        f"but return #{idx} has {rc} values ({module_name})"
                    )

    return errors


# ── Stage types ───────────────────────────────────────────────────────


@dataclass
class StageResult:
    """Structured result from running a pipeline stage."""
    success: bool
    elapsed: float
    ir_lines: int
    ir_snapshot_path: str | None = None
    error: str | None = None
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class Stage:
    """A single stage in the lowering pipeline.

    Can be either a PassManager pipeline string or a custom action
    callable.  If ``action`` is provided, it is called with the module
    as the sole argument.  Otherwise, ``pipeline`` is parsed and run
    via ``PassManager``.

    Each stage has a per-stage timeout; on failure or timeout, an IR
    snapshot is saved to ``outputs/logs/pipeline/``.
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
        ir_text_before = str(module)
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

            # ── Stage-level IR verification ──
            verify_warnings = _verify_stage_output(ir_text_before, str(module), self.name)
            if verify_warnings:
                for w in verify_warnings:
                    _log.warning("  Stage '%s' verification: %s", self.name, w)

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
                             self.name, error_msg, exc_info=True)
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
