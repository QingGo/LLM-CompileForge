#!/usr/bin/env python3
"""MLIR-aware test-case reducer with function-level reduction and global counter.

Uses MLIR Python API to handle both standard (``func.func @name``) and
generic (``"func.func"()``) serialization formats uniformly.

Usage:
    python scripts/diagnostics/reduce_mlir.py model.lowered.mlir \\
        --interestingness "./check_hang.sh {}" --output reduced.mlir
    python scripts/diagnostics/reduce_mlir.py model.lowered.mlir \\
        --pass "convert-vector-to-llvm" --output reduced.mlir
    python scripts/diagnostics/reduce_mlir.py model.lowered.mlir \\
        --interestingness "./check.sh {}" --metric lines
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

_log = logging.getLogger("reduce_mlir")


# =========================================================================
# MLIR context (shared, lazily initialized)
# =========================================================================

_mlir_ctx: ir.Context | None = None  # type: ignore[name-defined]


def _get_mlir_ctx():
    """Return a shared MLIR context with all dialects registered."""
    global _mlir_ctx
    if _mlir_ctx is not None:
        return _mlir_ctx
    import mlir.ir as ir

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    ctx.load_all_available_dialects()
    try:
        from mlir._mlir_libs import _mlirRegisterEverything

        reg = ir.DialectRegistry()
        _mlirRegisterEverything.register_dialects(reg)
        ctx.append_dialect_registry(reg)
    except Exception:
        pass
    try:
        from mlir_sf._mlir_libs._sfDialectsNanobind import sf
        sf.register_dialects(ctx._CAPIPtr, load=True)
    except Exception:
        pass
    _mlir_ctx = ctx
    return ctx


def _preprocess_mlir(mlir_text: str) -> str:
    """Strip sf.* module attributes and sf.* attrs on func.func so ir.Module.parse() can handle it."""
    import re
    # Strip sf.chain_order / sf.exec_plan_data from module attributes
    mlir_text = re.sub(r'\bsf\.\w+\s*=\s*\[[^\]]*\]\s*;?\s*', '', mlir_text)
    # Strip sf.chain_order in attributes { ... }
    mlir_text = re.sub(r'\bsf\.\w+\s*=\s*"[^"]*"\s*;?\s*', '', mlir_text)
    # Clean up double semicolons and trailing whitespace in attributes
    mlir_text = re.sub(r';\s*;', ';', mlir_text)
    mlir_text = re.sub(r'\{\s*;', '{', mlir_text)
    return mlir_text


# =========================================================================
# MLIR API–based module manipulation
# =========================================================================


def _get_func_names(module) -> list[str]:
    """Extract function names from a parsed MLIR module."""
    body = module.operation.regions[0].blocks[0]
    names = []
    for op in body.operations:
        if str(op.operation.name) == "func.func":
            attr = op.operation.attributes.get("sym_name")
            names.append(str(attr).strip('"') if attr else "?")
    return names


def _build_module_with(mlir_text: str, keep_names: set[str]) -> str:
    """Parse original MLIR, erase functions not in *keep_names*, serialize."""
    import mlir.ir as ir

    ctx = _get_mlir_ctx()
    parseable = _preprocess_mlir(mlir_text)
    with ir.Location.unknown(ctx):
        mod = ir.Module.parse(parseable, ctx)
    body = mod.operation.regions[0].blocks[0]
    for op in list(body.operations):
        if str(op.operation.name) == "func.func":
            attr = op.operation.attributes.get("sym_name")
            name = str(attr).strip('"') if attr else ""
            if name and name not in keep_names:
                op.operation.erase()
    return str(mod)


# =========================================================================
# Reduction strategies
# =========================================================================


def reduce_functions(
    mlir_text: str,
    interesting: Callable[[str], bool],
    retry: bool = True,
) -> str:
    """One-at-a-time function deletion using MLIR API.

    Returns the reduced MLIR text.
    """
    import mlir.ir as ir

    ctx = _get_mlir_ctx()
    parseable = _preprocess_mlir(mlir_text)
    with ir.Location.unknown(ctx):
        func_names = _get_func_names(ir.Module.parse(parseable, ctx))

    current = list(func_names)
    n_original = len(current)
    i = 0
    iterations = 0
    deletions = 0

    while i < len(current):
        iterations += 1
        keep = set(current[:i] + current[i + 1 :])
        candidate = _build_module_with(mlir_text, keep)

        if interesting(candidate):
            current = current[:i] + current[i + 1 :]
            deletions += 1
            if retry:
                i = 0
        else:
            i += 1

    _log.info(
        "  Function reduction: %d → %d functions (%d deletions in %d iterations)",
        n_original, len(current), deletions, iterations,
    )
    return _build_module_with(mlir_text, set(current))


def reduce_by_binary_search(
    mlir_text: str,
    interesting: Callable[[str], bool],
) -> str:
    """Binary search: find minimal function prefix that triggers bug."""
    import mlir.ir as ir

    ctx = _get_mlir_ctx()
    parseable = _preprocess_mlir(mlir_text)
    with ir.Location.unknown(ctx):
        func_names = _get_func_names(ir.Module.parse(parseable, ctx))

    lo, hi = 0, len(func_names) - 1
    n_original = len(func_names)
    iterations = 0

    while lo < hi:
        iterations += 1
        mid = (lo + hi) // 2
        keep = set(func_names[: mid + 1])
        candidate = _build_module_with(mlir_text, keep)

        if interesting(candidate):
            hi = mid
        else:
            lo = mid + 1

    _log.info(
        "  Binary search: first function at index %d/%d (%d iterations)",
        lo, n_original, iterations,
    )
    return _build_module_with(mlir_text, set(func_names[: lo + 1]))


# =========================================================================
# Interestingness test (abstract + implementations)
# =========================================================================


class InterestingnessTest(ABC):
    @abstractmethod
    def is_interesting(self, mlir_text: str) -> bool:
        ...


class ShellInterestingness(InterestingnessTest):
    def __init__(self, command: str, timeout: float = 60.0):
        self._command = command
        self._timeout = timeout

    def is_interesting(self, mlir_text: str) -> bool:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mlir", delete=False
        ) as f:
            f.write(mlir_text)
            f.flush()
            tmp_path = f.name
        try:
            cmd = self._command.replace("{}", tmp_path)
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=self._timeout,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


class PassInterestingness(InterestingnessTest):
    def __init__(self, pass_pipeline: str, timeout: float = 30.0):
        self._pass_pipeline = pass_pipeline
        self._timeout = timeout

    def is_interesting(self, mlir_text: str) -> bool:
        from concurrent.futures import ThreadPoolExecutor
        from concurrent.futures import TimeoutError as FutureTimeout

        import mlir.ir as ir
        import mlir.passmanager as pm

        ctx = _get_mlir_ctx()
        try:
            with ir.Location.unknown(ctx):
                mod = ir.Module.parse(mlir_text, ctx)
        except Exception:
            return False
        try:
            with ir.Location.unknown(ctx):
                p = pm.PassManager.parse(
                    f"builtin.module({self._pass_pipeline})", ctx
                )
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(p.run, mod.operation)
                    try:
                        future.result(timeout=self._timeout)
                    except FutureTimeout:
                        return True
        except Exception:
            return True
        return False


# =========================================================================
# Global counter (P1)
# =========================================================================


class GlobalCounter:
    _COUNTER_DIR = Path("/tmp/reduce_mlir_counters")

    def __init__(
        self, inner: InterestingnessTest, metric: str = "lines",
        input_hash: str = "",
    ):
        self._inner = inner
        self._metric = metric
        self._input_hash = input_hash or hashlib.md5(
            str(time.time()).encode()
        ).hexdigest()[:8]
        self._counter_path = self._COUNTER_DIR / f"best_{self._input_hash}.txt"
        self._counter_path.parent.mkdir(parents=True, exist_ok=True)
        self._best: int | None = self._load_best()

    def _load_best(self) -> int | None:
        if self._counter_path.exists():
            try:
                return int(self._counter_path.read_text().strip())
            except (ValueError, OSError):
                return None
        return None

    def _save_best(self, value: int) -> None:
        self._counter_path.write_text(str(value))

    def _measure(self, mlir_text: str) -> int:
        if self._metric == "lines":
            return len(mlir_text.splitlines())
        if self._metric == "ops":
            count = 0
            for line in mlir_text.splitlines():
                s = line.strip()
                if s and not s.startswith("//") and not s.startswith("#"):
                    if re.search(r"\b[a-zA-Z_]\w*\.[a-zA-Z_]\w*", s):
                        count += 1
            return count
        if self._metric == "bytes":
            return len(mlir_text.encode())
        return len(mlir_text.splitlines())

    def is_interesting(self, mlir_text: str) -> bool:
        if not self._inner.is_interesting(mlir_text):
            return False
        current = self._measure(mlir_text)
        if self._best is None:
            self._best = current
            self._save_best(current)
            return True
        if current > self._best:
            return False
        self._best = current
        self._save_best(current)
        return True

    @property
    def best_value(self) -> int | None:
        return self._best


# =========================================================================
# CLI
# =========================================================================


def build_interestingness(args: argparse.Namespace) -> InterestingnessTest:
    if args.pass_name:
        return PassInterestingness(args.pass_name, args.pass_timeout)
    if args.interestingness:
        return ShellInterestingness(args.interestingness, args.timeout)
    raise ValueError("Must specify either --interestingness or --pass")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MLIR-aware test-case reducer (MLIR API, both formats)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", help="Input MLIR file")
    parser.add_argument("--output", "-o", default="",
                        help="Output file (default: <input>.reduced.mlir)")
    parser.add_argument("--interestingness", metavar="CMD",
                        help="Shell command, {} = temp file path, exit 0 = bug present")
    parser.add_argument("--pass", dest="pass_name", metavar="PIPELINE",
                        help="MLIR pass pipeline; crash/hang = bug present")
    parser.add_argument("--pass-timeout", type=float, default=30.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--metric", choices=["lines", "ops", "bytes"],
                        help="Global counter metric (P1)")
    parser.add_argument("--strategy", choices=["function", "binary"],
                        default="function")
    parser.add_argument("--no-retry", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",
    )

    input_path = Path(args.input)
    if not input_path.exists():
        _log.error("Input file not found: %s", input_path)
        return 1

    mlir_text = input_path.read_text()
    original_lines = len(mlir_text.splitlines())
    _log.info("Input: %s (%d lines)", input_path, original_lines)

    func_names = _get_func_names(
        __import__("mlir.ir", fromlist=["ir"]).Module.parse(
            _preprocess_mlir(mlir_text), _get_mlir_ctx()
        )
    )
    _log.info("Functions: %d", len(func_names))
    for i, name in enumerate(func_names):
        _log.debug("  [%2d] %s", i, name)

    if not func_names:
        _log.warning("No functions found — nothing to reduce")
        return 0

    try:
        interesting = build_interestingness(args)
    except ValueError as e:
        _log.error(str(e))
        return 1

    if args.metric:
        ihash = hashlib.md5(input_path.read_bytes()).hexdigest()[:8]
        interesting = GlobalCounter(interesting, args.metric, ihash)
        _log.info("Global counter enabled: metric=%s", args.metric)

    _log.info("Validating initial input is interesting...")
    if not interesting.is_interesting(mlir_text):
        _log.error("Initial input is NOT interesting — check your interestingness test")
        return 1
    _log.info("  Initial input is interesting ✓")

    t0 = time.perf_counter()
    if args.strategy == "function":
        reduced_text = reduce_functions(
            mlir_text, interesting.is_interesting,
            retry=not args.no_retry,
        )
    else:
        reduced_text = reduce_by_binary_search(
            mlir_text, interesting.is_interesting,
        )
    elapsed = time.perf_counter() - t0

    reduced_lines = len(reduced_text.splitlines())
    reduction_pct = (1 - reduced_lines / original_lines) * 100 if original_lines else 0
    kept_names = _get_func_names(
        __import__("mlir.ir", fromlist=["ir"]).Module.parse(
            _preprocess_mlir(reduced_text), _get_mlir_ctx()
        )
    )

    output_path = args.output or str(input_path).replace(".mlir", ".reduced.mlir")
    Path(output_path).write_text(reduced_text)

    _log.info("=" * 60)
    _log.info("Reduction complete in %.1fs", elapsed)
    _log.info("  Functions: %d → %d", len(func_names), len(kept_names))
    _log.info("  Lines:     %d → %d (%.1f%% reduction)",
              original_lines, reduced_lines, reduction_pct)
    if args.metric and isinstance(interesting, GlobalCounter):
        _log.info("  Best %s:  %d", args.metric, interesting.best_value)
    _log.info("  Output:    %s", output_path)
    _log.info("Kept functions: %s", ", ".join(kept_names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
