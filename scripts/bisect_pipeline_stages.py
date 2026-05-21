"""Bisect BUILTIN_STAGES to find which stage causes argmax mismatch.

Usage:
    python scripts/bisect_pipeline_stages.py [--skip <name>] [--stages <N>]
    python scripts/bisect_pipeline_stages.py --auto          # automated bisect
    python scripts/bisect_pipeline_stages.py --skip ensure-filled-outputs
    python scripts/bisect_pipeline_stages.py --skip tile_matmuls
    python scripts/bisect_pipeline_stages.py --skip fma-fusion
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compiler.mlir_dialect.llvm_backend import (
    _has_bindings,
)
from utils.logging import init_logging

_log = logging.getLogger("bisect_pipeline")

COMPILED_DIR = "compiled/opt_125m_fresh"
LOWERED_MLIR = os.path.join(COMPILED_DIR, "model.lowered.mlir")
ORIG_DYLIB = os.path.join(COMPILED_DIR, "libopt_125m.dylib")
RUST_DIR = "rust"
EXPECTED_ARGMAX = 1437
WRONG_ARGMAX = 6

TEST_NAME = "test_forward_matches_python_argmax"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.ravel().astype(np.float64)
    b_f = b.ravel().astype(np.float64)
    denom = np.linalg.norm(a_f) * np.linalg.norm(b_f)
    return float(np.dot(a_f, b_f) / (denom + 1e-12))


def _setup_mlir() -> None:
    from compiler.mlir_dialect.compile_utils import _setup_mlir_path as _do_setup
    _do_setup()


def parse_stages() -> list:
    from compiler.mlir_dialect.pipeline_stages import BUILTIN_STAGES, Stage
    return [Stage(**{f.name: getattr(s, f.name) for f in s.__dataclass_fields__.values()}) for s in BUILTIN_STAGES]


def compile_with_stages(
    stages: list,
    out_dir: str,
    label: str,
) -> bool:
    """Run pipeline stages on model.lowered.mlir, compile to dylib. Returns True on success."""
    import mlir.ir as ir

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True

    reg = ir.DialectRegistry()
    try:
        from mlir._mlir_libs import _mlirRegisterEverything
        _mlirRegisterEverything.register_dialects(reg)
    except (ImportError, AttributeError) as e:
        _log.warning(f"Dialect registry failed: {e}")
    ctx.append_dialect_registry(reg)

    with ir.Location.unknown(ctx):
        with open(LOWERED_MLIR) as f:
            text = f.read()
        module = ir.Module.parse(text, ctx)

        from compiler.mlir_dialect.pipeline_stages import run_stages
        results = run_stages(module, ctx, stages, log_dir=f"logs/bisect_{label}")

        # Check all stages succeeded (or warn_only failed, which is OK)
        failed = [r for r in results if not r.success]
        if failed:
            failed_names = []
            for r in failed:
                name = r.context.get("stage_name", "?")
                failed_names.append(name)
            _log.warning(f"  {len(failed)} stage(s) failed/skipped: {failed_names}")

        LLVM_IR = os.path.join(out_dir, f"model_{label}.ll")

    from compiler.mlir_dialect.compile_utils import emit_llvm_ir_to_file, link_dylib, llc_compile
    try:
        emit_llvm_ir_to_file(module, LLVM_IR)
        obj_path = llc_compile(LLVM_IR, output=os.path.join(out_dir, f"model_{label}.o"))
        dylib_path = os.path.join(out_dir, f"lib_{label}.dylib")
        link_dylib([obj_path], dylib_path)
        return dylib_path
    except Exception as e:
        _log.error(f"  Compilation failed: {e}")
        return None


def run_rust_test(dylib_path: str) -> dict:
    """Run the Rust argmax test and parse the result."""
    backup = ORIG_DYLIB + ".bak"
    if os.path.exists(ORIG_DYLIB):
        shutil.copy2(ORIG_DYLIB, backup)

    try:
        shutil.copy2(dylib_path, ORIG_DYLIB)
        result = subprocess.run(
            ["cargo", "test", TEST_NAME, "--", "--nocapture"],
            cwd=RUST_DIR,
            capture_output=True, text=True,
            timeout=120,
        )
        output = result.stdout + "\n" + result.stderr
        argmax = None
        for line in output.splitlines():
            if "argmax" in line.lower():
                parts = line.split()
                for _i, p in enumerate(parts):
                    if "argmax" in p.lower():
                        # Find the number after ':'
                        after_colon = p.split(":")[-1].rstrip(",")
                        try:
                            argmax = int(after_colon)
                        except ValueError:
                            pass
        passed = result.returncode == 0
        return {"passed": passed, "argmax": argmax, "output": output}
    finally:
        if os.path.exists(backup):
            shutil.move(backup, ORIG_DYLIB)


def make_pipeline_without(stages: list, skip_index: int) -> list:
    """Return a new pipeline with stage at skip_index removed."""
    return [s for i, s in enumerate(stages) if i != skip_index]


def make_pipeline_to(stages: list, upto: int) -> list:
    """Return stages [0..upto] (inclusive)."""
    return stages[:upto + 1]


def auto_bisect(stages: list, out_dir: str):
    """Binary search to find the first stage that introduces the error."""
    lo, hi = 0, len(stages) - 1

    while lo < hi:
        mid = (lo + hi) // 2
        trial = stages[:mid + 1]
        label = f"stages_0_to_{mid}"
        _log.info(f"\n[二分] stages [0..{mid}] ({len(trial)} stages)")

        dylib_path = compile_with_stages(trial, out_dir, label)
        if dylib_path is None:
            _log.error("  Compilation failed — trying later midpoint")
            lo = mid + 1
            continue

        result = run_rust_test(dylib_path)
        _log.info(f"  Rust result: argmax={result['argmax']}, passed={result['passed']}")

        if result.get("argmax") == EXPECTED_ARGMAX:
            _log.info(f"  Stages [0..{mid}] are OK (argmax={EXPECTED_ARGMAX})")
            lo = mid + 1
        else:
            _log.info(f"  Stages [0..{mid}] have bug (argmax={result.get('argmax')})")
            hi = mid

    _log.info(f"\n✅ First bad stage index: {lo} — {stages[lo].name}")
    _log.info(f"  Pipeline text: {stages[lo].pipeline or '(custom action)'}")
    return lo


def skip_and_test(stages: list, out_dir: str, skip_indices: list[int], label: str):
    """Skip specific stages and test."""
    trial = [s for i, s in enumerate(stages) if i not in skip_indices]

    skip_names = [stages[i].name for i in skip_indices]
    _log.info(f"\n[SKIP] Skipping stages {skip_indices}: {skip_names}")
    _log.info(f"  Running {len(trial)}/{len(stages)} stages")

    dylib_path = compile_with_stages(trial, out_dir, label)
    if dylib_path is None:
        _log.error("  Compilation failed")
        return None

    result = run_rust_test(dylib_path)
    _log.info(f"  Rust argmax={result['argmax']}, passed={result['passed']}")
    if result.get("argmax") == EXPECTED_ARGMAX:
        _log.info(f"  ✅ SKIPPING {label} FIXES THE BUG!")
    return result


def baseline_test(out_dir: str):
    """Test the full pipeline to confirm the bug is present."""
    stages = parse_stages()
    _log.info(f"\n[基线] Full pipeline ({len(stages)} stages) — expecting argmax={WRONG_ARGMAX}")
    dylib_path = compile_with_stages(stages, out_dir, "baseline")
    if dylib_path:
        result = run_rust_test(dylib_path)
        _log.info(f"  Baseline argmax={result['argmax']}, passed={result['passed']}")
        return result
    return None


# =====================================================================
# Ctypes mode
# =====================================================================

CTYPES_HIGH_COS = 0.99  # cos above this = "correct" for binary search


def _ctypes_baseline(oracle, stages, out_dir: str) -> float:
    """Compile full pipeline and measure baseline cosine."""
    _log.info("\n[基线] Full pipeline — measuring reference cosine")
    dylib_path = compile_with_stages(stages, out_dir, "baseline")
    if dylib_path is None:
        _log.error("Baseline compilation failed")
        return 0.0
    cos = oracle.compare(dylib_path)
    _log.info("  Baseline cos(ctypes, Python executor): %.10f", cos)
    return cos


def _ctypes_auto_bisect(oracle, stages, out_dir: str):
    """Binary search to find first stage that degrades cosine."""
    lo, hi = 0, len(stages) - 1

    while lo < hi:
        mid = (lo + hi) // 2
        trial = stages[: mid + 1]
        _log.info("\n[二分] stages [0..%d] (%d stages)", mid, len(trial))

        dylib_path = compile_with_stages(trial, out_dir, f"stages_0_to_{mid}")
        if dylib_path is None:
            _log.error("  Compilation failed — trying later midpoint")
            lo = mid + 1
            continue

        cos = oracle.compare(dylib_path)
        _log.info("  cos(ctypes, Python executor): %.10f", cos)

        if cos > CTYPES_HIGH_COS:
            _log.info("  Stages [0..%d] are OK (cos > %.2f)", mid, CTYPES_HIGH_COS)
            lo = mid + 1
        else:
            _log.info("  Stages [0..%d] have degradation (cos ≤ %.2f)", mid, CTYPES_HIGH_COS)
            hi = mid

    _log.info("\n✅ First bad stage index: %d — %s", lo, stages[lo].name)
    _log.info("  Pipeline text: %s", stages[lo].pipeline or "(custom action)")
    return lo


def _ctypes_skip_and_test(oracle, stages, out_dir: str, skip_indices: list[int], label: str):
    """Skip specific stages and measure cosine."""
    trial = [s for i, s in enumerate(stages) if i not in skip_indices]
    skip_names = [stages[i].name for i in skip_indices]
    _log.info("\n[SKIP] Skipping stages %s: %s", skip_indices, skip_names)
    _log.info("  Running %d/%d stages", len(trial), len(stages))

    dylib_path = compile_with_stages(trial, out_dir, label)
    if dylib_path is None:
        _log.error("  Compilation failed")
        return None

    cos = oracle.compare(dylib_path)
    _log.info("  cos(ctypes, Python executor): %.10f", cos)
    return cos


def _check_signal_saturation(results: list[dict], threshold: float = 0.01) -> None:
    cos_values = [r["cos"] for r in results if "cos" in r]
    if len(cos_values) < 3:
        return
    spread = max(cos_values) - min(cos_values)
    if spread < threshold:
        print(f"\n  ⚠️ SIGNAL SATURATION: All variants produce cos within {spread:.4f}")
        print("     The bisect target may not be in the pipeline stages being tested.")
        print("     Consider testing: weight loading, position embedding, or bufferization.")


def _run_ctypes_bisect(args, stages, out_dir: str) -> int:
    """Handle all --ctypes modes."""
    from scripts.ctypes_oracle import CtypesOracle

    oracle = CtypesOracle(COMPILED_DIR)

    if args.baseline:
        _ctypes_baseline(oracle, stages, out_dir)
        return 0

    if args.auto:
        _ctypes_auto_bisect(oracle, stages, out_dir)
        return 0

    if args.skip:
        skip_names = [n.strip() for n in args.skip.split(",")]
        skip_indices = [
            i for i, s in enumerate(stages) if any(n in s.name for n in skip_names)
        ]
        if not skip_indices:
            _log.error("No stages matched: %s", skip_names)
            return 1
        label = f"skip_{'_'.join(skip_names)}"
        _ctypes_skip_and_test(oracle, stages, out_dir, skip_indices, label)
        return 0

    if args.suspects:
        # Test each suspect group individually using ctypes oracle
        suspects = {
            "ensure_filled": {"indices": [5], "label": "skip_ensure_filled"},
            "tile_matmuls": {"indices": [2, 3], "label": "skip_tile"},
            "fma_fusion": {"indices": [26], "label": "skip_fma"},
        }

        _log.info("\n=== Baseline (--ctypes) ===")
        baseline_cos = _ctypes_baseline(oracle, stages, out_dir)
        results = []

        for name, cfg in suspects.items():
            _log.info("\n=== Testing: skip %s ===", name)
            cos = _ctypes_skip_and_test(
                oracle, stages, out_dir, cfg["indices"], cfg["label"]
            )
            if cos is not None:
                delta = cos - baseline_cos
                flag = " ← SIGNIFICANT" if abs(delta) > args.threshold else ""
                _log.info(
                    "  => cos=%.6f  Δ=%+.6f%s", cos, delta, flag
                )
                results.append({"stage": name, "cos": cos})

        _check_signal_saturation(results)
        _log.info("\n=== All suspect tests done ===")
        return 0

    # Default --ctypes mode: skip each stage individually
    _log.info("\n=== Baseline (--ctypes) ===")
    baseline_cos = _ctypes_baseline(oracle, stages, out_dir)
    _log.info("Threshold for significance: |Δ| > %.4f", args.threshold)

    _log.info("\n=== Testing each stage by skipping it ===")
    results = []
    for i, s in enumerate(stages):
        trial = make_pipeline_without(stages, i)
        dylib_path = compile_with_stages(
            trial, out_dir, f"skip_stage_{i}"
        )
        if dylib_path is None:
            _log.warning("  [%2d] %s — compilation failed, skipping", i, s.name)
            continue

        cos = oracle.compare(dylib_path)
        delta = cos - baseline_cos
        flag = " ← SIGNIFICANT" if abs(delta) > args.threshold else ""
        _log.info("  [%2d] %-30s skip   cos=%.6f  Δ=%+.6f%s", i, s.name, cos, delta, flag)
        results.append({"stage": i, "cos": cos})

    _check_signal_saturation(results)
    _log.info("\n=== All stage-skip tests done ===")
    return 0


def main():
    init_logging()
    logging.getLogger().setLevel(logging.INFO)
    # Ensure handler level matches (init_logging defaults to WARNING handler)
    for _handler in logging.getLogger().handlers:
        _handler.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Bisect BUILTIN_STAGES")
    parser.add_argument("--auto", action="store_true", help="Automatic binary search")
    parser.add_argument("--skip", type=str, default=None,
                        help="Skip stages matching name (comma-sep: 'tile,fma')")
    parser.add_argument("--baseline", action="store_true", help="Run baseline test")
    parser.add_argument("--suspects", action="store_true", default=True,
                        help="Test suspect stages (default)")
    parser.add_argument("--ctypes", action="store_true",
                        help="Use ctypes oracle (cosine comparison) instead of Rust argmax test")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Cosine difference threshold for significance (default: 0.01)")
    args = parser.parse_args()

    _setup_mlir()
    if not _has_bindings():
        _log.error("MLIR bindings not available")
        return 1

    out_dir = "compiled/bisect_work"
    os.makedirs(out_dir, exist_ok=True)

    stages = parse_stages()
    _log.info(f"BUILTIN_STAGES: {len(stages)} stages")
    for i, s in enumerate(stages):
        _log.info(f"  [{i:2d}] {s.name:30s} pipeline={s.pipeline or '(action)'!r:40s} timeout={s.timeout} warn={s.warn_only}")

    if args.ctypes:
        return _run_ctypes_bisect(args, stages, out_dir)

    if args.baseline:
        baseline_test(out_dir)
        return 0

    if args.auto:
        auto_bisect(stages, out_dir)
        return 0

    if args.skip:
        skip_names = [n.strip() for n in args.skip.split(",")]
        skip_indices = [i for i, s in enumerate(stages)
                        if any(n in s.name for n in skip_names)]
        if not skip_indices:
            _log.error(f"No stages matched: {skip_names}")
            return 1
        skip_and_test(stages, out_dir, skip_indices, f"skip_{'_'.join(skip_names)}")
        return 0

    # Default: test suspect stages individually
    suspects = {
        "ensure_filled": {"indices": [5], "label": "skip_ensure_filled"},
        "tile_matmuls": {"indices": [2, 3], "label": "skip_tile"},
        "fma_fusion": {"indices": [26], "label": "skip_fma"},
    }

    _log.info("\n=== Baseline test ===")
    base = baseline_test(out_dir)
    if base is None:
        _log.error("Baseline failed — cannot proceed")
        return 1

    for name, cfg in suspects.items():
        _log.info(f"\n=== Testing: skip {name} ===")
        skip_and_test(stages, out_dir, cfg["indices"], cfg["label"])

    _log.info("\n=== All suspect tests done ===")
    return 0


if __name__ == "__main__":
    main()
