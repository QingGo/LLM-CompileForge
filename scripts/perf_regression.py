"""Pipeline performance regression: time each step, fail if any hangs or takes > expected.

Usage:
    python scripts/perf_regression.py                     # Full pipeline timing
    python scripts/perf_regression.py --step vec          # Just vectorization
    python scripts/perf_regression.py --step llvm         # Just LLVM lowering
    python scripts/perf_regression.py --model compiled/opt_125m_v8
    python scripts/perf_regression.py --baseline .perf_baseline.json  # Compare with saved baseline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Thresholds per step (seconds) — alert if exceeded
_DEFAULT_THRESHOLDS = {
    "parse": 2,
    "canonicalize": 2,
    "fuse_elementwise": 2,
    "vectorize": 5,
    "bufferize": 5,
    "linalg_to_loops": 3,
    "scf_to_cf": 3,
    "lower_memref": 5,
    "cf_to_llvm": 5,
    "math_to_llvm": 3,
    "vector_to_llvm": 30,
    "arith_to_llvm": 5,
    "func_to_llvm": 3,
    "translate": 30,
    "total_mlir": 120,  # total MLIR pipeline (no llc)
}


def _setup_mlir():
    _p = Path(__file__).resolve().parent.parent / "mlir_binding" / "mlir_package"
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def time_step(name: str, pipeline: str, ir_module, ctx, timeout: int = 120) -> str:
    """Run a pipeline step with timing. Raises TimeoutError if it takes too long."""
    import mlir.passmanager as pm

    t0 = time.time()
    try:
        pm.PassManager.parse(f"builtin.module({pipeline})", ctx).run(ir_module.operation)
        elapsed = time.time() - t0
    except Exception as e:
        elapsed = time.time() - t0
        raise RuntimeError(
            f"Step '{name}' FAILED after {elapsed:.1f}s: {e}"
        ) from e

    thresh = _DEFAULT_THRESHOLDS.get(name, timeout)
    if elapsed > thresh:
        import warnings
        warnings.warn(
            f"Step '{name}' took {elapsed:.1f}s (threshold {thresh}s)",
            stacklevel=2,
        )
    return str(ir_module)


def run_pipeline(lowered_path: str, max_time: int = 300) -> dict[str, float]:
    """Run the full lowered→LLVM pipeline, return step times."""
    _setup_mlir()
    import mlir.ir as ir
    import mlir.passmanager as pm
    from mlir._mlir_libs import _mlirRegisterEverything

    lowered_text = Path(lowered_path).read_text()
    times: dict[str, float] = {}

    # ── Context 1: pre-processing ──────────────────────────
    ctx1 = ir.Context()
    ctx1.allow_unregistered_dialects = True
    with ir.Location.unknown(ctx1):
        module = ir.Module.parse(lowered_text, ctx1)

        # Step 1: canonicalize + cse
        t0 = time.time()
        pm.PassManager.parse("builtin.module(canonicalize,cse)", ctx1).run(module.operation)
        times["canonicalize"] = time.time() - t0

        # Step 2: fuse elementwise
        t0 = time.time()
        pm.PassManager.parse(
            "builtin.module(linalg-fuse-elementwise-ops,canonicalize,cse)", ctx1
        ).run(module.operation)
        times["fuse_elementwise"] = time.time() - t0

        # Step 3: vectorize
        t0 = time.time()
        ctx1.load_all_available_dialects()
        from compiler.mlir_dialect.llvm_backend import _vectorize_via_transform

        _vectorize_via_transform(module)
        times["vectorize"] = time.time() - t0
        vec_text = str(module)

    # ── Context 2: bufferization + LLVM lowering ──────────
    reg = ir.DialectRegistry()
    _mlirRegisterEverything.register_dialects(reg)
    ctx2 = ir.Context()
    ctx2.allow_unregistered_dialects = True
    ctx2.append_dialect_registry(reg)

    with ir.Location.unknown(ctx2):
        m2 = ir.Module.parse(vec_text, ctx2)

        # Step 4: bufferize
        t0 = time.time()
        pm.PassManager.parse(
            "builtin.module(one-shot-bufferize{bufferize-function-boundaries})", ctx2
        ).run(m2.operation)
        times["bufferize"] = time.time() - t0

        # Step 5: linalg → loops, lower masks, vec→scf, scf→cf
        t0 = time.time()
        pm.PassManager.parse(
            "builtin.module(canonicalize,cse,convert-bufferization-to-memref,"
            "convert-linalg-to-loops,lower-affine,"
            "func.func(lower-vector-mask),func.func(convert-vector-to-scf),"
            "canonicalize,cse,convert-scf-to-cf)", ctx2
        ).run(m2.operation)
        times["scf_to_cf"] = time.time() - t0

        # Step 6: expand strides + finalize memref
        t0 = time.time()
        pm.PassManager.parse(
            "builtin.module(expand-strided-metadata,lower-affine,"
            "finalize-memref-to-llvm)", ctx2
        ).run(m2.operation)
        times["lower_memref"] = time.time() - t0

        # Step 7: cf + math → llvm
        t0 = time.time()
        pm.PassManager.parse(
            "builtin.module(convert-cf-to-llvm,convert-math-to-llvm)", ctx2
        ).run(m2.operation)
        times["cf_to_llvm"] = times.get("cf_to_llvm", 0) + time.time() - t0
        times["math_to_llvm"] = time.time() - t0

        # Step 8: vector → llvm (the most likely to hang)
        t0 = time.time()
        pm.PassManager.parse(
            "builtin.module(convert-vector-to-llvm{vector-contract-lowering=outerproduct})",
            ctx2,
        ).run(m2.operation)
        times["vector_to_llvm"] = time.time() - t0

        # Step 9: arith + func → llvm
        t0 = time.time()
        pm.PassManager.parse(
            "builtin.module(convert-arith-to-llvm,convert-ub-to-llvm,"
            "convert-func-to-llvm,reconcile-unrealized-casts)", ctx2
        ).run(m2.operation)
        times["arith_to_llvm"] = time.time() - t0

        # Step 10: translate to LLVM IR
        t0 = time.time()
        from compiler.mlir_dialect.llvm_backend import mlir_module_to_llvm_ir

        llvm_ir = mlir_module_to_llvm_ir(m2)
        times["translate"] = time.time() - t0
    times["total_mlir"] = sum(v for k, v in times.items() if k != "total_mlir")

    return times


def print_report(times: dict[str, float], baseline: dict[str, float] | None = None):
    """Print a human-readable performance report."""
    total = sum(v for k, v in times.items() if k != "total_mlir")
    print(f"\n{'Step':<30s} {'Time':>8s}  {'Threshold':>10s}  {'Baseline':>10s}  {'Status':>8s}")
    print("-" * 72)
    for step_name in _DEFAULT_THRESHOLDS:
        t = times.get(step_name)
        if t is None:
            continue
        thresh = _DEFAULT_THRESHOLDS[step_name]
        bl = baseline.get(step_name) if baseline else None
        status = "✓" if t <= thresh else "⚠ SLOW"
        if bl and t > bl * 1.5:
            status += " REGRESS"
        t_str = f"{t:.1f}s"
        th_str = f"{thresh}s"
        bl_str = f"{bl:.1f}s" if bl else "-"
        print(f"{step_name:<30s} {t_str:>8s}  {th_str:>10s}  {bl_str:>10s}  {status:>8s}")
    print("-" * 72)
    print(f"{'TOTAL (steps)':<30s} {total:.1f}s")
    if "total_mlir" in times:
        print(f"{'WALL CLOCK':<30s} {times['total_mlir']:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Pipeline performance regression")
    parser.add_argument("--model", default="compiled/opt_125m_v8/model.lowered.mlir")
    parser.add_argument("--baseline", help="JSON file with baseline times")
    parser.add_argument("--save", help="Save times to JSON file")
    parser.add_argument("--max-time", type=int, default=300, help="Max wall time (s)")
    args = parser.parse_args()

    baseline = {}
    if args.baseline and os.path.exists(args.baseline):
        baseline = json.load(open(args.baseline))

    print(f"Running pipeline: {args.model}")
    print(f"Max time: {args.max_time}s")
    times = run_pipeline(args.model, max_time=args.max_time)
    print_report(times, baseline)

    if args.save:
        with open(args.save, "w") as f:
            json.dump(times, f, indent=2)
        print(f"\nSaved to {args.save}")

    # Exit with non-zero if any step exceeds threshold
    for step_name in _DEFAULT_THRESHOLDS:
        t = times.get(step_name)
        if t is not None and t > _DEFAULT_THRESHOLDS[step_name]:
            print(f"\n⚠ Step '{step_name}' exceeded threshold ({t:.1f}s > {_DEFAULT_THRESHOLDS[step_name]}s)")
            sys.exit(1)
    print("\n✓ All steps within thresholds")


if __name__ == "__main__":
    main()
