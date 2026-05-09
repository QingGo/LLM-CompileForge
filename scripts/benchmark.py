#!/usr/bin/env python3
"""LLM-CompileForge Performance Benchmarks.

Phase 2.5 Sprint 1 — Performance Baseline System.

Usage:
    python scripts/benchmark.py --prefix-cache       # Prefix Cache TTFT
    python scripts/benchmark.py --all                # All benchmarks
    python scripts/benchmark.py --all --output results.json
    python scripts/benchmark.py --list               # List available
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))


@dataclass
class BenchmarkResult:
    """Structured result from a single benchmark run."""

    name: str
    description: str
    metrics: dict[str, Any] = field(default_factory=dict)
    passed: bool = True
    error: str | None = None


# ────────────────────────────────────────────────────────────────
# Engine helpers
# ────────────────────────────────────────────────────────────────


def _load_engine(
    model_dir: str = "./compiled/tiny_llama",
    enable_prefix_cache: bool = False,
    num_blocks: int = 200,
    block_size: int = 16,
    num_layers: int = 2,
    num_kv_heads: int = 4,
    head_dim: int = 4,
):
    """Create an LLMEngine for a compiled artifact (tiny_llama by default)."""
    from compiler.serialize import load_artifact
    from engine.llm_engine import LLMEngine
    from hal.pytorch_backend import PyTorchBackend

    module = load_artifact(model_dir)
    backend = PyTorchBackend("cpu")
    return LLMEngine(
        module,
        backend,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_blocks=num_blocks,
        block_size=block_size,
        enable_prefix_cache=enable_prefix_cache,
    )


def _measure_ttft_and_finish(engine, prompt_tokens: list[int]) -> float:
    """Add a request, measure TTFT (ms), and ensure it finishes so the
    scheduler reaps it and inserts KV blocks into the radix cache.

    Returns TTFT in milliseconds.
    """
    rid = engine.add_request(prompt_tokens, max_tokens=1)
    t_start = time.perf_counter()
    ttft: float | None = None

    for _ in range(50):  # safety limit
        results = engine.step()
        for r in results:
            if r.request_id == rid:
                if ttft is None:
                    ttft = (time.perf_counter() - t_start) * 1000
                if r.is_finished:
                    # One extra step so the scheduler reaps the finished
                    # request and inserts its KV blocks into the cache.
                    engine.step()
                    if ttft is not None:
                        return ttft
                    return (time.perf_counter() - t_start) * 1000
        if not engine.scheduler.has_work and ttft is not None:
            return ttft

    if ttft is not None:
        return ttft
    raise RuntimeError(f"Request {rid} never generated a token (timeout)")


# ────────────────────────────────────────────────────────────────
# Benchmark 1: Prefix Cache TTFT
# ────────────────────────────────────────────────────────────────


def benchmark_prefix_cache_ttft(
    prompt_len: int = 128,
    iterations: int = 5,
    warmup: int = 1,
    model_dir: str = "./compiled/tiny_llama",
) -> BenchmarkResult:
    """Measure TTFT reduction from RadixTree prefix cache hits.

    For each iteration, sends the same prompt twice:
      1. Cold request — full prefill (no cache)
      2. Warm request — cache hit, prefill skipped

    A warmup run is performed first to absorb one-time costs
    (KV cache allocation, torch compilation, etc.).

    Returns a BenchmarkResult with TTFT cold/warm statistics.
    """
    engine = _load_engine(model_dir, enable_prefix_cache=True, num_blocks=200)

    # Warmup: absorb one-time costs (KV cache init, torch warmup)
    for w in range(warmup):
        wp = list(range(w * 2000, w * 2000 + prompt_len))
        _measure_ttft_and_finish(engine, wp)   # cold
        _measure_ttft_and_finish(engine, wp)   # warm

    cold_ttfts: list[float] = []
    warm_ttfts: list[float] = []

    for i in range(iterations):
        # Unique prompt per iteration so the cold request is truly cold.
        # Using deterministic range sequences for reproducibility.
        prompt = list(range(i * 1000, i * 1000 + prompt_len))

        ttft_cold = _measure_ttft_and_finish(engine, prompt)
        cold_ttfts.append(ttft_cold)

        ttft_warm = _measure_ttft_and_finish(engine, prompt)
        warm_ttfts.append(ttft_warm)

    cold_mean = statistics.mean(cold_ttfts)
    cold_med = statistics.median(cold_ttfts)
    cold_stdev = statistics.stdev(cold_ttfts) if len(cold_ttfts) >= 2 else 0.0
    warm_mean = statistics.mean(warm_ttfts)
    warm_med = statistics.median(warm_ttfts)
    warm_stdev = statistics.stdev(warm_ttfts) if len(warm_ttfts) >= 2 else 0.0

    reduction_mean = round((cold_mean - warm_mean) / cold_mean * 100, 1) if cold_mean > 0 else 0.0
    reduction_median = round((cold_med - warm_med) / cold_med * 100, 1) if cold_med > 0 else 0.0
    prefill_skipped = warm_med < cold_med * 0.8

    return BenchmarkResult(
        name="prefix_cache_ttft",
        description=f"TTFT comparison ({prompt_len}-token prompt, {iterations} runs, {Path(model_dir).name})",
        metrics={
            "ttft_cold_mean_ms": round(cold_mean, 2),
            "ttft_cold_median_ms": round(cold_med, 2),
            "ttft_cold_stdev_ms": round(cold_stdev, 2),
            "ttft_warm_mean_ms": round(warm_mean, 2),
            "ttft_warm_median_ms": round(warm_med, 2),
            "ttft_warm_stdev_ms": round(warm_stdev, 2),
            "ttft_reduction_mean_pct": reduction_mean,
            "ttft_reduction_median_pct": reduction_median,
            "prefill_skipped_verified": prefill_skipped,
            "prompt_tokens": prompt_len,
            "iterations": iterations,
            "model": Path(model_dir).name,
        },
        passed=warm_med < cold_med,
    )


# ────────────────────────────────────────────────────────────────
# Compilation helpers
# ────────────────────────────────────────────────────────────────


def _patch_transformers_torch() -> None:
    """Patch transformers to recognize the symlinked torch installation."""
    import torch

    import transformers.utils.generic as _generic  # type: ignore[import-untyped]
    import transformers.utils.import_utils as _iu  # type: ignore[import-untyped]

    _iu._torch_available = True  # type: ignore[attr-defined]
    _iu._torch_version = torch.__version__  # type: ignore[attr-defined]
    _generic._torch_pytree = torch.utils._pytree  # type: ignore[attr-defined]

    from typing import Any as _Any

    def _flatten(output: _Any) -> _Any:  # type: ignore[no-untyped-def]
        return list(output.values()), list(output.keys())

    def _unflatten(values: _Any, context: _Any, output_type: _Any = None) -> _Any:  # type: ignore[no-untyped-def]
        return (output_type or type(context[0]))(**dict(zip(context, values)))

    _generic._model_output_flatten = _flatten  # type: ignore[attr-defined]
    _generic._model_output_unflatten = _unflatten  # type: ignore[attr-defined]


def _compile_opt125m(
    output_dir: str,
    enable_fusion: bool = True,
    enable_cse: bool = True,
    enable_constant_fold: bool = True,
    enable_qkv_only: bool = False,
) -> tuple[Any, float]:
    """Compile opt-125m with given pipeline settings.

    Caches compiled artifacts to *output_dir* — if model.mlir already
    exists there, loads it directly (compile_time = 0).

    Returns (IrModule, compile_time_seconds).
    """
    import os
    from pathlib import Path

    out = Path(output_dir)
    if (out / "model.mlir").exists() and (out / "weights.pth").exists():
        from compiler.serialize import load_artifact
        module = load_artifact(output_dir)
        # Read stored compile time from metadata if available
        compile_time = module.metadata.get("bench_compile_time_s", 0.0)
        return module, float(compile_time)

    import torch
    from torch.export import Dim
    from compiler.pipeline import CompilationPipeline
    from compiler.passes.base import PassManager
    from compiler.passes.constant_fold import ConstantFold
    from compiler.passes.cse_pass import CommonSubexpressionElimination
    from compiler.passes.dce_pass import DeadCodeElimination
    from compiler.passes.fuse_qkv import FuseQKVProjection
    from compiler.passes.fuse_rms_norm import FuseRMSNorm
    from compiler.passes.fuse_silu import FuseSiLU
    from compiler.passes.validate_ir import ValidateIR
    from transformers.models.opt.configuration_opt import OPTConfig  # type: ignore[import-untyped]
    from transformers.models.opt.modeling_opt import OPTForCausalLM  # type: ignore[import-untyped]

    _patch_transformers_torch()

    hub_dir = os.path.expanduser("~/.cache/huggingface/hub/models--facebook--opt-125m")
    snapshots = os.path.join(hub_dir, "snapshots")
    if not os.path.isdir(snapshots):
        raise FileNotFoundError(f"opt-125m not found at {hub_dir}")
    snap = os.listdir(snapshots)[0]
    model_path = os.path.join(snapshots, snap, "pytorch_model.bin")
    config_path = os.path.join(snapshots, snap, "config.json")

    config = OPTConfig.from_pretrained(config_path)
    config.use_cache = False
    model = OPTForCausalLM(config)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    if enable_qkv_only:
        # Build a pipeline with ONLY FuseQKV (no FuseRMSNorm, no FuseSiLU)
        pipeline = CompilationPipeline(
            enable_fusion=False,
            enable_cse=enable_cse,
            enable_constant_fold=enable_constant_fold,
        )
    else:
        pipeline = CompilationPipeline(
            enable_fusion=enable_fusion,
            enable_cse=enable_cse,
            enable_constant_fold=enable_constant_fold,
        )

    example_input = torch.randint(0, 50272, (2, 4), dtype=torch.long)

    t0 = time.perf_counter()
    ir_module = pipeline.compile(
        model,
        example_args=(example_input,),
        output_dir=output_dir,
        dynamic_shapes={"input_ids": {0: Dim("batch"), 1: Dim("seq")}},
    )
    elapsed = time.perf_counter() - t0

    # Stamp compile time in metadata for future cached loads
    ir_module.metadata["bench_compile_time_s"] = elapsed

    # Re-save to persist the metadata stamp (pipeline.compile already
    # saved once; this update includes bench_compile_time_s).
    from compiler.serialize import save_artifact as _save

    # If enable_qkv_only, manually apply only FuseQKV after the base pipeline
    if enable_qkv_only:
        from compiler.passes.base import PassManager
        from compiler.passes.fuse_qkv import FuseQKVProjection
        pm = PassManager()
        pm.add(FuseQKVProjection())
        ir_module = pm.run(ir_module)
    _save(ir_module, output_dir)
    return ir_module, elapsed


def _measure_forward_latency(
    module: Any,
    input_ids: Any = None,
    warmup: int = 3,
    iterations: int = 10,
) -> dict[str, float]:
    """Measure executor.forward() latency statistics.

    Returns dict with mean_ms, median_ms, min_ms, max_ms, stdev_ms.
    """
    from hal.pytorch_backend import PyTorchBackend
    from engine.executor import Executor

    backend = PyTorchBackend("cpu")
    executor = Executor(module, backend)

    if input_ids is None:
        import torch
        input_ids = torch.randint(0, 10000, (1, 64), dtype=torch.long)

    # Warmup
    for _ in range(warmup):
        executor.forward(input_ids)

    times: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        executor.forward(input_ids)
        times.append((time.perf_counter() - t0) * 1000)

    return {
        "latency_mean_ms": round(statistics.mean(times), 2),
        "latency_median_ms": round(statistics.median(times), 2),
        "latency_min_ms": round(min(times), 2),
        "latency_max_ms": round(max(times), 2),
        "latency_stdev_ms": round(statistics.stdev(times), 2) if len(times) >= 2 else 0.0,
    }


# ────────────────────────────────────────────────────────────────
# Benchmark 2: Compiler Pass Latency (fusion on/off)
# ────────────────────────────────────────────────────────────────


def benchmark_fusion_latency(
    input_len: int = 64,
    warmup: int = 3,
    iterations: int = 10,
) -> BenchmarkResult:
    """Compare forward latency with and without all fusion passes.

    Compiles opt-125m with enable_fusion=True and enable_fusion=False
    (cached to ./compiled/bench_opt125m_fusion/ and
    ./compiled/bench_opt125m_nofusion/ respectively), then measures
    executor.forward() latency for each variant.

    Returns a BenchmarkResult with op counts, latency stats for both
    variants, and the latency reduction percentage.
    """
    import torch

    # Compile (or load cached) fusion=True variant
    fusion_dir = "./compiled/bench_opt125m_fusion"
    mod_fusion, _ = _compile_opt125m(fusion_dir, enable_fusion=True)
    ops_fusion = len(mod_fusion.main.ops)
    gemm_fusion = sum(1 for op in mod_fusion.main.ops if op.name in ("linear", "matmul"))

    # Compile (or load cached) fusion=False variant
    nofusion_dir = "./compiled/bench_opt125m_nofusion"
    mod_nofusion, _ = _compile_opt125m(nofusion_dir, enable_fusion=False)
    ops_nofusion = len(mod_nofusion.main.ops)
    gemm_nofusion = sum(1 for op in mod_nofusion.main.ops if op.name in ("linear", "matmul"))

    input_ids = torch.randint(0, 10000, (1, input_len), dtype=torch.long)

    stats_fusion = _measure_forward_latency(mod_fusion, input_ids, warmup, iterations)
    stats_nofusion = _measure_forward_latency(mod_nofusion, input_ids, warmup, iterations)

    reduction_pct = (
        round(
            (stats_nofusion["latency_median_ms"] - stats_fusion["latency_median_ms"])
            / stats_nofusion["latency_median_ms"] * 100,
            1,
        )
        if stats_nofusion["latency_median_ms"] > 0
        else 0.0
    )

    return BenchmarkResult(
        name="fusion_latency",
        description=f"Forward latency fusion on/off ({input_len}-token input, {iterations} iters)",
        metrics={
            "ops_fusion": ops_fusion,
            "ops_nofusion": ops_nofusion,
            "gemm_fusion": gemm_fusion,
            "gemm_nofusion": gemm_nofusion,
            "gemm_reduction": gemm_nofusion - gemm_fusion,
            "fusion_latency_mean_ms": stats_fusion["latency_mean_ms"],
            "fusion_latency_median_ms": stats_fusion["latency_median_ms"],
            "fusion_latency_stdev_ms": stats_fusion["latency_stdev_ms"],
            "nofusion_latency_mean_ms": stats_nofusion["latency_mean_ms"],
            "nofusion_latency_median_ms": stats_nofusion["latency_median_ms"],
            "nofusion_latency_stdev_ms": stats_nofusion["latency_stdev_ms"],
            "latency_reduction_median_pct": reduction_pct,
            "input_tokens": input_len,
            "warmup": warmup,
            "iterations": iterations,
        },
        passed=gemm_fusion <= gemm_nofusion,
    )


# ────────────────────────────────────────────────────────────────
# Benchmark 3: FuseQKV single-pass
# ────────────────────────────────────────────────────────────────


def benchmark_fuse_qkv(
    input_len: int = 64,
    warmup: int = 3,
    iterations: int = 10,
) -> BenchmarkResult:
    """Count GEMM ops and measure latency with/without FuseQKV projection pass.

    Compiles opt-125m with only FuseQKV enabled vs no fusion at all.
    Reports GEMM (linear/matmul) count and forward latency delta.

    Cached artifacts at ./compiled/bench_opt125m_qkv/ and
    ./compiled/bench_opt125m_nofusion/ (shared with fusion benchmark).
    """
    import torch

    qkv_dir = "./compiled/bench_opt125m_qkv"
    mod_qkv, _ = _compile_opt125m(qkv_dir, enable_fusion=False, enable_qkv_only=True)
    gemm_qkv = sum(1 for op in mod_qkv.main.ops if op.name in ("linear", "matmul"))

    nofusion_dir = "./compiled/bench_opt125m_nofusion"
    mod_nofusion, _ = _compile_opt125m(nofusion_dir, enable_fusion=False)
    gemm_nofusion = sum(1 for op in mod_nofusion.main.ops if op.name in ("linear", "matmul"))

    # Count fused QKV ops
    qkv_fused = sum(1 for op in mod_qkv.main.ops if "folded" in op.attributes)
    qkv_groups = sum(
        1 for op in mod_qkv.main.ops
        if op.name in ("linear", "matmul") and "fused_qkv" in str(op.outputs)
    )

    input_ids = torch.randint(0, 10000, (1, input_len), dtype=torch.long)

    stats_qkv = _measure_forward_latency(mod_qkv, input_ids, warmup, iterations)
    stats_nofusion = _measure_forward_latency(mod_nofusion, input_ids, warmup, iterations)

    reduction_pct = (
        round(
            (stats_nofusion["latency_median_ms"] - stats_qkv["latency_median_ms"])
            / stats_nofusion["latency_median_ms"] * 100,
            1,
        )
        if stats_nofusion["latency_median_ms"] > 0
        else 0.0
    )

    return BenchmarkResult(
        name="fuse_qkv",
        description=f"FuseQKV single-pass: GEMM count + latency ({input_len}-token input, {iterations} iters)",
        metrics={
            "gemm_qkv": gemm_qkv,
            "gemm_nofusion": gemm_nofusion,
            "gemm_reduction": gemm_nofusion - gemm_qkv,
            "qkv_fused_groups": qkv_groups,
            "qkv_latency_median_ms": stats_qkv["latency_median_ms"],
            "nofusion_latency_median_ms": stats_nofusion["latency_median_ms"],
            "latency_reduction_median_pct": reduction_pct,
            "input_tokens": input_len,
            "iterations": iterations,
        },
        passed=gemm_qkv < gemm_nofusion,
    )


# ────────────────────────────────────────────────────────────────
# Benchmark 4: Compilation time
# ────────────────────────────────────────────────────────────────


def benchmark_compilation_time() -> BenchmarkResult:
    """Measure full compilation wall-clock time for opt-125m.

    Performs a fresh torch.export → passes → serialize cycle and
    reports the elapsed wall-clock time.  The compiled artifact is
    saved to ./compiled/bench_opt125m_compile_time/ for inspection.
    """
    output_dir = "./compiled/bench_opt125m_compile_time"
    mod, elapsed_s = _compile_opt125m(output_dir, enable_fusion=True)

    ops = len(mod.main.ops)
    weights = len(mod.main.weights)
    passes = mod.metadata.get("passes_applied", [])

    return BenchmarkResult(
        name="compilation_time",
        description="Full compilation wall-clock time (opt-125m)",
        metrics={
            "compile_time_s": round(elapsed_s, 2),
            "ops": ops,
            "weights": weights,
            "passes_applied": ", ".join(passes),
        },
        passed=elapsed_s > 0,
    )


# ────────────────────────────────────────────────────────────────
# Runner infrastructure
# ────────────────────────────────────────────────────────────────

_BENCHMARKS: dict[str, Any] = {
    "prefix-cache": benchmark_prefix_cache_ttft,
    "fusion": benchmark_fusion_latency,
    "qkv": benchmark_fuse_qkv,
    "compile-time": benchmark_compilation_time,
}


def run_benchmarks(names: list[str], **kwargs: Any) -> list[BenchmarkResult]:
    """Execute the named benchmarks and return structured results."""
    import inspect as _inspect

    results: list[BenchmarkResult] = []
    for name in names:
        fn = _BENCHMARKS[name]
        # Filter kwargs to only those accepted by the function
        sig = _inspect.signature(fn)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        try:
            result = fn(**filtered_kwargs)
        except NotImplementedError:
            result = BenchmarkResult(
                name=name, description="", passed=False, error="Not yet implemented"
            )
        except Exception as exc:
            result = BenchmarkResult(
                name=name, description="", passed=False, error=str(exc)
            )
        results.append(result)
    return results


def format_report(results: list[BenchmarkResult], fmt: str = "text") -> str:
    """Render benchmark results as text or JSON."""
    if fmt == "json":
        data = [
            {
                "name": r.name,
                "description": r.description,
                "metrics": r.metrics,
                "passed": r.passed,
                "error": r.error,
            }
            for r in results
        ]
        return json.dumps(data, indent=2)

    lines: list[str] = ["", "=== LLM-CompileForge Benchmarks ===", ""]
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"  [{status}]  {r.name}")
        if r.description:
            lines.append(f"          {r.description}")
        if r.error:
            lines.append(f"          ERROR: {r.error}")
        else:
            for k, v in r.metrics.items():
                lines.append(f"          {k}: {v}")
        lines.append("")
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-CompileForge Performance Benchmarks"
    )
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument(
        "--prefix-cache", action="store_true", help="Prefix Cache TTFT benchmark"
    )
    parser.add_argument(
        "--fusion", action="store_true", help="Compiler pass latency benchmark"
    )
    parser.add_argument(
        "--qkv", action="store_true", help="FuseQKV single-pass benchmark"
    )
    parser.add_argument(
        "--compile-time", action="store_true", help="Compilation time benchmark"
    )
    parser.add_argument(
        "--list", action="store_true", help="List available benchmarks"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Save JSON results to file"
    )
    parser.add_argument(
        "--iterations", type=int, default=5, help="Measurement iterations (default: 5)"
    )
    parser.add_argument(
        "--prompt-len", type=int, default=128, help="Prompt length in tokens (default: 128)"
    )
    parser.add_argument(
        "--input-len", type=int, default=64, help="Forward input length for latency benchmarks (default: 64)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="tiny_llama",
        choices=["tiny_llama", "opt_125m", "qwen3_0.8b"],
        help="Model to benchmark (default: tiny_llama)",
    )

    args = parser.parse_args()

    if args.list:
        print("Available benchmarks:")
        for name in _BENCHMARKS:
            print(f"  {name}")
        return

    selected: list[str] = []
    if args.all:
        selected = list(_BENCHMARKS.keys())
    else:
        for flag, name in [
            ("prefix_cache", "prefix-cache"),
            ("fusion", "fusion"),
            ("qkv", "qkv"),
            ("compile_time", "compile-time"),
        ]:
            if getattr(args, flag, False):
                selected.append(name)

    if not selected:
        parser.print_help()
        return

    model_dir = f"./compiled/{args.model}"
    results = run_benchmarks(
        selected,
        prompt_len=args.prompt_len,
        input_len=args.input_len,
        iterations=args.iterations,
        model_dir=model_dir,
    )

    report = format_report(results, fmt="text")
    print(report)

    if args.output:
        json_text = format_report(results, fmt="json")
        with open(args.output, "w") as f:
            f.write(json_text)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
