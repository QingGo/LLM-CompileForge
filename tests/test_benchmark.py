"""Benchmark tests — Phase 2.5 Sprint 1 performance baseline.

All tests marked with @pytest.mark.benchmark and NOT included in
make test-unit or make test-integration. Run explicitly with:

    make test-benchmark
    pytest tests/test_benchmark.py -m benchmark -v
"""

from __future__ import annotations

import pytest


@pytest.mark.benchmark
def test_prefix_cache_reduces_ttft() -> None:
    """Warm TTFT (cache hit) must be strictly lower than cold TTFT."""
    from scripts.benchmark import benchmark_prefix_cache_ttft

    result = benchmark_prefix_cache_ttft(prompt_len=256, iterations=3)
    assert result.passed, f"Prefix cache TTFT benchmark failed: {result.error}"
    assert result.error is None
    m = result.metrics
    assert m["ttft_warm_median_ms"] < m["ttft_cold_median_ms"], (
        f"Warm TTFT ({m['ttft_warm_median_ms']}ms) >= cold TTFT ({m['ttft_cold_median_ms']}ms)"
    )
    assert m["prefill_skipped_verified"], (
        f"Prefill not skipped (reduction={m['ttft_reduction_median_pct']}%)"
    )
    assert m["ttft_reduction_median_pct"] > 0, "TTFT reduction must be positive"


@pytest.mark.benchmark
def test_prefix_cache_multiple_iterations() -> None:
    """Multiple iterations should all show cache benefit."""
    from scripts.benchmark import benchmark_prefix_cache_ttft

    result = benchmark_prefix_cache_ttft(prompt_len=128, iterations=5)
    assert result.passed
    m = result.metrics
    assert m["ttft_reduction_median_pct"] > 0
    assert m["prefill_skipped_verified"]


@pytest.mark.benchmark
def test_compilation_time_completes() -> None:
    """Compilation should finish and produce a valid artifact."""
    from scripts.benchmark import benchmark_compilation_time

    result = benchmark_compilation_time()
    assert result.passed, f"Compilation time benchmark failed: {result.error}"
    assert result.error is None
    assert result.metrics["compile_time_s"] > 0
    assert result.metrics["ops"] > 0
    assert result.metrics["weights"] > 0


@pytest.mark.benchmark
def test_fusion_reduces_gemm_ops() -> None:
    """Fusion must reduce GEMM (linear/matmul) op count."""
    from scripts.benchmark import benchmark_fusion_latency

    result = benchmark_fusion_latency(input_len=32, warmup=1, iterations=3)
    assert result.passed, f"Fusion latency benchmark failed: {result.error}"
    assert result.error is None
    m = result.metrics
    assert m["gemm_fusion"] < m["gemm_nofusion"], (
        f"Fusion GEMM ({m['gemm_fusion']}) >= nofusion GEMM ({m['gemm_nofusion']})"
    )
    assert m["gemm_reduction"] > 0


@pytest.mark.benchmark
def test_fuse_qkv_reduces_gemm_ops() -> None:
    """FuseQKV must reduce GEMM count."""
    from scripts.benchmark import benchmark_fuse_qkv

    result = benchmark_fuse_qkv(input_len=32, warmup=1, iterations=3)
    assert result.passed, f"FuseQKV benchmark failed: {result.error}"
    assert result.error is None
    m = result.metrics
    assert m["gemm_qkv"] < m["gemm_nofusion"], (
        f"QKV GEMM ({m['gemm_qkv']}) >= nofusion GEMM ({m['gemm_nofusion']})"
    )
    assert m["gemm_reduction"] > 0
