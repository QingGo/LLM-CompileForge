"""Quantization toolchain — Phase 2 Module A.

Provides calibration, smoothing, and quantization pipelines for
LLM weight compression integrated with the MLIR compilation pipeline.

Sub-modules:
  - smoothquant.py  — SmoothQuant (W8A8) calibration + quantization
  - awq.py          — AWQ (W4A16) salient-channel weight quantization
  - fp8_kv_cache.py — FP8 per-block KV cache quantization
  - mixed_precision.py — Per-layer precision strategy config

Reference: design-phase2.md §2.1
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "SmoothQuantCalibrator",
    "AWQQuantizer",
    "FP8KVCacheQuantizer",
    "MixedPrecisionConfig",
]

_LAZY_ATTRS = frozenset(__all__)


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        import compiler.quantize.awq as _awq
        import compiler.quantize.fp8_kv_cache as _fp8
        import compiler.quantize.mixed_precision as _mp
        import compiler.quantize.smoothquant as _sq

        _globals = {
            "SmoothQuantCalibrator": _sq.SmoothQuantCalibrator,
            "AWQQuantizer": _awq.AWQQuantizer,
            "FP8KVCacheQuantizer": _fp8.FP8KVCacheQuantizer,
            "MixedPrecisionConfig": _mp.MixedPrecisionConfig,
        }
        if name in _globals:
            value = _globals[name]
            globals()[name] = value
            return value
    raise AttributeError(f"module 'compiler.quantize' has no attribute '{name}'")
