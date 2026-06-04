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

from compiler._lazy_imports import lazy_imports

lazy_imports(__name__, globals(), {
    "SmoothQuantCalibrator": ("compiler.quantize.smoothquant", "SmoothQuantCalibrator"),
    "AWQQuantizer": ("compiler.quantize.awq", "AWQQuantizer"),
    "FP8KVCacheQuantizer": ("compiler.quantize.fp8_kv_cache", "FP8KVCacheQuantizer"),
    "MixedPrecisionConfig": ("compiler.quantize.mixed_precision", "MixedPrecisionConfig"),
})
