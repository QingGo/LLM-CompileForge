"""Quantized GEMM kernels for LLM-ServeForge Phase 2.

Phase 2 kernels:
  - w4a16_gemm.py  — W4A16 GEMM: INT4-packed weights × FP16 activations
  - w8a8_gemm.py   — W8A8 GEMM: INT8 weights × INT8 activations

All kernels support CPU fallback (PyTorch reference) when Triton
is not available (e.g. macOS, non-GPU environments).

Usage:
    from kernels.quantize.w4a16_gemm import w4a16_gemm
    output = w4a16_gemm(activation, weight_packed, scale, zero)
"""

from __future__ import annotations

try:
    import triton  # noqa: F401

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
