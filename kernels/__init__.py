"""Triton custom kernels for LLM-ServeForge.

Phase 1 kernels:
  - flash_attention.py  — FlashAttention-2 forward pass
  - paged_attention.py  — PagedAttention with block table
  - rms_norm.py         — RMSNorm + residual fusion

Phase 2 kernels (quantize/):
  - w4a16_gemm.py       — W4A16 GEMM: INT4 weights × FP16 activations
  - w8a8_gemm.py        — W8A8 GEMM: INT8 weights × INT8 activations

All kernels support CPU fallback (PyTorch reference) when Triton
is not available (e.g. macOS, non-GPU environments).

Usage:
    from kernels.flash_attention import flash_attention_fwd
    output = flash_attention_fwd(q, k, v, causal=True)
"""

from __future__ import annotations

try:
    import triton  # noqa: F401
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
