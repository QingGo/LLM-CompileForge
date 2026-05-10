"""Triton custom kernels for LLM-ServeForge.

Phase 1 kernels:
  - flash_attention.py  — FlashAttention-2 forward pass
  - paged_attention.py  — PagedAttention with block table
  - rms_norm.py         — RMSNorm + residual fusion

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
