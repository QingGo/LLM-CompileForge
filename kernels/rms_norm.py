# ruff: noqa: N803,N806  — Triton convention: uppercase constexpr arguments
"""RMSNorm + residual connection fusion kernel.

Fuses three operations into a single kernel:
  x = input + residual          # residual connection
  rms = sqrt(mean(x²) + eps)    # RMS normalization
  output = (x / rms) * weight   # scale by learned weight

This eliminates two intermediate HBM round-trips compared to
running the operations separately.

Reference: design-phase1.md §3.4.3
"""

from __future__ import annotations

import torch


def _fused_rms_norm_add_cpu(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """CPU / fallback implementation."""
    x = input_tensor + residual
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return (x / rms) * weight


def _fused_rms_norm_add_triton(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Triton GPU kernel — only compiled when Triton is importable."""
    import triton  # noqa: F811
    import triton.language as tl

    @triton.jit
    def _kernel(
        input_ptr,
        residual_ptr,
        weight_ptr,
        output_ptr,
        n_cols: tl.constexpr,
        eps_val: tl.constexpr,
        block_size: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * block_size + tl.arange(0, block_size)
        mask = offs < n_cols

        inp = tl.load(input_ptr + offs, mask=mask, other=0.0)
        res = tl.load(residual_ptr + offs, mask=mask, other=0.0)
        x = inp + res

        rms = tl.sqrt(tl.sum(x * x) / n_cols + eps_val)
        x_norm = x / rms

        w = tl.load(weight_ptr + offs, mask=mask, other=0.0)
        out = x_norm * w

        tl.store(output_ptr + offs, out, mask=mask)

    n_cols = input_tensor.shape[-1]
    output = torch.empty_like(input_tensor)
    grid = (input_tensor.numel() // n_cols,)
    BLOCK_SIZE = min(1024, triton.next_power_of_2(n_cols))

    _kernel[grid](
        input_tensor,
        residual,
        weight,
        output,
        n_cols=n_cols,
        eps_val=eps,
        block_size=BLOCK_SIZE,
    )
    return output


def fused_rms_norm_add(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """RMSNorm + residual connection, fused.

    Args:
        input_tensor: Input tensor of shape (..., hidden_dim).
        residual:    Residual tensor of same shape.
        weight:      Learnable scale of shape (hidden_dim,).
        eps:         Numerical stability constant.

    Returns:
        Normalized and scaled output, same shape as input.
    """
    from kernels import HAS_TRITON

    if HAS_TRITON and input_tensor.is_cuda:
        return _fused_rms_norm_add_triton(input_tensor, residual, weight, eps)
    return _fused_rms_norm_add_cpu(input_tensor, residual, weight, eps)
