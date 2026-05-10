"""W8A8 GEMM kernel — INT8 weights × INT8 activations.

CPU fallback: dequantize both to FP32 → matmul.
Triton kernel (Phase 2): fused INT8 dot product on GPU Tensor Cores.

Reference: design-phase2.md §2.3.2, §2.1.1 (SmoothQuant)
"""

from __future__ import annotations

import torch

from kernels.quantize._utils import dequantize_per_channel


def w8a8_gemm_cpu(
    activation: torch.Tensor,
    act_scale: torch.Tensor | None,
    weight_int8: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """W8A8 GEMM on CPU: dequantize weights and activations, then matmul.

    Args:
        activation:  FP32 activation to be quantized [..., in_features].
        act_scale:   Per-channel activation scale, shape [1, in_features] or None.
                     If None, activation is used as-is (already FP32).
        weight_int8: Per-channel INT8 weights [out_features, in_features].
        weight_scale: Per-channel FP32 scale [out_features, 1].

    Returns:
        output: [..., out_features] in FP32.
    """
    weight_fp = dequantize_per_channel(weight_int8, weight_scale)

    if act_scale is not None:
        act_fp = activation.to(torch.float32) * act_scale
    else:
        act_fp = activation.to(torch.float32)

    return act_fp @ weight_fp.T


def w8a8_gemm(
    activation: torch.Tensor,
    act_scale: torch.Tensor | None,
    weight_int8: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """W8A8 GEMM: INT8 weights × INT8/FP32 activations.

    Dispatches to CPU fallback.  Triton kernel will be added in Phase 2
    for GPU environments.
    """
    return w8a8_gemm_cpu(activation, act_scale, weight_int8, weight_scale)
