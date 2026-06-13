"""W4A16 GEMM kernel — INT4 weights × FP16 activations.

CPU fallback: unpack INT4 → dequantize to FP32 → matmul with FP16 act.
Triton kernel (Phase 2): fused unpack + dequant + tile-based matmul on GPU.

Reference: design-phase2.md §2.3.2, §2.1.2 (AWQ)
"""

from __future__ import annotations

import torch

from kernels.quantize._utils import dequantize_groupwise


def w4a16_gemm_cpu(
    activation: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_zero: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """W4A16 GEMM on CPU: dequantize weights then standard matmul.

    Args:
        activation:  FP16/FP32 activation tensor [..., in_features].
        weight_packed: Packed INT4 weights [out_features, in_features//2] uint8.
        weight_scale:  Per-group FP32 scales [out_features, ceil(in_f/group_size)].
        weight_zero:   Per-group zero points (unused for symmetric quant).
        group_size:    Quantization group size (default 128).

    Returns:
        output: [..., out_features] in activation dtype.
    """
    out_features = weight_packed.size(0)
    in_features = weight_packed.size(1) * 2

    weight_fp = dequantize_groupwise(
        weight_packed,
        weight_scale,
        group_size,
        out_features=out_features,
        in_features=in_features,
    )

    return activation.to(torch.float32) @ weight_fp.T


def w4a16_gemm(
    activation: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_zero: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """W4A16 GEMM: INT4 weights × FP16 activations.

    Dispatches to CPU fallback.  Triton kernel will be added in Phase 2
    for GPU environments.
    """
    return w4a16_gemm_cpu(activation, weight_packed, weight_scale, weight_zero, group_size)
