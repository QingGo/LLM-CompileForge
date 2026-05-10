"""Shared quantization utilities for quantized GEMM kernels.

Provides low-level tensor packing, unpacking, quantization and
dequantization helpers used by w4a16_gemm, w8a8_gemm, and the
compiler quantize sub-module (smoothquant.py, awq.py).

Reference: design-phase2.md §2.1.2, §2.3.2
"""

from __future__ import annotations

import torch


def pack_int4(weight: torch.Tensor) -> torch.Tensor:
    """Pack a tensor of INT4 values (stored in uint8 range 0-15) into
    two values per byte.  Expects even-sized last dimension.

    Args:
        weight: uint8 tensor with values in 0..15.

    Returns:
        uint8 tensor with last dimension halved, each byte carrying
        two 4-bit values (low nibble = even index, high nibble = odd).
    """
    if weight.dtype != torch.uint8:
        raise ValueError(f"pack_int4 expects uint8 input, got {weight.dtype}")
    if weight.size(-1) % 2 != 0:
        raise ValueError(f"Last dim must be even for INT4 packing, got {weight.size(-1)}")

    w_low = weight[..., 0::2] & 0x0F
    w_high = (weight[..., 1::2] & 0x0F) << 4
    return w_low | w_high


def unpack_int4(packed: torch.Tensor, out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Unpack a packed INT4 tensor back to individual values.

    Args:
        packed: uint8 tensor with two 4-bit values per byte.
        out_dtype: desired output dtype (default float32).

    Returns:
        Tensor of shape (*, packed_last_dim * 2) in out_dtype.
    """
    if packed.dtype != torch.uint8:
        raise ValueError(f"unpack_int4 expects uint8 input, got {packed.dtype}")

    low = (packed & 0x0F).to(out_dtype)
    high = ((packed >> 4) & 0x0F).to(out_dtype)

    last_dim = low.size(-1)
    shape = packed.shape[:-1] + (last_dim * 2,)
    result = torch.empty(shape, dtype=out_dtype, device=packed.device)
    result[..., 0::2] = low
    result[..., 1::2] = high
    return result


def quantize_per_channel_int8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize weight per-channel (along dim=0) to INT8.

    scale = max(|weight[i]|) / 127  for each channel i.

    Args:
        weight: FP32 or FP16 tensor of shape [out_features, in_features].

    Returns:
        (quant_int8, scale_fp): quantized tensor (int8) and scale (fp32,
        shape [out_features, 1]).
    """
    w_max = weight.abs().amax(dim=1, keepdim=True)
    scale = w_max / 127.0
    scale = torch.clamp(scale, min=1e-9)
    quant = (weight / scale).round().clamp(-128, 127).to(torch.int8)
    return quant, scale.to(torch.float32)


def dequantize_per_channel(quant: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize per-channel INT8 or per-group INT4 weights back to FP32.

    Args:
        quant: INT8 or INT32 tensor of quantized values.
        scale: FP32 scale tensor, broadcastable with quant along last dim.

    Returns:
        FP32 tensor of dequantized values.
    """
    return quant.to(torch.float32) * scale


def quantize_groupwise_int4(
    weight: torch.Tensor, group_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group-wise INT4 quantization of weight tensor.

    Splits in_features dimension into groups of group_size.  For each
    group: scale = max(|group|) / 7, zero = 0 (symmetric quantization).

    Args:
        weight: FP32 or FP16 tensor of shape [out_features, in_features].
        group_size: number of features per quant group (default 128).

    Returns:
        (quant_uint8, scale_fp, zero_uint8):
          quant_uint8  — packed INT4 values (uint8, half last dim).
          scale_fp     — FP32 scales [out_features, ceil(in_features/group_size)].
          zero_uint8   — zero point per group (all zeros for symmetric).
    """
    out_f, in_f = weight.shape
    num_groups = (in_f + group_size - 1) // group_size
    padded_in = num_groups * group_size

    if padded_in != in_f:
        weight_padded = torch.zeros(out_f, padded_in, dtype=weight.dtype, device=weight.device)
        weight_padded[:, :in_f] = weight
    else:
        weight_padded = weight

    weight_r = weight_padded.view(out_f, num_groups, group_size)
    w_max = weight_r.abs().amax(dim=2)
    scale = w_max / 7.0
    scale = torch.clamp(scale, min=1e-9)

    quant = (weight_r / scale.unsqueeze(2)).round().add(8).clamp(0, 15).to(torch.uint8)
    quant_flat = quant.view(out_f, padded_in)
    if padded_in != in_f:
        quant_flat = quant_flat[:, :in_f]
    quant_packed = pack_int4(quant_flat)

    zero = torch.full_like(scale, 8, dtype=torch.uint8)
    return quant_packed, scale.to(torch.float32), zero


def dequantize_groupwise(
    quant_packed: torch.Tensor,
    scale: torch.Tensor,
    group_size: int = 128,
    out_features: int | None = None,
    in_features: int | None = None,
) -> torch.Tensor:
    """Dequantize group-wise INT4 packed weights back to FP32.

    Args:
        quant_packed: uint8 packed INT4, shape [out_features, in_features//2].
        scale: FP32 scale per group, shape [out_features, ceil(in_features/group_size)].
        group_size: number of features per quant group.
        out_features: optional, inferred from tensor shape if None.
        in_features: optional, inferred from tensor shape if None.

    Returns:
        FP32 tensor of shape [out_features, in_features].
    """
    out_f = out_features or quant_packed.size(0)
    in_f_packed = quant_packed.size(1)
    in_f = in_features or (in_f_packed * 2)
    num_groups = (in_f + group_size - 1) // group_size

    quant_flat = unpack_int4(quant_packed, out_dtype=torch.float32)

    padded_in = num_groups * group_size
    if padded_in != in_f:
        quant_padded = torch.zeros(out_f, padded_in, dtype=torch.float32, device=quant_flat.device)
        quant_padded[:, :in_f] = quant_flat
    else:
        quant_padded = quant_flat

    quant_r = quant_padded.view(out_f, num_groups, group_size)
    deq = (quant_r - 8.0) * scale.unsqueeze(2)
    deq_flat = deq.view(out_f, padded_in)
    if padded_in != in_f:
        deq_flat = deq_flat[:, :in_f]

    return deq_flat
