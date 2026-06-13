"""FP8 KV Cache quantization for memory-efficient KV cache storage.

Implements per-block FP8 (E4M3 format) quantization and dequantization
for KV cache blocks.  Quantizing from FP16 to FP8 halves the KV cache
memory footprint, enabling larger batch sizes or longer context lengths.

Hardware support: H100, H200, RTX 4090, RTX 6000 ADA, MI300.
On unsupported hardware, quantization is performed via torch.float8_e4m3fn
or falls back to per-block INT8.

Reference: design-phase2.md §2.1.3
"""

from __future__ import annotations

import torch


class FP8KVCacheQuantizer:
    """Per-block FP8 quantization/dequantization for KV cache tensors.

    Each KV block (e.g. 16 tokens × num_kv_heads × head_dim) is
    independently quantized with its own scale factor.  This avoids
    cross-block outlier contamination and preserves attention quality.

    The E4M3 format max representable value is 448.0.

    Args:
        block_size: Number of tokens per KV block (default 16).
    """

    SUPPORTED_HARDWARE: tuple[str, ...] = (
        "H100",
        "H200",
        "RTX_4090",
        "RTX_6000_ADA",
        "MI300",
    )

    def __init__(self, block_size: int = 16) -> None:
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        self.block_size = block_size
        self.fp8_max: float = 448.0

    def check_hardware_support(self, device_name: str) -> bool:
        """Check whether the given device name indicates FP8 hardware support.

        Args:
            device_name: String describing the device (e.g. "NVIDIA H100").
                          Partial substring matching is used.

        Returns:
            True if the device likely supports FP8 natively.
        """
        return any(hw in device_name for hw in self.SUPPORTED_HARDWARE)

    def _has_fp8_dtype(self) -> bool:
        """Check whether torch.float8_e4m3fn is available (PyTorch >= 2.1)."""
        try:
            _ = torch.float8_e4m3fn  # noqa: F841
            return True
        except AttributeError:
            return False

    def quantize_block(self, kv_block: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize an FP16 KV block to FP8 (or INT8 fallback).

        Per-block quantization: scale = max(|block|) / fp8_max.
        Each block gets its own scale factor.

        Args:
            kv_block: FP16/FP32 tensor of shape [block_size, num_kv_heads, head_dim]
                      or [block_size, num_kv_heads, head_dim].

        Returns:
            (quant_block, scale):
              quant_block — FP8 tensor or INT8 fallback.
              scale — FP32 scale, shape broadcastable along last dim.
        """
        amax = kv_block.float().abs().max()
        if amax == 0:
            scale = torch.tensor(1.0, dtype=torch.float32)
        else:
            scale = amax / self.fp8_max
            scale = torch.clamp(scale, min=1e-12)

        if self._has_fp8_dtype():
            quant = (kv_block.float() / scale).to(torch.float8_e4m3fn)
        else:
            quant = (kv_block.float() / scale).to(torch.int8)

        return quant, scale

    def dequantize_block(self, quant_block: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize an FP8 (or INT8) KV block back to FP32.

        Args:
            quant_block: Quantized tensor (float8_e4m3fn or int8).
            scale: Per-block scale factor (FP32 scalar).

        Returns:
            FP32 tensor of same shape as the original block.
        """
        return quant_block.float() * scale

    def quantize_kv(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize both K and V tensors for a single layer.

        Args:
            k: Key tensor [batch, num_heads, head_dim].
            v: Value tensor [batch, num_heads, head_dim].

        Returns:
            (k_quant, k_scale, v_quant, v_scale):
              Quantized K and V with their per-block scales.
        """
        k_quant, k_scale = self.quantize_block(k)
        v_quant, v_scale = self.quantize_block(v)
        return k_quant, k_scale, v_quant, v_scale

    def dequantize_kv(
        self,
        k_quant: torch.Tensor,
        k_scale: torch.Tensor,
        v_quant: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequantize both K and V tensors.

        Args:
            k_quant, k_scale: Quantized K and its scale.
            v_quant, v_scale: Quantized V and its scale.

        Returns:
            (k_fp, v_fp): Dequantized K and V in FP32.
        """
        return self.dequantize_block(k_quant, k_scale), self.dequantize_block(v_quant, v_scale)
