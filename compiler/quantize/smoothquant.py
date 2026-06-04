"""SmoothQuant calibration and W8A8 quantization.

Implements the SmoothQuant algorithm (Xiao et al., 2023):
  1. Calibrate: collect per-channel activation statistics.
  2. Smooth: compute smoothing factors s_j and apply to weights.
  3. Quantize: per-channel INT8 quantization of weights.

The core transform: Y = (X · diag(s)⁻¹) · (diag(s) · W)
This migrates quantization difficulty from activations to weights
while preserving the mathematical result.

Reference: design-phase2.md §2.1.1
"""

from __future__ import annotations

import torch
import torch.nn as nn

from compiler.quantize._utils import (
    collect_activation_stats,
    get_layer_weight,
    set_layer_weight,
)
from compiler.quantize.base import BaseQuantizer
from kernels.quantize._utils import quantize_per_channel_int8


class SmoothQuantCalibrator(BaseQuantizer):
    """SmoothQuant (W8A8) calibration and quantization.

    Args:
        model: PyTorch nn.Module to quantize.
        alpha: Migration factor (0.0 = all difficulty stays on activation;
               1.0 = all difficulty migrated to weight; 0.5 = balanced).
    """

    def __init__(self, model: nn.Module, alpha: float = 0.5) -> None:
        super().__init__(model, {"alpha": alpha})
        self.smoothing_factors: dict[str, torch.Tensor] = {}
        self.activation_scales: dict[str, torch.Tensor] = {}
        self.weight_scales: dict[str, torch.Tensor] = {}
        self.weight_quant: dict[str, torch.Tensor] = {}

    def _validate_config(self) -> None:
        """Validate SmoothQuant-specific config parameters."""
        alpha = self.config.get("alpha", 0.5)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha

    def calibrate(
        self,
        dataloader: list[tuple[torch.Tensor, ...]] | None = None,
        num_samples: int = 512,
    ) -> None:
        """Run the three-stage SmoothQuant pipeline on the given model.

        Step 1: Collect per-channel activation absmax statistics.
        Step 2: Compute smoothing factors: s_j = max(|X_j|)^α / max(|W_j|)^(1-α).
        Step 3: Apply smoothing transform: W *= s (weight) and store 1/s for activation.
        """
        act_stats = collect_activation_stats(
            self.model, dataloader, num_samples, capture_input=True
        )

        for layer_name, stats in act_stats.items():
            weight = get_layer_weight(layer_name, self.model)
            if weight is None:
                continue

            weight_absmax = weight.float().abs().amax(dim=0).detach()
            act_absmax = stats["absmax"].float().detach()

            if act_absmax.shape[0] != weight_absmax.shape[0]:
                continue

            s = self._compute_smoothing_factor(act_absmax, weight_absmax)
            self.smoothing_factors[layer_name] = s

        for layer_name, s in self.smoothing_factors.items():
            weight = get_layer_weight(layer_name, self.model)
            if weight is None:
                continue

            new_weight = weight.float() * s
            set_layer_weight(self.model, layer_name, new_weight)

            self.activation_scales[layer_name] = 1.0 / s

    def _compute_smoothing_factor(
        self,
        act_absmax: torch.Tensor,
        weight_absmax: torch.Tensor,
    ) -> torch.Tensor:
        """Compute SmoothQuant scaling factors.

        Formula: s_j = max(|X_j|)^α / max(|W_j|)^(1-α)
        where both X_j and W_j are per-input-channel.

        Args:
            act_absmax: Per-channel activation max [in_features].
            weight_absmax: Per-channel weight max over output channels [in_features].

        Returns:
            Smoothing factors [in_features].
        """
        s = torch.pow(act_absmax.float(), self.alpha) / torch.pow(
            weight_absmax.float(), 1.0 - self.alpha
        )
        s = torch.clamp(s, min=1e-8)
        return s.to(torch.float32)

    def quantize(self) -> None:
        """Apply per-channel INT8 quantization to all smoothed linear layers.

        Stores quantized weights and scales as registered buffers on each layer.
        """
        for layer_name in self.smoothing_factors:
            layer = _resolve_layer(self.model, layer_name)
            if layer is None or not hasattr(layer, "weight"):
                continue

            w: torch.Tensor = layer.weight  # type: ignore[assignment]
            w_int8, w_scale = quantize_per_channel_int8(w.float())
            self.weight_quant[layer_name] = w_int8
            self.weight_scales[layer_name] = w_scale

            layer.register_buffer("weight_quant", w_int8)
            layer.register_buffer("weight_scale", w_scale)

            if layer_name in self.activation_scales:
                act_scale = self.activation_scales[layer_name]
                layer.register_buffer("activation_scale", act_scale)

    @property
    def num_layers_processed(self) -> int:
        return len(self.smoothing_factors)


def _resolve_layer(model: nn.Module, layer_name: str) -> nn.Module | None:
    from compiler.quantize._utils import get_layer_by_name

    return get_layer_by_name(model, layer_name)
