"""AWQ (Activation-aware Weight Quantization) — W4A16 weight quantization.

Identifies ~1% salient weight channels (those whose activations have the
largest magnitudes), finds optimal per-channel scaling factors via grid
search to minimize quantization error, then applies group-wise INT4
quantization.

The key insight: not all weight channels are equally important. AWQ
identifies salient channels by their activation magnitudes and protects
them with increased scaling factors before quantization.

Reference: design-phase2.md §2.1.2; Lin et al., AWQ (2024)
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from compiler.quantize._utils import (
    collect_activation_stats,
    get_layer_weight,
    set_layer_weight,
)
from kernels.quantize._utils import quantize_groupwise_int4


class AWQQuantizer:
    """AWQ W4A16 weight quantizer.

    Pipeline:
      1. identify_salient_channels: find top-1% channels by activation magnitude.
      2. find_optimal_scales: grid-search scaling factor per layer
         to minimise L2 distance between original and quantized output.
      3. quantize: apply optimal scales and group-wise INT4 quantization.

    Args:
        group_size: number of features per quantization group (default 128).
        salient_fraction: fraction of channels deemed salient (default 0.01 = 1%).
    """

    def __init__(
        self,
        group_size: int = 128,
        salient_fraction: float = 0.01,
    ) -> None:
        if not 0.0 < salient_fraction <= 1.0:
            raise ValueError(f"salient_fraction must be in (0, 1], got {salient_fraction}")
        self.group_size = group_size
        self.salient_fraction = salient_fraction
        self.salient_channels: dict[str, torch.Tensor] = {}
        self.optimal_scales: dict[str, float] = {}
        self.weight_quant: dict[str, torch.Tensor] = {}
        self.weight_scales: dict[str, torch.Tensor] = {}
        self.weight_zeros: dict[str, torch.Tensor] = {}

    def identify_salient_channels(
        self,
        model: nn.Module,
        dataloader: list[tuple[torch.Tensor, ...]] | None = None,
        num_samples: int = 128,
    ) -> None:
        """Identify the 1% most salient channels per layer.

        Salience is determined by per-channel activation magnitude
        (absolute value) averaged over the calibration set.

        Args:
            model: PyTorch nn.Module to analyze.
            dataloader: Calibration data.  If None, uses a single random input.
            num_samples: Maximum number of calibration samples.
        """
        act_stats = collect_activation_stats(
            model, dataloader, num_samples, capture_input=False
        )

        for layer_name, stats in act_stats.items():
            means = stats["absmax"]
            num_channels = len(means)
            n_salient = max(1, int(num_channels * self.salient_fraction))
            _, indices = torch.topk(means.float(), n_salient)
            self.salient_channels[layer_name] = indices

    def find_optimal_scales(
        self,
        model: nn.Module,
        dataloader: list[tuple[torch.Tensor, ...]] | None = None,
        scale_range: tuple[float, float] = (1.0, 1.3),
        n_grid: int = 20,
    ) -> None:
        """Find the optimal per-layer scaling factor via grid search.

        For each layer, scales the salient weight channels by each
        candidate factor, quantizes, dequantizes, and computes the
        L2 output error.  Selects the scale that minimizes error.

        Args:
            model: PyTorch nn.Module with original weights.
            dataloader: Calibration inputs for computing L2 error.
                If None, uses a single random input.
            scale_range: (min_scale, max_scale) search bounds.
            n_grid: Number of search points in [min_scale, max_scale].
        """
        layer_inputs: dict[str, torch.Tensor] = {}
        input_hooks: list[Any] = []

        def _capture_input(layer_name: str) -> Any:
            def _hook(_m: nn.Module, _in: Any, _out: Any) -> None:
                if isinstance(_in, tuple) and len(_in) > 0:
                    inp = _in[0]
                else:
                    inp = _in
                if isinstance(inp, torch.Tensor):
                    layer_inputs[layer_name] = inp.detach().float()
            return _hook

        for name, module in model.named_modules():
            if name in self.salient_channels:
                input_hooks.append(module.register_forward_hook(_capture_input(name)))

        try:
            if dataloader is None:
                dummy = torch.randn(1, 32)
                model.eval()
                with torch.no_grad():
                    try:
                        model(dummy)
                    except Exception:
                        _log.debug("Model rejected dummy input during AWQ calibration (expected for non-forward models)", exc_info=True)
            else:
                model.eval()
                with torch.no_grad():
                    for batch in dataloader:
                        if isinstance(batch, (list, tuple)):
                            model(*batch)
                        elif isinstance(batch, dict):
                            model(**batch)
                        else:
                            model(batch)
                        break
        finally:
            for hook in input_hooks:
                hook.remove()

        for layer_name, salient_idx in self.salient_channels.items():
            weight = get_layer_weight(layer_name, model)
            if weight is None or layer_name not in layer_inputs:
                continue

            x_inp: torch.Tensor = layer_inputs[layer_name]
            w_orig = weight.detach().clone().float()
            ref = x_inp @ w_orig.T
            best_scale = 1.0
            best_loss = float("inf")

            for scale in np.linspace(scale_range[0], scale_range[1], n_grid):
                w_scaled = w_orig.clone()
                w_scaled[salient_idx] *= scale

                qp, qs, _qz = quantize_groupwise_int4(w_scaled, self.group_size)
                from kernels.quantize._utils import dequantize_groupwise

                w_deq = dequantize_groupwise(
                    qp, qs, self.group_size,
                    out_features=w_scaled.size(0), in_features=w_scaled.size(1),
                )
                loss = cast(float, F.mse_loss(x_inp @ w_deq.T, ref).item())
                if loss < best_loss:
                    best_loss = loss
                    best_scale = scale

            self.optimal_scales[layer_name] = best_scale

    def quantize(self, model: nn.Module) -> None:
        """Apply optimal scales and group-wise INT4 quantization.

        Modifies model weights in-place: applies per-channel scaling,
        then replaces weights with INT4 quantized versions stored as
        registered buffers.
        """
        for layer_name, scale in self.optimal_scales.items():
            weight = get_layer_weight(layer_name, model)
            if weight is None:
                continue

            w = weight.detach().clone().float()
            if layer_name in self.salient_channels:
                salient_idx = self.salient_channels[layer_name]
                w[salient_idx] *= scale

            qp, qs, qz = quantize_groupwise_int4(w, self.group_size)
            self.weight_quant[layer_name] = qp
            self.weight_scales[layer_name] = qs
            self.weight_zeros[layer_name] = qz

            layer: nn.Module = get_layer_by_name_inline(model, layer_name)
            layer.register_buffer("weight_quant", qp)
            layer.register_buffer("weight_scale", qs)
            layer.register_buffer("weight_zero", qz)

            set_layer_weight(model, layer_name, w)

    @property
    def num_layers_processed(self) -> int:
        return len(self.optimal_scales)


def get_layer_by_name_inline(model: nn.Module, layer_name: str) -> nn.Module:
    from compiler.quantize._utils import get_layer_by_name

    layer = get_layer_by_name(model, layer_name)
    if layer is None:
        raise ValueError(f"Layer '{layer_name}' not found in model")
    return layer
