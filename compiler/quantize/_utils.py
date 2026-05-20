"""Shared utilities for quantization calibration and weight manipulation.

Provides activation statistics collection, layer name resolution,
and weight access/modification helpers used by SmoothQuant, AWQ,
and FP8 KV Cache.

Reference: design-phase2.md §2.1
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def collect_activation_stats(
    model: nn.Module,
    dataloader: list[tuple[torch.Tensor, ...]] | None,
    num_samples: int = 512,
    target_layer_types: tuple[type, ...] = (nn.Linear,),
    capture_input: bool = True,
) -> dict[str, dict[str, torch.Tensor]]:
    """Run forward passes over a calibration dataset and collect per-layer
    activation statistics (per-channel absmax).

    Args:
        model: PyTorch nn.Module to calibrate.
        dataloader: List of input batches (tuples).  Each element is forwarded
                    through the model.  If None, uses a single random input.
        num_samples: Maximum number of samples to process.
        target_layer_types: Layer classes to instrument (default: nn.Linear).
        capture_input: If True, capture module input activations (for SmoothQuant
                       per-input-channel stats).  If False, capture module output
                       activations (for AWQ per-output-channel stats).

    Returns:
        dict:  layer_name → {"absmax": tensor [features], "count": int}
    """
    hooks: list[Any] = []
    stats: dict[str, dict[str, Any]] = {}

    def _make_hook(layer_name: str) -> Any:
        def _hook(_module: nn.Module, _input: Any, _output: Any) -> None:
            if capture_input:
                if isinstance(_input, tuple) and len(_input) > 0:
                    act = _input[0]
                else:
                    act = _input
            else:
                if isinstance(_output, tuple):
                    act = _output[0]
                else:
                    act = _output
            if act is None or not isinstance(act, torch.Tensor) or act.numel() == 0:
                return
            act_f = act.float().detach()
            # per-channel absmax over batch and sequence dims
            if act_f.dim() >= 2:
                reduce_dims = tuple(range(act_f.dim() - 1))
                ch_absmax = act_f.abs().amax(dim=reduce_dims)
            else:
                ch_absmax = act_f.abs()
            if layer_name not in stats:
                stats[layer_name] = {"absmax": ch_absmax.clone(), "count": 1}
            else:
                stats[layer_name]["absmax"] = torch.maximum(
                    stats[layer_name]["absmax"], ch_absmax
                )
                stats[layer_name]["count"] += 1
        return _hook

    for name, module in model.named_modules():
        if isinstance(module, target_layer_types):
            hooks.append(module.register_forward_hook(_make_hook(name)))  # type: ignore[attr-defined]

    try:
        if dataloader is None:
            dummy = torch.randn(1, 32)
            model.eval()
            with torch.no_grad():
                try:
                    model(dummy)
                except Exception:
                    _log.debug("Model rejected dummy input during calibration (expected for non-forward models)", exc_info=True)
        else:
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    if i >= num_samples:
                        break
                    if isinstance(batch, (list, tuple)):
                        model(*batch)
                    elif isinstance(batch, dict):
                        model(**batch)
                    else:
                        model(batch)
    finally:
        for hook in hooks:
            hook.remove()

    return stats


def get_layer_by_name(model: nn.Module, layer_name: str) -> nn.Module | None:
    """Resolve a dotted layer name within a PyTorch module hierarchy.

    Args:
        model: Root module.
        layer_name: Dotted path, e.g. "transformer.h.0.attn.q_proj".

    Returns:
        The submodule or None if not found.
    """
    parts = layer_name.split(".")
    current: nn.Module = model
    for part in parts:
        if hasattr(current, part):
            current = getattr(current, part)
        elif isinstance(current, nn.ModuleList):
            try:
                idx = int(part)
                current = current[idx]
            except (ValueError, IndexError):
                return None
        elif isinstance(current, nn.Sequential):
            try:
                idx = int(part)
                current = current[idx]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return current


def get_layer_weight(layer_name: str, model: nn.Module) -> torch.Tensor | None:
    """Get the weight tensor of a named linear layer.

    Args:
        layer_name: Dotted path to the layer.
        model: Root module.

    Returns:
        Weight tensor or None if layer not found or has no 'weight'.
    """
    layer = get_layer_by_name(model, layer_name)
    if layer is None or not hasattr(layer, "weight"):
        return None
    w: torch.Tensor = layer.weight  # type: ignore[assignment]
    return w.data


def set_layer_weight(model: nn.Module, layer_name: str, new_weight: torch.Tensor) -> bool:
    """Replace the weight tensor of a named linear layer in-place.

    Args:
        model: Root module.
        layer_name: Dotted path to the layer.
        new_weight: New weight tensor (must match shape).

    Returns:
        True if succeeded, False otherwise.
    """
    layer = get_layer_by_name(model, layer_name)
    if layer is None or not hasattr(layer, "weight"):
        return False
    w: torch.Tensor = layer.weight  # type: ignore[assignment]
    if w.shape != new_weight.shape:
        raise ValueError(
            f"Weight shape mismatch for {layer_name}: "
            f"expected {w.shape}, got {new_weight.shape}"
        )
    w.data = new_weight.to(
        dtype=w.dtype, device=w.device
    )
    return True
