"""torch.export → ExportedProgram capture.

Phase 1 wraps torch.export to trace a model's forward pass and capture its
FX graph. The resulting ExportedProgram is fed to fx_to_ir.py for conversion.

Export results are cached to disk (~/.cache/serveforge/exports/) keyed
by model directory, input shapes, and dynamic shapes configuration.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.export import ExportedProgram

_CACHE_DIR = Path.home() / ".cache" / "serveforge" / "exports"


def _cache_key(
    model_dir: str,
    example_args: tuple[Any, ...],
    dynamic_shapes: dict[str, Any] | None,
) -> str:
    """Build a deterministic cache key from export parameters."""
    shapes: list[Any] = []
    for a in example_args:
        if isinstance(a, torch.Tensor):
            shapes.append(list(a.shape))
        else:
            shapes.append(str(type(a).__name__))
    payload = json.dumps({
        "model_dir": os.path.abspath(model_dir),
        "shapes": shapes,
        "dynamic_shapes": str(dynamic_shapes) if dynamic_shapes else "static",
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def export_model(
    model: torch.nn.Module,
    example_args: tuple[Any, ...] | None = None,
    example_kwargs: dict[str, Any] | None = None,
    dynamic_shapes: dict[str, Any] | None = None,
    model_dir: str = "",
    cache: bool = False,
) -> ExportedProgram:
    """Export a PyTorch model to ExportedProgram via torch.export.

    Args:
        model: The PyTorch nn.Module to export.
        example_args: Positional arguments for a dummy forward call.
        example_kwargs: Keyword arguments for a dummy forward call.
        dynamic_shapes: Shape constraints for dynamic dimensions.
        model_dir: Directory containing model weights (used for cache key).
        cache: If True, cache the exported program to disk.

    Returns:
        ExportedProgram ready for IR conversion.
    """
    args = example_args or ()
    kwargs = example_kwargs or {}

    # Check disk cache
    if cache and model_dir:
        key = _cache_key(model_dir, args, dynamic_shapes)
        cache_path = _CACHE_DIR / f"{key}.pt"
        if cache_path.exists():
            return torch.export.load(cache_path)

    with torch.no_grad():
        if dynamic_shapes is not None:
            program = torch.export.export(model, args, kwargs=kwargs, dynamic_shapes=dynamic_shapes)
        elif kwargs:
            program = torch.export.export(model, args, kwargs=kwargs)
        else:
            program = torch.export.export(model, args)

    # Save to disk cache
    if cache and model_dir:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        key = _cache_key(model_dir, args, dynamic_shapes)
        torch.export.save(program, _CACHE_DIR / f"{key}.pt")

    return program


def get_signature(program: ExportedProgram) -> dict[str, Any]:
    """Extract input/output signature from an ExportedProgram.

    Returns a dict with keys:
      - "inputs": list of (name, shape, dtype)
      - "outputs": list of (name, shape, dtype)
    """
    sig = program.graph_signature
    graph = program.graph_module.graph

    inputs = []
    for inp in sig.user_inputs:
        node = None
        for n in graph.nodes:
            if n.name == inp:
                node = n
                break
        fake_tensor = node.meta.get("val") if node else None
        shape = tuple(fake_tensor.shape) if fake_tensor is not None else ()
        dtype = str(fake_tensor.dtype) if fake_tensor is not None else "float32"
        inputs.append((inp, shape, dtype))

    outputs = []
    for out in sig.user_outputs:
        node = None
        for n in graph.nodes:
            if n.name == out:
                node = n
                break
        fake_tensor = node.meta.get("val") if node else None
        shape = tuple(fake_tensor.shape) if fake_tensor is not None else ()
        dtype = str(fake_tensor.dtype) if fake_tensor is not None else "float32"
        outputs.append((out, shape, dtype))

    return {"inputs": inputs, "outputs": outputs}
