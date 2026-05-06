"""torch.export → ExportedProgram capture.

Phase 1 wraps torch.export to trace a model's forward pass and capture its
FX graph. The resulting ExportedProgram is fed to fx_to_ir.py for conversion.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.export import ExportedProgram


def export_model(
    model: torch.nn.Module,
    example_args: tuple[Any, ...] | None = None,
    example_kwargs: dict[str, Any] | None = None,
) -> ExportedProgram:
    """Export a PyTorch model to ExportedProgram via torch.export.

    The resulting ExportedProgram contains:
      - graph_module: torch.fx.GraphModule with aten-level ops
      - graph_signature: input/output specifications
      - state_dict: model weights

    Args:
        model: The PyTorch nn.Module to export.
        example_args: Positional arguments for a dummy forward call.
        example_kwargs: Keyword arguments for a dummy forward call.

    Returns:
        ExportedProgram ready for IR conversion.

    Raises:
        RuntimeError: If torch.export fails (unsupported ops, dynamic control flow, etc.).
    """
    args = example_args or ()
    kwargs = example_kwargs or {}
    return torch.export.export(model, args, kwargs=kwargs)


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
