#!/usr/bin/env python3
"""Per-layer HF model vs IR executor comparison for Qwen3.5-0.8B.

Captures hidden states at each layer boundary via HF forward hooks,
then matches them against IR SSA values by shape and cosine similarity.
Identifies the first layer where the IR output diverges from HF.

Usage:
    python scripts/debug_qwen_layers.py
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _patch_transformers_torch() -> None:
    """Patch transformers to work with the conda-symlinked torch."""
    try:
        from importlib import metadata as _meta

        import transformers
        for _pkg_name in list(_meta.packages_distributions().keys()):
            if _pkg_name == "torch":
                if not hasattr(transformers, "is_torch_available"):
                    transformers.is_torch_available = lambda: True
                if not hasattr(transformers.utils, "is_torch_available"):
                    transformers.utils.is_torch_available = lambda: True
        _has_flag = getattr(transformers, "_torch_available", None)
        if _has_flag is not None and callable(_has_flag):
            transformers._torch_available = True
        if hasattr(transformers.utils, "_torch_available"):
            transformers.utils._torch_available = True
        if hasattr(transformers, "is_torch_available"):
            if not transformers.is_torch_available():
                transformers.is_torch_available = lambda: True
        from torch._functorch import pytree
        if not hasattr(transformers, "_torch_pytree"):
            transformers._torch_pytree = pytree
        if hasattr(transformers, "modeling_utils"):
            if not getattr(transformers.modeling_utils, "_torch_pytree", None):
                transformers.modeling_utils._torch_pytree = pytree
        if hasattr(transformers, "utils"):
            if hasattr(transformers.utils, "is_torch_fx_available"):
                transformers.utils.is_torch_fx_available = lambda: True
            if hasattr(transformers.utils, "is_torch_export_available"):
                transformers.utils.is_torch_export_available = lambda: True
        _maybe_fx_imp = getattr(transformers, "_torch_fx_imported", None)
        if _maybe_fx_imp is not None:
            transformers._torch_fx_imported = True
        _has_dyn = hasattr(transformers, "dynamic_module_utils")
        if _has_dyn and hasattr(transformers.dynamic_module_utils, "_torch_fx_imported"):
            transformers.dynamic_module_utils._torch_fx_imported = True
        if hasattr(transformers, "modeling_utils"):
            if hasattr(transformers.modeling_utils, "_model_output_unflatten"):
                def _noop_flatten(x: Any, context: Any = None) -> Any:
                    return x
                transformers.modeling_utils._model_output_flatten = _noop_flatten
                transformers.modeling_utils._model_output_unflatten = _noop_flatten
    except ImportError:
        pass


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float().reshape(-1)
    b_f = b.float().reshape(-1)
    if a_f.numel() == 0 or b_f.numel() == 0:
        return 0.0
    c = torch.nn.functional.cosine_similarity(a_f, b_f, dim=0)
    if torch.isnan(c) or torch.isinf(c):
        return -1.0
    return float(c)


def _register_hooks(model: Any, hidden_size: int) -> tuple[list[dict[str, Any]], Any]:
    """Register forward hooks to capture hidden states at each layer boundary.

    Returns (hook_outputs, hook_handles) where hook_outputs is a list of dicts
    with keys: name, tensor.
    """
    captured: list[dict[str, Any]] = []

    def _make_hook(name: str):
        def _hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            if isinstance(out, torch.Tensor):
                captured.append({"name": name, "tensor": out.detach(), "shape": list(out.shape)})
        return _hook

    handles: list[Any] = []

    # Embedding output
    if hasattr(model.model, 'embed_tokens'):
        h = model.model.embed_tokens.register_forward_hook(_make_hook("embed_tokens"))
        handles.append(h)

    # Each layer output (hidden states after the layer)
    if hasattr(model.model, 'layers'):
        for i, layer in enumerate(model.model.layers):
            h = layer.register_forward_hook(_make_hook(f"layer_{i}"))
            handles.append(h)

    # Final norm output
    if hasattr(model.model, 'norm'):
        h = model.model.norm.register_forward_hook(_make_hook("final_norm"))
        handles.append(h)

    return captured, handles


def main() -> None:
    _patch_transformers_torch()

    from transformers import AutoConfig, AutoModelForCausalLM

    from compiler.fx_to_mlir import fx_graph_to_mlir
    from hal.pytorch_backend import PyTorchBackend

    model_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "Qwen", "Qwen3.5-0.8B",
    )

    if not os.path.isdir(model_dir):
        print(f"ERROR: Model directory not found: {model_dir}")
        sys.exit(1)

    print(f"Loading Qwen3.5-0.8B from: {model_dir}")
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    if hasattr(config, "text_config") and hasattr(config.text_config, "use_cache"):
        config.text_config.use_cache = False
    elif hasattr(config, "use_cache"):
        config.use_cache = False

    hidden_size = (
        config.hidden_size if hasattr(config, "hidden_size")
        else config.text_config.hidden_size
    )
    num_layers = (
        config.num_hidden_layers if hasattr(config, "num_hidden_layers")
        else config.text_config.num_hidden_layers
    )
    print(f"  hidden_size={hidden_size}, num_layers={num_layers}")

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    model.eval()

    example_input = torch.randint(0, 248320, (1, 64), dtype=torch.long)

    # ── Step 1: Run HF model with hooks ──────────────────────
    print("\n=== Step 1: Running HF model with hooks ===")
    hook_outputs, hook_handles = _register_hooks(model, hidden_size)
    t0 = time.time()
    with torch.no_grad():
        model(example_input)
    elapsed = time.time() - t0
    for h in hook_handles:
        h.remove()

    print(f"  Captured {len(hook_outputs)} layer outputs in {elapsed:.1f}s")
    for entry in hook_outputs:
        print(f"    {entry['name']:20s} shape={entry['shape']}")

    # ── Step 2: Export and convert to IR ─────────────────────
    print("\n=== Step 2: Exporting and converting to IR ===")
    t0 = time.time()
    with torch.no_grad():
        program = torch.export.export(model, (example_input,))
    print(f"  Export: {time.time() - t0:.1f}s")

    t0 = time.time()
    ir_module = fx_graph_to_mlir(program)
    ir_func = ir_module.functions[0]
    print(f"  IR conversion: {time.time() - t0:.1f}s")
    print(f"  IR ops: {len(ir_func.ops)}, weights: {len(ir_func.weights)}")

    # ── Step 3: Run IR executor, capture all SSA values ──────
    print("\n=== Step 3: Running IR executor ===")
    backend = PyTorchBackend("cpu")
    ssa_values: dict[str, torch.Tensor] = {}
    input_names = [name for name, _ in ir_func.inputs]
    if input_names:
        ssa_values[input_names[0]] = example_input
    for named_input in input_names[1:]:
        ssa_values[named_input] = torch.zeros(1, dtype=torch.long)

    all_weights = dict(ir_func.weights)
    missing = 0
    exec_errors = 0

    t0 = time.time()
    for _i, op in enumerate(ir_func.ops):
        if op.name == "constant":
            if op.inputs and op.inputs[0] in all_weights:
                result = all_weights[op.inputs[0]]
            else:
                result = None
        else:
            ti = []
            for inp in op.inputs:
                if inp in ssa_values:
                    ti.append(ssa_values[inp])
                elif inp in all_weights:
                    ti.append(all_weights[inp])
                elif inp in ir_func.weights:
                    ti.append(ir_func.weights[inp])
                else:
                    missing += 1
                    ti = []
                    break
            if not ti and missing:
                continue
            try:
                attrs = {k: v for k, v in op.attributes.items() if k != "source_node"}
                if ti:
                    result = backend.execute(op.name, ti, **attrs)
                else:
                    result = backend.execute(op.name, [], **attrs)
            except Exception as e:
                print(f"  WARNING: op execution failed: {e}")
                exec_errors += 1
                result = None

        if result is not None and op.outputs:
            ssa_values[op.outputs[0]] = result

    elapsed = time.time() - t0
    print(f"  Executed in {elapsed:.1f}s")
    print(f"  Missing inputs: {missing}, exec errors: {exec_errors}")
    print(f"  SSA values produced: {len(ssa_values)}")

    # ── Step 4: Match HF layer outputs to IR SSA values ──────
    print("\n=== Step 4: Matching HF layer outputs to IR SSA values ===")
    print(f"  Hidden state shape: [1, 64, {hidden_size}]")

    # Collect SSA values with matching shape
    target_shape = [1, 64, hidden_size]
    candidates = []
    for name, tensor in ssa_values.items():
        if list(tensor.shape) == target_shape:
            candidates.append((name, tensor))

    print(f"  SSA values with shape {target_shape}: {len(candidates)}")

    # For each HF layer output, find the best-matching SSA value
    print(f"\n{'Layer':<20s} {'Best SSA match':<40s} {'Cosine':>10s} {'Status'}")
    print("-" * 85)

    all_good = True
    first_bad_layer = -1

    for entry in hook_outputs:
        hf_name = entry["name"]
        hf_tensor = entry["tensor"]

        best_cos = -2.0
        best_name = ""
        for ssa_name, ssa_tensor in candidates:
            cos = _cosine(hf_tensor, ssa_tensor)
            if cos > best_cos:
                best_cos = cos
                best_name = ssa_name

        status = "OK" if best_cos >= 0.999 else "DIVERGED" if best_cos < 0.99 else "WARN"
        if best_cos < 0.999 and first_bad_layer < 0 and hf_name.startswith("layer"):
            first_bad_layer = int(hf_name.split("_")[1])
            all_good = False if best_cos < 0.99 else all_good

        print(f"{hf_name:<20s} {best_name:<40s} {best_cos:>10.6f}  {status}")

    # ── Also check final output ──────────────────────────────
    print("\n=== Final output comparison ===")
    output_names = [n for n, _, _ in ir_func.outputs]
    if output_names and output_names[0] in ssa_values:
        ir_logits = ssa_values[output_names[0]]
    else:
        ir_logits = list(ssa_values.values())[-1]

    with torch.no_grad():
        hf_raw = model(example_input)
        hf_logits = hf_raw.logits if hasattr(hf_raw, "logits") else hf_raw

    final_cos = _cosine(hf_logits, ir_logits)
    print(f"  IR vs HF logits: cos={final_cos:.6f}")

    if first_bad_layer >= 0:
        print(f"\n*** First diverged layer: {first_bad_layer} ***")
        print(f"    Focus debugging on layer {first_bad_layer}'s computation chain.")


if __name__ == "__main__":
    main()
