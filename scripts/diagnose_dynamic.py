#!/usr/bin/env python3
"""Diagnose why torch.export dynamic_shapes fails on real models.

Tests multiple configurations and reports the exact error for each.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Any

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))


def _patch_transformers_torch() -> None:
    import torch
    import transformers.utils.generic as _generic  # type: ignore[import-untyped]
    import transformers.utils.import_utils as _iu  # type: ignore[import-untyped]

    _iu._torch_available = True  # type: ignore[attr-defined]
    _iu._torch_version = torch.__version__  # type: ignore[attr-defined]
    _generic._torch_pytree = torch.utils._pytree  # type: ignore[attr-defined]

    def _flatten(output: Any) -> Any:
        return list(output.values()), list(output.keys())

    def _unflatten(values: Any, context: Any, output_type: Any = None) -> Any:
        return (output_type or type(context[0]))(**dict(zip(context, values)))

    _generic._model_output_flatten = _flatten  # type: ignore[attr-defined]
    _generic._model_output_unflatten = _unflatten  # type: ignore[attr-defined]


def _load_tiny_llama():
    from transformers.models.llama.configuration_llama import LlamaConfig  # type: ignore[import-untyped]
    from transformers.models.llama.modeling_llama import LlamaForCausalLM  # type: ignore[import-untyped]

    hub_dir = os.path.expanduser(
        "~/.cache/huggingface/hub/models--hf-internal-testing--tiny-random-LlamaForCausalLM"
    )
    snapshots = os.path.join(hub_dir, "snapshots")
    snap = os.listdir(snapshots)[0]
    model_path = os.path.join(snapshots, snap, "model.safetensors")
    config_path = os.path.join(snapshots, snap, "config.json")

    config = LlamaConfig.from_pretrained(config_path) if os.path.exists(config_path) else LlamaConfig()
    model = LlamaForCausalLM(config)

    import safetensors.torch
    state_dict = safetensors.torch.load_file(model_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def _load_opt125m():
    import torch
    from transformers.models.opt.configuration_opt import OPTConfig  # type: ignore[import-untyped]
    from transformers.models.opt.modeling_opt import OPTForCausalLM  # type: ignore[import-untyped]

    hub_dir = os.path.expanduser("~/.cache/huggingface/hub/models--facebook--opt-125m")
    snapshots = os.path.join(hub_dir, "snapshots")
    snap = os.listdir(snapshots)[0]
    model_path = os.path.join(snapshots, snap, "pytorch_model.bin")

    config_path = os.path.join(snapshots, snap, "config.json")
    config = OPTConfig.from_pretrained(config_path) if os.path.exists(config_path) else OPTConfig()

    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    model = OPTForCausalLM(config)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def _build_simple_model():
    """Build a minimal transformer-like model for baseline testing."""
    import torch.nn as nn

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(1000, 64)
            self.linear = nn.Linear(64, 100)
            self.norm = nn.LayerNorm(64)

        def forward(self, x):
            x = self.embed(x)
            x = self.norm(x)
            x = self.linear(x)
            return x

    return TinyModel().eval()


def test_config(label: str, model_fn, model_name: str,
                input_shape: tuple[int, ...],
                dynamic_shapes: Any = None, strict: bool = True) -> bool:
    """Test one dynamic_shapes configuration. Returns True on success."""
    import torch

    print(f"\n{'='*60}")
    print(f"  TEST: {label}")
    print(f"  Model: {model_name}")
    print(f"  Input shape: {list(input_shape)}")
    print(f"  dynamic_shapes: {dynamic_shapes}")
    print(f"  strict: {strict}")
    print(f"{'='*60}")

    try:
        model = model_fn()
        dummy_input = torch.randint(0, 500, input_shape, dtype=torch.long)

        ep = torch.export.export(
            model,
            (dummy_input,),
            kwargs={},
            dynamic_shapes=dynamic_shapes,
            strict=strict,
        )

        # Check what the exported graph looks like
        gm = ep.graph_module
        placeholder = None
        for node in gm.graph.nodes:
            if node.op == "placeholder" and node.name not in ("",):
                placeholder = node
                break

        if placeholder is not None and "val" in placeholder.meta:
            fake = placeholder.meta["val"]
            print(f"  SUCCESS! Exported input shape: {list(fake.shape)}")
            print(f"  Shape type: {fake.shape}")
            # Check if symbolic
            has_sym = any(hasattr(s, 'node') for s in fake.shape)
            print(f"  Has symbolic dims: {has_sym}")
        else:
            print("  SUCCESS! (no placeholder info)")

        print(f"  Graph nodes: {len(list(gm.graph.nodes))}")
        return True

    except Exception as e:
        print("  FAILED!")
        print(f"  Error type: {type(e).__name__}")
        print(f"  Error message: {str(e)[:500]}")
        print("\n  Full traceback (last 30 lines):")
        tb_lines = traceback.format_exc().splitlines()
        for line in tb_lines[-30:]:
            print(f"    {line}")
        return False


def main() -> None:
    _patch_transformers_torch()

    from torch.export import Dim

    results: dict[str, bool] = {}

    # ── Baseline: simple model, no dynamic_shapes ──
    results["simple_static"] = test_config(
        "Simple model (static)", _build_simple_model,
        "TinyModel(1000→64→100)", (1, 4),
        dynamic_shapes=None, strict=True,
    )

    # ── Baseline: simple model WITH dynamic_shapes ──
    results["simple_dynamic"] = test_config(
        "Simple model (dynamic batch+seq)",
        _build_simple_model, "TinyModel(1000→64→100)", (1, 4),
        dynamic_shapes={"x": {0: Dim("batch"), 1: Dim("seq")}},
        strict=True,
    )

    # ── tiny_llama: no dynamic_shapes (baseline) ──
    results["tinyllama_static_1x4"] = test_config(
        "tiny_llama (static, [1,4])",
        _load_tiny_llama, "hf-internal-testing/tiny-random-LlamaForCausalLM",
        (1, 4), dynamic_shapes=None, strict=True,
    )

    results["tinyllama_static_1x1"] = test_config(
        "tiny_llama (static, [1,1])",
        _load_tiny_llama, "hf-internal-testing/tiny-random-LlamaForCausalLM",
        (1, 1), dynamic_shapes=None, strict=True,
    )

    # ── tiny_llama: dynamic_shapes variants ──
    results["tinyllama_dyn_seq_only"] = test_config(
        "tiny_llama (dynamic seq only, [1,4])",
        _load_tiny_llama, "hf-internal-testing/tiny-random-LlamaForCausalLM",
        (1, 4),
        dynamic_shapes={"input_ids": {1: Dim("seq")}},
        strict=True,
    )

    results["tinyllama_dyn_batch_seq"] = test_config(
        "tiny_llama (dynamic batch+seq, [1,4])",
        _load_tiny_llama, "hf-internal-testing/tiny-random-LlamaForCausalLM",
        (1, 4),
        dynamic_shapes={"input_ids": {0: Dim("batch"), 1: Dim("seq")}},
        strict=True,
    )

    results["tinyllama_dyn_batch_seq_strict_false"] = test_config(
        "tiny_llama (dynamic batch+seq, strict=False, [1,4])",
        _load_tiny_llama, "hf-internal-testing/tiny-random-LlamaForCausalLM",
        (1, 4),
        dynamic_shapes={"input_ids": {0: Dim("batch"), 1: Dim("seq")}},
        strict=False,
    )

    # ── opt-125m: no dynamic_shapes (baseline) ──
    results["opt125m_static_1x1"] = test_config(
        "opt-125m (static, [1,1])",
        _load_opt125m, "facebook/opt-125m", (1, 1),
        dynamic_shapes=None, strict=True,
    )

    results["opt125m_static_1x4"] = test_config(
        "opt-125m (static, [1,4])",
        _load_opt125m, "facebook/opt-125m", (1, 4),
        dynamic_shapes=None, strict=True,
    )

    # ── opt-125m: dynamic_shapes variants ──
    results["opt125m_dyn_seq_only"] = test_config(
        "opt-125m (dynamic seq only, [1,4])",
        _load_opt125m, "facebook/opt-125m", (1, 4),
        dynamic_shapes={"input_ids": {1: Dim("seq")}},
        strict=True,
    )

    results["opt125m_dyn_batch_seq"] = test_config(
        "opt-125m (dynamic batch+seq, [1,1])",
        _load_opt125m, "facebook/opt-125m", (1, 1),
        dynamic_shapes={"input_ids": {0: Dim("batch"), 1: Dim("seq")}},
        strict=True,
    )

    results["opt125m_dyn_batch_seq_strict_false"] = test_config(
        "opt-125m (dynamic batch+seq, strict=False, [1,1])",
        _load_opt125m, "facebook/opt-125m", (1, 1),
        dynamic_shapes={"input_ids": {0: Dim("batch"), 1: Dim("seq")}},
        strict=False,
    )

    # ── Summary ──
    print(f"\n\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for name, ok in results.items():
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}")
    print(f"\n  Total: {sum(results.values())}/{len(results)} passed")


if __name__ == "__main__":
    main()
