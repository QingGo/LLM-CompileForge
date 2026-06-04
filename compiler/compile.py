#!/usr/bin/env python3
"""Compile a PyTorch model through the LLM-ServeForge compiler pipeline.

Usage:
    python compiler/compile.py opt-125m    # Compile facebook/opt-125m
    python compiler/compile.py opt-125m --output-dir ./outputs/compiled/opt125m
    python compiler/compile.py --help
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

# Ensure the project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))


from compiler.mlir_dialect.lowering.compile_utils import _patch_transformers_torch


def compile_opt125m(output_dir: str, apply_lowering: bool = False,
                    cache_policy: Any = None) -> None:
    """Compile facebook/opt-125m through the full pipeline.

    Args:
        cache_policy: Optional CachePolicy for KV cache strategy.
    """
    import os

    import torch

    _patch_transformers_torch()

    from torch.export import Dim
    from transformers.models.opt.configuration_opt import OPTConfig  # type: ignore[import-untyped]
    from transformers.models.opt.modeling_opt import OPTForCausalLM  # type: ignore[import-untyped]

    from compiler.pipeline import compile_mlir

    # HF cache path for opt-125m
    hub_dir = os.path.expanduser("~/.cache/huggingface/hub/models--facebook--opt-125m")
    snapshots = os.path.join(hub_dir, "snapshots")
    if not os.path.isdir(snapshots):
        raise FileNotFoundError(f"Model not found at {hub_dir}")
    snap = os.listdir(snapshots)[0]
    model_path = os.path.join(snapshots, snap, "pytorch_model.bin")

    print(f"Loading weights from: {model_path}")
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)

    print("Building OPT-125M model...")
    config_path = os.path.join(snapshots, snap, "config.json")
    config = OPTConfig.from_pretrained(config_path) if os.path.exists(config_path) else OPTConfig()
    config.use_cache = False
    model = OPTForCausalLM(config)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    example_input = torch.randint(0, 50272, (2, 4), dtype=torch.long)
    print(f"Exporting with example input shape: {list(example_input.shape)} (dynamic batch + seq)")

    pipeline = compile_mlir
    mlir_mod = pipeline(
        model,
        example_args=(example_input,),
        output_dir=output_dir,
        model_dir=os.path.join(snapshots, snap),
        dynamic_shapes={
            "input_ids": {0: Dim("batch"), 1: Dim("seq")},
        },
        cache_policy=cache_policy,
    )

    op_count = len(mlir_mod.functions[0].ops)
    weight_count = len(mlir_mod.functions[0].weights)
    print(f"Compiled: {op_count} ops, {weight_count} weight tensors")
    print(f"Artifact saved to: {output_dir}")


def compile_tiny_llama(output_dir: str, apply_lowering: bool = False) -> None:
    """Compile hf-internal-testing/tiny-random-LlamaForCausalLM."""
    import os

    import torch

    _patch_transformers_torch()

    from torch.export import Dim
    from transformers.models.llama.configuration_llama import LlamaConfig  # type: ignore[import-untyped]
    from transformers.models.llama.modeling_llama import LlamaForCausalLM  # type: ignore[import-untyped]

    from compiler.pipeline import compile_mlir

    model_name = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    print(f"Loading {model_name} weights...")

    hub_dir = os.path.expanduser(
        "~/.cache/huggingface/hub/models--hf-internal-testing--tiny-random-LlamaForCausalLM"
    )
    snapshots = os.path.join(hub_dir, "snapshots")
    snap = os.listdir(snapshots)[0]
    model_path = os.path.join(snapshots, snap, "model.safetensors")

    config_path = os.path.join(snapshots, snap, "config.json")
    config = LlamaConfig.from_pretrained(config_path) if os.path.exists(config_path) else LlamaConfig()
    config.use_cache = False
    model = LlamaForCausalLM(config)
    # Load safetensors
    import safetensors.torch
    state_dict = safetensors.torch.load_file(model_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    example_input = torch.randint(0, 32000, (2, 4), dtype=torch.long)
    position_ids = torch.arange(0, 4, dtype=torch.long).unsqueeze(0).expand(2, -1)
    print(f"Exporting with example input shape: {list(example_input.shape)} (dynamic batch + seq)")

    pipeline = compile_mlir
    mlir_mod = pipeline(
        model,
        example_args=(example_input,),
        example_kwargs={"position_ids": position_ids},
        output_dir=output_dir,
        dynamic_shapes={
            "input_ids":    {0: Dim("batch"), 1: Dim("seq")},
            "position_ids": {0: Dim("batch"), 1: Dim("seq")},
        },
    )

    op_count = len(mlir_mod.functions[0].ops)
    weight_count = len(mlir_mod.functions[0].weights)
    print(f"Compiled: {op_count} ops, {weight_count} weight tensors")
    print(f"Artifact saved to: {output_dir}")


def compile_qwen(output_dir: str, apply_lowering: bool = False) -> None:
    """Compile Qwen/Qwen3.5-0.8B through the full pipeline."""
    import os

    import torch

    _patch_transformers_torch()

    from transformers import AutoConfig, AutoModelForCausalLM  # type: ignore[import-untyped]

    from compiler.cache_policy import CachePolicy
    from compiler.pipeline import compile_mlir

    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(_root, "models", "Qwen", "Qwen3.5-0.8B")
    model_dir = os.path.abspath(model_dir)

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    print(f"Loading Qwen3.5-0.8B from: {model_dir}")

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    tc = config.text_config if hasattr(config, "text_config") else config
    tc.use_cache = False
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    if hasattr(model, "lm_head") and hasattr(model.model, "embed_tokens"):
        if getattr(config, "tie_word_embeddings", False):
            model.lm_head.weight = model.model.embed_tokens.weight
    model.eval()

    num_layers = tc.num_hidden_layers if hasattr(tc, "num_hidden_layers") else 24
    num_heads = tc.num_attention_heads if hasattr(tc, "num_attention_heads") else 8
    head_dim = getattr(tc, "head_dim", 256)
    cache_policy = CachePolicy.for_llama(num_layers=num_layers, num_kv_heads=num_heads, head_dim=head_dim)

    example_input = torch.randint(0, 248320, (1, 64), dtype=torch.long)
    print(f"Exporting with example input shape: {list(example_input.shape)} (static shape)")

    mlir_mod = compile_mlir(
        model,
        example_args=(example_input,),
        output_dir=output_dir,
        model_dir=model_dir,
        cache_export=False,
        cache_policy=cache_policy,
    )

    op_count = len(mlir_mod.functions[0].ops)
    weight_count = len(mlir_mod.functions[0].weights)
    print(f"Compiled: {op_count} ops, {weight_count} weight tensors")
    print(f"Artifact saved to: {output_dir}")

    # ── Self-check ───────────────────────────────────────
    from compiler.serialize import load_artifact
    from python_runtime.engine.mlir_executor import MlirExecutor
    from python_runtime.hal.pytorch_backend import PyTorchBackend
    reloaded = load_artifact(output_dir)
    ex = MlirExecutor(reloaded, PyTorchBackend("cpu"))
    test_in = torch.randint(0, 248320, (1, 64), dtype=torch.long)
    out = ex.forward(test_in)
    assert not torch.isnan(out).any(), "NaN in output"
    assert not torch.isinf(out).any(), "Inf in output"
    print(f"Self-check OK: output {list(out.shape)}, no NaN/Inf")


def compile_llama_1b(output_dir: str, apply_lowering: bool = False) -> None:
    """Compile Llama-3.2-1B from models/LLM-Research/Llama-3.2-1B."""
    import os

    import torch

    _patch_transformers_torch()


    from compiler.cache_policy import CachePolicy
    from compiler.pipeline import compile_mlir

    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "models", "LLM-Research", "Llama-3.2-1B")
    model_dir = os.path.abspath(model_dir)

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    print(f"Loading Llama-3.2-1B from: {model_dir}")

    from transformers import AutoConfig, AutoModelForCausalLM  # type: ignore[import-untyped]

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=False)
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, config=config, torch_dtype=torch.bfloat16,
    )
    model.eval()

    # Handle tied weights: safetensors omits lm_head.weight when tie_word_embeddings=True,
    # but torch.export captures both as separate graph inputs. Ensure lm_head has its own
    # tensor reference so the export sees it and weight_source can locate the canonical key.
    if hasattr(model, "lm_head") and hasattr(model.model, "embed_tokens"):
        if config.tie_word_embeddings:
            model.lm_head.weight = model.model.embed_tokens.weight

    example_input = torch.randint(0, 128256, (1, 8), dtype=torch.long)
    position_ids = torch.arange(0, 8, dtype=torch.long).unsqueeze(0).expand(1, -1)
    print(f"Exporting with example input shape: {list(example_input.shape)} (dynamic seq)")

    cache_policy = CachePolicy.for_llama(
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_attention_heads,
        head_dim=config.head_dim,
    )

    from torch.export import Dim as _Dim
    dynamic_shapes = {
        "input_ids":    {1: _Dim("seq", min=1, max=256)},
        "position_ids": {1: _Dim("seq", min=1, max=256)},
    }

    mlir_mod = compile_mlir(
        model,
        example_args=(example_input,),
        example_kwargs={"position_ids": position_ids},
        output_dir=output_dir,
        dynamic_shapes=dynamic_shapes,
        model_dir=model_dir,
        cache_export=False,
        cache_policy=cache_policy,
    )

    op_count = len(mlir_mod.functions[0].ops)
    weight_count = len(mlir_mod.functions[0].weights)
    print(f"Compiled: {op_count} ops, {weight_count} weight tensors")

    mlir_text = mlir_mod.metadata.get("mlir", "")
    mlir_lines = len(mlir_text.splitlines()) if mlir_text else 0
    print(f"MLIR output: {mlir_lines} lines")
    print(f"Artifact saved to: {output_dir}")

    # ── Self-check: reload and run quick forward ───────
    from compiler.serialize import load_artifact
    from python_runtime.engine.mlir_executor import MlirExecutor
    from python_runtime.hal.pytorch_backend import PyTorchBackend

    reloaded = load_artifact(output_dir)
    be = PyTorchBackend("cpu")
    ex = MlirExecutor(reloaded, be)
    test_in = torch.randint(0, 128256, (1, 8), dtype=torch.long)
    out = ex.forward(test_in)
    assert out.shape[-1] == config.vocab_size, f"Bad vocab {out.shape[-1]}"
    assert not torch.isnan(out).any(), "NaN in output"
    assert not torch.isinf(out).any(), "Inf in output"
    print(f"Self-check OK: output {list(out.shape)}, no NaN/Inf")

    return mlir_mod


def compile_llama_3b(output_dir: str, apply_lowering: bool = False) -> None:
    """Compile Llama-3.2-3B from models/LLM-Research/Llama-3.2-3B."""
    import os

    import torch

    _patch_transformers_torch()

    from torch.export import Dim as _Dim
    from transformers import AutoConfig, AutoModelForCausalLM  # type: ignore[import-untyped]

    from compiler.cache_policy import CachePolicy
    from compiler.pipeline import compile_mlir

    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "models", "LLM-Research", "Llama-3.2-3B")
    model_dir = os.path.abspath(model_dir)

    print(f"Loading Llama-3.2-3B from: {model_dir}")
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=False)
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, config=config, torch_dtype=torch.bfloat16,
    )
    if hasattr(model, "lm_head") and hasattr(model.model, "embed_tokens"):
        if config.tie_word_embeddings:
            model.lm_head.weight = model.model.embed_tokens.weight
    model.eval()

    example_input = torch.randint(0, 128256, (1, 8), dtype=torch.long)
    position_ids = torch.arange(0, 8, dtype=torch.long).unsqueeze(0).expand(1, -1)
    print(f"Exporting with example input shape: {list(example_input.shape)} (dynamic seq)")

    cache_policy = CachePolicy.for_llama(
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_attention_heads,
        head_dim=config.head_dim,
    )
    dynamic_shapes = {
        "input_ids":    {1: _Dim("seq", min=1, max=256)},
        "position_ids": {1: _Dim("seq", min=1, max=256)},
    }

    mlir_mod = compile_mlir(
        model,
        example_args=(example_input,),
        example_kwargs={"position_ids": position_ids},
        output_dir=output_dir,
        dynamic_shapes=dynamic_shapes,
        model_dir=model_dir,
        cache_export=False,
        cache_policy=cache_policy,
    )

    op_count = len(mlir_mod.functions[0].ops)
    weight_count = len(mlir_mod.functions[0].weights)
    print(f"Compiled: {op_count} ops, {weight_count} weight tensors")
    print(f"Artifact saved to: {output_dir}")

    # ── Self-check ───────────────────────────────────────
    from compiler.serialize import load_artifact
    from python_runtime.engine.mlir_executor import MlirExecutor
    from python_runtime.hal.pytorch_backend import PyTorchBackend

    reloaded = load_artifact(output_dir)
    ex = MlirExecutor(reloaded, PyTorchBackend("cpu"))
    test_in = torch.randint(0, 128256, (1, 8), dtype=torch.long)
    out = ex.forward(test_in)
    assert out.shape[-1] == config.vocab_size, f"Bad vocab {out.shape[-1]}"
    assert not torch.isnan(out).any(), "NaN in output"
    assert not torch.isinf(out).any(), "Inf in output"
    print(f"Self-check OK: output {list(out.shape)}, no NaN/Inf")

    return mlir_mod


def compile_rwkv(output_dir: str, apply_lowering: bool = False) -> None:
    """Compile RWKV-7 g1d-0.4b from models/RWKV/."""
    import json
    import os

    import torch

    _patch_transformers_torch()

    from compiler.cache_policy import CachePolicy
    from compiler.pipeline import compile_mlir
    from models.RWKV.rwkv_model import RWKV7Config, RWKV7Model

    pth_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "RWKV", "rwkv7-g1",
    )
    pth_dir = os.path.abspath(pth_dir)
    pth_path = os.path.join(pth_dir, "rwkv7-g1d-0.4b-20260210-ctx8192.pth")
    if not os.path.isfile(pth_path):
        raise FileNotFoundError(f"Model weights not found: {pth_path}")

    print(f"Loading RWKV-7 weights from: {pth_path}")
    config = RWKV7Config(vocab_size=65536, hidden_size=1024, num_layers=24)
    model = RWKV7Model(config)
    model.load_weights_from_pth(pth_path)
    model.eval()

    example_input = torch.randint(0, config.vocab_size, (1, 4), dtype=torch.long)
    print(f"Exporting RWKV-7 with input shape: {list(example_input.shape)} (static seq=4)")

    cache_policy = CachePolicy.for_rwkv(
        num_layers=config.num_layers,
        state_dim=config.hidden_size,
    )

    mlir_mod = compile_mlir(
        model,
        example_args=(example_input,),
        output_dir=output_dir,
        cache_export=False,
        cache_policy=cache_policy,
    )

    # Write weight_source with name mapping
    meta_path = os.path.join(output_dir, "metadata.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        meta["weight_source"] = {
            "path": pth_path,
            "format": "pytorch_bin",
            "name_mapping": {
                "blocks_0_ln0_weight": "ln0_weight",
                "blocks_0_ln0_bias": "ln0_bias",
            },
        }
        # Embed sf-dialect git hash for staleness detection
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD:sf-dialect/"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                _hash = result.stdout.strip()
                if len(_hash) == 40 and all(c in "0123456789abcdef" for c in _hash):
                    meta["sf_dialect_hash"] = _hash
                else:
                    print("WARNING: invalid sf_dialect hash, skipping", file=sys.stderr)
            else:
                print("WARNING: could not get sf_dialect hash", file=sys.stderr)
        except Exception:
            print("WARNING: exception getting sf_dialect hash, skipping", file=sys.stderr)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    op_count = len(mlir_mod.functions[0].ops)
    weight_count = len(mlir_mod.functions[0].weights)
    print(f"Compiled: {op_count} ops, {weight_count} weight tensors")
    print(f"Artifact saved to: {output_dir}")
    return mlir_mod


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile a PyTorch model through LLM-ServeForge compiler",
    )
    parser.add_argument(
        "model",
        choices=["opt-125m", "tiny-llama", "qwen", "llama-1b", "llama-3b", "rwkv"],
        help="Model to compile",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for compiled artifacts",
    )
    parser.add_argument(
        "--cache-policy",
        action="store_true",
        default=False,
        help="Enable CachePolicy for models that support it (opt-125m: for_llama, etc.)",
    )
    args = parser.parse_args()

    targets = {
        "opt-125m": (compile_opt125m, "./outputs/compiled/opt_125m"),
        "tiny-llama": (compile_tiny_llama, "./outputs/compiled/tiny_llama"),
        "qwen": (compile_qwen, "./outputs/compiled/qwen3_0.8b"),
        "llama-1b": (compile_llama_1b, "./outputs/compiled/llama_1b"),
        "llama-3b": (compile_llama_3b, "./outputs/compiled/llama_3b"),
        "rwkv": (compile_rwkv, "./outputs/compiled/rwkv7_g1d_0.4b"),
    }

    func, default_dir = targets[args.model]
    output_dir = args.output_dir or default_dir

    # Build CachePolicy for models that support it
    cache_policy = None
    if args.cache_policy:
        from compiler.cache_policy import CachePolicy

        if args.model == "opt-125m":
            cache_policy = CachePolicy.for_llama(
                num_layers=12, num_kv_heads=12, head_dim=64,
            )
            print("CachePolicy enabled: opt-125m (12 layers, 12 KV heads, head_dim=64)")
        else:
            print(f"WARNING: --cache-policy not yet implemented for {args.model}, ignoring")

    if args.model == "opt-125m":
        func(output_dir, cache_policy=cache_policy)
    else:
        func(output_dir)


if __name__ == "__main__":
    main()
