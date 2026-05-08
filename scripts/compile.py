#!/usr/bin/env python3
"""Compile a PyTorch model through the LLM-ServeForge compiler pipeline.

Usage:
    python scripts/compile.py opt-125m    # Compile facebook/opt-125m
    python scripts/compile.py opt-125m --output-dir ./compiled/opt125m
    python scripts/compile.py --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# Ensure the project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))


def _patch_transformers_torch() -> None:
    """Patch transformers to recognize the symlinked torch installation."""
    import torch

    import transformers.utils.generic as _generic  # type: ignore[import-untyped]
    import transformers.utils.import_utils as _iu  # type: ignore[import-untyped]

    _iu._torch_available = True  # type: ignore[attr-defined]
    _iu._torch_version = torch.__version__  # type: ignore[attr-defined]

    _generic._torch_pytree = torch.utils._pytree  # type: ignore[attr-defined]

    def _flatten(output: Any) -> Any:  # type: ignore[no-untyped-def]
        return list(output.values()), list(output.keys())

    def _unflatten(values: Any, context: Any, output_type: Any = None) -> Any:  # type: ignore[no-untyped-def]
        return (output_type or type(context[0]))(**dict(zip(context, values)))

    _generic._model_output_flatten = _flatten  # type: ignore[attr-defined]
    _generic._model_output_unflatten = _unflatten  # type: ignore[attr-defined]


def compile_opt125m(output_dir: str) -> None:
    """Compile facebook/opt-125m through the full pipeline."""
    import os
    import torch

    _patch_transformers_torch()

    from compiler.pipeline import default_pipeline
    from torch.export import Dim
    from transformers.models.opt.configuration_opt import OPTConfig  # type: ignore[import-untyped]
    from transformers.models.opt.modeling_opt import OPTForCausalLM  # type: ignore[import-untyped]

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

    pipeline = default_pipeline()
    ir_module = pipeline.compile(
        model,
        example_args=(example_input,),
        output_dir=output_dir,
        dynamic_shapes={"input_ids": {0: Dim("batch"), 1: Dim("seq")}},
    )

    op_count = len(ir_module.main.ops)
    weight_count = len(ir_module.main.weights)
    print(f"Compiled: {op_count} ops, {weight_count} weight tensors")
    print(f"Artifact saved to: {output_dir}")


def compile_tiny_llama(output_dir: str) -> None:
    """Compile hf-internal-testing/tiny-random-LlamaForCausalLM."""
    import os
    import torch

    _patch_transformers_torch()

    from compiler.pipeline import default_pipeline
    from torch.export import Dim
    from transformers.models.llama.configuration_llama import LlamaConfig  # type: ignore[import-untyped]
    from transformers.models.llama.modeling_llama import LlamaForCausalLM  # type: ignore[import-untyped]

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
    print(f"Exporting with example input shape: {list(example_input.shape)} (dynamic batch + seq)")

    pipeline = default_pipeline()
    ir_module = pipeline.compile(
        model,
        example_args=(example_input,),
        output_dir=output_dir,
        dynamic_shapes={"input_ids": {0: Dim("batch"), 1: Dim("seq")}},
    )

    op_count = len(ir_module.main.ops)
    weight_count = len(ir_module.main.weights)
    print(f"Compiled: {op_count} ops, {weight_count} weight tensors")
    print(f"Artifact saved to: {output_dir}")


def compile_qwen(output_dir: str) -> None:
    """Compile Qwen/Qwen3.5-0.8B through the full pipeline."""
    import os
    import torch

    _patch_transformers_torch()

    from compiler.pipeline import default_pipeline
    from torch.export import Dim
    from transformers import AutoConfig, AutoModelForCausalLM  # type: ignore[import-untyped]

    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "Qwen", "Qwen3.5-0.8B")
    model_dir = os.path.abspath(model_dir)

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    print(f"Loading Qwen3.5-0.8B from: {model_dir}")

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    # Qwen3_5 stores use_cache in text_config sub-config
    if hasattr(config, "text_config") and hasattr(config.text_config, "use_cache"):
        config.text_config.use_cache = False
    elif hasattr(config, "use_cache"):
        config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(model_dir, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.eval()

    example_input = torch.randint(0, 248320, (1, 64), dtype=torch.long)
    print(f"Exporting with example input shape: {list(example_input.shape)} (static shape due to linear attention constraints)")

    pipeline = default_pipeline()
    pipeline.cache_export = True
    ir_module = pipeline.compile(
        model,
        example_args=(example_input,),
        output_dir=output_dir,
        dynamic_shapes=None,
        model_dir=model_dir,
    )

    op_count = len(ir_module.main.ops)
    weight_count = len(ir_module.main.weights)
    print(f"Compiled: {op_count} ops, {weight_count} weight tensors")

    mlir_text = ir_module.metadata.get("mlir", "")
    mlir_lines = len(mlir_text.splitlines()) if mlir_text else 0
    print(f"MLIR output: {mlir_lines} lines")
    print(f"Artifact saved to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile a PyTorch model through LLM-ServeForge compiler",
    )
    parser.add_argument(
        "model",
        choices=["opt-125m", "tiny-llama", "qwen"],
        help="Model to compile (opt-125m, tiny-llama, or qwen)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for compiled artifacts",
    )
    args = parser.parse_args()

    targets = {
        "opt-125m": (compile_opt125m, "./compiled/opt_125m"),
        "tiny-llama": (compile_tiny_llama, "./compiled/tiny_llama"),
        "qwen": (compile_qwen, "./compiled/qwen3_0.8b"),
    }

    func, default_dir = targets[args.model]
    output_dir = args.output_dir or default_dir
    func(output_dir)


if __name__ == "__main__":
    main()
