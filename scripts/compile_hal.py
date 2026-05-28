#!/usr/bin/env python3
"""HAL IR v0 integration pipeline: compile_mlir → SF Normalize → LowerSFToHal → EmitRust.

Usage:
    python scripts/compile_hal.py opt-125m              # Compile + HAL pipeline for opt-125m
    python scripts/compile_hal.py opt-125m --no-compile  # Re-process existing model.mlir

Pipeline steps:
  1. (optional) Source model → compile_mlir() → model.mlir + metadata.json
  2. SF Normalize — decompose complex SF ops (SDPA, linear, layer_norm)
  3. LowerSFToHal — map primitive SF ops to HAL IR JSON (hal_ir.json)
  4. EmitRust — generate hal_ops_cpu.rs from HAL IR
  5. Summary
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

_log = logging.getLogger("compile_hal")


# ── Pipeline step implementations ──────────────────────────────────────


def step_normalize(mlir_text: str, output_dir: str) -> str:
    """Step 2: SF Normalize — decompose complex SF ops into primitives.

    Returns the normalized MLIR text.
    """
    from compiler.mlir_dialect.hal_ir.sf_normalize import normalize_sf_mlir

    _log.info("Step [2/4]: SF Normalize — decomposing complex SF ops ...")
    normalized = normalize_sf_mlir(mlir_text)
    norm_path = os.path.join(output_dir, "model.normalized.mlir")
    with open(norm_path, "w") as f:
        f.write(normalized)
    _log.info("  Wrote normalized MLIR: %s (%d bytes)", norm_path, len(normalized))

    # Summary of decomposition
    sdpa_before = mlir_text.count('"sf.scaled_dot_product_attention"')
    linear_before = mlir_text.count('"sf.linear"')
    ln_before = mlir_text.count('"sf.layer_norm"')
    sdpa_after = normalized.count('"sf.scaled_dot_product_attention"')
    linear_after = normalized.count('"sf.linear"')
    ln_after = normalized.count('"sf.layer_norm"')
    decomposed = (sdpa_before - sdpa_after) + (linear_before - linear_after) + (ln_before - ln_after)
    _log.info(
        "  Decomposition: %d SDPA, %d linear, %d layer_norm = %d total decomposed",
        sdpa_before - sdpa_after, linear_before - linear_after, ln_before - ln_after, decomposed,
    )
    return normalized


def step_lower_to_hal(output_dir: str) -> str:
    """Step 3: LowerSFToHal — map primitive SF ops to HAL IR JSON.

    Returns the path to ``hal_ir.json``.
    """
    from compiler.mlir_dialect.hal_ir.lower_sf_to_hal import lower_sf_to_hal_file

    _log.info("Step [3/4]: LowerSFToHal — generating HAL IR JSON ...")
    mlir_path = os.path.join(output_dir, "model.normalized.mlir")
    meta_path = os.path.join(output_dir, "metadata.json")

    result_path = lower_sf_to_hal_file(
        mlir_path=mlir_path,
        metadata_path=meta_path if os.path.isfile(meta_path) else None,
    )

    # Load the JSON to report stats
    with open(result_path) as f:
        hal_ir = json.load(f)
    total_ops = sum(len(fn["ops"]) for fn in hal_ir.get("functions", []))
    op_types: set[str] = set()
    for fn in hal_ir.get("functions", []):
        for op in fn.get("ops", []):
            op_types.add(op["op"])
    _log.info(
        "  HAL IR: %d functions, %d total ops, %d unique op types: %s",
        hal_ir.get("num_functions", 0),
        total_ops,
        len(op_types),
        sorted(op_types),
    )
    return result_path


def step_emit_rust(hal_ir_path: str, output_dir: str) -> str:
    """Step 4: EmitRust — generate hal_ops_cpu.rs from HAL IR.

    Returns the path to the generated Rust file.
    """
    from compiler.mlir_dialect.hal_ir.emit_rust import emit_rust

    _log.info("Step [4/4]: EmitRust — generating hal_ops_cpu.rs ...")
    rust_path = emit_rust(
        hal_ir_path=hal_ir_path,
        output_path=os.path.join(output_dir, "hal_ops_cpu.rs"),
    )
    _log.info("  Generated: %s", rust_path)
    return rust_path


# ── Model compile helpers (mirrors scripts/compile.py) ─────────────────


def compile_opt125m(output_dir: str) -> None:
    """Compile facebook/opt-125m through compile_mlir()."""
    import torch
    from torch.export import Dim
    from transformers.models.opt.configuration_opt import OPTConfig  # type: ignore[import-untyped]
    from transformers.models.opt.modeling_opt import OPTForCausalLM  # type: ignore[import-untyped]

    from compiler.mlir_dialect.compile_utils import _patch_transformers_torch
    from compiler.pipeline import compile_mlir

    _patch_transformers_torch()

    hub_dir = os.path.expanduser("~/.cache/huggingface/hub/models--facebook--opt-125m")
    snapshots = os.path.join(hub_dir, "snapshots")
    if not os.path.isdir(snapshots):
        raise FileNotFoundError(f"Model not found at {hub_dir}")
    snap = os.listdir(snapshots)[0]
    model_path = os.path.join(snapshots, snap, "pytorch_model.bin")

    _log.info("Loading weights from: %s", model_path)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)

    _log.info("Building OPT-125M model...")
    config_path = os.path.join(snapshots, snap, "config.json")
    config = OPTConfig.from_pretrained(config_path) if os.path.exists(config_path) else OPTConfig()
    config.use_cache = False
    model = OPTForCausalLM(config)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    example_input = torch.randint(0, 50272, (2, 4), dtype=torch.long)
    position_ids = torch.arange(0, 4, dtype=torch.long).unsqueeze(0).expand(2, -1)
    _log.info("Exporting with example input shape: %s (dynamic batch + seq)", list(example_input.shape))

    mlir_mod = compile_mlir(
        model,
        example_args=(example_input,),
        example_kwargs={"position_ids": position_ids},
        output_dir=output_dir,
        model_dir=os.path.join(snapshots, snap),
        dynamic_shapes={
            "input_ids":    {0: Dim("batch"), 1: Dim("seq")},
            "position_ids": {0: Dim("batch"), 1: Dim("seq")},
        },
    )

    op_count = sum(len(f.ops) for f in mlir_mod.functions)
    weight_count = sum(len(f.weights) for f in mlir_mod.functions)
    _log.info("  Compiled: %d ops, %d weight tensors → %s", op_count, weight_count, output_dir)


def compile_tiny_llama(output_dir: str) -> None:
    """Compile hf-internal-testing/tiny-random-LlamaForCausalLM."""
    import torch
    from torch.export import Dim
    from transformers.models.llama.configuration_llama import LlamaConfig  # type: ignore[import-untyped]
    from transformers.models.llama.modeling_llama import LlamaForCausalLM  # type: ignore[import-untyped]

    from compiler.mlir_dialect.compile_utils import _patch_transformers_torch
    from compiler.pipeline import compile_mlir

    _patch_transformers_torch()

    model_name = "hf-internal-testing/tiny-random-LlamaForCausalLM"
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
    import safetensors.torch
    state_dict = safetensors.torch.load_file(model_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    example_input = torch.randint(0, 32000, (2, 4), dtype=torch.long)
    position_ids = torch.arange(0, 4, dtype=torch.long).unsqueeze(0).expand(2, -1)
    _log.info("Exporting with example input shape: %s (dynamic batch + seq)", list(example_input.shape))

    compile_mlir(
        model,
        example_args=(example_input,),
        example_kwargs={"position_ids": position_ids},
        output_dir=output_dir,
        dynamic_shapes={
            "input_ids":    {0: Dim("batch"), 1: Dim("seq")},
            "position_ids": {0: Dim("batch"), 1: Dim("seq")},
        },
    )


def compile_qwen(output_dir: str) -> None:
    """Compile Qwen model (placeholder — not fully tested)."""
    import torch
    from torch.export import Dim as _Dim

    from compiler.mlir_dialect.compile_utils import _patch_transformers_torch
    from compiler.pipeline import compile_mlir

    _patch_transformers_torch()

    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(_root, "models", "Qwen", "Qwen3.5-0.8B")
    model_dir = os.path.abspath(model_dir)

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    from transformers import AutoConfig, AutoModelForCausalLM  # type: ignore[import-untyped]

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    tc = config.text_config if hasattr(config, "text_config") else config
    tc.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    if hasattr(model, "lm_head") and hasattr(model.model, "embed_tokens"):
        if getattr(config, "tie_word_embeddings", False):
            model.lm_head.weight = model.model.embed_tokens.weight
    model.eval()

    from compiler.cache_policy import CachePolicy
    num_layers = tc.num_hidden_layers if hasattr(tc, "num_hidden_layers") else 24
    num_heads = tc.num_attention_heads if hasattr(tc, "num_attention_heads") else 8
    head_dim = getattr(tc, "head_dim", 256)
    cache_policy = CachePolicy.for_llama(num_layers=num_layers, num_kv_heads=num_heads, head_dim=head_dim)

    example_input = torch.randint(0, 248320, (1, 64), dtype=torch.long)
    _log.info("Exporting Qwen with input shape: %s", list(example_input.shape))

    compile_mlir(
        model,
        example_args=(example_input,),
        output_dir=output_dir,
        model_dir=model_dir,
        cache_export=False,
        cache_policy=cache_policy,
    )


# ── Main ────────────────────────────────────────────────────────────────


def main() -> None:
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="HAL IR v0 integration pipeline",
    )
    parser.add_argument(
        "model",
        nargs="?",
        default="opt-125m",
        choices=["opt-125m", "tiny-llama", "qwen"],
        help="Model to compile (default: opt-125m)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: ./compiled/<model>_hal)",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        default=False,
        help="Skip model compilation; re-process existing artifacts",
    )
    args = parser.parse_args()

    # Ensure project root on sys.path
    _project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_project_root))

    # Determine default output dir
    model_dirs = {
        "opt-125m": "./compiled/opt_125m_hal",
        "tiny-llama": "./compiled/tiny_llama_hal",
        "qwen": "./compiled/qwen_hal",
    }
    output_dir = args.output_dir or model_dirs[args.model]
    os.makedirs(output_dir, exist_ok=True)

    mlir_path = os.path.join(output_dir, "model.mlir")

    # ── Step 1: Compile model (optional) ──────────────────────────────
    if not args.no_compile:
        _log.info("Step [1/4]: Compiling model → MLIR ...")
        compilers = {
            "opt-125m": compile_opt125m,
            "tiny-llama": compile_tiny_llama,
            "qwen": compile_qwen,
        }
        compilers[args.model](output_dir)
        if not os.path.isfile(mlir_path):
            _log.error("  Compilation did not produce %s", mlir_path)
            sys.exit(1)
        _log.info("  Model MLIR: %s", mlir_path)
    else:
        if not os.path.isfile(mlir_path):
            _log.error(
                "  --no-compile but %s not found. Compile first or remove --no-compile.",
                mlir_path,
            )
            sys.exit(1)
        _log.info("Step [1/4]: SKIP (--no-compile). Using existing: %s", mlir_path)

    # Read MLIR text
    with open(mlir_path) as f:
        mlir_text = f.read()

    # ── Step 2: SF Normalize ──────────────────────────────────────────
    normalized = step_normalize(mlir_text, output_dir)

    # ── Step 3: LowerSFToHal ──────────────────────────────────────────
    hal_ir_path = step_lower_to_hal(output_dir)

    # ── Step 4: EmitRust ──────────────────────────────────────────────
    rust_path = step_emit_rust(hal_ir_path, output_dir)

    # ── Summary ────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  HAL IR v0 Pipeline Complete")
    print("=" * 60)
    print(f"  Model:          {args.model}")
    print(f"  Output dir:     {output_dir}")
    print()

    # Load HAL IR for final stats
    with open(hal_ir_path) as f:
        hal_ir = json.load(f)
    total_ops = sum(len(fn["ops"]) for fn in hal_ir.get("functions", []))
    op_types: set[str] = set()
    for fn in hal_ir.get("functions", []):
        for op in fn.get("ops", []):
            op_types.add(op["op"])
    print(f"  Functions:      {hal_ir.get('num_functions', 0)}")
    print(f"  Total ops:      {total_ops}")
    print(f"  Unique op types: {len(op_types)} → {sorted(op_types)}")

    rust_size = os.path.getsize(rust_path)
    print(f"  Generated Rust: {rust_path} ({rust_size} bytes)")

    # Check for uncovered ops (no impl in emit_rust)
    from compiler.mlir_dialect.hal_ir.emit_rust import OP_IMPLS
    uncovered = op_types - set(OP_IMPLS.keys())
    if uncovered:
        print(f"  ⚠ Uncovered ops (no Rust impl): {sorted(uncovered)}")
    else:
        print("  ✓ All ops have Rust implementations")

    print()


if __name__ == "__main__":
    main()
