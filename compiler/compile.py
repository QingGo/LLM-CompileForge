#!/usr/bin/env python3
"""Compile a PyTorch model through the LLM-ServeForge compiler pipeline.

Usage:
    python compiler/compile.py opt-125m    # Compile facebook/opt-125m
    python compiler/compile.py opt-125m --output-dir ./outputs/compiled/opt125m
    python compiler/compile.py --help
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

# Ensure the project root is on sys.path before any compiler imports.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from compiler.backend.compile_utils import _patch_transformers_torch  # noqa: E402

# ── Shared compilation helper ─────────────────────────────────────────


@dataclass
class CompileConfig:
    """Configuration for a single model compilation run.

    All model-specific behavior is encoded here; ``_compile_model()``
    executes the common pipeline: build → compile → print → self-check.
    """

    name: str
    output_dir: str
    example_input: torch.Tensor
    dynamic_shapes: dict[str, Any]
    example_kwargs: dict[str, Any] = field(default_factory=dict)

    # Model building (function or pre-built module)
    build_model: Callable[[], tuple[torch.nn.Module, str | None]] | None = None
    model: torch.nn.Module | None = None
    model_dir: str = ""

    cache_policy: Any = None
    cache_policy_from_config: bool = False
    cache_export: bool = False

    # Post-compile hooks
    post_hooks: list[Callable[[str, Any], None]] = field(default_factory=list)

    # LLaMA no-mask path: force HF's internally-created causal mask to None
    # so SDPA receives ``is_causal=True`` + ``enable_gqa=True`` instead of a
    # materialized 2D mask (A.2 contract).
    patch_causal_mask_to_none: bool = False


def _patch_llama_causal_mask_to_none() -> list[tuple[Any, Any]]:
    """Replace HF LLaMA's materialized causal mask with ``None``.

    The model's SDPA attention interface then takes the ``is_causal`` path and
    exports ``aten.scaled_dot_product_attention(..., None, 0.0, True)`` with
    ``enable_gqa=True``.  Returns the previous callables for restoration.
    """
    originals: list[tuple[Any, Any]] = []
    try:
        import transformers.masking_utils as _masking_utils
        import transformers.models.llama.modeling_llama as _llama_modeling
    except ImportError:
        return originals

    def _none(*args: Any, **kwargs: Any) -> None:
        return None

    for _mod in (_masking_utils, _llama_modeling):
        _mod_any: Any = _mod
        if hasattr(_mod_any, "create_causal_mask"):
            originals.append((_mod_any, _mod_any.create_causal_mask))
            _mod_any.create_causal_mask = _none
    return originals


def _restore_llama_causal_mask(originals: list[tuple[Any, Any]]) -> None:
    for _mod, _orig in originals:
        _mod.create_causal_mask = _orig


def _cache_policy_from_model(model: torch.nn.Module) -> Any:
    """Resolve a CachePolicy from the model config; never from literals.

    Mixed linear/full attention (Qwen3.5 Gated DeltaNet) is not yet
    representable by the SDPA-only policy (E11).  Export-only preflights
    therefore run with no cache policy instead of a silently wrong one.
    """
    from compiler.cache_policy import CachePolicy

    config = getattr(model, "config", None)
    if config is None:
        print("CachePolicy skipped: model has no .config attribute", file=sys.stderr)
        return CachePolicy.none()
    try:
        policy = CachePolicy.for_config(config)
        # E11/P1: the mixed Qwen policy is now describable, but the runtime
        # op-plan/linear-attn executor does not consume recurrent/conv-state
        # slabs yet.  Keep export-only compile on the no-cache path until
        # the stateful kernel is wired into execution.
        if any(s.slab_id in ("recurrent_state", "conv_state") for s in policy.slabs):
            print(
                "CachePolicy skipped: recurrent/conv-state slabs not yet consumed "
                "by the runtime; using no-cache for export-only preflight",
                file=sys.stderr,
            )
            return CachePolicy.none()
        return policy
    except NotImplementedError as exc:
        print(f"CachePolicy skipped: {exc}", file=sys.stderr)
        return CachePolicy.none()


def _compile_model(cfg: CompileConfig) -> None:
    """Execute the full compilation pipeline for a given config."""
    _patch_transformers_torch()
    from compiler.pipeline import compile_mlir

    # Build model (or use pre-built)
    if cfg.build_model is not None:
        model, model_dir = cfg.build_model()
    elif cfg.model is not None:
        model, model_dir = cfg.model, cfg.model_dir
    else:
        raise ValueError("Either build_model or model must be provided")
    model.eval()

    if cfg.cache_policy_from_config:
        if cfg.cache_policy is not None:
            raise ValueError("cache_policy and cache_policy_from_config are mutually exclusive")
        cfg.cache_policy = _cache_policy_from_model(model)

    causal_patch = _patch_llama_causal_mask_to_none() if cfg.patch_causal_mask_to_none else []
    try:
        mlir_mod = compile_mlir(
            model,
            example_args=(cfg.example_input,),
            example_kwargs=cfg.example_kwargs,
            output_dir=cfg.output_dir,
            model_dir=model_dir or cfg.model_dir,
            dynamic_shapes=cfg.dynamic_shapes,
            cache_export=cfg.cache_export,
            cache_policy=cfg.cache_policy,
        )
    finally:
        _restore_llama_causal_mask(causal_patch)

    op_count = len(mlir_mod.functions[0].ops) if mlir_mod.functions else 0
    weight_count = len(mlir_mod.functions[0].weights) if mlir_mod.functions else 0
    print(f"Compiled: {op_count} ops, {weight_count} weight tensors")
    print(f"Artifact saved to: {cfg.output_dir}")

    for hook in cfg.post_hooks:
        hook(cfg.output_dir, mlir_mod)


# ── Model builders ────────────────────────────────────────────────────


def _build_opt125m() -> tuple[torch.nn.Module, str | None]:
    from transformers.models.opt.configuration_opt import OPTConfig
    from transformers.models.opt.modeling_opt import OPTForCausalLM

    hub_dir = os.path.expanduser("~/.cache/huggingface/hub/models--facebook--opt-125m")
    snapshots = os.path.join(hub_dir, "snapshots")
    if not os.path.isdir(snapshots):
        raise FileNotFoundError(f"Model not found at {hub_dir}")
    snap = os.listdir(snapshots)[0]
    model_path = os.path.join(snapshots, snap, "pytorch_model.bin")

    print(f"Loading weights from: {model_path}")
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    config_path = os.path.join(snapshots, snap, "config.json")
    config = OPTConfig.from_pretrained(config_path) if os.path.exists(config_path) else OPTConfig()
    config.use_cache = False
    model = OPTForCausalLM(config)  # type: ignore[no-untyped-call]
    model.load_state_dict(state_dict, strict=False)
    return model, os.path.join(snapshots, snap)


def _build_tiny_llama() -> tuple[torch.nn.Module, str | None]:
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    model_name = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    print(f"Loading {model_name} weights...")
    hub_dir = os.path.expanduser("~/.cache/huggingface/hub/models--hf-internal-testing--tiny-random-LlamaForCausalLM")
    snapshots = os.path.join(hub_dir, "snapshots")
    snap = os.listdir(snapshots)[0]
    model_path = os.path.join(snapshots, snap, "model.safetensors")

    import safetensors.torch

    state_dict = safetensors.torch.load_file(model_path)
    config_path = os.path.join(snapshots, snap, "config.json")
    config = LlamaConfig.from_pretrained(config_path) if os.path.exists(config_path) else LlamaConfig()
    config.use_cache = False
    model = LlamaForCausalLM(config)  # type: ignore[no-untyped-call]
    model.load_state_dict(state_dict, strict=False)
    return model, os.path.join(snapshots, snap)


def _build_qwen() -> tuple[torch.nn.Module, str | None]:
    from transformers import AutoConfig, AutoModelForCausalLM

    model_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "Qwen", "Qwen3.5-0.8B"
    )
    model_dir = os.path.abspath(model_dir)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    print(f"Loading Qwen3.5-0.8B from: {model_dir}")
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    if hasattr(config, "text_config"):
        config.text_config.use_cache = False
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    _fix_tied_weights(model, config)
    return model, model_dir


def _build_llama(
    variant: str,
    torch_dtype: torch.dtype | None = None,
) -> tuple[torch.nn.Module, str | None]:
    from transformers import AutoConfig, AutoModelForCausalLM

    model_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "LLM-Research", f"Llama-3.2-{variant}"
    )
    model_dir = os.path.abspath(model_dir)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    print(f"Loading Llama-3.2-{variant} from: {model_dir}")
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=False)
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        config=config,
        torch_dtype=torch_dtype or torch.bfloat16,
    )
    _fix_tied_weights(model, config)
    return model, model_dir


def _build_rwkv() -> tuple[torch.nn.Module, str | None]:
    from models.RWKV.rwkv_model import RWKV7Config, RWKV7Model

    pth_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models",
        "RWKV",
        "rwkv7-g1",
    )
    pth_dir = os.path.abspath(pth_dir)
    pth_path = os.path.join(pth_dir, "rwkv7-g1d-0.4b-20260210-ctx8192.pth")
    if not os.path.isfile(pth_path):
        raise FileNotFoundError(f"Model weights not found: {pth_path}")
    print(f"Loading RWKV-7 weights from: {pth_path}")
    config = RWKV7Config(vocab_size=65536, hidden_size=1024, num_layers=24)
    model = RWKV7Model(config)
    model.load_weights_from_pth(pth_path)
    return model, pth_dir


# ── Helpers ────────────────────────────────────────────────────────────


def _fix_tied_weights(model: torch.nn.Module, config: Any) -> None:
    """Handle tied weights: safetensors omits lm_head.weight when
    tie_word_embeddings=True, but torch.export captures both as separate
    graph inputs. Ensure lm_head has its own tensor reference.
    """
    if (
        hasattr(model, "lm_head")
        and hasattr(model.model, "embed_tokens")
        and getattr(config, "tie_word_embeddings", False)
    ):
        assert isinstance(model.lm_head, torch.nn.Module)
        assert isinstance(model.model.embed_tokens, torch.nn.Module)
        model.lm_head.weight = model.model.embed_tokens.weight


def _rwkv_post_hook(output_dir: str, mlir_mod: Any) -> None:
    """Write weight_source metadata and sf-dialect git hash for RWKV models."""
    pth_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models",
        "RWKV",
        "rwkv7-g1",
    )
    pth_path = os.path.join(os.path.abspath(pth_dir), "rwkv7-g1d-0.4b-20260210-ctx8192.pth")
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
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD:sf-dialect/"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                sha = result.stdout.strip()
                if len(sha) == 40 and all(c in "0123456789abcdef" for c in sha):
                    meta["sf_dialect_hash"] = sha
        except Exception:
            pass
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)


def _make_position_ids(example_input: torch.Tensor) -> torch.Tensor:
    """Create position_ids matching the input batch/seq shape."""
    batch, seq = example_input.shape[0], example_input.shape[1]
    return torch.arange(0, seq, dtype=torch.long).unsqueeze(0).expand(batch, -1)


# ── Model compilation functions ────────────────────────────────────────


def compile_opt125m(
    output_dir: str,
    apply_lowering: bool = False,
    cache_policy: Any = None,
    cache_policy_from_config: bool = False,
) -> None:
    from torch.export import Dim

    example_input = torch.randint(0, 50272, (2, 4), dtype=torch.long)
    # Contract: position_ids is an explicit graph input (not computed from
    # arange(0, seq) inside the dylib). During decode the runtime feeds the
    # absolute position of the current token; an in-graph arange would
    # restart at 0 and corrupt the position embedding for every decode step.
    position_ids = _make_position_ids(example_input)
    _compile_model(
        CompileConfig(
            name="opt-125m",
            output_dir=output_dir,
            build_model=_build_opt125m,
            example_input=example_input,
            example_kwargs={"position_ids": position_ids},
            dynamic_shapes={
                "input_ids": {0: Dim("batch"), 1: Dim("seq")},
                "position_ids": {0: Dim("batch"), 1: Dim("seq")},
            },
            cache_policy=cache_policy,
            cache_policy_from_config=cache_policy_from_config,
        )
    )


def compile_tiny_llama(output_dir: str, apply_lowering: bool = False, cache_policy: Any = None) -> None:
    from torch.export import Dim

    example_input = torch.randint(0, 32000, (2, 4), dtype=torch.long)
    position_ids = _make_position_ids(example_input)
    _compile_model(
        CompileConfig(
            name="tiny-llama",
            output_dir=output_dir,
            build_model=_build_tiny_llama,
            example_input=example_input,
            example_kwargs={"position_ids": position_ids},
            dynamic_shapes={
                "input_ids": {0: Dim("batch"), 1: Dim("seq")},
                "position_ids": {0: Dim("batch"), 1: Dim("seq")},
            },
        )
    )


def compile_qwen(output_dir: str, apply_lowering: bool = False) -> None:
    example_input = torch.randint(0, 248320, (1, 64), dtype=torch.long)
    _compile_model(
        CompileConfig(
            name="qwen",
            output_dir=output_dir,
            build_model=_build_qwen,
            example_input=example_input,
            dynamic_shapes={},
            cache_policy_from_config=True,
            cache_export=False,
        )
    )


def compile_llama_1b(output_dir: str, apply_lowering: bool = False) -> None:
    from torch.export import Dim as _Dim

    # A-phase correctness path: f32 weights + no materialized attention mask.
    # SDPA receives is_causal=True + enable_gqa=True; scalar dropout_p is an
    # attribute, never a tensor operand (see test_sdpa_attention_contract.py).
    example_input = torch.randint(0, 128256, (1, 8), dtype=torch.long)
    position_ids = _make_position_ids(example_input)
    _compile_model(
        CompileConfig(
            name="llama-1b",
            output_dir=output_dir,
            build_model=lambda: _build_llama("1B", torch_dtype=torch.float32),
            example_input=example_input,
            example_kwargs={"position_ids": position_ids},
            dynamic_shapes={
                "input_ids": {1: _Dim("seq", min=1, max=256)},
                "position_ids": {1: _Dim("seq", min=1, max=256)},
            },
            cache_policy_from_config=True,
            cache_export=False,
            patch_causal_mask_to_none=True,
        )
    )


def compile_llama_3b(output_dir: str, apply_lowering: bool = False) -> None:
    from torch.export import Dim as _Dim

    example_input = torch.randint(0, 128256, (1, 8), dtype=torch.long)
    position_ids = _make_position_ids(example_input)
    _compile_model(
        CompileConfig(
            name="llama-3b",
            output_dir=output_dir,
            build_model=lambda: _build_llama("3B"),
            example_input=example_input,
            example_kwargs={"position_ids": position_ids},
            dynamic_shapes={
                "input_ids": {1: _Dim("seq", min=1, max=256)},
                "position_ids": {1: _Dim("seq", min=1, max=256)},
            },
            cache_policy_from_config=True,
            cache_export=False,
            patch_causal_mask_to_none=True,
        )
    )


def compile_rwkv(output_dir: str, apply_lowering: bool = False) -> None:
    from compiler.cache_policy import CachePolicy

    def _build_and_make_input() -> tuple[torch.nn.Module, str | None]:
        model, model_dir = _build_rwkv()
        # Track config from the model for CachePolicy
        _build_and_make_input.raw_model = model  # type: ignore[attr-defined]
        return model, model_dir

    example_input = torch.randint(0, 65536, (1, 4), dtype=torch.long)
    _compile_model(
        CompileConfig(
            name="rwkv",
            output_dir=output_dir,
            build_model=_build_rwkv,
            example_input=example_input,
            dynamic_shapes={},
            cache_policy=CachePolicy.for_rwkv(
                num_layers=24,
                state_dim=1024,
            ),
            cache_export=False,
            post_hooks=[_rwkv_post_hook],
        )
    )


# ── CLI ────────────────────────────────────────────────────────────────


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
        help="Enable CachePolicy for models that support it",
    )
    args = parser.parse_args()

    targets: dict[str, tuple[Callable[..., Any], str]] = {
        "opt-125m": (compile_opt125m, "./outputs/compiled/opt_125m"),
        "tiny-llama": (compile_tiny_llama, "./outputs/compiled/tiny_llama"),
        "qwen": (compile_qwen, "./outputs/compiled/qwen3_0.8b"),
        "llama-1b": (compile_llama_1b, "./outputs/compiled/llama_1b"),
        "llama-3b": (compile_llama_3b, "./outputs/compiled/llama_3b"),
        "rwkv": (compile_rwkv, "./outputs/compiled/rwkv7_g1d_0.4b"),
    }

    func, default_dir = targets[args.model]
    output_dir = args.output_dir or default_dir

    cache_policy = None
    cache_policy_from_config = False
    if args.cache_policy and args.model == "opt-125m":
        cache_policy_from_config = True
        print("CachePolicy enabled: opt-125m (config-driven)")

    if args.model == "opt-125m":
        func(output_dir, cache_policy=cache_policy, cache_policy_from_config=cache_policy_from_config)
    else:
        func(output_dir)


if __name__ == "__main__":
    main()
