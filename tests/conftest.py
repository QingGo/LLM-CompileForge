"""Shared pytest fixtures — eliminates repeated model/artifact loading.

When compiled models are missing, fixtures skip with clear instructions
on how to generate them.  This lets ``make test-unit`` pass on a fresh
clone without any compiled artifacts.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from tests.helpers import patch_transformers_torch


def pytest_runtest_setup(item: Any) -> None:
    if item.get_closest_marker("unit"):
        item.own_markers = list(item.own_markers)
        item.own_markers.append(pytest.mark.timeout(1))


_MODEL_SETUP_NOTE = (
    "No compiled model found.  Run one of:\n"
    "  bash scripts/setup.sh --with-models\n"
    "  .venv/bin/python scripts/compile.py tiny-llama"
)


def _require_compiled_model(model_dir: str) -> Path:
    path = Path(model_dir)
    if not path.is_dir():
        pytest.skip(f"{model_dir} not found.  {_MODEL_SETUP_NOTE}")
    return path


# ── MLIR context fixture — shared across all tests to avoid
#    nanobind Context create/destroy cycle issues ─────────────

@pytest.fixture(scope="session")
def mlir_context() -> Any:
    """Session-scoped MLIR Context.

    Shared across all tests to avoid creating/destroying many Context
    objects, which can trigger nanobind type registry instability in
    LLVM 22.x.  Threading is disabled — unit tests don't need parallelism.
    """
    import mlir.ir as ir

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    return ctx


# ── Compiled artifact fixtures ─────────────────────────────────


@pytest.fixture(scope="session")
def module_tiny_llama():
    from compiler.serialize import load_artifact
    p = _require_compiled_model("./compiled/tiny_llama")
    return load_artifact(str(p))


@pytest.fixture(scope="session")
def module_opt_125m():
    from compiler.serialize import load_artifact
    p = _require_compiled_model("./compiled/opt_125m")
    return load_artifact(str(p))


@pytest.fixture(scope="session")
def module_opt_125m_dynamic():
    from compiler.serialize import load_artifact
    p = _require_compiled_model("./compiled/opt_125m_dynamic")
    return load_artifact(str(p))


@pytest.fixture(scope="session")
def module_qwen():
    from compiler.serialize import load_artifact
    p = _require_compiled_model("./compiled/qwen3_0.8b")
    return load_artifact(str(p))


# ── HF model fixtures ──────────────────────────────────────────


def _require_hf_cache(model_name: str, hub_path: str) -> str:
    hub_dir = os.path.expanduser(hub_path)
    snapshots = os.path.join(hub_dir, "snapshots")
    if not os.path.isdir(snapshots):
        pytest.skip(
            f"{model_name} not found in HF cache ({hub_dir}).  "
            "Download it first via huggingface-cli or a test run with network access."
        )
    snap = os.listdir(snapshots)[0]
    return os.path.join(snapshots, snap)


@pytest.fixture(scope="session")
def hf_tiny_llama():
    import safetensors.torch
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    patch_transformers_torch()

    model_dir = _require_hf_cache(
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "~/.cache/huggingface/hub/models--hf-internal-testing--tiny-random-LlamaForCausalLM",
    )
    model_path = os.path.join(model_dir, "model.safetensors")
    config_path = os.path.join(model_dir, "config.json")

    config = LlamaConfig.from_pretrained(config_path)
    model = LlamaForCausalLM(config)
    state_dict = safetensors.torch.load_file(model_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


@pytest.fixture(scope="session")
def hf_opt_125m():
    import torch
    from transformers.models.opt.configuration_opt import OPTConfig
    from transformers.models.opt.modeling_opt import OPTForCausalLM

    patch_transformers_torch()

    model_dir = _require_hf_cache(
        "facebook/opt-125m",
        "~/.cache/huggingface/hub/models--facebook--opt-125m",
    )
    model_path = os.path.join(model_dir, "pytorch_model.bin")

    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    config_path = os.path.join(model_dir, "config.json")
    config = OPTConfig.from_pretrained(config_path)
    model = OPTForCausalLM(config)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


@pytest.fixture(scope="session")
def hf_qwen():
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM  # type: ignore[import-untyped]

    patch_transformers_torch()

    model_dir = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen", "Qwen3.5-0.8B")
    model_dir = os.path.abspath(model_dir)
    if not os.path.isdir(model_dir):
        pytest.skip(
            f"Qwen model not found at {model_dir}.  "
            "Clone it from HuggingFace into models/Qwen/Qwen3.5-0.8B."
        )
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    if hasattr(config, "text_config") and hasattr(config.text_config, "use_cache"):
        config.text_config.use_cache = False
    elif hasattr(config, "use_cache"):
        config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    model.eval()
    return model
