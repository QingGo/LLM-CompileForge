"""Shared pytest fixtures — eliminates repeated model/artifact loading."""

import os

import pytest


def pytest_runtest_setup(item):
    """Enforce 1s timeout on unit-marked tests at setup time."""
    if item.get_closest_marker("unit"):
        item.own_markers = list(item.own_markers)
        item.own_markers.append(pytest.mark.timeout(1))


# ── Compiled artifact fixtures ─────────────────────────────────


@pytest.fixture(scope="session")
def module_tiny_llama():
    """Load compiled tiny_llama IrModule once per test session."""
    from compiler.serialize import load_artifact
    return load_artifact("./compiled/tiny_llama")


@pytest.fixture(scope="session")
def module_opt_125m():
    """Load compiled opt_125m IrModule once per test session."""
    from compiler.serialize import load_artifact
    return load_artifact("./compiled/opt_125m")


@pytest.fixture(scope="session")
def module_opt_125m_dynamic():
    """Load compiled opt_125m_dynamic IrModule once per test session."""
    from compiler.serialize import load_artifact
    return load_artifact("./compiled/opt_125m_dynamic")


@pytest.fixture(scope="session")
def module_qwen():
    """Load compiled Qwen3.5-0.8B IrModule once per test session."""
    from compiler.serialize import load_artifact
    return load_artifact("./compiled/qwen3_0.8b")


# ── HF model fixtures (session-scoped, shared across tests) ────


def _patch_transformers_torch():
    """Patch transformers to recognize the symlinked torch installation."""
    from importlib import metadata as _meta

    import transformers

    for _pkg_name in list(_meta.packages_distributions().keys()):
        if _pkg_name == "torch":
            if not hasattr(transformers, "is_torch_available"):
                transformers.is_torch_available = lambda: True  # type: ignore
            if not hasattr(transformers.utils, "is_torch_available"):
                transformers.utils.is_torch_available = lambda: True  # type: ignore
    _has_flag = getattr(transformers, "_torch_available", None)
    if _has_flag is not None and callable(_has_flag):
        transformers._torch_available = True  # type: ignore
    if hasattr(transformers.utils, "_torch_available"):
        transformers.utils._torch_available = True  # type: ignore
    if hasattr(transformers, "is_torch_available"):
        if not transformers.is_torch_available():  # type: ignore
            transformers.is_torch_available = lambda: True  # type: ignore
    try:
        from torch._functorch import pytree  # type: ignore[import-untyped]
    except ImportError:
        pytree = None  # type: ignore[assignment]
    if pytree is not None:
        if not hasattr(transformers, "_torch_pytree"):
            transformers._torch_pytree = pytree  # type: ignore
        if hasattr(transformers, "modeling_utils"):
            if not getattr(transformers.modeling_utils, "_torch_pytree", None):
                transformers.modeling_utils._torch_pytree = pytree  # type: ignore
    if hasattr(transformers, "utils"):
        if hasattr(transformers.utils, "is_torch_fx_available"):
            transformers.utils.is_torch_fx_available = lambda: True  # type: ignore
        if hasattr(transformers.utils, "is_torch_export_available"):
            transformers.utils.is_torch_export_available = lambda: True  # type: ignore
    if hasattr(transformers, "modeling_utils"):
        if hasattr(transformers.modeling_utils, "_model_output_unflatten"):

            def _noop_flatten(x, context=None):  # type: ignore[no-untyped-def]
                return x

            transformers.modeling_utils._model_output_flatten = _noop_flatten  # type: ignore
            transformers.modeling_utils._model_output_unflatten = _noop_flatten  # type: ignore


@pytest.fixture(scope="session")
def hf_tiny_llama():
    """Load tiny_llama HF model once per test session."""
    import safetensors.torch
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    _patch_transformers_torch()

    hub_dir = os.path.expanduser(
        "~/.cache/huggingface/hub/models--hf-internal-testing--tiny-random-LlamaForCausalLM"
    )
    snapshots = os.path.join(hub_dir, "snapshots")
    snap = os.listdir(snapshots)[0]
    model_path = os.path.join(snapshots, snap, "model.safetensors")
    config_path = os.path.join(snapshots, snap, "config.json")

    config = LlamaConfig.from_pretrained(config_path)
    model = LlamaForCausalLM(config)
    state_dict = safetensors.torch.load_file(model_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


@pytest.fixture(scope="session")
def hf_opt_125m():
    """Load opt_125m HF model once per test session."""
    import torch
    from transformers.models.opt.configuration_opt import OPTConfig
    from transformers.models.opt.modeling_opt import OPTForCausalLM

    _patch_transformers_torch()

    hub_dir = os.path.expanduser("~/.cache/huggingface/hub/models--facebook--opt-125m")
    snapshots = os.path.join(hub_dir, "snapshots")
    snap = os.listdir(snapshots)[0]
    model_path = os.path.join(snapshots, snap, "pytorch_model.bin")

    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    config_path = os.path.join(snapshots, snap, "config.json")
    config = OPTConfig.from_pretrained(config_path)
    model = OPTForCausalLM(config)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


@pytest.fixture(scope="session")
def hf_qwen():
    """Load Qwen3.5-0.8B HF model once per test session."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM  # type: ignore[import-untyped]

    _patch_transformers_torch()

    model_dir = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen", "Qwen3.5-0.8B")
    model_dir = os.path.abspath(model_dir)
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
