"""HF logits cosine similarity — Phase 1b gate #1.

Verifies the compiled model produces logits within cosine_similarity > 0.999
of the reference HuggingFace model, demonstrating compilation correctness.
"""

from __future__ import annotations

import os

import pytest
import torch


def _patch_transformers_torch():
    """Patch transformers to recognize the symlinked torch installation."""
    import transformers.utils.generic as _generic
    import transformers.utils.import_utils as _iu

    _iu._torch_available = True
    _iu._torch_version = torch.__version__

    _generic._torch_pytree = torch.utils._pytree

    def _flatten(output):
        return list(output.values()), list(output.keys())

    def _unflatten(values, context, output_type=None):
        return (output_type or type(context[0]))(**dict(zip(context, values, strict=False)))

    _generic._model_output_flatten = _flatten
    _generic._model_output_unflatten = _unflatten


def _load_hf_tiny_llama():
    """Load hf-internal-testing/tiny-random-LlamaForCausalLM from local cache."""
    _patch_transformers_torch()
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    hub_dir = os.path.expanduser(
        "~/.cache/huggingface/hub/models--hf-internal-testing--tiny-random-LlamaForCausalLM"
    )
    snapshots = os.path.join(hub_dir, "snapshots")
    snap = os.listdir(snapshots)[0]
    model_path = os.path.join(snapshots, snap, "model.safetensors")
    config_path = os.path.join(snapshots, snap, "config.json")

    config = LlamaConfig.from_pretrained(config_path)
    model = LlamaForCausalLM(config)
    import safetensors.torch

    state_dict = safetensors.torch.load_file(model_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def _load_hf_opt_125m():
    """Load facebook/opt-125m from local cache."""
    _patch_transformers_torch()
    from transformers.models.opt.configuration_opt import OPTConfig
    from transformers.models.opt.modeling_opt import OPTForCausalLM

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


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two flattened tensors."""
    a_flat = a.reshape(-1).float()
    b_flat = b.reshape(-1).float()
    return float(
        torch.nn.functional.cosine_similarity(a_flat, b_flat, dim=0).item()
    )


@pytest.mark.integration
class TestHFCosineTinyLlama:
    """Cosine similarity test with compiled tiny_llama."""

    @pytest.mark.timeout(120)
    def test_cosine_similarity_exceeds_threshold(self):
        """Compiled tiny_llama logits must match HF within cos > 0.999."""
        from compiler.serialize import load_artifact
        from engine.executor import Executor
        from hal.pytorch_backend import PyTorchBackend

        hf_model = _load_hf_tiny_llama()
        compiled_module = load_artifact("./compiled/tiny_llama")
        backend = PyTorchBackend("cpu")
        executor = Executor(compiled_module, backend)

        input_ids = torch.randint(0, 1000, (1, 4), dtype=torch.long)

        with torch.no_grad():
            hf_output = hf_model(input_ids)
            hf_logits = hf_output.logits

        compiled_logits = executor.forward(input_ids)

        assert hf_logits.shape == compiled_logits.shape, (
            f"Shape mismatch: HF {hf_logits.shape} vs compiled {compiled_logits.shape}"
        )

        similarity = _cosine_similarity(hf_logits, compiled_logits)
        print(f"\n  tiny_llama cosine similarity: {similarity:.8f}")
        assert similarity > 0.999, (
            f"Cosine similarity {similarity:.8f} below threshold 0.999"
        )

    @pytest.mark.timeout(120)
    def test_cosine_similarity_decode_shape(self):
        """Compiled tiny_llama decode shape [1,1] should also match HF."""
        from compiler.serialize import load_artifact
        from engine.executor import Executor
        from hal.pytorch_backend import PyTorchBackend

        hf_model = _load_hf_tiny_llama()
        compiled_module = load_artifact("./compiled/tiny_llama")
        backend = PyTorchBackend("cpu")
        executor = Executor(compiled_module, backend)

        input_ids = torch.randint(0, 1000, (1, 1), dtype=torch.long)

        with torch.no_grad():
            hf_output = hf_model(input_ids)
            hf_logits = hf_output.logits

        compiled_logits = executor.forward(input_ids)

        assert hf_logits.shape == compiled_logits.shape

        similarity = _cosine_similarity(hf_logits, compiled_logits)
        print(f"\n  tiny_llama decode cos: {similarity:.8f}")
        assert similarity > 0.999


@pytest.mark.integration
class TestHFCosineOpt125M:
    """Cosine similarity test with compiled opt_125m."""

    @pytest.mark.timeout(300)
    def test_cosine_similarity_exceeds_threshold(self):
        """Compiled opt_125m logits must match HF within cos > 0.999."""
        from compiler.serialize import load_artifact
        from engine.executor import Executor
        from hal.pytorch_backend import PyTorchBackend

        hf_model = _load_hf_opt_125m()
        compiled_module = load_artifact("./compiled/opt_125m")
        backend = PyTorchBackend("cpu")
        executor = Executor(compiled_module, backend)

        input_ids = torch.randint(0, 1000, (1, 4), dtype=torch.long)

        with torch.no_grad():
            hf_output = hf_model(input_ids)
            hf_logits = hf_output.logits

        compiled_logits = executor.forward(input_ids)

        assert hf_logits.shape == compiled_logits.shape, (
            f"Shape mismatch: HF {hf_logits.shape} vs compiled {compiled_logits.shape}"
        )

        similarity = _cosine_similarity(hf_logits, compiled_logits)
        print(f"\n  opt_125m cosine similarity: {similarity:.8f}")
        assert similarity > 0.999, (
            f"Cosine similarity {similarity:.8f} below threshold 0.999"
        )

    @pytest.mark.timeout(300)
    def test_cosine_similarity_decode_shape(self):
        """Compiled opt_125m decode shape [1,1] should also match HF."""
        from compiler.serialize import load_artifact
        from engine.executor import Executor
        from hal.pytorch_backend import PyTorchBackend

        hf_model = _load_hf_opt_125m()
        compiled_module = load_artifact("./compiled/opt_125m")
        backend = PyTorchBackend("cpu")
        executor = Executor(compiled_module, backend)

        input_ids = torch.randint(0, 1000, (1, 1), dtype=torch.long)

        with torch.no_grad():
            hf_output = hf_model(input_ids)
            hf_logits = hf_output.logits

        compiled_logits = executor.forward(input_ids)

        assert hf_logits.shape == compiled_logits.shape

        similarity = _cosine_similarity(hf_logits, compiled_logits)
        print(f"\n  opt_125m decode cos: {similarity:.8f}")
        assert similarity > 0.999
