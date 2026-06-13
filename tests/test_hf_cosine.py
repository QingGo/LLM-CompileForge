"""HF logits cosine similarity — Phase 1b gate #1.

Verifies the compiled model produces logits within cosine_similarity > 0.999
of the reference HuggingFace model, demonstrating compilation correctness.
"""

from __future__ import annotations

import os

import pytest
import torch

from compiler.backend.compile_utils import _patch_transformers_torch
from tests.helpers import cosine_similarity


def _load_hf_tiny_llama():
    """Load hf-internal-testing/tiny-random-LlamaForCausalLM from local cache."""
    _patch_transformers_torch()
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    hub_dir = os.path.expanduser("~/.cache/huggingface/hub/models--hf-internal-testing--tiny-random-LlamaForCausalLM")
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


@pytest.mark.integration
@pytest.mark.baseline
class TestHFCosineTinyLlama:
    """Cosine similarity test with compiled tiny_llama."""

    @pytest.mark.timeout(120)
    def test_cosine_similarity_exceeds_threshold(self):
        """Compiled tiny_llama logits must match HF within cos > 0.999."""
        from compiler.serialize import load_artifact
        from python_runtime.engine.mlir_executor import MlirExecutor
        from python_runtime.hal.pytorch_backend import PyTorchBackend

        hf_model = _load_hf_tiny_llama()
        compiled_module = load_artifact("./outputs/compiled/tiny_llama")
        backend = PyTorchBackend("cpu")
        executor = MlirExecutor(compiled_module, backend)

        input_ids = torch.randint(0, 1000, (1, 4), dtype=torch.long)

        with torch.no_grad():
            hf_output = hf_model(input_ids)
            hf_logits = hf_output.logits

        compiled_logits = executor.forward(input_ids)

        assert hf_logits.shape == compiled_logits.shape, (
            f"Shape mismatch: HF {hf_logits.shape} vs compiled {compiled_logits.shape}"
        )

        similarity = cosine_similarity(hf_logits, compiled_logits)
        print(f"\n  tiny_llama cosine similarity: {similarity:.8f}")
        assert similarity > 0.999, f"Cosine similarity {similarity:.8f} below threshold 0.999"

    @pytest.mark.timeout(120)
    def test_cosine_similarity_decode_shape(self):
        """Compiled tiny_llama decode shape [1,1] should also match HF."""
        from compiler.serialize import load_artifact
        from python_runtime.engine.mlir_executor import MlirExecutor
        from python_runtime.hal.pytorch_backend import PyTorchBackend

        hf_model = _load_hf_tiny_llama()
        compiled_module = load_artifact("./outputs/compiled/tiny_llama")
        backend = PyTorchBackend("cpu")
        executor = MlirExecutor(compiled_module, backend)

        input_ids = torch.randint(0, 1000, (1, 1), dtype=torch.long)

        with torch.no_grad():
            hf_output = hf_model(input_ids)
            hf_logits = hf_output.logits

        compiled_logits = executor.forward(input_ids)

        assert hf_logits.shape == compiled_logits.shape

        similarity = cosine_similarity(hf_logits, compiled_logits)
        print(f"\n  tiny_llama decode cos: {similarity:.8f}")
        assert similarity > 0.999


@pytest.mark.integration
@pytest.mark.baseline
class TestHFCosineOpt125M:
    """Cosine similarity test with compiled opt_125m."""

    @pytest.mark.timeout(300)
    def test_cosine_similarity_exceeds_threshold(self):
        """Compiled opt_125m logits must match HF within cos > 0.999."""
        from compiler.serialize import load_artifact
        from python_runtime.engine.mlir_executor import MlirExecutor
        from python_runtime.hal.pytorch_backend import PyTorchBackend

        hf_model = _load_hf_opt_125m()
        compiled_module = load_artifact("./outputs/compiled/opt_125m")
        backend = PyTorchBackend("cpu")
        executor = MlirExecutor(compiled_module, backend)

        input_ids = torch.randint(0, 1000, (1, 4), dtype=torch.long)

        with torch.no_grad():
            hf_output = hf_model(input_ids)
            hf_logits = hf_output.logits

        compiled_logits = executor.forward(input_ids)

        assert hf_logits.shape == compiled_logits.shape, (
            f"Shape mismatch: HF {hf_logits.shape} vs compiled {compiled_logits.shape}"
        )

        similarity = cosine_similarity(hf_logits, compiled_logits)
        print(f"\n  opt_125m cosine similarity: {similarity:.8f}")
        assert similarity > 0.999, f"Cosine similarity {similarity:.8f} below threshold 0.999"

    @pytest.mark.timeout(300)
    def test_cosine_similarity_decode_shape(self):
        """Compiled opt_125m decode shape [1,1] should also match HF."""
        from compiler.serialize import load_artifact
        from python_runtime.engine.mlir_executor import MlirExecutor
        from python_runtime.hal.pytorch_backend import PyTorchBackend

        hf_model = _load_hf_opt_125m()
        compiled_module = load_artifact("./outputs/compiled/opt_125m")
        backend = PyTorchBackend("cpu")
        executor = MlirExecutor(compiled_module, backend)

        input_ids = torch.randint(0, 1000, (1, 1), dtype=torch.long)

        with torch.no_grad():
            hf_output = hf_model(input_ids)
            hf_logits = hf_output.logits

        compiled_logits = executor.forward(input_ids)

        assert hf_logits.shape == compiled_logits.shape

        similarity = cosine_similarity(hf_logits, compiled_logits)
        print(f"\n  opt_125m decode cos: {similarity:.8f}")
        assert similarity > 0.999


@pytest.mark.integration
@pytest.mark.baseline
class TestHFCosineOpt125MDynamic:
    """Cosine similarity test with dynamically-compiled opt_125m (batch+seq)."""

    @pytest.mark.timeout(300)
    def test_cosine_similarity_exceeds_threshold(self):
        """Compiled opt_125m_dynamic logits must match HF within cos > 0.999."""
        from compiler.serialize import load_artifact
        from python_runtime.engine.mlir_executor import MlirExecutor
        from python_runtime.hal.pytorch_backend import PyTorchBackend

        hf_model = _load_hf_opt_125m()
        compiled_module = load_artifact("./outputs/compiled/opt_125m_dynamic")
        backend = PyTorchBackend("cpu")
        executor = MlirExecutor(compiled_module, backend)

        input_ids = torch.randint(0, 1000, (1, 4), dtype=torch.long)

        with torch.no_grad():
            hf_output = hf_model(input_ids)
            hf_logits = hf_output.logits

        compiled_logits = executor.forward(input_ids)

        assert hf_logits.shape == compiled_logits.shape, (
            f"Shape mismatch: HF {hf_logits.shape} vs compiled {compiled_logits.shape}"
        )

        similarity = cosine_similarity(hf_logits, compiled_logits)
        print(f"\n  opt_125m_dynamic cosine similarity: {similarity:.8f}")
        assert similarity > 0.999, f"Cosine similarity {similarity:.8f} below threshold 0.999"

    @pytest.mark.timeout(300)
    def test_cosine_similarity_batch2(self):
        """Compiled opt_125m_dynamic batch=2 — should not crash and produce valid output."""
        from compiler.serialize import load_artifact
        from python_runtime.engine.mlir_executor import MlirExecutor
        from python_runtime.hal.pytorch_backend import PyTorchBackend

        compiled_module = load_artifact("./outputs/compiled/opt_125m_dynamic")
        backend = PyTorchBackend("cpu")
        executor = MlirExecutor(compiled_module, backend)

        input_ids = torch.randint(0, 1000, (2, 4), dtype=torch.long)
        compiled_logits = executor.forward(input_ids)

        assert compiled_logits.shape == (2, 4, 50272), f"Expected (2,4,50272), got {compiled_logits.shape}"
        assert not torch.allclose(compiled_logits[0], compiled_logits[1], atol=1e-6), (
            "Different batch elements should produce different logits"
        )
        assert not torch.allclose(compiled_logits[0], compiled_logits[1], atol=1e-6), (
            "Different batch elements should produce different logits"
        )
        assert not torch.isnan(compiled_logits).any(), "No NaN in output"
        assert not torch.isinf(compiled_logits).any(), "No Inf in output"


# --- Qwen3.5-0.8B ---
# (hf_qwen and module_qwen fixtures are in conftest.py)


@pytest.mark.integration
@pytest.mark.baseline
class TestHFCosineQwen:
    """Cosine similarity test with compiled Qwen3.5-0.8B."""

    @pytest.mark.timeout(300)
    def test_cosine_similarity_exceeds_threshold(self, hf_qwen, module_qwen):
        """Compiled Qwen3.5-0.8B logits must match HF within cos > 0.999."""
        from python_runtime.engine.mlir_executor import MlirExecutor
        from python_runtime.hal.pytorch_backend import PyTorchBackend

        hf_model = hf_qwen
        executor = MlirExecutor(module_qwen, PyTorchBackend("cpu"))

        input_ids = torch.randint(0, 1000, (1, 64), dtype=torch.long)
        with torch.no_grad():
            hf_logits = hf_model(input_ids).logits
        compiled_logits = executor.forward(input_ids)

        assert hf_logits.shape == compiled_logits.shape
        similarity = cosine_similarity(hf_logits, compiled_logits)
        print(f"\n  qwen3_0.8b cosine similarity: {similarity:.8f}")
        assert similarity > 0.999

    @pytest.mark.timeout(300)
    def test_cosine_similarity_decode_shape(self, hf_qwen, module_qwen):
        """Compiled Qwen3.5-0.8B seq=64 — static shape limitation."""
        from python_runtime.engine.mlir_executor import MlirExecutor
        from python_runtime.hal.pytorch_backend import PyTorchBackend

        executor = MlirExecutor(module_qwen, PyTorchBackend("cpu"))
        input_ids = torch.randint(0, 1000, (1, 64), dtype=torch.long)
        with torch.no_grad():
            hf_logits = hf_qwen(input_ids).logits
        compiled_logits = executor.forward(input_ids)

        assert hf_logits.shape == compiled_logits.shape
        similarity = cosine_similarity(hf_logits, compiled_logits)
        assert similarity > 0.999


@pytest.mark.integration
@pytest.mark.baseline
class TestHFCosineLlama1B:
    """Cosine similarity test with compiled Llama-3.2-1B."""

    @pytest.mark.timeout(600)
    def test_cosine_similarity_exceeds_threshold(self) -> None:
        _patch_transformers_torch()
        from transformers import AutoConfig, AutoModelForCausalLM

        from compiler.serialize import load_artifact
        from python_runtime.engine.mlir_executor import MlirExecutor
        from python_runtime.hal.pytorch_backend import PyTorchBackend

        model_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "models", "LLM-Research", "Llama-3.2-1B")
        )
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=False)
        config.use_cache = False
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            config=config,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        hf_model.eval()

        compiled = load_artifact("./outputs/compiled/llama_1b")
        executor = MlirExecutor(compiled, PyTorchBackend("cpu"))

        input_ids = torch.randint(0, 5000, (1, 4), dtype=torch.long)
        with torch.no_grad():
            hf_logits = hf_model(input_ids).logits.to(torch.float32)
        compiled_logits = executor.forward(input_ids).to(torch.float32)

        assert hf_logits.shape == compiled_logits.shape
        similarity = cosine_similarity(hf_logits, compiled_logits)
        print(f"\n  llama_1b cosine similarity: {similarity:.8f}")
        assert similarity > 0.999


@pytest.mark.integration
@pytest.mark.baseline
class TestHFCosineLlama3B:
    """Cosine similarity test with compiled Llama-3.2-3B."""

    @pytest.mark.timeout(900)
    def test_cosine_similarity_exceeds_threshold(self) -> None:
        _patch_transformers_torch()
        from transformers import AutoConfig, AutoModelForCausalLM

        from compiler.serialize import load_artifact
        from python_runtime.engine.mlir_executor import MlirExecutor
        from python_runtime.hal.pytorch_backend import PyTorchBackend

        model_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "models", "LLM-Research", "Llama-3.2-3B")
        )
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=False)
        config.use_cache = False
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            config=config,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        hf_model.eval()

        compiled = load_artifact("./outputs/compiled/llama_3b")
        executor = MlirExecutor(compiled, PyTorchBackend("cpu"))

        input_ids = torch.randint(0, 5000, (1, 4), dtype=torch.long)
        with torch.no_grad():
            hf_logits = hf_model(input_ids).logits.to(torch.float32)
        compiled_logits = executor.forward(input_ids).to(torch.float32)

        assert hf_logits.shape == compiled_logits.shape
        similarity = cosine_similarity(hf_logits, compiled_logits)
        print(f"\n  llama_3b cosine similarity: {similarity:.8f}")
        assert similarity > 0.999
