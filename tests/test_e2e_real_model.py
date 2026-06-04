"""End-to-end test with real compiled models (dynamic seq).

Verifies the full pipeline: compiled model → Executor → LLMEngine → generate().
"""

from __future__ import annotations

import pytest
import torch


@pytest.mark.integration
class TestE2ERealModel:
    """E2E test with compiled tiny_llama (fast, 2-layer random model)."""

    @pytest.mark.timeout(120)
    def test_compiled_forward_varying_seq(self, module_tiny_llama):
        """Compiled model should accept different seq lengths without error."""
        from python_runtime.engine.mlir_executor import MlirExecutor
        from python_runtime.hal.pytorch_backend import PyTorchBackend

        backend = PyTorchBackend("cpu")
        executor = MlirExecutor(module_tiny_llama, backend)

        for seq in [1, 4, 8]:
            x = torch.randint(0, 100, (1, seq))
            logits = executor.forward(x)
            assert logits.numel() > 0
            assert logits.dtype == torch.float32

    @pytest.mark.timeout(120)
    def test_engine_generate_runs_without_error(self, module_tiny_llama):
        """LLMEngine.generate() should complete without crash."""
        from python_runtime.engine.llm_engine import LLMEngine
        from python_runtime.hal.pytorch_backend import PyTorchBackend

        backend = PyTorchBackend("cpu")
        engine = LLMEngine(module_tiny_llama, backend, max_batch_size=4, chunk_size=8)

        # Simple tokenizer for testing
        class _SimpleTokenizer:
            def encode(self, text):
                return [ord(c) % 100 for c in text]
            def decode(self, tokens):
                return " ".join(str(t) for t in tokens)

        engine.set_tokenizer(_SimpleTokenizer())

        # Generate with a short prompt
        result = engine.generate("hello world", max_tokens=2, temperature=0.0)
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.timeout(120)
    def test_engine_step_returns_results(self, module_tiny_llama):
        """step() should produce GenerationResult entries."""
        from python_runtime.engine.llm_engine import LLMEngine
        from python_runtime.hal.pytorch_backend import PyTorchBackend

        backend = PyTorchBackend("cpu")
        engine = LLMEngine(module_tiny_llama, backend, max_batch_size=4, chunk_size=8)

        class _SimpleTokenizer:
            def encode(self, text):
                return [ord(c) % 100 for c in text]
            def decode(self, tokens):
                return " ".join(str(t) for t in tokens)

        engine.set_tokenizer(_SimpleTokenizer())
        engine.add_request("test", max_tokens=3, temperature=0.0)

        results = engine.step()
        # Should produce at least one result (prefill token sampling)
        assert len(results) == 1
        assert results[0].new_tokens  # non-empty
        assert not results[0].is_finished  # still generating


@pytest.mark.integration
class TestE2EOpt125M:
    """E2E test with compiled opt-125m (12-layer, larger model)."""

    @pytest.mark.timeout(300)
    def test_compiled_forward_varying_seq(self, module_opt_125m):
        """opt-125m should accept different seq lengths."""
        from python_runtime.engine.mlir_executor import MlirExecutor
        from python_runtime.hal.pytorch_backend import PyTorchBackend

        backend = PyTorchBackend("cpu")
        executor = MlirExecutor(module_opt_125m, backend)

        for seq in [1, 4]:
            x = torch.randint(0, 1000, (1, seq))
            logits = executor.forward(x)
            assert logits.numel() > 0
            assert logits.dtype == torch.float32

    @pytest.mark.timeout(300)
    def test_engine_step_runs_without_error(self, module_opt_125m):
        """LLMEngine.step() should work with compiled opt-125m."""
        from python_runtime.engine.llm_engine import LLMEngine
        from python_runtime.hal.pytorch_backend import PyTorchBackend

        backend = PyTorchBackend("cpu")
        engine = LLMEngine(module_opt_125m, backend, max_batch_size=2, chunk_size=4)

        class _SimpleTokenizer:
            def encode(self, text):
                return [ord(c) % 5000 for c in text]
            def decode(self, tokens):
                return " ".join(str(t) for t in tokens)

        engine.set_tokenizer(_SimpleTokenizer())
        engine.add_request("hi", max_tokens=2, temperature=0.0)

        results = engine.step()
        assert len(results) == 1
        assert results[0].new_tokens
