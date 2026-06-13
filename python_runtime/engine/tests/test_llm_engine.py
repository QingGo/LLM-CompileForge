"""Tests for LLMEngine — top-level orchestration layer.

Covers constructor validation, executor auto-selection, generate()
with/without tokenizer, add_request() edge cases, and step() loop
termination.
"""

from __future__ import annotations

import pytest

from compiler.artifact import MlirFunction, MlirModule, MlirOp
from python_runtime.engine.llm_engine import LLMEngine
from python_runtime.engine.mlir_executor import MlirExecutor
from python_runtime.hal.pytorch_backend import PyTorchBackend


def _make_test_mlir() -> MlirModule:
    """MlirModule: %input → sf.identity → %logits."""
    return MlirModule(
        functions=[
            MlirFunction(
                name="main",
                inputs=[("%input_ids", "tensor<?xf32>")],
                outputs=[("%logits", "tensor<?xf32>", False)],
                ops=[
                    MlirOp(
                        name="sf.identity",
                        dialect="sf",
                        op_name="identity",
                        operands=["%input_ids"],
                        results=["%logits"],
                    ),
                ],
            )
        ]
    )


@pytest.mark.unit
class TestLLMEngineInit:
    """Constructor validation and executor selection."""

    def test_creates_mli_executor(self):
        mod = _make_test_mlir()
        engine = LLMEngine(mod, PyTorchBackend("cpu"))
        assert isinstance(engine.executor, MlirExecutor)

    def test_explicit_executor_override(self):
        mod = _make_test_mlir()
        backend = PyTorchBackend("cpu")
        mlir_exe = MlirExecutor(_make_test_mlir(), backend)
        engine = LLMEngine(mod, backend, executor=mlir_exe)
        assert engine.executor is mlir_exe

    def test_max_batch_size_zero_raises(self):
        with pytest.raises(ValueError, match="max_batch_size"):
            LLMEngine(_make_test_mlir(), PyTorchBackend("cpu"), max_batch_size=0)

    def test_defaults_idle(self):
        engine = LLMEngine(_make_test_mlir(), PyTorchBackend("cpu"))
        assert engine.is_idle
        assert engine.num_running == 0
        assert engine.num_waiting == 0


@pytest.mark.unit
class TestLLMEngineGenerate:
    """generate() method — text and token ID modes."""

    @staticmethod
    def _tokenizer():
        from tests.helpers import SimpleTokenizer

        return SimpleTokenizer()

    def test_generate_with_token_ids(self):
        engine = LLMEngine(_make_test_mlir(), PyTorchBackend("cpu"), max_batch_size=4, chunk_size=2)
        result = engine.generate([1, 2, 3], max_tokens=2, temperature=0)
        assert isinstance(result, str)

    def test_generate_requires_tokenizer_for_text(self):
        engine = LLMEngine(_make_test_mlir(), PyTorchBackend("cpu"))
        with pytest.raises(RuntimeError, match="requires a tokenizer"):
            engine.generate("hello world")

    def test_generate_with_tokenizer(self):
        engine = LLMEngine(_make_test_mlir(), PyTorchBackend("cpu"), max_batch_size=4, chunk_size=2)
        engine.set_tokenizer(self._tokenizer())
        result = engine.generate("hello world", max_tokens=2, temperature=0)
        assert isinstance(result, str)

    def test_generate_mlir_executor_path(self):
        engine = LLMEngine(_make_test_mlir(), PyTorchBackend("cpu"), max_batch_size=4, chunk_size=2)
        result = engine.generate([1, 2], max_tokens=2, temperature=0)
        assert isinstance(result, str)


@pytest.mark.unit
class TestLLMEngineAddRequest:
    """add_request() edge cases."""

    def test_add_text_requires_tokenizer(self):
        engine = LLMEngine(_make_test_mlir(), PyTorchBackend("cpu"))
        with pytest.raises(RuntimeError, match="requires a tokenizer"):
            engine.add_request("hello")

    def test_add_token_ids_returns_request_id(self):
        engine = LLMEngine(_make_test_mlir(), PyTorchBackend("cpu"))
        rid = engine.add_request([1, 2, 3])
        assert rid.startswith("req_")
        assert not engine.is_idle

    def test_add_with_sampling_params(self):
        engine = LLMEngine(_make_test_mlir(), PyTorchBackend("cpu"))
        rid = engine.add_request([1, 2], temperature=0.5, top_p=0.9, priority=1)
        assert rid.startswith("req_")


@pytest.mark.unit
class TestLLMEngineStep:
    """step() loop behaviour."""

    def test_step_idle_returns_empty(self):
        engine = LLMEngine(_make_test_mlir(), PyTorchBackend("cpu"))
        assert engine.step() == []

    def test_step_returns_generation_results(self):
        engine = LLMEngine(_make_test_mlir(), PyTorchBackend("cpu"), max_batch_size=4, chunk_size=2)
        engine.add_request([1, 2, 3], max_tokens=2, temperature=0)
        results = engine.step()
        assert len(results) > 0
        for r in results:
            assert r.request_id.startswith("req_")

    def test_step_exhausts_requests(self):
        engine = LLMEngine(_make_test_mlir(), PyTorchBackend("cpu"), max_batch_size=4, chunk_size=2)
        engine.add_request([1, 2], max_tokens=1, temperature=0)
        for _ in range(10):
            engine.step()
            if engine.is_idle:
                break
        assert engine.num_running == 0

    def test_multiple_add_has_work(self):
        engine = LLMEngine(_make_test_mlir(), PyTorchBackend("cpu"), max_batch_size=4)
        engine.add_request([1, 2, 3], max_tokens=5, temperature=0)
        engine.add_request([4, 5], max_tokens=3, temperature=0)
        assert engine.num_waiting == 2


@pytest.mark.unit
class TestLLMEngineTokenizer:
    """Tokenizer integration."""

    @staticmethod
    def _tokenizer():
        from tests.helpers import SimpleTokenizer

        return SimpleTokenizer()

    def test_set_tokenizer(self):
        engine = LLMEngine(_make_test_mlir(), PyTorchBackend("cpu"))
        engine.set_tokenizer(self._tokenizer(), eos_token_id=99)
        engine.add_request("test prompt")
        assert not engine.is_idle

    def test_eos_stops_generation(self):
        engine = LLMEngine(_make_test_mlir(), PyTorchBackend("cpu"), max_batch_size=4, chunk_size=2)
        engine.set_tokenizer(self._tokenizer(), eos_token_id=0)
        engine.add_request([1, 2], max_tokens=100, temperature=0)
        for _ in range(20):
            results = engine.step()
            for r in results:
                if r.is_finished:
                    return
            if engine.is_idle:
                return


@pytest.mark.unit
class TestLLMEngineCachePolicyIntegration:
    """P0-5: CachePolicy num_blocks injection from engine metadata."""

    def test_num_blocks_injected_into_metadata(self) -> None:
        """Creating engine with cache_policy should set num_blocks in metadata."""
        from compiler.cache_policy import CachePolicy

        mod = _make_test_mlir()
        mod.metadata["cache_policy"] = CachePolicy.for_llama(4, 8, 64).to_dict()
        assert "num_blocks" not in mod.metadata

        LLMEngine(mod, PyTorchBackend("cpu"), num_blocks=500, block_size=16)
        assert mod.metadata["num_blocks"] == 500

    def test_executor_uses_cache_manager(self) -> None:
        """Engine auto-creates executor with cache manager when policy present."""
        from compiler.cache_policy import CachePolicy

        mod = _make_test_mlir()
        mod.metadata["cache_policy"] = CachePolicy.for_llama(4, 8, 64).to_dict()
        assert "num_blocks" not in mod.metadata

        engine = LLMEngine(mod, PyTorchBackend("cpu"), num_blocks=32, block_size=16)
        assert engine.executor._uses_cache_manager
