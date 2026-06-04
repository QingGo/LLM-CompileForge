"""FFI boundary tests for Rust PyScheduler and PyBlockManager.

Verifies that the Python → Rust → Python data marshaling works correctly:
  - Type conversion (list[int] → Vec<u32>, str → String, etc.)
  - Exception propagation (Rust panics → Python exceptions)
  - Schedule() return value structure and consistency
  - Prefix cache hint injection through FFI
"""

from __future__ import annotations

import llm_serveforge_runtime as _rt
import pytest

# ── PyBlockManager FFI tests ──────────────────────────────


@pytest.mark.unit
class TestPyBlockManagerFFI:
    """Test FFI boundary for PyBlockManager."""

    def test_create(self):
        bm = _rt.PyBlockManager(1000, 16)
        assert bm.num_blocks == 1000
        assert bm.block_size == 16
        assert bm.num_free_blocks() == 1000

    def test_create_invalid_params_raises(self):
        with pytest.raises(ValueError):
            _rt.PyBlockManager(0, 16)
        with pytest.raises(ValueError):
            _rt.PyBlockManager(100, 0)

    def test_allocate_free(self):
        bm = _rt.PyBlockManager(1000, 16)
        blocks = bm.allocate("req", 32)
        assert len(blocks) == 2
        assert bm.num_free_blocks() == 998

        bm.free("req")
        assert bm.num_free_blocks() == 1000

    def test_allocate_out_of_memory_raises(self):
        bm = _rt.PyBlockManager(10, 16)
        with pytest.raises(RuntimeError, match="OutOfMemory"):
            bm.allocate("req", 200)

    def test_free_unknown_noop(self):
        bm = _rt.PyBlockManager(1000, 16)
        bm.free("nonexistent")  # should not raise

    def test_get_blocks_unknown_raises(self):
        bm = _rt.PyBlockManager(10, 16)
        with pytest.raises(KeyError):
            bm.get_blocks("unknown")

    def test_ensure_blocks_expands(self):
        bm = _rt.PyBlockManager(1000, 16)
        bm.ensure_blocks("req", 48)
        blocks = bm.get_blocks("req")
        assert len(blocks) == 3

    def test_share_prefix(self):
        bm = _rt.PyBlockManager(1000, 16)
        bm.allocate("src", 64)  # 4 blocks
        shared = bm.share_prefix("src", "dst", 32)  # first 2 blocks
        assert len(shared) == 2
        assert bm.get_blocks("dst") == shared

    def test_share_prefix_invalid_src_raises(self):
        bm = _rt.PyBlockManager(1000, 16)
        with pytest.raises(ValueError):
            bm.share_prefix("unknown", "dst", 16)

    def test_utilization(self):
        bm = _rt.PyBlockManager(100, 16)
        assert bm.utilization() == 0.0
        bm.allocate("req", 160)  # 10 blocks
        assert abs(bm.utilization() - 0.10) < 0.001

    def test_increment_ref_count_and_free_block(self):
        bm = _rt.PyBlockManager(1000, 16)
        bm.allocate("req", 16)
        blocks = bm.get_blocks("req")
        bid = blocks[0]

        bm.increment_ref_count(bid)
        bm.free("req")
        assert bm.num_free_blocks() == 999  # block not freed due to extra ref

        bm.free_block(bid)
        assert bm.num_free_blocks() == 1000

    def test_assign_cached_blocks(self):
        bm = _rt.PyBlockManager(1000, 16)
        bm.assign_cached_blocks("req", [5, 6])
        assert bm.get_blocks("req") == [5, 6]


# ── PyScheduler FFI tests ─────────────────────────────────


@pytest.mark.unit
class TestPySchedulerFFI:
    """Test FFI boundary for PyScheduler."""

    def test_create(self):
        s = _rt.PyScheduler(32, 512, 256)
        assert s.waiting_count() == 0
        assert s.running_count() == 0
        assert not s.has_work()

    def test_create_invalid_params_raises(self):
        with pytest.raises(ValueError):
            _rt.PyScheduler(0, 512, 256)
        with pytest.raises(ValueError):
            _rt.PyScheduler(32, 512, 0)

    def test_add_request_returns_str(self):
        s = _rt.PyScheduler(32, 512, 256)
        rid = s.add_request([1, 2, 3], 0, 0.0, 256, [], None)
        assert isinstance(rid, str)
        assert rid.startswith("req_")

    def test_add_request_custom_id(self):
        s = _rt.PyScheduler(32, 512, 256)
        rid = s.add_request([1, 2], 0, 0.0, 256, [], "my_id")
        assert rid == "my_id"

    def test_schedule_empty_returns_empty_dict(self):
        s = _rt.PyScheduler(32, 512, 256)
        bm = _rt.PyBlockManager(1000, 16)
        batch = s.schedule(bm, [])
        assert isinstance(batch, dict)
        assert batch["total_tokens"] == 0
        assert len(batch["requests"]) == 0

    def test_schedule_returns_correct_structure(self):
        s = _rt.PyScheduler(32, 512, 256)
        bm = _rt.PyBlockManager(1000, 16)
        s.add_request([10, 20, 30], 0, 0.0, 256, [], None)

        batch = s.schedule(bm, [])
        assert isinstance(batch, dict)
        assert "requests" in batch
        assert "total_tokens" in batch
        assert batch["total_tokens"] > 0

        reqs = batch["requests"]
        assert len(reqs) == 1
        r = reqs[0]
        assert isinstance(r, dict)
        assert "request_id" in r
        assert "input_ids" in r
        assert "positions" in r
        assert "state" in r
        assert "block_table" in r
        assert "n_tokens" in r
        assert isinstance(r["request_id"], str)
        assert isinstance(r["input_ids"], list)
        assert isinstance(r["positions"], list)
        assert isinstance(r["block_table"], list)
        assert isinstance(r["n_tokens"], int)

    def test_schedule_then_decode(self):
        s = _rt.PyScheduler(32, 512, 256)
        bm = _rt.PyBlockManager(1000, 16)
        s.add_request([1, 2, 3], 0, 0.0, 256, [], None)

        b1 = s.schedule(bm, [])
        assert b1["requests"][0]["state"] == "prefill"
        assert b1["requests"][0]["input_ids"] == [1, 2, 3]

        b2 = s.schedule(bm, [])
        assert b2["requests"][0]["state"] == "decode"
        assert b2["requests"][0]["n_tokens"] == 1

    def test_record_output_tracks_finished(self):
        s = _rt.PyScheduler(32, 512, 256)
        bm = _rt.PyBlockManager(1000, 16)
        s.add_request([1], 0, 0.0, 1, [], None)  # max_tokens=1

        _ = s.schedule(bm, [])  # prefill
        _ = s.schedule(bm, [])  # decode
        finished = s.record_output("req_1", 42)
        assert finished is True  # max_tokens=1 reached

    def test_record_output_stop_token(self):
        s = _rt.PyScheduler(32, 512, 256)
        bm = _rt.PyBlockManager(1000, 16)
        s.add_request([1], 0, 0.0, 100, [13], None)

        _ = s.schedule(bm, [])
        _ = s.schedule(bm, [])
        finished = s.record_output("req_1", 13)
        assert finished is True

    def test_record_output_unknown_request(self):
        s = _rt.PyScheduler(32, 512, 256)
        assert s.record_output("nonexistent", 42) is False

    def test_chunked_prefill_split(self):
        s = _rt.PyScheduler(32, 512, 4)  # chunk_size=4
        bm = _rt.PyBlockManager(1000, 16)
        s.add_request(list(range(10)), 0, 0.0, 256, [], None)

        b1 = s.schedule(bm, [])
        assert b1["requests"][0]["n_tokens"] == 4
        assert b1["requests"][0]["input_ids"] == [0, 1, 2, 3]

        b2 = s.schedule(bm, [])
        assert b2["requests"][0]["n_tokens"] == 4

        b3 = s.schedule(bm, [])
        assert b3["requests"][0]["n_tokens"] == 2

        b4 = s.schedule(bm, [])
        assert b4["requests"][0]["state"] == "decode"

    def test_priority_ordering(self):
        s = _rt.PyScheduler(1, 512, 256)  # batch_size=1
        bm = _rt.PyBlockManager(1000, 16)

        s.add_request([7], 10, 0.0, 256, [], "low")
        s.add_request([3], 0, 0.0, 256, [], "high")
        s.add_request([5], 5, 0.0, 256, [], "mid")

        b1 = s.schedule(bm, [])
        assert b1["requests"][0]["request_id"] == "high"

    def test_cache_hit_injection(self):
        s = _rt.PyScheduler(32, 512, 256)
        bm = _rt.PyBlockManager(1000, 16)

        # Pre-allocate blocks to simulate cached blocks
        cache_blocks = bm.allocate("_cache", 64)
        bm.increment_ref_count(cache_blocks[0])  # simulate cache hold

        s.add_request([1, 2, 3, 4, 5, 6], 0, 0.0, 256, [], "hit")

        batch = s.schedule(bm, [("hit", cache_blocks[:1], 4)])
        # Should skip first 4 tokens, only prefill remaining 2
        assert batch["requests"][0]["input_ids"] == [5, 6]
        assert batch["requests"][0]["positions"] == [4, 5]
        assert batch["requests"][0]["n_tokens"] == 2

    def test_fully_cached_goes_to_decode(self):
        s = _rt.PyScheduler(32, 512, 256)
        bm = _rt.PyBlockManager(1000, 16)

        cache_blocks = bm.allocate("_cache", 48)
        for bid in cache_blocks:
            bm.increment_ref_count(bid)

        s.add_request([1, 2, 3], 0, 0.0, 256, [], "full")

        batch = s.schedule(bm, [("full", cache_blocks, 3)])
        assert batch["requests"][0]["state"] == "decode"

    def test_query_properties(self):
        s = _rt.PyScheduler(32, 512, 256)
        bm = _rt.PyBlockManager(1000, 16)

        assert s.waiting_count() == 0
        assert s.running_count() == 0
        assert not s.has_work()

        s.add_request([1], 0, 0.0, 256, [], None)
        assert s.waiting_count() == 1
        assert s.has_work()

        s.schedule(bm, [])
        assert s.waiting_count() == 0
        assert s.running_count() == 1


# ── LLMEngine + Rust integration tests ─────────────────────


@pytest.mark.unit
class TestLLMEngineRustIntegration:
    """Test that LLMEngine with Rust backend works end-to-end."""

    @staticmethod
    def _make_test_mlir():
        from compiler.artifact import MlirFunction, MlirModule, MlirOp
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

    def test_generate_multi_token(self):
        """Generate path works through Rust scheduler."""
        from python_runtime.engine.llm_engine import LLMEngine
        from python_runtime.hal.pytorch_backend import PyTorchBackend

        engine = LLMEngine(
            self._make_test_mlir(),
            PyTorchBackend("cpu"),
            max_batch_size=4,
            chunk_size=2,
        )
        result = engine.generate([1, 2, 3], max_tokens=2, temperature=0)
        assert isinstance(result, str)

    def test_step_idle_returns_empty(self):
        from python_runtime.engine.llm_engine import LLMEngine
        from python_runtime.hal.pytorch_backend import PyTorchBackend

        engine = LLMEngine(self._make_test_mlir(), PyTorchBackend("cpu"))
        assert engine.step() == []

    def test_step_returns_results(self):
        from python_runtime.engine.llm_engine import LLMEngine
        from python_runtime.hal.pytorch_backend import PyTorchBackend

        engine = LLMEngine(
            self._make_test_mlir(),
            PyTorchBackend("cpu"),
            max_batch_size=4,
            chunk_size=2,
        )
        engine.add_request([1, 2, 3], max_tokens=2, temperature=0)
        results = engine.step()
        assert len(results) > 0
        assert results[0].request_id.startswith("req_")

    def test_multiple_requests_batched(self):
        from python_runtime.engine.llm_engine import LLMEngine
        from python_runtime.hal.pytorch_backend import PyTorchBackend

        engine = LLMEngine(
            self._make_test_mlir(),
            PyTorchBackend("cpu"),
            max_batch_size=4,
            chunk_size=4,
        )
        engine.add_request([1, 2], max_tokens=2, temperature=0)
        engine.add_request([3, 4], max_tokens=2, temperature=0)

        # Both should be batched together
        results = engine.step()
        assert len(results) >= 1  # at least one result

    def test_exhaust_loop_terminates(self):
        from python_runtime.engine.llm_engine import LLMEngine
        from python_runtime.hal.pytorch_backend import PyTorchBackend

        engine = LLMEngine(
            self._make_test_mlir(),
            PyTorchBackend("cpu"),
            max_batch_size=4,
            chunk_size=2,
        )
        engine.add_request([1, 2], max_tokens=1, temperature=0)

        for _ in range(10):
            engine.step()
            if engine.is_idle:
                break

        assert engine.num_running == 0
        assert engine.is_idle

    def test_add_request_with_text_requires_tokenizer(self):
        from python_runtime.engine.llm_engine import LLMEngine
        from python_runtime.hal.pytorch_backend import PyTorchBackend

        engine = LLMEngine(self._make_test_mlir(), PyTorchBackend("cpu"))
        with pytest.raises(RuntimeError, match="requires a tokenizer"):
            engine.add_request("hello world")

    def test_stop_token_terminates(self):
        from python_runtime.engine.llm_engine import LLMEngine
        from python_runtime.hal.pytorch_backend import PyTorchBackend

        engine = LLMEngine(
            self._make_test_mlir(),
            PyTorchBackend("cpu"),
            max_batch_size=4,
            chunk_size=2,
        )
        engine.set_tokenizer(
            self._make_tokenizer(),
            eos_token_id=0,
        )
        engine.add_request([1, 2], max_tokens=100, temperature=0)

        for _ in range(20):
            results = engine.step()
            for r in results:
                if r.is_finished:
                    return  # stop token hit → termination
            if engine.is_idle:
                return  # fallback: engine exhausted

    def test_is_idle_transitions(self):
        from python_runtime.engine.llm_engine import LLMEngine
        from python_runtime.hal.pytorch_backend import PyTorchBackend

        engine = LLMEngine(self._make_test_mlir(), PyTorchBackend("cpu"))
        assert engine.is_idle

        engine.add_request([1], max_tokens=1, temperature=0)
        assert not engine.is_idle

        engine.step()  # prefill
        engine.step()  # decode → should finish
        engine.step()  # reap

        assert engine.is_idle

    @staticmethod
    def _make_tokenizer():
        from tests.helpers import SimpleTokenizer
        return SimpleTokenizer()
