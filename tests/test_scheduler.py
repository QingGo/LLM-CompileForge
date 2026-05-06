import pytest

from engine.batch import SamplingParams
from engine.block_manager import BlockManager
from engine.scheduler import Scheduler

# ═══════════════════════════════════════════════════════════
# Scheduler
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
class TestSchedulerCreation:
    def test_defaults(self):
        s = Scheduler()
        assert s.max_batch_size == 32
        assert s.chunk_size == 256

    def test_custom_params(self):
        s = Scheduler(max_batch_size=8, chunk_size=64)
        assert s.max_batch_size == 8

    def test_invalid_params_raises(self):
        with pytest.raises(ValueError):
            Scheduler(max_batch_size=0)
        with pytest.raises(ValueError):
            Scheduler(chunk_size=0)


@pytest.mark.unit
class TestAddRequest:
    def test_add_single_request(self):
        s = Scheduler(max_batch_size=8)
        rid = s.add_request(prompt_tokens=[1, 2, 3])
        assert rid.startswith("req_")
        assert s.waiting_count == 1

    def test_add_with_custom_id(self):
        s = Scheduler()
        rid = s.add_request(prompt_tokens=[1], request_id="my_id")
        assert rid == "my_id"

    def test_add_with_sampling_params(self):
        s = Scheduler()
        sp = SamplingParams(temperature=0.7, max_tokens=50)
        s.add_request(prompt_tokens=[1, 2], sampling_params=sp)
        assert s.waiting_count == 1


@pytest.mark.unit
class TestSchedule:
    def test_empty_schedule(self):
        s = Scheduler()
        bm = BlockManager(num_blocks=100)
        batch = s.schedule(bm)
        assert batch.is_empty

    def test_single_request_prefill(self):
        s = Scheduler(max_batch_size=8, chunk_size=64)
        bm = BlockManager(num_blocks=100)
        s.add_request(prompt_tokens=list(range(50)))
        assert s.waiting_count == 1

        batch = s.schedule(bm)
        assert not batch.is_empty
        assert batch.size == 1
        assert batch.input_ids is not None

    def test_request_transitions_to_decode(self):
        s = Scheduler(max_batch_size=8, chunk_size=256)
        bm = BlockManager(num_blocks=100)

        # Add a short prompt
        s.add_request(prompt_tokens=[1, 2, 3])

        # First schedule — prefill
        batch = s.schedule(bm)
        assert not batch.is_empty
        assert batch.requests[0].state == "decode"  # All tokens prefill'd in one step

    def test_chunked_prefill_splits_long_prompt(self):
        s = Scheduler(max_batch_size=8, chunk_size=2, max_tokens_per_step=10)
        bm = BlockManager(num_blocks=100)

        s.add_request(prompt_tokens=[1, 2, 3, 4, 5])  # 5 tokens, chunk=2

        # First step: prefill 2 tokens
        batch1 = s.schedule(bm)
        assert batch1.requests[0].state == "prefill"

        # Second step: prefill next 2 tokens
        batch2 = s.schedule(bm)
        assert batch2.requests[0].state == "prefill"

        # Third step: prefill last 1 token → transition to decode
        batch3 = s.schedule(bm)
        assert batch3.requests[0].state == "decode"

    def test_priority_queue_order(self):
        s = Scheduler(max_batch_size=2)
        bm = BlockManager(num_blocks=100)

        s.add_request(prompt_tokens=[1], priority=5)  # Lower priority (higher value)
        s.add_request(prompt_tokens=[2], priority=1)  # Higher priority (lower value)

        batch = s.schedule(bm)
        # Both admitted since max_batch=2, but priority=1 should be first
        assert len(batch.requests) == 2

    def test_has_work(self):
        s = Scheduler()
        assert not s.has_work
        s.add_request(prompt_tokens=[1])
        assert s.has_work

    def test_finished_request_reaped(self):
        s = Scheduler(max_batch_size=8)
        bm = BlockManager(num_blocks=100)

        s.add_request(prompt_tokens=[1])
        batch = s.schedule(bm)
        req = batch.requests[0]
        req.mark_finished()
        assert s.running_count == 1

        # Next schedule should reap the finished request
        s.schedule(bm)
        assert s.running_count == 0
