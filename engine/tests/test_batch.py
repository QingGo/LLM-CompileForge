import pytest
import torch

from engine.batch import GenerationResult, Request, SamplingParams, SequenceGroup

# ═══════════════════════════════════════════════════════════
# SamplingParams
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
class TestSamplingParams:
    def test_defaults(self):
        sp = SamplingParams()
        assert sp.temperature == 1.0
        assert sp.top_p == 1.0
        assert sp.top_k == 0
        assert sp.max_tokens == 256

    def test_bad_temperature_raises(self):
        with pytest.raises(ValueError):
            SamplingParams(temperature=-1)

    def test_bad_top_p_raises(self):
        with pytest.raises(ValueError):
            SamplingParams(top_p=1.5)


# ═══════════════════════════════════════════════════════════
# Request
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
class TestRequest:
    def test_creation(self):
        req = Request(request_id="test", prompt_tokens=[1, 2, 3])
        assert req.request_id == "test"
        assert req.state == "waiting"

    def test_lifecycle(self):
        req = Request(request_id="r1", prompt_tokens=[1, 2], state="prefill")
        assert not req.is_finished
        req.mark_finished()
        assert req.is_finished

    def test_append_token(self):
        req = Request(request_id="r1", prompt_tokens=[1, 2])
        req.append_token(3)
        assert req.output_tokens == [3]
        req.append_token(4)
        assert req.output_tokens == [3, 4]

    def test_tokens_remaining(self):
        req = Request(request_id="r1", prompt_tokens=[1, 2, 3])
        assert req.tokens_remaining == 3

    def test_num_processed_tokens(self):
        req = Request(request_id="r1", prompt_tokens=[1, 2, 3], state="prefill")
        assert req.num_processed_tokens == 0


# ═══════════════════════════════════════════════════════════
# SequenceGroup
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
class TestSequenceGroup:
    def test_empty_group(self):
        sg = SequenceGroup()
        assert sg.is_empty
        assert sg.size == 0
        assert sg.total_tokens == 0

    def test_with_requests(self):
        r1 = Request(request_id="a", prompt_tokens=[1])
        r2 = Request(request_id="b", prompt_tokens=[2])
        sg = SequenceGroup(
            requests=[r1, r2],
            input_ids=torch.tensor([1, 2]),
        )
        assert sg.size == 2
        assert not sg.is_empty
        assert sg.total_tokens == 2

    def test_block_tables(self):
        sg = SequenceGroup(block_tables={"r1": [0, 1], "r2": [2]})
        assert sg.block_tables["r1"] == [0, 1]


# ═══════════════════════════════════════════════════════════
# GenerationResult
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
class TestGenerationResult:
    def test_creation(self):
        r = GenerationResult(request_id="x", new_tokens=[42], is_finished=False)
        assert r.request_id == "x"
        assert r.new_tokens == [42]
        assert not r.is_finished
