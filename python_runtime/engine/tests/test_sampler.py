import pytest
import torch

from python_runtime.engine.sampler import greedy, sample

# ═══════════════════════════════════════════════════════════
# Greedy
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
class TestGreedy:
    def test_single(self):
        logits = torch.tensor([0.1, 0.5, 0.4])
        token = greedy(logits)
        assert token.item() == 1

    def test_batch(self):
        logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        tokens = greedy(logits)
        assert torch.equal(tokens, torch.tensor([1, 0]))


# ═══════════════════════════════════════════════════════════
# Sample
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
class TestSample:
    def test_temperature_zero_is_greedy(self):
        logits = torch.tensor([[0.1, 5.0, 2.0]])
        token = sample(logits, temperature=0.0)
        assert token.item() == 1

    def test_temperature_scaling(self):
        logits = torch.tensor([[1.0, 100.0, 1.0]])  # Very peaked
        # With low temperature, highest logit is even more dominant
        # Multiple runs should all return argmax
        all_tokens = [sample(logits, temperature=0.1).item() for _ in range(20)]
        # Almost always returns 1
        assert all(t == 1 for t in all_tokens)

    def test_top_k(self):
        logits = torch.tensor([[1.0, 5.0, 3.0, 4.0, 2.0]])
        # top_k=3 should only sample from top 3 (indices 1, 3, 2)
        tokens = [sample(logits, temperature=1.0, top_k=3).item() for _ in range(50)]
        assert all(t in {1, 2, 3} for t in tokens)
        assert 0 not in tokens  # index 0 (value 1.0) is below top 3
        assert 4 not in tokens  # index 4 (value 2.0) is below top 3

    def test_top_p(self):
        logits = torch.tensor([[0.1, 10.0, 0.1, 0.1, 0.1]])
        # With top_p=0.5, only the most likely token should survive
        tokens = [sample(logits, temperature=1.0, top_p=0.5).item() for _ in range(30)]
        assert all(t == 1 for t in tokens)

    def test_1d_input(self):
        logits = torch.tensor([0.1, 0.8, 0.1])
        token = sample(logits, temperature=0.0)
        assert token.dim() == 0
        assert token.item() == 1

    def test_batch_sampling(self):
        logits = torch.tensor([[1.0, 10.0], [10.0, 1.0]])
        tokens = sample(logits, temperature=0.0)
        assert tokens.dim() == 1
        assert tokens[0].item() == 1
        assert tokens[1].item() == 0
