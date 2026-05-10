"""Tests for speculative decoding (Phase 2 Sprint 2).

Verifies MTP/EAGLE draft proposers and rejection-sampling verifier.
All tests use synthetic models and deterministic seeds for
reproducible results.

Reference: design-phase2.md §5.2 — Token序列完全一致 (SpecDec vs 纯自回归)
"""

from __future__ import annotations

import pytest
import torch

# ── MTP Proposer ──────────────────────────────────────────


@pytest.mark.unit
class TestMTPProposer:
    def test_propose_returns_correct_shape(self) -> None:
        from engine.speculative.mtp_proposer import MTPProposer

        torch.manual_seed(42)
        batch, hidden, vocab = 2, 128, 1000
        proposer = MTPProposer(hidden, vocab, num_mtp_layers=4)

        h = torch.randn(batch, hidden)
        input_ids = torch.randint(0, vocab, (batch, 1))
        drafts = proposer.propose(h, input_ids, num_tokens=3)

        assert drafts.shape == (batch, 3)
        assert drafts.dtype == torch.int64

    def test_propose_capped_to_num_layers(self) -> None:
        from engine.speculative.mtp_proposer import MTPProposer

        torch.manual_seed(1)
        proposer = MTPProposer(64, 500, num_mtp_layers=2)
        h = torch.randn(3, 64)
        drafts = proposer.propose(h, torch.randint(0, 500, (3, 1)), num_tokens=10)
        assert drafts.shape == (3, 2)

    def test_propose_deterministic(self) -> None:
        from engine.speculative.mtp_proposer import MTPProposer

        torch.manual_seed(99)
        proposer = MTPProposer(64, 200, num_mtp_layers=3)
        h = torch.randn(1, 64)

        torch.manual_seed(99)
        d1 = proposer.propose(h, torch.tensor([[0]]), num_tokens=3)

        torch.manual_seed(99)
        d2 = proposer.propose(h, torch.tensor([[0]]), num_tokens=3)

        assert torch.equal(d1, d2)


# ── EAGLE Proposer ────────────────────────────────────────


@pytest.mark.unit
class TestEAGLEProposer:
    def test_propose_returns_correct_shape(self) -> None:
        from engine.speculative.eagle_proposer import EAGLEProposer

        torch.manual_seed(7)
        batch, hidden, vocab = 3, 128, 500
        proposer = EAGLEProposer(hidden, vocab, num_spec_tokens=4)

        h = torch.randn(batch, hidden)
        drafts = proposer.propose(h, torch.tensor([[0]]), num_tokens=3)

        assert drafts.shape == (batch, 3)
        assert drafts.dtype == torch.int64

    def test_propose_single_token(self) -> None:
        from engine.speculative.eagle_proposer import EAGLEProposer

        torch.manual_seed(3)
        proposer = EAGLEProposer(64, 256, num_spec_tokens=3)
        h = torch.randn(1, 64)
        drafts = proposer.propose(h, torch.tensor([[5]]), num_tokens=1)
        assert drafts.shape == (1, 1)


# ── Verifier (Greedy) ─────────────────────────────────────


@pytest.mark.unit
class TestSpeculativeVerifierGreedy:
    def test_all_accepted_when_match(self) -> None:
        from engine.speculative.verifier import SpeculativeVerifier

        verifier = SpeculativeVerifier()
        draft = torch.tensor([[5, 10, 15]])
        # Logits: shape [1, 1+k, vocab] where k=3
        logits = torch.zeros(1, 4, 20)
        logits[0, 0, 5] = 100.0   # Correct token 0
        logits[0, 1, 10] = 100.0  # Draft matches target
        logits[0, 2, 15] = 100.0  # Draft matches target

        accepted, all_ok = verifier.verify_greedy(draft, logits)
        assert all_ok
        assert len(accepted) == 3

    def test_rejects_on_mismatch(self) -> None:
        from engine.speculative.verifier import SpeculativeVerifier

        verifier = SpeculativeVerifier()
        draft = torch.tensor([[2, 8]])
        logits = torch.zeros(1, 3, 20)
        logits[0, 0, 5] = 100.0   # Target says 5, draft says 2 → mismatch
        logits[0, 1, 8] = 100.0   # Draft 8 matches target 8
        logits[0, 2, 8] = 100.0

        accepted, all_ok = verifier.verify_greedy(draft, logits)
        assert not all_ok
        # On mismatch at pos 0, draft is replaced with target token
        assert accepted[0][0].item() == 5

    def test_verify_greedy_matches_manual(self) -> None:
        from engine.speculative.verifier import SpeculativeVerifier

        torch.manual_seed(42)
        verifier = SpeculativeVerifier()
        draft = torch.tensor([[3, 7, 2, 9]])

        logits = torch.randn(1, 5, 50)
        accepted, all_ok = verifier.verify_greedy(draft, logits)

        # Verify accepted tokens match target argmax or are replaced
        for i, tok in enumerate(accepted):
            target_argmax = logits[0, i].argmax().item()
            if tok[0].item() != target_argmax:
                pytest.fail(f"Position {i}: draft {tok[0].item()} ≠ target {target_argmax}")


# ── Verifier (Rejection Sampling) ─────────────────────────


@pytest.mark.unit
class TestSpeculativeVerifierRejection:
    def test_deterministic_verification(self) -> None:
        from engine.speculative.verifier import SpeculativeVerifier

        verifier = SpeculativeVerifier(temperature=1.0)

        draft = torch.tensor([[10, 20]])
        logits = torch.zeros(1, 3, 100)
        logits[0, 0, 10] = 10.0
        logits[0, 1, 20] = 10.0
        logits[0, 2, 20] = 10.0

        accepted, ok = verifier.verify(draft, logits)
        assert len(accepted) >= 1
        for tok in accepted:
            assert tok.shape == (1,)


# ── Integration ───────────────────────────────────────────


@pytest.mark.unit
class TestSpeculativeIntegration:
    def test_mtp_with_verifier_end_to_end(self) -> None:
        from engine.speculative.mtp_proposer import MTPProposer
        from engine.speculative.verifier import SpeculativeVerifier

        torch.manual_seed(0)
        hidden, vocab = 64, 256
        proposer = MTPProposer(hidden, vocab, num_mtp_layers=3)
        verifier = SpeculativeVerifier()

        h = torch.randn(2, hidden)
        draft = proposer.propose(h, torch.tensor([[0], [0]]), num_tokens=3)

        logits = torch.randn(2, 4, vocab)
        accepted, all_ok = verifier.verify_greedy(draft, logits)

        assert len(accepted) >= 1
        for tok in accepted:
            assert tok.shape == (2,)
