"""Integration tests for speculative decoding (Phase 2 Sprint 3).

Verifies AdaptiveSpeculator dynamics and SpeculativeRunner integration
with LLMEngine-like patterns.
"""

from __future__ import annotations

import pytest
import torch

# ── Adaptive Speculator ───────────────────────────────────


@pytest.mark.unit
class TestAdaptiveSpeculator:
    def test_initial_draft_length(self) -> None:
        from python_runtime.engine.speculative.adaptive import AdaptiveSpeculator
        from python_runtime.engine.speculative.mtp_proposer import MTPProposer
        from python_runtime.engine.speculative.verifier import SpeculativeVerifier

        proposer = MTPProposer(64, 256, num_mtp_layers=5)
        verifier = SpeculativeVerifier()
        adaptive = AdaptiveSpeculator(proposer, verifier, max_spec_tokens=5)

        assert adaptive.current_spec_tokens == 5

    def test_avg_acceptance_rate_initially_one(self) -> None:
        from python_runtime.engine.speculative.adaptive import AdaptiveSpeculator
        from python_runtime.engine.speculative.mtp_proposer import MTPProposer
        from python_runtime.engine.speculative.verifier import SpeculativeVerifier

        proposer = MTPProposer(64, 256, num_mtp_layers=3)
        verifier = SpeculativeVerifier()
        adaptive = AdaptiveSpeculator(proposer, verifier)
        assert adaptive.avg_acceptance_rate == 1.0

    def test_step_produces_accepted_tokens(self) -> None:
        from python_runtime.engine.speculative.adaptive import AdaptiveSpeculator
        from python_runtime.engine.speculative.mtp_proposer import MTPProposer
        from python_runtime.engine.speculative.verifier import SpeculativeVerifier

        torch.manual_seed(0)
        proposer = MTPProposer(64, 256, num_mtp_layers=3)
        verifier = SpeculativeVerifier()
        adaptive = AdaptiveSpeculator(proposer, verifier, max_spec_tokens=3)

        h = torch.randn(2, 64)
        accepted, all_ok = adaptive.step(h, torch.zeros(2, 1).long())

        assert len(accepted) >= 1
        assert adaptive.step_count == 1

    def test_reset_stats(self) -> None:
        from python_runtime.engine.speculative.adaptive import AdaptiveSpeculator
        from python_runtime.engine.speculative.mtp_proposer import MTPProposer
        from python_runtime.engine.speculative.verifier import SpeculativeVerifier

        proposer = MTPProposer(64, 256, num_mtp_layers=2)
        adaptive = AdaptiveSpeculator(proposer, SpeculativeVerifier())
        adaptive.accept_history.append(0.5)
        adaptive.total_drafted = 10
        adaptive.step_count = 5

        adaptive.reset_stats()

        assert len(adaptive.accept_history) == 0
        assert adaptive.total_drafted == 0
        assert adaptive.step_count == 0
        assert adaptive.current_spec_tokens == adaptive.max_spec_tokens


# ── SpeculativeRunner ─────────────────────────────────────


@pytest.mark.unit
class TestSpeculativeRunner:
    def test_creates_and_runs(self) -> None:
        from python_runtime.engine.speculative.integration import SpeculativeRunner
        from python_runtime.engine.speculative.mtp_proposer import MTPProposer

        torch.manual_seed(1)
        proposer = MTPProposer(64, 100, num_mtp_layers=2)
        runner = SpeculativeRunner(proposer)

        h = torch.randn(2, 64)
        logits = torch.randn(2, 4, 100)
        accepted, all_ok, n_gen = runner.run_speculative_step(h, torch.zeros(2, 1).long(), logits, num_spec_tokens=2)

        assert n_gen >= 1
        assert len(accepted) == n_gen

    def test_disabled_falls_back_to_first_token(self) -> None:
        from python_runtime.engine.speculative.integration import SpeculativeRunner
        from python_runtime.engine.speculative.mtp_proposer import MTPProposer

        proposer = MTPProposer(32, 50, num_mtp_layers=2)
        runner = SpeculativeRunner(proposer, enabled=False)

        logits = torch.zeros(1, 3, 50)
        logits[0, 0, 7] = 100.0

        accepted, all_ok, n_gen = runner.run_speculative_step(
            torch.randn(1, 32), torch.tensor([[1]]), logits, num_spec_tokens=2
        )

        assert n_gen == 1
        assert accepted[0][0].item() == 7

    def test_get_first_token(self) -> None:
        from python_runtime.engine.speculative.integration import SpeculativeRunner
        from python_runtime.engine.speculative.mtp_proposer import MTPProposer

        proposer = MTPProposer(32, 50, num_mtp_layers=1)
        runner = SpeculativeRunner(proposer)

        logits = torch.zeros(2, 3, 50)
        logits[:, 0, 3] = 100.0
        tok = runner.get_first_token(logits)
        assert torch.equal(tok, torch.tensor([3, 3]))
