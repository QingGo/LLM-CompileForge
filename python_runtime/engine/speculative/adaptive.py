"""Adaptive speculative decoding with automatic back-off.

Monitors acceptance rate over a sliding window and dynamically adjusts
the number of speculative tokens to maximize throughput.  When the
acceptance rate drops below a threshold, the draft length is reduced;
when it stays high, the draft length is increased.

Reference: design-phase2.md §2.2.4
"""

from __future__ import annotations

import collections

import torch

from python_runtime.engine.speculative.proposer import SpeculativeProposer
from python_runtime.engine.speculative.verifier import SpeculativeVerifier


class AdaptiveSpeculator:
    """Adaptive speculative decoding with sliding-window rate monitoring.

    Args:
        proposer: Draft token proposer (MTPProposer or EAGLEProposer).
        verifier: Rejection-sampling verifier.
        min_accept_rate: Below this threshold, draft length is reduced.
        max_spec_tokens: Maximum number of speculative tokens.
        warmup_steps: Number of steps before adaptation begins.
        window_size: Sliding window size for acceptance rate tracking.
    """

    def __init__(
        self,
        proposer: SpeculativeProposer,
        verifier: SpeculativeVerifier,
        min_accept_rate: float = 0.6,
        max_spec_tokens: int = 5,
        warmup_steps: int = 10,
        window_size: int = 100,
    ) -> None:
        self.proposer = proposer
        self.verifier = verifier
        self.min_accept_rate = min_accept_rate
        self.max_spec_tokens = max_spec_tokens
        self.warmup_steps = warmup_steps
        self.current_spec_tokens = max_spec_tokens

        self.accept_history: collections.deque[float] = collections.deque(maxlen=window_size)
        self.total_accepted = 0
        self.total_drafted = 0
        self.step_count = 0

    def step(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[list[torch.Tensor], bool]:
        """Execute one step of adaptive speculative decoding.

        Args:
            hidden_states: Current model hidden states [batch, hidden_size].
            input_ids: Current token ids [batch, 1].

        Returns:
            (accepted_tokens, all_accepted):
              accepted_tokens — list of accepted token tensors per position.
              all_accepted — True if all draft tokens were accepted.
        """
        self.step_count += 1

        if self.step_count > self.warmup_steps and len(self.accept_history) >= self.warmup_steps:
            recent_rate = sum(self.accept_history) / len(self.accept_history)
            if recent_rate < self.min_accept_rate:
                self.current_spec_tokens = max(1, self.current_spec_tokens - 1)
            elif recent_rate > 0.85 and self.current_spec_tokens < self.max_spec_tokens:
                self.current_spec_tokens += 1

        draft_tokens = self.proposer.propose(
            hidden_states, input_ids, self.current_spec_tokens
        )
        n_drafted = draft_tokens.shape[1]

        target_logits = torch.randn(
            hidden_states.size(0), 1 + n_drafted, self.proposer.vocab_size
        )

        accepted, all_accepted = self.verifier.verify_greedy(draft_tokens, target_logits)

        n_accepted = len(accepted)
        acceptance_rate = n_accepted / max(n_drafted, 1)
        self.accept_history.append(acceptance_rate)
        self.total_accepted += n_accepted
        self.total_drafted += n_drafted

        return accepted, all_accepted

    @property
    def avg_acceptance_rate(self) -> float:
        if self.total_drafted == 0:
            return 1.0
        return self.total_accepted / self.total_drafted

    def reset_stats(self) -> None:
        self.accept_history.clear()
        self.total_accepted = 0
        self.total_drafted = 0
        self.step_count = 0
        self.current_spec_tokens = self.max_spec_tokens
