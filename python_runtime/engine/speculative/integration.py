"""Speculative decoding integration with LLMEngine.

Provides a SpeculativeEngine wrapper that extends LLMEngine's step()
with draft proposal and verification, automatically falling back to
standard decoding when speculation is not beneficial.

Reference: design-phase2.md §2.2, §3.1
"""

from __future__ import annotations

import torch

from python_runtime.engine.speculative.proposer import SpeculativeProposer
from python_runtime.engine.speculative.verifier import SpeculativeVerifier


class SpeculativeRunner:
    """Stateless speculative decoding runner.

    Wrap the proposal/verification cycle for one step of speculative
    decoding.  Suitable for integration into LLMEngine.step() without
    modifying the engine's core logic.

    Args:
        proposer: Draft token proposer.
        verifier: Rejection-sampling verifier.
        enabled: Whether speculation is active (can be toggled at runtime).
    """

    def __init__(
        self,
        proposer: SpeculativeProposer,
        verifier: SpeculativeVerifier | None = None,
        enabled: bool = True,
    ) -> None:
        self.proposer = proposer
        self.verifier = verifier or SpeculativeVerifier()
        self.enabled = enabled

    def run_speculative_step(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        target_logits: torch.Tensor,
        num_spec_tokens: int = 3,
    ) -> tuple[list[torch.Tensor], bool, int]:
        """Run one speculative decoding step.

        Args:
            hidden_states: Model hidden states [batch, hidden_size].
            input_ids: Current input token ids [batch, 1].
            target_logits: Target model logits for all positions
                          [batch, 1 + num_spec_tokens, vocab_size].
            num_spec_tokens: Number of draft tokens to generate.

        Returns:
            (accepted_tokens, all_accepted, num_generated):
              accepted_tokens — list of accepted tokens.
              all_accepted — True if all draft tokens accepted.
              num_generated — total number of tokens generated (including bonus).
        """
        if not self.enabled:
            # Fallback: just return the target model's first token
            first_tok = target_logits[:, 0, :].argmax(dim=-1).unsqueeze(1)
            return [first_tok], True, 1

        draft_tokens = self.proposer.propose(
            hidden_states, input_ids, num_spec_tokens
        )
        accepted, all_accepted = self.verifier.verify_greedy(
            draft_tokens, target_logits
        )
        num_generated = len(accepted)
        return accepted, all_accepted, num_generated

    def get_first_token(
        self, target_logits: torch.Tensor
    ) -> torch.Tensor:
        """Extract the first (non-speculative) token from target logits.

        Args:
            target_logits: [batch, 1 + k, vocab_size].

        Returns:
            First token [batch].
        """
        return target_logits[:, 0, :].argmax(dim=-1)
