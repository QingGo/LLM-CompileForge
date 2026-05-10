"""Abstract speculative proposer interface.

Defines the contract for draft token generation: given the current
model state, propose k future tokens for verification.

Reference: design-phase2.md §2.2
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class SpeculativeProposer(ABC):
    """Abstract base for draft token proposers.

    Concrete implementations: MTPProposer (DeepSeek-style multi-token
    prediction), EAGLEProposer (generic hidden-state extrapolation).
    """

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Vocabulary size used by this proposer."""
        ...

    @abstractmethod
    def propose(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        """Propose k draft tokens given the current model state.

        Args:
            hidden_states: Model hidden states at current position
                           [batch, hidden_size] or [batch, seq_len, hidden_size].
            input_ids: Current input token ids [batch, 1].
            num_tokens: Number of draft tokens to generate.

        Returns:
            Draft token ids [batch, num_tokens].
        """
        ...
