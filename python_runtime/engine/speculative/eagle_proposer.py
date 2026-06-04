"""EAGLE draft proposer — generic hidden-state extrapolation.

EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency)
uses the target model's own intermediate hidden states to predict
future tokens through lightweight extrapolation layers.  Unlike MTP,
EAGLE does not require model pre-training with multi-token heads.

The proposer picks a middle Transformer layer (default: layer 2),
applies trainable extrapolation layers, and iteratively predicts
multiple future tokens.

Reference: design-phase2.md §2.2.2
"""

from __future__ import annotations

import torch
import torch.nn as nn

from python_runtime.engine.speculative.proposer import SpeculativeProposer


class EAGLEProposer(SpeculativeProposer):
    """EAGLE draft token proposer.

    Uses lightweight extrapolation layers (LayerNorm + Linear) from
    an intermediate transformer layer's hidden states to predict
    future tokens.

    Args:
        hidden_size: Main model hidden dimension.
        vocab_size: Vocabulary size.
        num_spec_tokens: Maximum number of speculative tokens to predict.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_spec_tokens: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self._vocab_size = vocab_size
        self.num_spec_tokens = num_spec_tokens

        self.eagle_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, vocab_size, bias=False),
            )
            for _ in range(num_spec_tokens)
        ])

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def propose(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        """Generate num_tokens draft tokens from hidden states.

        Uses the first layer's hidden state to predict draft tokens
        via extrapolation.

        Args:
            hidden_states: [batch, hidden_size] — from an intermediate layer.
            input_ids: [batch, 1] — current input token (used as fallback context).
            num_tokens: Number of draft tokens to predict.

        Returns:
            Draft token ids [batch, num_tokens].
        """
        num_tokens = min(num_tokens, self.num_spec_tokens)
        drafts: list[torch.Tensor] = []

        h = hidden_states
        for i in range(num_tokens):
            logits = self.eagle_layers[i](h)
            token = torch.argmax(logits, dim=-1)
            drafts.append(token)

        return torch.stack(drafts, dim=1)
