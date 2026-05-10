"""MTP (Multi-Token Prediction) draft proposer.

MTP is DeepSeek V3/V4's native multi-token prediction mechanism.  Each
Transformer layer has dedicated MTP modules (extra projection layers
and LayerNorm) that predict future tokens from the current hidden states.
MTP parameters account for only ~1-2% of the main model.

Reference: design-phase2.md §2.2.1
"""

from __future__ import annotations

import torch
import torch.nn as nn

from engine.speculative.proposer import SpeculativeProposer


class MTPProposer(SpeculativeProposer):
    """MTP draft token proposer.

    Uses lightweight projection layers on top of the main model's
    hidden states to predict multiple future tokens.  The model must
    have MTP heads pre-trained (DeepSeek V3/V4).

    For testing without a real MTP-trained model, this can operate
    in 'mock' mode using random projection layers.

    Args:
        hidden_size: Main model hidden dimension.
        vocab_size: Vocabulary size.
        num_mtp_layers: Number of MTP prediction heads (one per draft token).
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_mtp_layers: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self._vocab_size = vocab_size
        self.num_mtp_layers = num_mtp_layers

        self.mtp_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_mtp_layers)
        ])
        self.mtp_projections = nn.ModuleList([
            nn.Linear(hidden_size, vocab_size, bias=False)
            for _ in range(num_mtp_layers)
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

        Args:
            hidden_states: [batch, hidden_size] — last position hidden state.
            input_ids: [batch, 1] — current token (unused in MTP; draft tokens
                       are predicted purely from hidden states).
            num_tokens: Number of draft tokens to predict (capped to num_mtp_layers).

        Returns:
            Draft token ids [batch, num_tokens].
        """
        num_tokens = min(num_tokens, self.num_mtp_layers)
        drafts: list[torch.Tensor] = []

        h = hidden_states
        for i in range(num_tokens):
            h_norm = self.mtp_norms[i](h)
            logits = self.mtp_projections[i](h_norm)
            token = torch.argmax(logits, dim=-1)
            drafts.append(token)

        return torch.stack(drafts, dim=1)
