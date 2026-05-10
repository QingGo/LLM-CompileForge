"""Speculative decoding Verifier — rejection sampling.

Verifies draft tokens against the target model's logits using
rejection sampling, guaranteeing that the output distribution
matches the target model exactly.

The algorithm (per position i of k draft tokens):
  1. Get target model probability p and draft probability q for draft token.
  2. Draw random r ~ Uniform(0, 1).
  3. Accept if r < min(1, p / q).  Otherwise:
     - Reject from position i onwards.
     - Sample a new token from the adjusted target distribution.

Reference: design-phase2.md §2.2.3
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812


class SpeculativeVerifier:
    """Rejection-sampling verifier for speculative decoding.

    Verifies draft tokens against target model logits.  If a draft token
    is rejected, all subsequent draft tokens are discarded and a new token
    is sampled from the target distribution.

    Args:
        temperature: Sampling temperature for target model probabilities.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = temperature

    def verify(
        self,
        draft_tokens: torch.Tensor,
        target_logits: torch.Tensor,
        draft_probs: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], bool]:
        """Verify draft tokens against target model logits.

        Args:
            draft_tokens: [batch, k] — proposed draft token ids.
            target_logits: [batch, 1 + k, vocab_size] — target model logits
                           for the current + draft positions.
            draft_probs: [batch, k] — draft model probabilities for each token.
                         If None, assumes draft_probs = 1.0 (for greedy draft).

        Returns:
            (accepted_tokens, all_accepted):
              accepted_tokens — list of [batch] tensors, one per accepted position.
              all_accepted — True if all k draft tokens were accepted.
        """
        batch, k = draft_tokens.shape
        accepted: list[torch.Tensor] = []

        for i in range(k):
            draft_tok = draft_tokens[:, i]
            target_logit = target_logits[:, i, :]

            p_target = F.softmax(target_logit / self.temperature, dim=-1)

            target_probs_for_draft = p_target.gather(1, draft_tok.unsqueeze(1)).squeeze(1)

            if draft_probs is not None:
                q_draft = draft_probs[:, i]
            else:
                q_draft = torch.ones(batch, device=draft_tokens.device)

            rand = torch.rand(batch, device=draft_tokens.device)
            ratio = torch.minimum(
                torch.ones(batch, device=draft_tokens.device),
                target_probs_for_draft / (q_draft + 1e-12),
            )
            accept_mask = rand < ratio

            if not accept_mask.all():
                for b in range(batch):
                    if not accept_mask[b]:
                        adjusted_p = p_target[b].clone()
                        adjusted_p[draft_tok[b]] = 0.0
                        adjusted_p = adjusted_p / adjusted_p.sum()
                        new_tok = torch.multinomial(adjusted_p, 1).squeeze()
                        draft_tok[b] = new_tok
                accepted.append(draft_tok)
                return accepted, False

            accepted.append(draft_tok)

        return accepted, True

    def verify_greedy(
        self,
        draft_tokens: torch.Tensor,
        target_logits: torch.Tensor,
    ) -> tuple[list[torch.Tensor], bool]:
        """Greedy verification: accept draft tokens that match the target
        model's argmax.  Much faster than rejection sampling and always
        correct for greedy decoding.

        Args:
            draft_tokens: [batch, k] — proposed draft token ids.
            target_logits: [batch, 1 + k, vocab_size] — target model logits.

        Returns:
            (accepted_tokens, all_accepted).
        """
        batch, k = draft_tokens.shape
        accepted: list[torch.Tensor] = []

        for i in range(k):
            draft_tok = draft_tokens[:, i]
            target_tok = target_logits[:, i, :].argmax(dim=-1)

            match = draft_tok == target_tok
            if not match.all():
                for b in range(batch):
                    if not match[b]:
                        draft_tok[b] = target_tok[b]
                accepted.append(draft_tok)
                return accepted, False

            accepted.append(draft_tok)

        return accepted, True
