"""Token sampling module.

Implements greedy, temperature, top-p (nucleus), and top-k sampling
strategies for logits-to-token conversion.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812


def sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
) -> torch.Tensor:
    """Sample token IDs from logits.

    Supporting table:
      - temperature = 0: greedy (argmax).
      - temperature > 0: scale logits, then optionally filter with top_p / top_k.

    Args:
        logits: shape [batch_size, vocab_size] or [vocab_size].
        temperature: softmax temperature. 0.0 = greedy.
        top_p: nucleus sampling cumulative threshold. 1.0 = off.
        top_k: top-k filtering. 0 = off.

    Returns:
        Sampled token IDs, shape [] or [batch_size].
    """
    original_shape = logits.shape
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)  # [1, vocab_size]

    if temperature == 0.0:
        tokens = logits.argmax(dim=-1)
    else:
        logits = logits / temperature

        if top_k > 0:
            _apply_top_k(logits, top_k)

        if top_p < 1.0:
            _apply_top_p(logits, top_p)

        probs = F.softmax(logits, dim=-1)
        tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

    if len(original_shape) == 1:
        return tokens.squeeze(0)
    return tokens


def greedy(logits: torch.Tensor) -> torch.Tensor:
    """Greedy sampling: argmax."""
    return logits.argmax(dim=-1)


def _apply_top_k(logits: torch.Tensor, k: int) -> None:
    """Zero out logits below the top-k threshold (in-place)."""
    if k <= 0:
        return
    topk_vals, _ = torch.topk(logits, k, dim=-1)
    threshold = topk_vals[..., -1, None]
    logits[logits < threshold] = float("-inf")


def _apply_top_p(logits: torch.Tensor, p: float) -> None:
    """Apply nucleus (top-p) filtering (in-place)."""
    if p >= 1.0:
        return
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    # Only set -inf for indices where the mask is True
    for b in range(logits.size(0)):
        indices_to_remove = sorted_indices[b][sorted_indices_to_remove[b]]
        logits[b, indices_to_remove] = float("-inf")
