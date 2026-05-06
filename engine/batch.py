"""Batch-level data structures for the inference engine."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

# ── Sampling Parameters ─────────────────────────────────────


@dataclass
class SamplingParams:
    """Parameters controlling token generation.

    Attributes:
        temperature: Softmax temperature (0.0 = greedy). Range [0, ∞).
        top_p: Nucleus sampling cumulative probability threshold. 1.0 = off.
        top_k: Top-k sampling. 0 = off.
        max_tokens: Maximum number of tokens to generate.
        stop_token_ids: Token IDs that signal generation should stop.
    """

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    max_tokens: int = 256
    stop_token_ids: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if not (0.0 <= self.top_p <= 1.0):
            raise ValueError(f"top_p must be in [0, 1], got {self.top_p}")


# ── Request ─────────────────────────────────────────────────


@dataclass
class Request:
    """Inference request state tracked by the scheduler.

    Lifecycle: waiting → prefill → decode → finished
    """

    request_id: str
    prompt_tokens: list[int]
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    output_tokens: list[int] = field(default_factory=list)
    state: str = "waiting"
    priority: int = 0
    arrival_time: float = 0.0
    prefill_pos: int = 0  # Number of prompt tokens already processed

    @property
    def is_finished(self) -> bool:
        return self.state == "finished"

    @property
    def num_processed_tokens(self) -> int:
        """Number of prompt tokens already processed (for Chunked Prefill)."""
        return self.prefill_pos

    @property
    def tokens_remaining(self) -> int:
        """Number of prompt tokens still to prefill."""
        return max(0, len(self.prompt_tokens) - self.prefill_pos)

    def mark_finished(self) -> None:
        self.state = "finished"

    def append_token(self, token_id: int) -> None:
        self.output_tokens.append(token_id)


# ── Sequence Group (for batch construction) ─────────────────


@dataclass
class SequenceGroup:
    """A group of requests batched together for one forward pass.

    Maps each request to its position in the flattened tensors.
    """

    requests: list[Request] = field(default_factory=list)
    # Flattened tensors across all requests in the batch
    input_ids: torch.Tensor | None = None
    positions: torch.Tensor | None = None
    # request_id → list of physical block IDs (PagedAttention KV cache)
    block_tables: dict[str, list[int]] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.requests)

    @property
    def is_empty(self) -> bool:
        return self.size == 0

    @property
    def total_tokens(self) -> int:
        if self.input_ids is None:
            return 0
        return int(torch.numel(self.input_ids))


# ── Generation Result ───────────────────────────────────────


@dataclass
class GenerationResult:
    """Result of one scheduling step for a single request."""

    request_id: str
    new_tokens: list[int]
    is_finished: bool
    text: str = ""  # Decoded text (set by upper layers)
