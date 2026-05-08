"""Inference scheduler: Continuous Batching + Chunked Prefill.

The scheduler is the runtime's central decision-maker. Each step:
  1. Removes finished requests from the running batch.
  2. Adds new requests from the waiting queue (priority-sorted min-heap).
  3. For prefill requests, applies Chunked Prefill to limit per-step tokens.
  4. Builds a Batch with input_ids, positions, and block_tables.

Key scheduling strategies:
   - FCFS (default): requests processed in arrival order.
   - Priority Queue: heapq-based, lower priority value = higher priority.
   - Chunked Prefill: long prompts split into chunk_size-token slices.
   - Hybrid: prefill and decode requests mixed in a single step.
   - Prefix Cache: RadixTree-based KV block reuse for shared prefixes.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from engine.batch import GenerationResult, Request, SamplingParams, SequenceGroup
from engine.block_manager import BlockManager

if TYPE_CHECKING:
    from cache.radix_cache import RadixCache


@dataclass(order=True)
class _QueueEntry:
    """Entry in the waiting priority queue. Lower priority_value = higher priority."""

    priority_value: int
    request: Request = field(compare=False)


class Scheduler:
    """Continuous Batching scheduler with Chunked Prefill support.

    Configuration:
        max_batch_size: Maximum number of concurrent requests in a batch.
        max_tokens_per_step: Total tokens processed per forward pass (prefill + decode).
        chunk_size: Maximum prefill tokens per request per step.

    Thread safety: the scheduler lock is held only during schedule().
    The caller must release it before calling forward() so new requests
    can be added to the waiting queue during GPU execution.
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        max_tokens_per_step: int = 512,
        chunk_size: int = 256,
        radix_cache: RadixCache | None = None,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive, got {max_batch_size}")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")

        self.max_batch_size = max_batch_size
        self.max_tokens_per_step = max_tokens_per_step
        self.chunk_size = chunk_size
        self.radix_cache = radix_cache

        # Priority queue: list of _QueueEntry, heapq-maintained
        self._waiting: list[_QueueEntry] = []
        self._running: list[Request] = []
        self._request_counter = 0

    # ── Public API ──────────────────────────────────────────

    def add_request(
        self,
        prompt_tokens: list[int],
        sampling_params: SamplingParams | None = None,
        priority: int = 0,
        request_id: str | None = None,
    ) -> str:
        """Add a request to the waiting queue.

        Args:
            prompt_tokens: Tokenized prompt.
            sampling_params: Generation parameters.
            priority: Lower value = higher priority (for heapq).
            request_id: Optional custom ID. Auto-generated if None.

        Returns:
            The request_id (useful for tracking).
        """
        if request_id is None:
            self._request_counter += 1
            request_id = f"req_{self._request_counter}"

        req = Request(
            request_id=request_id,
            prompt_tokens=list(prompt_tokens),
            sampling_params=sampling_params or SamplingParams(),
            priority=priority,
            arrival_time=time.monotonic(),
        )
        heapq.heappush(self._waiting, _QueueEntry(priority, req))
        return request_id

    def schedule(self, block_manager: BlockManager) -> SequenceGroup:
        """Core scheduling method — called once per forward step.

        Algorithm:
          1. Remove finished requests from running batch, free their KV blocks.
          2. Transition waiting → running until max_batch_size.
          3. For prefill requests, chunk tokens to respect max_tokens_per_step.
          4. Build flattened input tensors and position arrays.

        Args:
            block_manager: BlockManager for KV cache allocation queries.

        Returns:
            SequenceGroup describing the batch for this step.
        """
        # Step 1: reap finished requests
        finished = [r for r in self._running if r.is_finished]
        for r in finished:
            self._running.remove(r)
            # Insert KV blocks into prefix cache before freeing
            if self.radix_cache is not None:
                all_tokens = r.prompt_tokens + r.output_tokens
                try:
                    blks = block_manager.get_blocks(r.request_id)
                    self.radix_cache.insert(all_tokens, blks)
                except KeyError:
                    pass
            block_manager.free(r.request_id)

        # Step 2: admit new requests
        while self._waiting and len(self._running) < self.max_batch_size:
            entry = heapq.heappop(self._waiting)
            req = entry.request
            req.state = "prefill"

            # ── Prefix cache lookup ──
            if self.radix_cache is not None:
                matched_blocks, matched_tokens = self.radix_cache.match_prefix(
                    req.prompt_tokens
                )
                if matched_tokens > 0:
                    req.prefill_pos = matched_tokens
                    # Assign cached blocks immediately so both prefill and
                    # decode paths can find them via get_blocks().
                    block_manager.assign_cached_blocks(req.request_id, matched_blocks)
                    if matched_tokens >= len(req.prompt_tokens):
                        req.state = "decode"

            self._running.append(req)

        if not self._running:
            return SequenceGroup()

        # Step 3: build the batch with Chunked Prefill
        batch_requests: list[Request] = []
        input_ids_list: list[torch.Tensor] = []
        positions_list: list[torch.Tensor] = []
        block_tables: dict[str, list[int]] = {}
        total_prefill_tokens = 0
        num_decode = 0

        for req in self._running:
            if req.state == "prefill":
                # Chunked Prefill: limit tokens per step
                remaining = req.tokens_remaining
                if remaining <= 0:
                    req.state = "decode"
                    # Fall through to decode logic
                else:
                    # How many tokens can this request prefill this step?
                    prefill_budget = self.max_tokens_per_step - total_prefill_tokens
                    if prefill_budget <= 0:
                        # Out of budget — skip this request this step
                        continue

                    n_tokens = min(remaining, self.chunk_size, prefill_budget)
                    start_pos = req.prefill_pos
                    end_pos = start_pos + n_tokens

                    chunk_input_ids = req.prompt_tokens[start_pos:end_pos]
                    total_prefill_tokens += n_tokens

                    req.prefill_pos = end_pos

                    batch_requests.append(req)
                    input_ids_list.append(torch.tensor(chunk_input_ids, dtype=torch.long))
                    positions_list.append(torch.arange(start_pos, end_pos, dtype=torch.long))

                    # Allocate KV blocks on first prefill step, reuse thereafter
                    try:
                        blocks = block_manager.get_blocks(req.request_id)
                        # Blocks exist (from cache or prior allocation) — check
                        # whether they cover the full prompt.
                        covered = len(blocks) * block_manager.block_size
                        if covered < len(req.prompt_tokens):
                            extra = len(req.prompt_tokens) - covered
                            n_extra = (
                                (extra + block_manager.block_size - 1)
                                // block_manager.block_size
                            )
                            for _ in range(n_extra):
                                bid = block_manager.free_blocks.pop()
                                block_manager.blocks[bid].ref_count = 1
                                block_manager.block_tables[req.request_id].append(bid)
                    except KeyError:
                        # Check if prefix cache already assigned blocks
                        existing = block_manager.block_tables.get(req.request_id, [])
                        existing_tokens = len(existing) * block_manager.block_size
                        suffix_tokens = max(0, len(req.prompt_tokens) - existing_tokens)
                        if suffix_tokens > 0:
                            block_manager.allocate(req.request_id, suffix_tokens)
                        else:
                            block_manager.block_tables.setdefault(req.request_id, [])
                        blocks = block_manager.get_blocks(req.request_id)
                    block_tables[req.request_id] = blocks

                    # If all prompt tokens consumed this step, transition to decode
                    if end_pos >= len(req.prompt_tokens):
                        req.state = "decode"
                    continue

            if req.state == "decode":
                # Decode: one token per step
                if total_prefill_tokens + num_decode + 1 > self.max_tokens_per_step:
                    continue  # Budget exhausted

                num_decode += 1
                batch_requests.append(req)

                if req.output_tokens:
                    # Use the last generated token as input
                    last_token = req.output_tokens[-1]
                    pos = len(req.prompt_tokens) + len(req.output_tokens) - 1
                else:
                    # First decode step after prefill — position is last prefill position
                    last_token = req.prompt_tokens[-1]
                    pos = len(req.prompt_tokens) - 1

                input_ids_list.append(torch.tensor([last_token], dtype=torch.long))
                positions_list.append(torch.tensor([pos], dtype=torch.long))

                # Ensure KV blocks exist (may have been allocated during prefill)
                try:
                    blocks = block_manager.get_blocks(req.request_id)
                except KeyError:
                    # First decode step without prior prefill — allocate now
                    blocks = block_manager.allocate(req.request_id, pos + 1)
                block_tables[req.request_id] = blocks

        if not batch_requests:
            return SequenceGroup()

        # Flatten tensors (for backward compatibility / batch forward)
        input_ids = torch.cat(input_ids_list) if input_ids_list else torch.tensor([], dtype=torch.long)
        positions = torch.cat(positions_list) if positions_list else torch.tensor([], dtype=torch.long)

        return SequenceGroup(
            requests=batch_requests,
            input_ids=input_ids,
            positions=positions,
            request_input_ids=input_ids_list,
            request_positions=positions_list,
            block_tables=block_tables,
        )

    def process_outputs(self, logits: torch.Tensor, batch: SequenceGroup) -> list[GenerationResult]:
        """Process model outputs: sample tokens and update request state.

        Returns a list of GenerationResult entries (defined in engine/batch.py).
        """
        from engine.sampler import sample

        results: list[GenerationResult] = []
        if batch.is_empty or logits.numel() == 0:
            return results

        # For decode steps: one logit vector per request
        # For prefill steps: we only need the last position's logits
        offset = 0
        for req in batch.requests:
            sp = req.sampling_params
            if req.state == "decode":
                req_logits = logits[offset : offset + 1]
                offset += 1
            elif req.state == "prefill":
                # Prefill: take the last token position logits
                processed = req.num_processed_tokens
                if processed > 0 and processed <= logits.size(0):
                    req_logits = logits[processed - 1 : processed]
                else:
                    req_logits = logits[0:1]
                # Don't advance offset for prefill (position mapping is implicit)
            else:
                continue

            token_id = sample(
                req_logits,
                temperature=sp.temperature,
                top_p=sp.top_p,
                top_k=sp.top_k,
            )

            token_val = int(token_id.item())
            req.append_token(token_val)

            # Check stop conditions
            is_finished = False
            if len(req.output_tokens) >= sp.max_tokens:
                is_finished = True
            if token_val in sp.stop_token_ids:
                is_finished = True

            if is_finished:
                req.mark_finished()

            results.append(
                GenerationResult(
                    request_id=req.request_id,
                    new_tokens=[token_val],
                    is_finished=is_finished,
                )
            )

        return results

    # ── Query ───────────────────────────────────────────────

    @property
    def waiting_count(self) -> int:
        return len(self._waiting)

    @property
    def running_count(self) -> int:
        return len(self._running)

    @property
    def has_work(self) -> bool:
        return len(self._waiting) > 0 or len(self._running) > 0
