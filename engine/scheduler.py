"""Inference scheduler: Continuous Batching + Chunked Prefill.

.. deprecated::
    The production scheduler is now the Rust `PyScheduler` in
    ``llm_serveforge_runtime``.  This Python implementation is retained
    as a reference and for standalone testing, but LLMEngine no longer
    uses it by default.


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
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from engine.batch import GenerationResult, Request, SamplingParams, SequenceGroup
from engine.block_manager import BlockManager

if TYPE_CHECKING:
    from cache.radix_cache import RadixCache


@dataclass(order=True)
class _QueueEntry:
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
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive, got {max_batch_size}")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")

        self.max_batch_size = max_batch_size
        self.max_tokens_per_step = max_tokens_per_step
        self.chunk_size = chunk_size
        self.radix_cache = radix_cache
        self._clock = clock

        self._waiting: list[_QueueEntry] = []
        self._running: list[Request] = []
        self._request_counter = 0

    def add_request(
        self,
        prompt_tokens: list[int],
        sampling_params: SamplingParams | None = None,
        priority: int = 0,
        request_id: str | None = None,
    ) -> str:
        if request_id is None:
            self._request_counter += 1
            request_id = f"req_{self._request_counter}"

        req = Request(
            request_id=request_id,
            prompt_tokens=list(prompt_tokens),
            sampling_params=sampling_params or SamplingParams(),
            priority=priority,
            arrival_time=self._clock(),
        )
        heapq.heappush(self._waiting, _QueueEntry(priority, req))
        return request_id

    def schedule(self, block_manager: BlockManager) -> SequenceGroup:
        self._reap_finished(block_manager)
        self._admit_requests(block_manager)

        if not self._running:
            return SequenceGroup()

        return self._build_sequence_group(block_manager)

    def _reap_finished(self, block_manager: BlockManager) -> None:
        finished = [r for r in self._running if r.is_finished]
        for r in finished:
            self._running.remove(r)
            if self.radix_cache is not None:
                all_tokens = r.prompt_tokens + r.output_tokens
                try:
                    blks = block_manager.get_blocks(r.request_id)
                    self.radix_cache.insert(all_tokens, blks)
                except KeyError:
                    pass
            block_manager.free(r.request_id)

    def _admit_requests(self, block_manager: BlockManager) -> None:
        while self._waiting and len(self._running) < self.max_batch_size:
            entry = heapq.heappop(self._waiting)
            req = entry.request
            req.state = "prefill"

            if self.radix_cache is not None:
                matched_blocks, matched_tokens = self.radix_cache.match_prefix(
                    req.prompt_tokens
                )
                if matched_tokens > 0:
                    req.prefill_pos = matched_tokens
                    block_manager.assign_cached_blocks(req.request_id, matched_blocks)
                    if matched_tokens >= len(req.prompt_tokens):
                        req.state = "decode"

            self._running.append(req)

    def _build_sequence_group(self, block_manager: BlockManager) -> SequenceGroup:
        batch_requests: list[Request] = []
        input_ids_list: list[torch.Tensor] = []
        positions_list: list[torch.Tensor] = []
        block_tables: dict[str, list[int]] = {}
        total_prefill_tokens = 0
        num_decode = 0

        for req in self._running:
            if req.state == "prefill":
                result = self._build_prefill_chunk(
                    req, block_manager, block_tables, total_prefill_tokens
                )
                if result is not None:
                    req_tensor, pos_tensor, n_tokens = result
                    batch_requests.append(req)
                    input_ids_list.append(req_tensor)
                    positions_list.append(pos_tensor)
                    total_prefill_tokens += n_tokens
                    if req.prefill_pos >= len(req.prompt_tokens):
                        req.state = "decode"
                continue

            if req.state == "decode":
                if total_prefill_tokens + num_decode + 1 > self.max_tokens_per_step:
                    continue

                num_decode += 1
                batch_requests.append(req)
                tok_tensor, pos_tensor, blocks = self._build_decode_step(req, block_manager)
                input_ids_list.append(tok_tensor)
                positions_list.append(pos_tensor)
                block_tables[req.request_id] = blocks

        if not batch_requests:
            return SequenceGroup()

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

    def _build_prefill_chunk(
        self,
        req: Request,
        block_manager: BlockManager,
        block_tables: dict[str, list[int]],
        total_prefill_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int] | None:
        remaining = req.tokens_remaining
        if remaining <= 0:
            req.state = "decode"
            return None

        prefill_budget = self.max_tokens_per_step - total_prefill_tokens
        if prefill_budget <= 0:
            return None

        n_tokens = min(remaining, self.chunk_size, prefill_budget)
        start_pos = req.prefill_pos
        end_pos = start_pos + n_tokens

        chunk_input_ids = req.prompt_tokens[start_pos:end_pos]
        req.prefill_pos = end_pos

        try:
            blocks = block_manager.get_blocks(req.request_id)
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
            existing = block_manager.block_tables.get(req.request_id, [])
            existing_tokens = len(existing) * block_manager.block_size
            suffix_tokens = max(0, len(req.prompt_tokens) - existing_tokens)
            if suffix_tokens > 0:
                block_manager.allocate(req.request_id, suffix_tokens)
            else:
                block_manager.block_tables.setdefault(req.request_id, [])
            blocks = block_manager.get_blocks(req.request_id)

        block_tables[req.request_id] = blocks

        return (
            torch.tensor(chunk_input_ids, dtype=torch.long),
            torch.arange(start_pos, end_pos, dtype=torch.long),
            n_tokens,
        )

    def _build_decode_step(
        self, req: Request, block_manager: BlockManager
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        if req.output_tokens:
            last_token = req.output_tokens[-1]
            pos = len(req.prompt_tokens) + len(req.output_tokens) - 1
        else:
            last_token = req.prompt_tokens[-1]
            pos = len(req.prompt_tokens) - 1

        try:
            blocks = block_manager.get_blocks(req.request_id)
        except KeyError:
            blocks = block_manager.allocate(req.request_id, pos + 1)

        return (
            torch.tensor([last_token], dtype=torch.long),
            torch.tensor([pos], dtype=torch.long),
            blocks,
        )

    def process_outputs(self, logits: torch.Tensor, batch: SequenceGroup) -> list[GenerationResult]:
        from engine.sampler import sample

        results: list[GenerationResult] = []
        if batch.is_empty or logits.numel() == 0:
            return results

        offset = 0
        for req in batch.requests:
            sp = req.sampling_params
            if req.state == "decode":
                req_logits = logits[offset : offset + 1]
                offset += 1
            elif req.state == "prefill":
                processed = req.num_processed_tokens
                if processed > 0 and processed <= logits.size(0):
                    req_logits = logits[processed - 1 : processed]
                else:
                    req_logits = logits[0:1]
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

    @property
    def waiting_count(self) -> int:
        return len(self._waiting)

    @property
    def running_count(self) -> int:
        return len(self._running)

    @property
    def has_work(self) -> bool:
        return len(self._waiting) > 0 or len(self._running) > 0
