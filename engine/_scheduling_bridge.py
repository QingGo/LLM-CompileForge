"""Scheduling bridge — wraps Rust PyScheduler + PyBlockManager + RadixCache.

Provides a unified Python interface for the scheduling subsystem,
hiding the Rust FFI details from the engine layer.

Responsibility boundary:
  - Owns: scheduler, block_manager, radix_cache lifecycle
  - Exposes: schedule(), add_request(), free_request(), get_blocks()
  - Does NOT own: executor, tokenizer, sampling
"""

from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger("engine.scheduling")


class SchedulingBridge:
    """Bridge between LLMEngine and the Rust scheduling runtime.

    Manages the lifecycle of PyScheduler, PyBlockManager, and optional
    RadixCache (prefix cache).
    """

    def __init__(
        self,
        num_blocks: int = 1024,
        block_size: int = 16,
        max_batch_size: int = 8,
        max_tokens_per_step: int = 2048,
        chunk_size: int = 64,
        enable_prefix_cache: bool = False,
    ) -> None:
        import llm_serveforge_runtime as _rt

        # ── Rust core runtime ──
        self._bm = _rt.PyBlockManager(num_blocks, block_size)
        self._scheduler = _rt.PyScheduler(max_batch_size, max_tokens_per_step, chunk_size)

        # ── Prefix Cache (Python) — only if enabled ──
        self._radix_cache = None
        if enable_prefix_cache:
            from cache.radix_cache import RadixCache
            self._radix_cache = RadixCache(self._bm)

        self._num_blocks = num_blocks
        self._block_size = block_size

    # ── Block Manager API ──

    @property
    def block_manager(self) -> Any:
        return self._bm

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def num_free_blocks(self) -> int:
        return self._bm.num_free_blocks()

    def get_blocks(self, request_id: str) -> list[int]:
        return self._bm.get_blocks(request_id)

    def free_request(self, request_id: str) -> None:
        self._bm.free(request_id)

    # ── Scheduler API ──

    def schedule(self, cache_hits: list[tuple[str, list[int], int]]) -> dict[str, Any]:
        """Run the scheduler, returning a batch dict from Rust PyScheduler."""
        return self._scheduler.schedule(self._bm, cache_hits)

    def add_request(
        self,
        request_id: str,
        prompt_tokens: list[int],
        max_tokens: int = 256,
    ) -> None:
        self._scheduler.add_request(prompt_tokens, 0, 0.0, max_tokens, [], request_id)

    def record_output(self, request_id: str, token_id: int) -> bool:
        """Notify scheduler of an output token. Returns True if request is finished."""
        return self._scheduler.record_output(request_id, token_id)

    @property
    def waiting_count(self) -> int:
        return self._scheduler.waiting_count()

    @property
    def running_count(self) -> int:
        return self._scheduler.running_count()

    def has_work(self) -> bool:
        return self._scheduler.has_work()

    # ── Prefix Cache API ──

    @property
    def prefix_cache(self) -> Any | None:
        return self._radix_cache

    def build_cache_hits(
        self, prompt_tokens: dict[str, list[int]], output_tokens: dict[str, list[int]]
    ) -> list[tuple[str, list[int], int]]:
        """Query RadixCache for pending requests that haven't started yet."""
        if self._radix_cache is None:
            return []
        hits: list[tuple[str, list[int], int]] = []
        for rid, prompt in list(prompt_tokens.items()):
            if rid not in output_tokens or len(output_tokens[rid]) == 0:
                matched_blocks, matched_tokens = self._radix_cache.match_prefix(prompt)
                if matched_tokens > 0:
                    hits.append((rid, matched_blocks, matched_tokens))
        return hits

    def insert_finished_to_cache(
        self, request_id: str, prompt_tokens: dict[str, list[int]], output_tokens: dict[str, list[int]]
    ) -> None:
        if self._radix_cache is None:
            return
        prompt = prompt_tokens.get(request_id)
        generated = output_tokens.get(request_id)
        if prompt is None:
            return
        all_tokens = prompt + (generated or [])
        try:
            blocks = self._bm.get_blocks(request_id)
            self._radix_cache.insert(all_tokens, blocks)
        except Exception:
            _log.warning("Failed to insert request %s into prefix cache", request_id, exc_info=True)

    def evict_cache_if_needed(self) -> int:
        """Evict from prefix cache if memory pressure is high. Returns blocks freed."""
        if self._radix_cache is None:
            return 0
        free = self._bm.num_free_blocks()
        if free < max(1, self._num_blocks // 10):
            return self._radix_cache.evict(max(1, self._num_blocks // 10 - free))
        return 0

    # ── Observer ──

    def __repr__(self) -> str:
        return (
            f"SchedulingBridge(waiting={self.waiting_count}, running={self.running_count}, "
            f"free_blocks={self.num_free_blocks}, prefix_cache={self._radix_cache is not None})"
        )
