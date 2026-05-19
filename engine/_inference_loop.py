"""Inference loop — step execution, batch dispatch, and sampling.

Owns per-request state dicts and KV cache lifecycle.
Called by LLMEngine after scheduling.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from engine._scheduling_bridge import SchedulingBridge
from engine.batch import GenerationResult, SamplingParams
from engine.sampler import sample
from utils.logging import get_logger, log_request_lifecycle, log_step_begin, log_step_end

_log = get_logger("engine.inference")


@dataclass
class _BatchRequest:
    """Lightweight per-request metadata passed through the batch forward."""
    request_id: str
    input_ids: torch.Tensor
    positions: torch.Tensor
    block_table: list[int]
    is_decode: bool

    @property
    def n_tokens(self) -> int:
        return int(torch.numel(self.input_ids))


class _Batch:
    """SequenceGroup-compatible batch built from Rust scheduler output."""

    def __init__(self, batch_requests: list[_BatchRequest]):
        self._requests = batch_requests

    @property
    def requests(self) -> list[_BatchRequest]:
        return self._requests

    @property
    def request_input_ids(self) -> list[torch.Tensor]:
        return [r.input_ids for r in self._requests]

    @property
    def request_positions(self) -> list[torch.Tensor]:
        return [r.positions for r in self._requests]

    @property
    def block_tables(self) -> dict[str, list[int]]:
        return {r.request_id: r.block_table for r in self._requests}

    @property
    def is_empty(self) -> bool:
        return len(self._requests) == 0

    @property
    def size(self) -> int:
        return len(self._requests)


class InferenceLoop:
    """Per-step inference loop: batch forward → sample → record.

    Separated from LLMEngine for testability. Owns per-request state
    dicts and KV cache lifecycle.  Mutates state in place.
    """

    def __init__(
        self,
        executor: Any,
        bridge: SchedulingBridge,
        num_layers: int = 0,
        num_kv_heads: int = 0,
        head_dim: int = 0,
        kv_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.executor = executor
        self._bridge = bridge

        # ── KV cache ──
        self._kv_cache: torch.Tensor | None = None
        self._num_layers = num_layers
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._kv_dtype = kv_dtype

        # ── Per-request state ──
        self._sampling_params: dict[str, SamplingParams] = {}
        self._prompt_tokens: dict[str, list[int]] = {}
        self._output_tokens: dict[str, list[int]] = {}

        self._step_id = 0

    # ── Convenience API ─────────────────────────────────────────

    def add_request(
        self,
        rid: str,
        prompt_tokens: list[int],
        sp: SamplingParams,
    ) -> None:
        """Register a request's tokens and sampling params."""
        self._sampling_params[rid] = sp
        self._prompt_tokens[rid] = prompt_tokens

    # ── KV Cache Lifecycle ──────────────────────────────────────

    def _uses_new_cache(self) -> bool:
        return (
            hasattr(self.executor, "_uses_cache_manager")
            and self.executor._uses_cache_manager
        )

    def _uses_static_model(self) -> bool:
        return (
            hasattr(self.executor, "_uses_static_shape")
            and getattr(self.executor, "_uses_static_shape", False)
        )

    def _ensure_kv_cache(self) -> torch.Tensor:
        if self._kv_cache is not None:
            return self._kv_cache
        if self._uses_new_cache() or self._uses_static_model():
            self._kv_cache = torch.zeros(1)
            return self._kv_cache
        if self._num_layers <= 0 or self._num_kv_heads <= 0 or self._head_dim <= 0:
            raise RuntimeError(
                "KV cache requires num_layers, num_kv_heads, head_dim. "
                "Set them on LLMEngine creation."
            )
        self._kv_cache = self.executor.prepare_kv_blocks(
            num_layers=self._num_layers,
            num_kv_heads=self._num_kv_heads,
            head_dim=self._head_dim,
            block_size=self._bridge.block_size,
            num_blocks=self._bridge.num_blocks,
            dtype=self._kv_dtype,
        )
        return self._kv_cache

    def _write_kv_outputs(
        self,
        kv_tensors: list[tuple[str, torch.Tensor]],
        positions: torch.Tensor,
        block_tables: dict[str, list[int]],
    ) -> None:
        if self._uses_new_cache() or self._uses_static_model():
            return
        if self._kv_cache is None:
            return
        self.executor.set_kv_cache(
            kv_cache=self._kv_cache,
            block_tables=block_tables,
            block_size=self._bridge.block_size,
            num_kv_heads=self._num_kv_heads,
            head_dim=self._head_dim,
        )
        for layer_idx in range(self._num_layers):
            ki = layer_idx * 2
            vi = layer_idx * 2 + 1
            if ki < len(kv_tensors) and vi < len(kv_tensors):
                _, key = kv_tensors[ki]
                _, value = kv_tensors[vi]
                if key.dim() >= 4 and key.shape[0] == 1:
                    key = key.squeeze(0)
                if value.dim() >= 4 and value.shape[0] == 1:
                    value = value.squeeze(0)
                flat_pos = positions.squeeze(0) if positions.dim() >= 2 else positions
                self.executor.write_kv_to_cache(
                    key, value, flat_pos, block_tables, layer_idx=layer_idx
                )

    # ── Prefix Cache helpers ────────────────────────────────────

    def _build_cache_hits(self) -> list[tuple[str, list[int], int]]:
        return self._bridge.build_cache_hits(self._prompt_tokens, self._output_tokens)

    def _insert_finished_to_cache(self, request_id: str) -> None:
        self._bridge.insert_finished_to_cache(request_id, self._prompt_tokens, self._output_tokens)

    def _cleanup_request(self, request_id: str) -> None:
        self._bridge.free_request(request_id)
        self._prompt_tokens.pop(request_id, None)
        self._output_tokens.pop(request_id, None)
        self._sampling_params.pop(request_id, None)

    # ── Core Loop ───────────────────────────────────────────────

    def step(self) -> list[GenerationResult]:
        self._step_id += 1
        step_id = self._step_id
        t0 = time.perf_counter()

        log_step_begin(_log, step_id, self._bridge.num_waiting, self._bridge.num_running)

        # ── Prefix Cache LRU eviction (under memory pressure) ──
        freed = self._bridge.evict_cache_if_needed()
        if freed > 0:
            _log.debug("step %d | prefix cache evicted %d blocks", step_id, freed)

        # ── Schedule (Rust) ─────────────────────────────────────
        cache_hits = self._build_cache_hits()
        batch_dict = self._bridge.schedule(cache_hits)
        reqs_data = batch_dict.get("requests", [])
        if not reqs_data:
            return []

        # ── Build batch objects ─────────────────────────────────
        batch_requests: list[_BatchRequest] = []
        for rd in reqs_data:
            rid = rd["request_id"]
            is_decode = rd["state"] == "decode"
            br = _BatchRequest(
                request_id=rid,
                input_ids=torch.tensor(rd["input_ids"], dtype=torch.long),
                positions=torch.tensor(rd["positions"], dtype=torch.long),
                block_table=rd["block_table"],
                is_decode=is_decode,
            )
            batch_requests.append(br)

        batch = _Batch(batch_requests)

        # ── Ensure KV cache is initialized ──────────────────────
        if self._uses_new_cache():
            self._ensure_kv_cache()

        kv = self._kv_cache
        bt = batch.block_tables
        use_kv = bool(kv is not None and bt)

        # ── Forward ─────────────────────────────────────────────
        can_batch = self._can_batch_forward(batch)
        if can_batch:
            results = self._step_batch(batch, use_kv)
        else:
            results = self._step_per_request(batch, use_kv)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        log_step_end(_log, step_id, elapsed_ms,
                     batch_size=batch.size,
                     total_tokens=sum(r.n_tokens for r in batch._requests),
                     results=len(results))
        return results

    def _can_batch_forward(self, batch: _Batch) -> bool:
        if len(batch._requests) <= 1:
            return len(batch._requests) == 1
        first_n = batch._requests[0].n_tokens
        for r in batch._requests[1:]:
            if r.n_tokens != first_n:
                return False
        return True

    def _step_batch(self, batch: _Batch, use_kv: bool) -> list[GenerationResult]:
        stacked_input = torch.stack([r.input_ids.flatten() for r in batch._requests])
        stacked_pos = torch.stack([r.positions.flatten() for r in batch._requests])
        batch_is_decode = batch._requests[0].is_decode if batch._requests else False

        if use_kv:
            self.executor.set_kv_cache(
                kv_cache=self._kv_cache,
                block_tables=batch.block_tables,
                block_size=self._bridge.block_size,
                num_kv_heads=self._num_kv_heads,
                head_dim=self._head_dim,
            )
            logits, kv_tensors = self.executor.forward_with_kv(
                stacked_input, positions=stacked_pos, is_decode=batch_is_decode,
            )
            self._write_kv_outputs(kv_tensors, stacked_pos, batch.block_tables)
        else:
            logits = self.executor.forward(stacked_input, positions=stacked_pos)

        return self._sample_from_logits(logits, batch)

    def _step_per_request(self, batch: _Batch, use_kv: bool) -> list[GenerationResult]:
        if use_kv:
            self.executor.set_kv_cache(
                kv_cache=self._kv_cache,
                block_tables=batch.block_tables,
                block_size=self._bridge.block_size,
                num_kv_heads=self._num_kv_heads,
                head_dim=self._head_dim,
            )

        results: list[GenerationResult] = []
        for i, br in enumerate(batch._requests):
            req_input = br.input_ids.unsqueeze(0) if br.input_ids.dim() == 1 else br.input_ids
            req_pos = br.positions.unsqueeze(0) if br.positions.dim() == 1 else br.positions

            if use_kv:
                logits, kv_tensors = self.executor.forward_with_kv(
                    req_input, positions=req_pos, is_decode=br.is_decode,
                )
                self._write_kv_outputs(kv_tensors, req_pos, batch.block_tables)
            else:
                logits = self.executor.forward(req_input, positions=req_pos)

            req_results = self._sample_from_logits(logits, batch, single_req=i)
            results.extend(req_results)

        return results

    def _sample_from_logits(
        self, logits_tensor: torch.Tensor, batch: _Batch, single_req: int | None = None
    ) -> list[GenerationResult]:
        results: list[GenerationResult] = []
        bsz = logits_tensor.shape[0] if logits_tensor.dim() >= 3 else 1
        req_start = 0 if single_req is None else single_req
        req_end = req_start + bsz if single_req is None else req_start + 1

        for req_idx, br in enumerate(batch._requests[req_start:req_end]):
            rid = br.request_id

            if logits_tensor.dim() >= 3:
                req_logits = logits_tensor[req_idx:req_idx + 1, -1, :]
            elif logits_tensor.dim() == 2:
                req_logits = logits_tensor[req_idx:req_idx + 1, :]
            else:
                req_logits = logits_tensor

            sp = self._sampling_params.get(rid, SamplingParams())
            token_id = sample(
                req_logits,
                temperature=sp.temperature,
                top_p=sp.top_p,
                top_k=sp.top_k,
            )
            token_val = int(token_id.item())

            self._output_tokens.setdefault(rid, []).append(token_val)
            finished = self._bridge.record_output(rid, token_val)

            if finished:
                self._insert_finished_to_cache(rid)
                self._cleanup_request(rid)
                log_request_lifecycle(_log, rid, "finished",
                                      output_tokens=len(self._output_tokens.get(rid, [])))

            results.append(
                GenerationResult(
                    request_id=rid,
                    new_tokens=[token_val],
                    is_finished=finished,
                )
            )

        return results
