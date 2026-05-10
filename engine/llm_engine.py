"""LLMEngine — top-level inference engine.

Integrates Rust Scheduler, Rust BlockManager, and Executor into a single
inference loop. Provides both a step() API for server integration and a
blocking generate() convenience method.

Architecture alignment:
  - vLLM V1: Engine → Scheduler → Worker → ModelRunner
  - LLM-ServeForge: Engine → Rust Scheduler + BlockManager + Executor (HAL)

The Engine owns the lifecycle of all subsystems and coordinates
the per-step scheduling-execution-sampling cycle.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch

from compiler.mlir_artifact import MlirModule
from engine._constants import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_BATCH_SIZE,
    DEFAULT_MAX_TOKENS_PER_STEP,
    DEFAULT_NUM_BLOCKS,
)
from engine.batch import GenerationResult, SamplingParams
from engine.sampler import sample
from hal.interface import OpExecutor
from hal.protocols import Tokenizer
from utils.logging import get_logger, log_request_lifecycle, log_step_begin, log_step_end

_log = get_logger("engine")


def _read_policy_dim(raw_policy: dict[str, Any], key: str) -> int:
    for slab in raw_policy.get("slabs", []):
        if key in slab.get("dims", {}):
            return int(slab["dims"][key])
    return 0


@runtime_checkable
class _ExecutorLike(Protocol):
    """Protocol for any executor compatible with LLMEngine."""

    _kv_cache: torch.Tensor | None
    _block_tables: dict[str, list[int]]

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor: ...
    def forward_with_kv(
        self, input_ids: torch.Tensor, **kwargs: Any
    ) -> tuple[torch.Tensor, list[tuple[str, torch.Tensor]]]: ...
    def set_kv_cache(
        self,
        kv_cache: torch.Tensor | None,
        block_tables: dict[str, list[int]] | None = None,
        block_size: int = 16,
        num_kv_heads: int = 0,
        head_dim: int = 0,
    ) -> None: ...
    def prepare_kv_blocks(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        num_blocks: int,
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor: ...
    def write_kv_to_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        block_tables: dict[str, list[int]],
        layer_idx: int = 0,
    ) -> None: ...
    def gather_kv_from_cache(
        self,
        block_tables: dict[str, list[int]],
        max_seq_len: int,
        layer_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


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


class LLMEngine:
    """Single-process inference engine.

    Coordinates the full inference loop:
        schedule → prepare inputs → forward → sample → return results

    Uses Rust PyScheduler and PyBlockManager for hot-path scheduling and
    memory management. Prefix caching (RadixCache) remains in Python.

    Usage:
        backend = PyTorchBackend("cpu")
        ir_module = load_artifact("./compiled/model")
        engine = LLMEngine(ir_module, backend)
        text = engine.generate("Explain quantum computing", max_tokens=100)
    """

    def __init__(
        self,
        module: MlirModule,
        hal_backend: OpExecutor,
        executor: _ExecutorLike | None = None,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        max_tokens_per_step: int = DEFAULT_MAX_TOKENS_PER_STEP,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        num_blocks: int = DEFAULT_NUM_BLOCKS,
        block_size: int = DEFAULT_BLOCK_SIZE,
        num_layers: int = 0,
        num_kv_heads: int = 0,
        head_dim: int = 0,
        dtype: torch.dtype | None = None,
        enable_prefix_cache: bool = False,
    ) -> None:
        import llm_serveforge_runtime as _rt

        self._module = module
        self._hal_backend = hal_backend

        # ── Auto-detect cache config from module metadata ─
        raw_policy = module.metadata.get("cache_policy") if module.metadata else None
        if raw_policy and num_layers <= 0:
            num_layers = _read_policy_dim(raw_policy, "layers")
        if raw_policy and num_kv_heads <= 0:
            num_kv_heads = _read_policy_dim(raw_policy, "heads")
        if raw_policy and head_dim <= 0:
            head_dim = _read_policy_dim(raw_policy, "dim")
        # Ensure the executor sees the correct block count for slab allocation
        if raw_policy and "num_blocks" not in module.metadata:
            module.metadata["num_blocks"] = num_blocks

        # ── Rust core runtime ────────────────────────────
        self._bm = _rt.PyBlockManager(num_blocks, block_size)
        self._scheduler = _rt.PyScheduler(max_batch_size, max_tokens_per_step, chunk_size)

        # ── Prefix Cache (Python) ────────────────────────
        self._radix_cache = None
        if enable_prefix_cache:
            from cache.radix_cache import RadixCache
            self._radix_cache = RadixCache(self._bm)

        # ── Executor ─────────────────────────────────────
        if executor is not None:
            self.executor: _ExecutorLike = executor
        else:
            from engine.mlir_executor import MlirExecutor
            self.executor = MlirExecutor(module, hal_backend)

        # ── Per-request state (Python side) ──────────────
        self._sampling_params: dict[str, SamplingParams] = {}
        self._prompt_tokens: dict[str, list[int]] = {}
        self._output_tokens: dict[str, list[int]] = {}

        # ── Tokenizer ────────────────────────────────────
        self._tokenizer: Tokenizer | None = None
        self._eos_token_id: int | None = None

        # ── KV cache ─────────────────────────────────────
        self._kv_cache: torch.Tensor | None = None
        self._num_layers = num_layers
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._kv_dtype = dtype or torch.float32

        # ── Observability ─────────────────────────────────
        self._step_id = 0

    # ── KV Cache Lifecycle ─────────────────────────────────

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
        if self._uses_new_cache():
            self._kv_cache = torch.zeros(1)
            return self._kv_cache
        if self._uses_static_model():
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
            block_size=self._bm.block_size,
            num_blocks=self._bm.num_blocks,
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
            block_size=self._bm.block_size,
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

    # ── Prefix Cache helpers ───────────────────────────────

    def _build_cache_hits(self) -> list[tuple[str, list[int], int]]:
        """Query RadixCache for pending requests that haven't started yet."""
        if self._radix_cache is None:
            return []
        hits: list[tuple[str, list[int], int]] = []
        for rid, prompt in list(self._prompt_tokens.items()):
            if rid not in self._output_tokens or len(self._output_tokens[rid]) == 0:
                matched_blocks, matched_tokens = self._radix_cache.match_prefix(prompt)
                if matched_tokens > 0:
                    hits.append((rid, matched_blocks, matched_tokens))
        return hits

    def _insert_finished_to_cache(self, request_id: str) -> None:
        """Insert a finished request's tokens into RadixCache."""
        if self._radix_cache is None:
            return
        prompt = self._prompt_tokens.get(request_id)
        generated = self._output_tokens.get(request_id)
        if prompt is None:
            return
        all_tokens = prompt + (generated or [])
        try:
            blocks = self._bm.get_blocks(request_id)
            self._radix_cache.insert(all_tokens, blocks)
        except Exception:
            import logging
            _log = logging.getLogger("engine.llm_engine")
            _log.warning("Failed to insert request %s into prefix cache", request_id, exc_info=True)

    def _cleanup_request(self, request_id: str) -> None:
        self._bm.free(request_id)
        self._prompt_tokens.pop(request_id, None)
        self._output_tokens.pop(request_id, None)
        self._sampling_params.pop(request_id, None)

    # ── Core Loop ───────────────────────────────────────────

    def step(self) -> list[GenerationResult]:
        self._step_id += 1
        step_id = self._step_id
        t0 = time.perf_counter()

        log_step_begin(_log, step_id, self.num_waiting, self.num_running)

        # ── Prefix Cache LRU eviction (under memory pressure) ──
        if self._radix_cache is not None:
            free = self._bm.num_free_blocks()
            total = self._bm.num_blocks
            if free < max(1, total // 10):
                freed = self._radix_cache.evict(max(1, total // 10 - free))
                _log.debug("step %d | prefix cache evicted %d blocks (free=%d/%d)",
                           step_id, freed, free, total)

        # ── Schedule (Rust) ──────────────────────────────
        cache_hits = self._build_cache_hits()
        batch_dict = self._scheduler.schedule(self._bm, cache_hits)
        reqs_data = batch_dict.get("requests", [])
        if not reqs_data:
            return []

        # ── Build batch objects ───────────────────────────
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
        # ── Ensure KV cache is initialized ────────────────
        if self._uses_new_cache():
            self._ensure_kv_cache()

        kv = self._kv_cache
        bt = batch.block_tables
        use_kv = bool(kv is not None and bt)

        # ── Forward ───────────────────────────────────────
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
                block_size=self._bm.block_size,
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
                block_size=self._bm.block_size,
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

            # Track local output tokens for RadixCache insertion on completion
            self._output_tokens.setdefault(rid, []).append(token_val)

            # Notify Rust scheduler of the output token
            finished = self._scheduler.record_output(rid, token_val)

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

    # ── Query ───────────────────────────────────────────────

    @property
    def is_idle(self) -> bool:
        return not self._scheduler.has_work()

    @property
    def num_running(self) -> int:
        return self._scheduler.running_count()  # type: ignore[no-any-return]

    @property
    def num_waiting(self) -> int:
        return self._scheduler.waiting_count()  # type: ignore[no-any-return]

    # ── Convenience API ─────────────────────────────────────

    def add_request(
        self,
        prompt: str | list[int],
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        priority: int = 0,
    ) -> str:
        if isinstance(prompt, str):
            if self._tokenizer is None:
                raise RuntimeError("Text prompt requires a tokenizer. Pass tokenized IDs instead.")
            prompt_tokens = self._tokenizer.encode(prompt)
        else:
            prompt_tokens = list(prompt)

        stop_token_ids = []
        if self._eos_token_id is not None:
            stop_token_ids.append(self._eos_token_id)

        rid: str = self._scheduler.add_request(
            prompt_tokens, priority, time.monotonic(), max_tokens, stop_token_ids, None
        )
        self._sampling_params[rid] = SamplingParams(
            temperature=temperature, top_p=top_p, top_k=top_k, max_tokens=max_tokens
        )
        self._prompt_tokens[rid] = prompt_tokens
        log_request_lifecycle(_log, rid, "admitted", prompt_len=len(prompt_tokens),
                              max_tokens=max_tokens, priority=priority)
        return rid

    def generate(
        self,
        prompt: str | list[int],
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> str:
        if self._uses_static_model():
            return self._generate_static_model(prompt, max_tokens, temperature, top_p, top_k)

        request_id = self.add_request(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        all_output_tokens: list[int] = []

        while True:
            results = self.step()
            for r in results:
                if r.request_id == request_id:
                    all_output_tokens.extend(r.new_tokens)
                    if r.is_finished:
                        if self._tokenizer is not None:
                            return str(self._tokenizer.decode(all_output_tokens))
                        return " ".join(str(t) for t in all_output_tokens)

            if self.is_idle:
                break

        if self._tokenizer is not None:
            return str(self._tokenizer.decode(all_output_tokens))
        return " ".join(str(t) for t in all_output_tokens)

    # ── Tokenizer Support ───────────────────────────────────

    def _generate_static_model(
        self,
        prompt: str | list[int],
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> str:
        if isinstance(prompt, str):
            if self._tokenizer is None:
                raise RuntimeError("Text prompt requires a tokenizer.")
            prompt_tokens = self._tokenizer.encode(prompt)
        else:
            prompt_tokens = list(prompt)

        # Determine model's expected seq_len from function input type
        import re
        expected_seq = None
        for _name, tp in self._module.main.inputs:
            m = re.search(r"tensor<(\d+)x(\d+)x", tp)
            if m and m.group(2) != "?":
                expected_seq = int(m.group(2))
                break

        all_tokens = list(prompt_tokens)
        eos_id = self._eos_token_id

        for _ in range(max_tokens):
            current_seq = all_tokens
            if expected_seq is not None:
                current_seq = current_seq[-expected_seq:]
                if len(current_seq) < expected_seq:
                    pad_id = getattr(self._tokenizer, "pad_token_id", 0) if self._tokenizer else 0
                    current_seq = current_seq + [pad_id] * (expected_seq - len(current_seq))

            inp = torch.tensor([current_seq], dtype=torch.long)
            logits = self.executor.forward(inp)
            last_pos = expected_seq - 1 if expected_seq else len(current_seq) - 1
            last_logits = logits[0, last_pos, :]

            from engine.sampler import sample
            sp = SamplingParams(temperature=temperature, top_p=top_p, top_k=top_k,
                                max_tokens=max_tokens)
            token_id = int(sample(last_logits.unsqueeze(0),
                                  temperature=sp.temperature, top_p=sp.top_p,
                                  top_k=sp.top_k).item())

            if eos_id is not None and token_id == eos_id:
                break
            all_tokens.append(token_id)

        output_tokens = all_tokens[len(prompt_tokens):]
        if self._tokenizer is not None:
            return str(self._tokenizer.decode(output_tokens))
        return " ".join(str(t) for t in output_tokens)

    def set_tokenizer(self, tokenizer: Tokenizer, eos_token_id: int | None = None) -> None:
        self._tokenizer = tokenizer
        self._eos_token_id = eos_token_id
