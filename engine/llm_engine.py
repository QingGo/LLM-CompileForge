"""LLMEngine — top-level inference engine.

Integrates Scheduler, BlockManager, and Executor into a single inference
loop. Provides both a step() API for server integration and a blocking
generate() convenience method.

Architecture alignment:
  - vLLM V1: Engine → Scheduler → Worker → ModelRunner
  - LLM-ServeForge: Engine → Scheduler + BlockManager + Executor (HAL)

The Engine owns the lifecycle of all subsystems and coordinates
the per-step scheduling-execution-sampling cycle.
"""

from __future__ import annotations

from typing import Any

import torch

from compiler.ir import IrModule
from engine.batch import GenerationResult
from engine.block_manager import BlockManager
from engine.executor import Executor
from engine.sampler import sample
from engine.scheduler import Scheduler
from hal.interface import OpExecutor


class LLMEngine:
    """Single-process inference engine.

    Coordinates the full inference loop:
        schedule → prepare inputs → forward → sample → return results

    Usage:
        backend = PyTorchBackend("cpu")
        ir_module = load_artifact("./compiled/model")
        engine = LLMEngine(ir_module, backend)
        text = engine.generate("Explain quantum computing", max_tokens=100)
    """

    def __init__(
        self,
        module: IrModule,
        hal_backend: OpExecutor,
        max_batch_size: int = 32,
        max_tokens_per_step: int = 512,
        chunk_size: int = 256,
        num_blocks: int = 1000,
        block_size: int = 16,
        num_layers: int = 0,
        num_kv_heads: int = 0,
        head_dim: int = 0,
        dtype: torch.dtype | None = None,
    ) -> None:
        self._module = module
        self._hal_backend = hal_backend

        self.scheduler = Scheduler(
            max_batch_size=max_batch_size,
            max_tokens_per_step=max_tokens_per_step,
            chunk_size=chunk_size,
        )
        self.block_manager = BlockManager(num_blocks=num_blocks, block_size=block_size)
        self.executor = Executor(module, hal_backend)

        # Tokenizer reference — set by the API server or user
        self._tokenizer: Any = None
        self._eos_token_id: int | None = None

        # KV cache — allocated on first use
        self._kv_cache: torch.Tensor | None = None
        self._num_layers = num_layers
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._kv_dtype = dtype or torch.float32

    # ── KV Cache Lifecycle ─────────────────────────────────

    def _ensure_kv_cache(self) -> torch.Tensor:
        """Allocate the KV cache tensor on first use."""
        if self._kv_cache is not None:
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
            block_size=self.block_manager.block_size,
            num_blocks=self.block_manager.num_blocks,
            dtype=self._kv_dtype,
        )
        return self._kv_cache

    def _write_kv_outputs(
        self,
        kv_tensors: list[tuple[str, torch.Tensor]],
        positions: torch.Tensor,
        block_tables: dict[str, list[int]],
    ) -> None:
        """Write K/V model outputs into the paged KV cache.

        During prefill, the executor's SDPA intercept already writes K/V
        to cache (covering the same tokens). This function serves as a
        safety net for any K/V output pairs not captured by SDPA intercept.
        """
        if self._kv_cache is None:
            return
        self.executor.set_kv_cache(
            kv_cache=self._kv_cache,
            block_tables=block_tables,
            block_size=self.block_manager.block_size,
            num_kv_heads=self._num_kv_heads,
            head_dim=self._head_dim,
        )
        for layer_idx in range(self._num_layers):
            ki = layer_idx * 2
            vi = layer_idx * 2 + 1
            if ki < len(kv_tensors) and vi < len(kv_tensors):
                _, key = kv_tensors[ki]
                _, value = kv_tensors[vi]
                # Squeeze batch dim: [1, seq, heads, head_dim] → [seq, heads, head_dim]
                if key.dim() >= 4 and key.shape[0] == 1:
                    key = key.squeeze(0)
                if value.dim() >= 4 and value.shape[0] == 1:
                    value = value.squeeze(0)
                flat_pos = positions.squeeze(0) if positions.dim() >= 2 else positions
                self.executor.write_kv_to_cache(
                    key, value, flat_pos, block_tables, layer_idx=layer_idx
                )

    # ── Core Loop ───────────────────────────────────────────

    def step(self) -> list[GenerationResult]:
        """Execute one scheduling cycle — per-request or batch forward.

        When all requests have compatible shapes (same number of tokens), they
        are combined into a single batch forward. Otherwise, falls back to
        per-request processing.
        """
        batch = self.scheduler.schedule(self.block_manager)
        if batch.is_empty or not batch.request_input_ids:
            return []

        kv = self._kv_cache
        bt = batch.block_tables
        use_kv = bool(kv is not None and bt)

        # Try batch forward if all requests have same num tokens
        can_batch = self._can_batch_forward(batch)
        if can_batch:
            return self._step_batch(batch, use_kv)
        else:
            return self._step_per_request(batch, use_kv)

    def _can_batch_forward(self, batch: Any) -> bool:
        """Check if requests can be combined into a single batch forward."""
        if len(batch.requests) <= 1:
            return len(batch.requests) == 1
        first_n = batch.request_input_ids[0].numel()
        for t in batch.request_input_ids[1:]:
            if t.numel() != first_n:
                return False
        return True

    def _step_batch(self, batch: Any, use_kv: bool) -> list[GenerationResult]:
        """Execute a single batch forward for all requests."""
        stacked_input = torch.stack([t.flatten() for t in batch.request_input_ids])
        stacked_pos = torch.stack([t.flatten() for t in batch.request_positions])

        if use_kv:
            self.executor.set_kv_cache(
                kv_cache=self._kv_cache,
                block_tables=batch.block_tables,
                block_size=self.block_manager.block_size,
                num_kv_heads=self._num_kv_heads,
                head_dim=self._head_dim,
            )
            logits, kv_tensors = self.executor.forward_with_kv(
                stacked_input, positions=stacked_pos
            )
            self._write_kv_outputs(kv_tensors, stacked_pos, batch.block_tables)
        else:
            logits = self.executor.forward(stacked_input, positions=stacked_pos)

        return self._sample_from_logits(logits, batch)

    def _step_per_request(self, batch: Any, use_kv: bool) -> list[GenerationResult]:
        """Process each request independently through executor.forward()."""
        if use_kv:
            self.executor.set_kv_cache(
                kv_cache=self._kv_cache,
                block_tables=batch.block_tables,
                block_size=self.block_manager.block_size,
                num_kv_heads=self._num_kv_heads,
                head_dim=self._head_dim,
            )

        results: list[GenerationResult] = []
        for i, _req in enumerate(batch.requests):
            req_input = batch.request_input_ids[i]
            req_pos = batch.request_positions[i]
            if req_input.dim() == 1:
                req_input = req_input.unsqueeze(0)
            if req_pos.dim() == 1:
                req_pos = req_pos.unsqueeze(0)

            if use_kv:
                logits, kv_tensors = self.executor.forward_with_kv(
                    req_input, positions=req_pos
                )
                self._write_kv_outputs(kv_tensors, req_pos, batch.block_tables)
            else:
                logits = self.executor.forward(req_input, positions=req_pos)

            req_results = self._sample_from_logits(logits, batch, single_req=i)
            results.extend(req_results)

        return results

    def _sample_from_logits(
        self, logits_tensor: torch.Tensor, batch: Any, single_req: int | None = None
    ) -> list[GenerationResult]:
        """Sample next tokens from logits and update requests."""
        results: list[GenerationResult] = []
        bsz = logits_tensor.shape[0] if logits_tensor.dim() >= 3 else 1
        req_start = 0 if single_req is None else single_req
        req_end = req_start + bsz if single_req is None else req_start + 1

        for req_idx, req in enumerate(batch.requests[req_start:req_end]):
            if logits_tensor.dim() >= 3:
                req_logits = logits_tensor[req_idx:req_idx + 1, -1, :]
            else:
                req_logits = logits_tensor

            sp = req.sampling_params
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
        """Add an inference request.

        Args:
            prompt: Text prompt (requires tokenizer) or tokenized list.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            top_p: Nucleus sampling threshold.
            top_k: Top-k filtering.
            priority: Queue priority (lower = higher).

        Returns:
            The request_id.
        """
        from engine.batch import SamplingParams

        if isinstance(prompt, str):
            if self._tokenizer is None:
                raise RuntimeError("Text prompt requires a tokenizer. Pass tokenized IDs instead.")
            prompt_tokens = self._tokenizer.encode(prompt)
        else:
            prompt_tokens = list(prompt)

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
        )

        if self._eos_token_id is not None:
            sampling_params.stop_token_ids.append(self._eos_token_id)

        return self.scheduler.add_request(prompt_tokens, sampling_params, priority=priority)

    def generate(
        self,
        prompt: str | list[int],
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> str:
        """Blocking synchronous generation.

        Adds a request and loops step() until the request completes,
        returning the full generated text.
        """
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

            if not self.scheduler.has_work:
                break

        if self._tokenizer is not None:
            return str(self._tokenizer.decode(all_output_tokens))
        return " ".join(str(t) for t in all_output_tokens)

    # ── Tokenizer Support ───────────────────────────────────

    def set_tokenizer(self, tokenizer: Any, eos_token_id: int | None = None) -> None:
        """Attach a tokenizer for text encode/decode.

        The tokenizer must support encode(text) → list[int] and decode(tokens) → str.
        """
        self._tokenizer = tokenizer
        self._eos_token_id = eos_token_id

    # ── Query ───────────────────────────────────────────────

    @property
    def is_idle(self) -> bool:
        return not self.scheduler.has_work

    @property
    def num_running(self) -> int:
        return self.scheduler.running_count

    @property
    def num_waiting(self) -> int:
        return self.scheduler.waiting_count
