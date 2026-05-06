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

    # ── Core Loop ───────────────────────────────────────────

    def step(self) -> list[GenerationResult]:
        """Execute one scheduling cycle — per-request forward pass.

        Each request is processed independently through executor.forward()
        with input shape [1, n_tokens]. This works with both static-shape
        compiled models (fed exact compile-time shapes) and dynamic-seq
        compiled models (accepting any [1, N] input).

        Returns a list of GenerationResult, one per request that produced output.
        """
        batch = self.scheduler.schedule(self.block_manager)
        if batch.is_empty or not batch.request_input_ids:
            return []

        # Configure PagedAttention if KV cache is available
        kv = self._kv_cache
        bt = batch.block_tables
        if kv is not None and bt:
            self.executor.set_kv_cache(
                kv_cache=kv,
                block_tables=bt,
                block_size=self.block_manager.block_size,
                num_kv_heads=self._num_kv_heads,
                head_dim=self._head_dim,
            )

        results: list[GenerationResult] = []
        for i, req in enumerate(batch.requests):
            req_input = batch.request_input_ids[i]
            req_pos = batch.request_positions[i]
            # Reshape [n] → [1, n] to match model's expected input layout
            if req_input.dim() == 1:
                req_input = req_input.unsqueeze(0)
            if req_pos.dim() == 1:
                req_pos = req_pos.unsqueeze(0)

            logits = self.executor.forward(req_input, positions=req_pos)

            # Take the last position's logits for next-token sampling
            if logits.dim() >= 3:
                req_logits = logits[0:1, -1, :]  # [batch, seq, vocab] → [1, vocab]
            else:
                req_logits = logits  # already [1, vocab] or [batch, vocab]

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
