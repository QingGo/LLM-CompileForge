"""LLMEngine — top-level inference engine.

Integrates Rust Scheduler, Rust BlockManager, Executor, and InferenceLoop
into a single orchestration layer.  The per-step inference logic lives
in ``_inference_loop.InferenceLoop`` for testability.

Architecture alignment:
  - vLLM V1: Engine → Scheduler → Worker → ModelRunner
  - LLM-ServeForge: Engine → SchedulingBridge → InferenceLoop → Executor (HAL)

The Engine owns the lifecycle of all subsystems and provides the public
``step()``, ``add_request()``, and ``generate()`` API.  Most internal
state (sampling params, output tokens, KV cache) lives in InferenceLoop.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from compiler.mlir_artifact import MlirModule
from engine._constants import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_BATCH_SIZE,
    DEFAULT_MAX_TOKENS_PER_STEP,
    DEFAULT_NUM_BLOCKS,
)
from engine._inference_loop import InferenceLoop
from engine._scheduling_bridge import SchedulingBridge
from engine.batch import GenerationResult, SamplingParams
from hal.interface import OpExecutor
from hal.protocols import Tokenizer
from utils.logging import get_logger, log_request_lifecycle

_log = get_logger("engine")


def _read_policy_dim(raw_policy: dict[str, Any], key: str) -> int:
    for slab in raw_policy.get("slabs", []):
        if key in slab.get("dims", {}):
            return int(slab["dims"][key])
    return 0


class _CacheParams:
    """KV cache configuration extracted from module metadata."""

    def __init__(self, module: MlirModule, **overrides: Any) -> None:
        raw = module.metadata.get("cache_policy") if module.metadata else {}
        if raw is None:
            raw = {}
        self.num_layers: int = overrides.get("num_layers", 0) or _read_policy_dim(raw, "layers")
        self.num_kv_heads: int = overrides.get("num_kv_heads", 0) or _read_policy_dim(raw, "heads")
        self.head_dim: int = overrides.get("head_dim", 0) or _read_policy_dim(raw, "dim")
        self.num_blocks: int = overrides.get("num_blocks", DEFAULT_NUM_BLOCKS)
        self.block_size: int = overrides.get("block_size", DEFAULT_BLOCK_SIZE)
        self.dtype: torch.dtype = overrides.get("dtype", torch.float32)

        if raw and "num_blocks" not in module.metadata:
            module.metadata["num_blocks"] = self.num_blocks


class LLMEngine:
    """Single-process inference engine.

    Thin orchestrator that wires together:
      SchedulingBridge (Rust scheduler + block manager + prefix cache)
      InferenceLoop (step dispatch + sampling + KV cache)
      Executor (MLIR/HAL forward)
      Tokenizer

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
        executor: Any = None,
        **kwargs: Any,
    ) -> None:
        self._module = module
        self._hal_backend = hal_backend

        # ── Auto-detect cache config from module metadata ─
        cp = _CacheParams(module, **kwargs)

        # ── Scheduling bridge (Rust Scheduler + BlockManager + RadixCache) ──
        self._bridge = SchedulingBridge(
            num_blocks=cp.num_blocks,
            block_size=cp.block_size,
            max_batch_size=kwargs.get("max_batch_size", DEFAULT_MAX_BATCH_SIZE),
            max_tokens_per_step=kwargs.get("max_tokens_per_step", DEFAULT_MAX_TOKENS_PER_STEP),
            chunk_size=kwargs.get("chunk_size", DEFAULT_CHUNK_SIZE),
            enable_prefix_cache=kwargs.get("enable_prefix_cache", False),
        )

        # ── Executor ─────────────────────────────────────
        if executor is not None:
            self.executor: Any = executor
        else:
            from engine.mlir_executor import MlirExecutor
            self.executor = MlirExecutor(module, hal_backend)

        # ── Inference loop ───────────────────────────────
        self._loop = InferenceLoop(
            executor=self.executor,
            bridge=self._bridge,
            num_layers=cp.num_layers,
            num_kv_heads=cp.num_kv_heads,
            head_dim=cp.head_dim,
            kv_dtype=cp.dtype,
        )

        # ── Tokenizer ────────────────────────────────────
        self._tokenizer: Tokenizer | None = None
        self._eos_token_id: int | None = None

        # ── Observability ─────────────────────────────────
        self._step_id = 0

    # ── Delegation to InferenceLoop ─────────────────────────────

    @property
    def _loop(self) -> InferenceLoop:
        return self.__loop  # type: ignore[has-type]

    @_loop.setter
    def _loop(self, loop: InferenceLoop) -> None:
        self.__loop = loop

    def step(self) -> list[GenerationResult]:
        return self.__loop.step()

    def _build_cache_hits(self) -> list[tuple[str, list[int], int]]:
        return self.__loop._build_cache_hits()

    # ── Tokenizer ─────────────────────────────────────────────────

    @property
    def is_idle(self) -> bool:
        return not self._bridge.has_work()

    @property
    def num_running(self) -> int:
        return self._bridge.running_count

    @property
    def num_waiting(self) -> int:
        return self._bridge.waiting_count

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

        rid = str(time.monotonic_ns())
        self._bridge.add_request(rid, prompt_tokens, max_tokens)
        self.__loop.add_request(rid, prompt_tokens, SamplingParams(
            temperature=temperature, top_p=top_p, top_k=top_k, max_tokens=max_tokens,
        ))
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
        if self.__loop._uses_static_model():
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
