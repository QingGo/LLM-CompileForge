"""Tests for LLMEngine.generate() KV-cache decode path.

Phase P1 (performance-decode-o1): generate() on a KV-capable executor must
run decode steps with a single-token input (seq=1) driven by the Rust
scheduler's KV mode, instead of re-running the whole sequence (O(n^2)).

The tests use a deterministic toy executor whose next-token logits are a
function of the full prefix, so the KV path and the legacy full-sequence
path must agree token-for-token — any position/context wiring bug shows
up as a divergence.  No compiled model is required for the fast tests;
the real-model consistency test is marked ``integration`` and skips when
the opt_125m_kv artifact is missing.
"""

from __future__ import annotations

import pytest
import torch

from compiler.artifact import MlirFunction, MlirModule, MlirOp
from python_runtime.engine.llm_engine import LLMEngine
from python_runtime.hal.pytorch_backend import PyTorchBackend

_VOCAB = 16


def _make_test_mlir() -> MlirModule:
    """Minimal MlirModule placeholder — execution is faked by the executor."""
    return MlirModule(
        functions=[
            MlirFunction(
                name="main",
                inputs=[("%input_ids", "tensor<?x?xi64>")],
                outputs=[("%logits", "tensor<?xf32>", False)],
                ops=[
                    MlirOp(
                        name="sf.identity",
                        dialect="sf",
                        op_name="identity",
                        operands=["%input_ids"],
                        results=["%logits"],
                    ),
                ],
            )
        ]
    )


class _ToyExecutor:
    """Deterministic fake executor.

    Maintains a position→token context map and emits logits as a
    one-hot at ``(sum(last 3 context tokens) + last_pos) % vocab``.
    This makes next-token prediction depend on the whole prefix, so the
    KV path (single-token decode + absolute positions) and the legacy
    path (full-sequence recompute) must produce identical token streams.
    """

    def __init__(self, use_cache_manager: bool):
        self._uses_cache_manager = use_cache_manager
        self._uses_static_shape = False
        self._seq: dict[int, int] = {}
        # (input_shape, positions_list, is_decode) per forward call
        self.forward_calls: list[tuple[tuple[int, ...], list[int] | None, bool]] = []

    def _update_context(self, input_ids: torch.Tensor, positions: torch.Tensor | None) -> None:
        ids = [int(t) for t in input_ids.reshape(-1).tolist()]
        if positions is None:
            start = max(self._seq.keys(), default=-1) + 1
            pos_list = list(range(start, start + len(ids)))
        else:
            pos_list = [int(p) for p in positions.reshape(-1).tolist()]
        for pos, tok in zip(pos_list, ids, strict=False):
            self._seq[pos] = tok

    def _logits(self, last_pos: int) -> torch.Tensor:
        ctx = [self._seq[p] for p in range(last_pos + 1)]
        target = (sum(ctx[-3:]) + last_pos) % _VOCAB
        logits = torch.zeros(1, 1, _VOCAB)
        logits[0, 0, target] = 1.0
        return logits

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor | None = None, **_: object) -> torch.Tensor:
        pos_list = None if positions is None else [int(p) for p in positions.reshape(-1).tolist()]
        self.forward_calls.append((tuple(input_ids.shape), pos_list, False))
        self._update_context(input_ids, positions)
        last = pos_list[-1] if pos_list else max(self._seq.keys())
        return self._logits(last)

    def forward_with_kv(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        is_decode: bool = False,
        **_: object,
    ) -> tuple[torch.Tensor, list[object]]:
        pos_list = None if positions is None else [int(p) for p in positions.reshape(-1).tolist()]
        self.forward_calls.append((tuple(input_ids.shape), pos_list, bool(is_decode)))
        self._update_context(input_ids, positions)
        last = pos_list[-1] if pos_list else max(self._seq.keys())
        return self._logits(last), []

    def set_kv_cache(self, **_: object) -> None:
        pass


def _make_engine(executor: _ToyExecutor, **kwargs: object) -> LLMEngine:
    mod = _make_test_mlir()
    engine = LLMEngine(mod, PyTorchBackend("cpu"), executor=executor, max_batch_size=4, **kwargs)
    return engine


@pytest.mark.unit
class TestGenerateKvShapes:
    """decode steps must run with seq=1 inputs (the O(1) shape assertion)."""

    def test_decode_steps_are_single_token(self) -> None:
        ex = _ToyExecutor(use_cache_manager=True)
        engine = _make_engine(ex)
        engine.generate([5, 6, 7], max_tokens=5, temperature=0)

        assert ex.forward_calls, "generate() must run at least the prefill"
        prefill = ex.forward_calls[0]
        assert prefill[0] == (1, 3), f"prefill shape must be [1, prompt_len], got {prefill[0]}"
        assert prefill[1] == [0, 1, 2], f"prefill positions must be absolute 0..2, got {prefill[1]}"

        decode_calls = ex.forward_calls[1:]
        assert len(decode_calls) == 4, f"expected 4 decode steps for max_tokens=5, got {len(decode_calls)}"
        for step, (shape, pos, is_decode) in enumerate(decode_calls):
            assert shape == (1, 1), f"decode step {step} must be seq=1, got {shape}"
            assert is_decode is True, f"decode step {step} must be flagged is_decode"
            assert pos == [3 + step], f"decode step {step} must feed absolute position {3 + step}, got {pos}"

    def test_prefill_is_not_flagged_decode(self) -> None:
        ex = _ToyExecutor(use_cache_manager=True)
        engine = _make_engine(ex)
        engine.generate([1, 2], max_tokens=2, temperature=0)
        prefill = ex.forward_calls[0]
        assert prefill[2] is False, "prefill chunk (n>1 tokens) must not be flagged decode"


@pytest.mark.unit
class TestGenerateKvConsistency:
    """KV path and legacy full-sequence path must agree token-for-token."""

    def test_kv_matches_legacy_tokens(self) -> None:
        prompt = [1, 2, 3, 4, 5, 6, 7]
        kv_engine = _make_engine(_ToyExecutor(use_cache_manager=True))
        legacy_engine = _make_engine(_ToyExecutor(use_cache_manager=False))

        out_kv = kv_engine.generate(prompt, max_tokens=8, temperature=0)
        out_legacy = legacy_engine.generate(prompt, max_tokens=8, temperature=0)
        assert out_kv == out_legacy, f"KV decode diverged from legacy: {out_kv!r} != {out_legacy!r}"

    def test_semantics_max_tokens_and_eos(self) -> None:
        """max_tokens caps generation; eos stops it early (loop path)."""
        # max_tokens: 1 prefill sample + 3 decode steps = 4 forwards.
        ex = _ToyExecutor(use_cache_manager=True)
        engine = _make_engine(ex)
        engine.generate([1, 2, 3], max_tokens=4, temperature=0)
        assert len(ex.forward_calls) == 4

        # eos: executor always emits token 0 == eos → stop after prefill.
        eos_ex = _EosExecutor()
        eos_engine = _make_engine(eos_ex)
        eos_engine.set_tokenizer(_tokenizer(), eos_token_id=0)
        out = eos_engine.generate([1, 2, 3], max_tokens=100, temperature=0)
        assert out == "<0>", f"eos must stop generation after one token, got {out!r}"
        assert len(eos_ex.forward_calls) == 1, "eos sampled at prefill — no decode steps expected"


class _EosExecutor(_ToyExecutor):
    """Toy executor that always emits token 0 (used as the eos id)."""

    def __init__(self) -> None:
        super().__init__(use_cache_manager=True)

    def _logits(self, last_pos: int) -> torch.Tensor:
        logits = torch.zeros(1, 1, _VOCAB)
        logits[0, 0, 0] = 1.0
        return logits


def _tokenizer() -> object:
    from tests.helpers import SimpleTokenizer

    return SimpleTokenizer()


@pytest.mark.unit
class TestGenerateLegacyFallback:
    """Executors without a cache manager keep the legacy full-sequence path."""

    def test_legacy_decode_sends_full_sequence(self) -> None:
        ex = _ToyExecutor(use_cache_manager=False)
        engine = _make_engine(ex)
        engine.generate([5, 6, 7], max_tokens=4, temperature=0)

        shapes = [shape for shape, _, _ in ex.forward_calls]
        assert shapes[0] == (1, 3), f"prefill shape, got {shapes[0]}"
        # Legacy path re-runs the whole sequence each decode step.
        assert shapes[1:] == [(1, 4), (1, 5), (1, 6)], f"legacy decode shapes, got {shapes[1:]}"

    def test_legacy_produces_tokens(self) -> None:
        ex = _ToyExecutor(use_cache_manager=False)
        engine = _make_engine(ex)
        out = engine.generate([1, 2], max_tokens=3, temperature=0)
        tokens = [int(t) for t in out.split()]
        assert len(tokens) == 3
        assert all(0 <= t < _VOCAB for t in tokens)


@pytest.mark.integration
@pytest.mark.timeout(600)
class TestGenerateKvRealModel:
    """Real-model KV vs legacy consistency on the opt_125m_kv artifact."""

    @staticmethod
    def _artifact_path() -> str:
        import os

        return os.path.join(
            os.path.dirname(__file__),
            "..", "..", "..", "outputs", "compiled", "opt_125m_kv",
        )

    @staticmethod
    def _load_engine(use_cache_manager: bool) -> LLMEngine:
        from compiler.serialize import load_artifact
        from python_runtime.engine.mlir_executor import MlirExecutor

        module = load_artifact(TestGenerateKvRealModel._artifact_path())
        backend = PyTorchBackend("cpu")
        executor = MlirExecutor(module, backend)
        if not use_cache_manager:
            executor._uses_cache_manager = False  # force legacy recompute path
            executor._cache_mgr = None
        return LLMEngine(module, backend, executor=executor, num_blocks=256, max_batch_size=2)

    @staticmethod
    def _tokens(out: str) -> list[int]:
        return [int(t) for t in out.split()]

    def test_kv_matches_legacy_token_for_token(self) -> None:
        import os

        if not os.path.isdir(self._artifact_path()):
            pytest.skip("opt_125m_kv artifact missing — run `make rebuild-kv` first")

        prompt = [2, 31414, 6, 232, 328, 7181, 87, 9]
        kv_engine = self._load_engine(use_cache_manager=True)
        legacy_engine = self._load_engine(use_cache_manager=False)

        # 18 tokens: crosses the 16-token block boundary, exercising the
        # scheduler's decode-time block-table growth.
        kv_out = self._tokens(kv_engine.generate(prompt, max_tokens=18, temperature=0))
        legacy_out = self._tokens(legacy_engine.generate(prompt, max_tokens=18, temperature=0))

        assert kv_out == legacy_out, f"real-model KV decode diverged: {kv_out} != {legacy_out}"
