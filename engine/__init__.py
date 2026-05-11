"""Inference Runtime Engine — Phase 1 MVP."""

from typing import Any

__all__ = [
    "GenerationResult",
    "LLMEngine",
    "Request",
    "SamplingParams",
    "SequenceGroup",
    "greedy",
    "sample",
]

_LAZY_ATTRS = frozenset(__all__)


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        import engine.batch as _batch
        import engine.llm_engine as _engine
        import engine.sampler as _sampler

        _globals: dict[str, Any] = {
            "GenerationResult": _batch.GenerationResult,
            "Request": _batch.Request,
            "SamplingParams": _batch.SamplingParams,
            "SequenceGroup": _batch.SequenceGroup,
            "LLMEngine": _engine.LLMEngine,
            "greedy": _sampler.greedy,
            "sample": _sampler.sample,
        }
        if name in _globals:
            value = _globals[name]
            globals()[name] = value
            return value
    raise AttributeError(f"module 'engine' has no attribute '{name}'")
