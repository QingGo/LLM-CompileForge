"""Speculative decoding — Phase 2 Module B.

Provides dual-draft speculative decoding engines (MTP + EAGLE) with
rejection-sampling verification and adaptive back-off strategies.

Reference: design-phase2.md §2.2
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "MTPProposer",
    "EAGLEProposer",
    "SpeculativeVerifier",
    "AdaptiveSpeculator",
]

_LAZY_ATTRS = frozenset(__all__)


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        import python_runtime.engine.speculative.adaptive as _adaptive
        import python_runtime.engine.speculative.eagle_proposer as _eagle
        import python_runtime.engine.speculative.mtp_proposer as _mtp
        import python_runtime.engine.speculative.verifier as _verifier

        _globals = {
            "MTPProposer": _mtp.MTPProposer,
            "EAGLEProposer": _eagle.EAGLEProposer,
            "SpeculativeVerifier": _verifier.SpeculativeVerifier,
            "AdaptiveSpeculator": _adaptive.AdaptiveSpeculator,
        }
        if name in _globals:
            value = _globals[name]
            globals()[name] = value
            return value
    raise AttributeError(f"module 'engine.speculative' has no attribute '{name}'")
