"""Runtime contract for auditable external n-gram memory fusion.

This module defines the Python reference for the external-memory logit
correction that the qwen35-ple project prototyped:

    fused_logits = base_logits + scale * log p_memory + bias

It is intentionally NumPy-only so the Rust runtime and compiler can mirror the
same math without depending on engram-peft.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def fuse_ngram_logits(
    base_logits: np.ndarray,
    ngram_probs: dict[int, float] | None,
    *,
    scale: float = 1.0,
    bias: float = 0.0,
    temperature: float = 1.0,
) -> np.ndarray:
    """Add a calibrated sparse n-gram log-prior to base logits."""
    out = base_logits.copy()
    if not ngram_probs:
        return out
    inv_t = 1.0 / max(temperature, 1e-6)
    for tok, p in ngram_probs.items():
        if 0 <= tok < len(out) and p > 0:
            out[tok] += scale * (math.log(p) * inv_t) + bias
    return out


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    m = float(np.max(logits))
    z = logits - m
    return z - math.log(float(np.exp(z).sum()))


def compute_memory_features(
    logits: Any,
    dist: dict[int, float] | None,
    *,
    matched_order: int | None = None,
) -> dict[str, float]:
    """Compute auditable memory features consumed by the token policy."""
    if not dist:
        return {
            "matched_order": float(matched_order or 0),
            "base_entropy": math.inf,
            "memory_entropy": math.inf,
            "density_ratio": -math.inf,
            "base_top1_prob": 0.0,
            "memory_top1_prob": 0.0,
            "memory_top1_agree_base": 0.0,
        }
    logits_np = _to_numpy(logits)
    log_pb = _log_softmax(logits_np)
    pb = np.exp(log_pb)
    total = sum(float(v) for v in dist.values() if v > 0)
    if total <= 0:
        pm = {}
    else:
        pm = {
            int(k): float(v) / total
            for k, v in dist.items()
            if int(k) >= 0 and v > 0
        }
    expected = sum(
        p * (math.log(p) - float(log_pb[tok]))
        for tok, p in pm.items()
        if 0 <= tok < len(log_pb) and p > 0
    )
    base_top1 = int(np.argmax(logits_np)) if len(logits_np) else 0
    mem_top1 = max(pm, key=pm.get) if pm else 0
    return {
        "matched_order": float(matched_order or 0),
        "base_entropy": float(-np.sum(pb * np.log(np.maximum(pb, 1e-12)))),
        "memory_entropy": float(-sum(p * math.log(p) for p in pm.values() if p > 0)),
        "density_ratio": float(expected),
        "base_top1_prob": float(pb[base_top1]) if len(pb) else 0.0,
        "memory_top1_prob": float(pm[mem_top1]) if pm else 0.0,
        "memory_top1_agree_base": 1.0 if mem_top1 == base_top1 else 0.0,
    }


class LogDensityRatioGate:
    """Gate that controls whether external memory fusion is active."""

    def __init__(self, mode: str = "expected_kl", threshold: float = 0.0) -> None:
        if mode not in {"expected_kl", "pseudo_label", "memory_top1", "hybrid"}:
            raise ValueError(f"unknown gate mode: {mode}")
        self.mode = mode
        self.threshold = float(threshold)

    def evaluate(
        self,
        base_logits: Any,
        ngram_probs: dict[int, float] | None,
        *,
        memory_order: int | None = None,
    ) -> dict[str, Any]:
        if not ngram_probs:
            return {"active": False, "mode": self.mode, "threshold": self.threshold}
        logits = _to_numpy(base_logits)
        log_pb = _log_softmax(logits)
        pm = {int(k): float(v) for k, v in ngram_probs.items() if int(k) >= 0}
        total = sum(pm.values())
        if total > 0:
            pm = {k: v / total for k, v in pm.items()}
        expected = sum(
            p * (math.log(p) - float(log_pb[tok]))
            for tok, p in pm.items()
            if 0 <= tok < len(log_pb) and p > 0
        )
        base_argmax = int(np.argmax(logits))
        mem_argmax = max(pm, key=pm.get) if pm else 0
        pseudo = (
            math.log(pm[base_argmax]) - float(log_pb[base_argmax])
            if base_argmax in pm and pm[base_argmax] > 0
            else -math.inf
        )
        mem_top1 = (
            math.log(pm[mem_argmax]) - float(log_pb[mem_argmax])
            if pm and 0 <= mem_argmax < len(log_pb)
            else -math.inf
        )
        if self.mode == "expected_kl":
            active = expected >= self.threshold
        elif self.mode == "pseudo_label":
            active = pseudo >= self.threshold
        elif self.mode == "memory_top1":
            active = mem_top1 >= self.threshold
        else:
            active = expected >= self.threshold and pseudo >= 0.0
        return {
            "active": active,
            "mode": self.mode,
            "threshold": self.threshold,
            "matched_order": memory_order,
            "expected_log_density_ratio": float(expected),
            "pseudo_label_log_density_ratio": float(pseudo),
            "memory_top1_log_density_ratio": float(mem_top1),
        }


class TokenPlePolicy:
    """Logistic token-level safety policy loaded from a JSON dict/file."""

    def __init__(self, data: dict[str, Any] | str) -> None:
        if isinstance(data, str):
            import json
            from pathlib import Path

            data = json.loads(Path(data).read_text(encoding="utf-8"))
        self.feature_names = list(data["feature_names"])
        self.mean = np.asarray(data["mean"], dtype=np.float64)
        self.std = np.asarray(data["std"], dtype=np.float64)
        self.weights = np.asarray(data["weights"], dtype=np.float64)
        self.bias = float(data["bias"])

    def predict(self, features: dict[str, Any]) -> float:
        x = np.asarray(
            [float(features.get(name, 0.0)) for name in self.feature_names],
            dtype=np.float64,
        )
        x = (x - self.mean) / self.std
        z = float(x @ self.weights + self.bias)
        return float(1.0 / (1.0 + math.exp(-max(min(z, 30.0), -30.0))))

    def should_apply(self, features: dict[str, Any], threshold: float = 0.5) -> bool:
        return self.predict(features) >= threshold


__all__ = [
    "LogDensityRatioGate",
    "TokenPlePolicy",
    "compute_memory_features",
    "fuse_ngram_logits",
]
