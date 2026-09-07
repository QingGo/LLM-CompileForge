"""Tests for the external memory fusion runtime contract."""

from __future__ import annotations

import numpy as np

from python_runtime.memory_fusion import (
    LogDensityRatioGate,
    TokenPlePolicy,
    compute_memory_features,
    fuse_ngram_logits,
)


def test_fuse_ngram_logits_support_only() -> None:
    base = np.zeros(5, dtype=np.float32)
    dist = {1: 0.8, 3: 0.2}
    fused = fuse_ngram_logits(base, dist, scale=1.0)
    np.testing.assert_allclose(fused[1], np.log(0.8), atol=1e-6)
    np.testing.assert_allclose(fused[3], np.log(0.2), atol=1e-6)
    assert fused[0] == 0.0


def test_compute_features() -> None:
    logits = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    features = compute_memory_features(logits, {1: 0.9, 2: 0.1}, matched_order=4)
    assert features["matched_order"] == 4.0
    assert features["memory_top1_agree_base"] == 1.0
    assert features["density_ratio"] > 0.0


def test_gate_activation() -> None:
    gate = LogDensityRatioGate()
    logits = np.array([0.0, 2.0, 0.0], dtype=np.float32)
    assert gate.evaluate(logits, {1: 0.9, 0: 0.1})["active"]


def test_token_policy() -> None:
    data = {
        "feature_names": ["matched_order"],
        "mean": [3.0],
        "std": [1.0],
        "weights": [5.0],
        "bias": -4.0,
        "metrics": {},
    }
    policy = TokenPlePolicy(data)
    assert policy.should_apply({"matched_order": 4.0})
    assert not policy.should_apply({"matched_order": 2.0})
