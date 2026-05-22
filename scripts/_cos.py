"""Shared cosine similarity with shape alignment.

Usage:
    from scripts._cos import cosine_similarity, safe_cos
"""

from __future__ import annotations

import logging

import numpy as np

_log = logging.getLogger("cos")


def cosine_similarity(a, b) -> float:
    """Cosine similarity between two arrays. Supports numpy arrays and torch tensors."""
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    if hasattr(b, "detach"):
        b = b.detach().cpu().numpy()
    a_f = np.asarray(a).ravel().astype(np.float64)
    b_f = np.asarray(b).ravel().astype(np.float64)
    dot = np.dot(a_f, b_f)
    norm_a = np.linalg.norm(a_f)
    norm_b = np.linalg.norm(b_f)
    if norm_a == 0.0 and norm_b == 0.0:
        return 1.0
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def safe_cos(a: np.ndarray, b: np.ndarray) -> float:  # type: ignore[type-arg]
    """Compute cosine similarity with shape alignment check.

    If shapes differ, warns but still attempts alignment:
    - Flattens both to 1-D
    - Truncates to min length
    - Casts to float64 for precision

    Args:
        a: First array.
        b: Second array.

    Returns:
        Cosine similarity in [0, 1].
    """
    if a.shape != b.shape:
        a_flat = a.ravel()
        b_flat = b.ravel()
        min_len = min(len(a_flat), len(b_flat))
        _log.warning(
            "cos(): shape mismatch %s vs %s — truncating to %d elements",
            a.shape, b.shape, min_len,
        )
        af = a_flat[:min_len].astype(np.float64)
        bf = b_flat[:min_len].astype(np.float64)
    else:
        af = a.ravel().astype(np.float64)
        bf = b.ravel().astype(np.float64)

    denom = np.linalg.norm(af) * np.linalg.norm(bf)
    return float(np.dot(af, bf) / (denom + 1e-12))
