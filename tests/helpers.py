"""Shared test utilities — use across test files to keep them DRY.

Import from here instead of copy-pasting helpers between test modules.
"""

from __future__ import annotations

import os
import sys
from typing import Any

# ── MLIR bindings detection ──────────────────────────────────


def has_mlir_bindings() -> bool:
    """Check if MLIR Python bindings are available."""
    try:
        import mlir.ir  # noqa: F401

        return True
    except ImportError:
        return False


# ── Transformers monkey-patch ─────────────────────────────────


def patch_transformers_torch() -> None:
    """Ensure transformers can find torch even when symlinked into .venv."""
    import torch

    if hasattr(torch, "__path__"):
        torch_path = os.path.dirname(os.path.abspath(torch.__file__))
        torch_root = os.path.dirname(torch_path)
        site_paths: list[str] = []
        for p in sys.path:
            if "site-packages" in p or "dist-packages" in p:
                site_paths.append(p)
        if torch_root not in sys.path:
            for sp in site_paths:
                if sp not in sys.path:
                    alt = os.path.join(sp, "torch")
                    if os.path.isdir(alt):
                        sys.modules.setdefault("torch", __import__("torch"))
                        break
    if "torch" not in sys.modules:
        sys.modules["torch"] = torch


# ── SSE helpers ───────────────────────────────────────────────


def collect_sse_events(response: Any) -> list[dict[str, Any]]:
    """Parse a streaming SSE response into a list of event dicts.

    Args:
        response: An httpx Response with iter_lines() or iter_text().

    Returns:
        List of parsed JSON events (``data:`` lines only).
    """
    import json

    events: list[dict[str, Any]] = []
    for line in response.iter_lines():
        if line.startswith("data: "):
            payload = line[len("data: ") :]
            if payload.strip() == "[DONE]":
                events.append({"done": True})
            else:
                try:
                    events.append(json.loads(payload))
                except json.JSONDecodeError:
                    pass
    return events


# ── Tokenizer helpers ─────────────────────────────────────────


class SimpleTokenizer:
    """Deterministic tokenizer for testing — maps tokens 1:1 with IDs.

    ``encode(text)`` splits on whitespace and maps to sequential token IDs.
    ``decode(ids)`` joins tokens with spaces.
    """

    _word_to_id: dict[str, int]
    _id_to_word: dict[int, str]
    next_id: int

    def __init__(self, vocab_size: int = 1000) -> None:
        self._word_to_id = {}
        self._id_to_word = {}
        self.next_id = 1

    def _get_id(self, word: str) -> int:
        if word not in self._word_to_id:
            self._word_to_id[word] = self.next_id
            self._id_to_word[self.next_id] = word
            self.next_id += 1
        return self._word_to_id[word]

    def encode(self, text: str) -> list[int]:
        words = text.split()
        return [self._get_id(w) for w in words]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self._id_to_word.get(i, f"<{i}>") for i in ids)


# ── Tensor helpers ────────────────────────────────────────────


def assert_tensors_close(
    a: Any,
    b: Any,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    msg: str = "",
) -> None:
    """Assert two tensors are element-wise close, with a helpful message."""
    import torch

    assert a.shape == b.shape, f"{msg} shape mismatch: {a.shape} != {b.shape}"
    assert torch.allclose(a, b, atol=atol, rtol=rtol), (
        f"{msg} tensors not close: max_diff={(a - b).abs().max().item():.6f}"
    )


def cosine_similarity(a, b) -> float:
    """Compute cosine similarity between two tensors. Returns a float in [-1, 1]."""
    import torch

    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return torch.nn.functional.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)).item()


def assert_cosine_above(
    a: Any,
    b: Any,
    threshold: float = 0.999,
    msg: str = "",
) -> None:
    """Assert cosine similarity between two tensors exceeds threshold."""
    import torch

    a_f = a.float().flatten()
    b_f = b.float().flatten()
    cos = torch.nn.functional.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)).item()
    assert cos >= threshold, f"{msg} cosine {cos:.6f} < {threshold}"


def assert_max_diff_below(
    a: Any,
    b: Any,
    threshold: float = 1e-3,
    msg: str = "",
) -> None:
    """Assert max absolute difference between two tensors is below threshold."""
    diff = float((a.float() - b.float()).abs().max().item())
    assert diff < threshold, f"{msg} max diff {diff:.6f} >= {threshold}"
