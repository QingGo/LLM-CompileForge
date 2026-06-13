"""Tests for CachePolicy metadata serialization in compiled artifacts.

Verifies that:
  1. When compile_mlir() receives a cache_policy, metadata.json contains it
  2. The cache_policy JSON matches CachePolicy.to_dict() output
  3. Without cache_policy, no cache_policy key in metadata (backward compat)
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest
import torch
import torch.nn as nn

from compiler.cache_policy import CachePolicy

# ── Test helpers ──────────────────────────────────────────────


class SimpleLinear(nn.Module):
    """Minimal exportable model for fast compilation tests."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _make_model() -> tuple[nn.Module, torch.Tensor]:
    model = SimpleLinear().eval()
    x = torch.randn(2, 8)
    return model, x


# ── Tests ─────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.timeout(15)
def test_cache_policy_in_metadata() -> None:
    """Compile with CachePolicy and verify metadata.json contains it."""
    from compiler.pipeline import compile_mlir

    model, x = _make_model()
    policy = CachePolicy.for_llama(num_layers=1, num_kv_heads=2, head_dim=8)
    policy_dict = policy.to_dict()

    with tempfile.TemporaryDirectory() as tmpdir:
        compile_mlir(
            model,
            example_args=(x,),
            output_dir=tmpdir,
            apply_fusion=False,
            cache_policy=policy,
        )
        meta_path = os.path.join(tmpdir, "metadata.json")
        meta = json.load(open(meta_path))

        # cache_policy key must be present
        assert "cache_policy" in meta, "metadata.json missing cache_policy key"

        cp = meta["cache_policy"]

        # Structure matches CachePolicy.to_dict()
        assert cp == policy_dict, f"cache_policy mismatch:\n  expected={policy_dict}\n  got={cp}"

        # Specific field checks
        assert cp["block_size"] == 16
        assert cp["max_requests"] == 256
        assert len(cp["slabs"]) == 2
        assert len(cp["intercepts"]) == 2
        assert cp["slabs"][0]["slab_id"] == "k"
        assert cp["slabs"][1]["slab_id"] == "v"
        assert cp["intercepts"][0]["slab_id"] == "k"
        assert cp["intercepts"][1]["slab_id"] == "v"

        # to_dict roundtrip
        recovered = CachePolicy.from_dict(cp)
        assert recovered.to_dict() == policy_dict


@pytest.mark.unit
@pytest.mark.timeout(15)
def test_cache_policy_backward_compat() -> None:
    """Without cache_policy, metadata.json must not have a cache_policy key."""
    from compiler.pipeline import compile_mlir

    model, x = _make_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        compile_mlir(
            model,
            example_args=(x,),
            output_dir=tmpdir,
            apply_fusion=False,
            # no cache_policy
        )
        meta_path = os.path.join(tmpdir, "metadata.json")
        meta = json.load(open(meta_path))

        assert "cache_policy" not in meta, "cache_policy key should be absent when no CachePolicy provided"


@pytest.mark.unit
@pytest.mark.timeout(15)
def test_cache_policy_none_explicit() -> None:
    """Explicit cache_policy=None must also omit cache_policy key."""
    from compiler.pipeline import compile_mlir

    model, x = _make_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        compile_mlir(
            model,
            example_args=(x,),
            output_dir=tmpdir,
            apply_fusion=False,
            cache_policy=None,
        )
        meta_path = os.path.join(tmpdir, "metadata.json")
        meta = json.load(open(meta_path))

        assert "cache_policy" not in meta, "cache_policy key should be absent when cache_policy=None"


@pytest.mark.unit
@pytest.mark.timeout(15)
def test_cache_policy_empty_policy() -> None:
    """Empty CachePolicy to_dict() must serialize correctly."""
    from compiler.pipeline import compile_mlir

    model, x = _make_model()
    policy = CachePolicy()  # empty — no slabs, no intercepts

    with tempfile.TemporaryDirectory() as tmpdir:
        compile_mlir(
            model,
            example_args=(x,),
            output_dir=tmpdir,
            apply_fusion=False,
            cache_policy=policy,
        )
        meta_path = os.path.join(tmpdir, "metadata.json")
        meta = json.load(open(meta_path))

        assert "cache_policy" in meta
        cp = meta["cache_policy"]
        assert cp["slabs"] == []
        assert cp["intercepts"] == []
        assert cp["block_size"] == 16
        assert cp["max_requests"] == 256
