"""Tests for engine/cache_manager.py — slab allocation, read/write, KV normalize."""

from __future__ import annotations

import pytest
import torch

from python_runtime.engine.cache_manager import CacheManager, _dict_to_proto_cache_policy


def _make_policy(num_layers=4, heads=8, dim=64):
    """Create a proto SfaCachePolicy via JSON fallback (exercises backward-compat path)."""
    return _dict_to_proto_cache_policy({
        "slabs": [
            {
                "slab_id": "k", "storage": "paged",
                "dims": {"layers": num_layers, "heads": heads, "dim": dim},
                "layout": "BNLD", "dtype": "float32",
            },
            {
                "slab_id": "v", "storage": "paged",
                "dims": {"layers": num_layers, "heads": heads, "dim": dim},
                "layout": "BNLD", "dtype": "float32",
            },
        ],
        "intercepts": [
            {
                "slab_id": "k", "op_name": "scaled_dot_product_attention",
                "direction": "read_write", "source": "operand[1]",
                "layer": "sequential",
            },
            {
                "slab_id": "v", "op_name": "scaled_dot_product_attention",
                "direction": "read_write", "source": "operand[2]",
                "layer": "sequential",
            },
        ],
        "block_size": 16,
        "max_requests": 256,
    })


def _make_mgr(num_blocks=16):
    return CacheManager(_make_policy(), num_blocks=num_blocks)


@pytest.mark.unit
class TestKvNormalize:
    """P0-4: KV normalize should handle all SDPA tensor formats."""

    def test_format_batch_heads_seq_dim(self) -> None:
        """[batch, heads, seq, dim] → squeeze batch → permute → [seq, heads, dim]."""
        mgr = _make_mgr()
        data = torch.randn(1, 8, 4, 64)
        norm = mgr._normalize_kv(data, "k")
        assert norm.shape == (4, 8, 64)

    def test_format_batch_seq_heads_dim(self) -> None:
        """[batch, seq, heads, dim] → squeeze batch → [seq, heads, dim] already correct."""
        mgr = _make_mgr()
        data = torch.randn(1, 4, 8, 64)
        norm = mgr._normalize_kv(data, "k")
        assert norm.shape == (4, 8, 64)

    def test_format_seq_heads_dim(self) -> None:
        """[seq, heads, dim] → pass through."""
        mgr = _make_mgr()
        data = torch.randn(4, 8, 64)
        norm = mgr._normalize_kv(data, "k")
        assert norm.shape == (4, 8, 64)

    def test_format_seq_hidden(self) -> None:
        """[seq, hidden] where hidden = heads * dim → reshape."""
        mgr = _make_mgr()
        data = torch.randn(4, 8 * 64)
        norm = mgr._normalize_kv(data, "k")
        assert norm.shape == (4, 8, 64)

    def test_bfloat16_preserved(self) -> None:
        """bfloat16 dtype should be preserved."""
        mgr = _make_mgr()
        data = torch.randn(1, 8, 4, 64, dtype=torch.bfloat16)
        norm = mgr._normalize_kv(data, "k")
        assert norm.dtype == torch.bfloat16


@pytest.mark.unit
class TestCacheManagerBasic:
    """CacheManager slab allocation and read/write roundtrip."""

    def test_slab_shape(self) -> None:
        mgr = CacheManager(_make_policy(4, 8, 64), num_blocks=10)
        assert mgr._slabs["k"].shape == (10, 4, 16, 8, 64)
        assert mgr._slabs["v"].shape == (10, 4, 16, 8, 64)

    def test_write_read_roundtrip(self) -> None:
        """Write K/V → read back → values match."""
        mgr = _make_mgr()
        bt = {"r0": [0, 1]}
        mgr.begin_step(bt)

        k_data = torch.randn(6, 8, 64)  # 6 tokens, prefill
        v_data = torch.randn(6, 8, 64)
        positions = torch.tensor([0, 1, 2, 3, 4, 5])

        mgr.write_paged("k", 0, k_data, positions)
        mgr.write_paged("v", 0, v_data, positions)

        k_read = mgr.read_paged("k", 0, max_seq_len=6)
        v_read = mgr.read_paged("v", 0, max_seq_len=6)
        assert k_read.shape == (1, 6, 8, 64)
        assert torch.allclose(k_read[0, :6], k_data)
        assert torch.allclose(v_read[0, :6], v_data)

    def test_layer_isolation(self) -> None:
        """Layer 0 and layer 1 should be independent."""
        mgr = _make_mgr()
        bt = {"r0": [0]}
        mgr.begin_step(bt)

        k0 = torch.ones(1, 8, 64)
        k1 = torch.ones(1, 8, 64) * 2.0
        pos = torch.tensor([0])

        mgr.write_paged("k", 0, k0, pos)
        mgr.write_paged("k", 1, k1, pos)

        r0 = mgr.read_paged("k", 0, max_seq_len=1)
        r1 = mgr.read_paged("k", 1, max_seq_len=1)
        assert torch.allclose(r0, torch.ones(1, 1, 8, 64) * 1.0)
        assert torch.allclose(r1, torch.ones(1, 1, 8, 64) * 2.0)

    def test_block_boundary(self) -> None:
        """Write across block boundary — positions correctly mapped."""
        mgr = _make_mgr()
        bt = {"r0": [0, 1]}
        mgr.begin_step(bt)

        k_data = torch.randn(32, 8, 64)  # spans 2 blocks of 16
        positions = torch.arange(0, 32)

        mgr.write_paged("k", 0, k_data, positions)
        k_read = mgr.read_paged("k", 0, max_seq_len=32)
        assert torch.allclose(k_read[0], k_data)

    def test_multi_request(self) -> None:
        """Two requests with sequential position ranges in read."""
        mgr = _make_mgr()
        # r0 uses block 0, r1 uses block 1. Position 0 writes r0, position 16 writes r1.
        bt = {"r0": [0, 1], "r1": [2]}
        mgr.begin_step(bt)

        k0 = torch.randn(16, 8, 64)
        k1 = torch.randn(1, 8, 64)

        mgr.write_paged("k", 0, k0, torch.arange(0, 16))
        mgr.write_paged("k", 0, k1, torch.tensor([16]))

        result = mgr.read_paged("k", 0, max_seq_len=17)
        assert result.shape[0] == 2
        assert torch.allclose(result[0, :16], k0)  # r0's first 16 tokens
        assert not torch.allclose(result[0, 0], result[1, 0])  # different requests

    def test_begin_step_resets_layers(self) -> None:
        """begin_step should reset layer counters."""
        mgr = _make_mgr()
        mgr.begin_step({})
        assert mgr._layer_counters["k"] == 0
        mgr.resolve_layer("k")
        mgr.resolve_layer("k")
        assert mgr._layer_counters["k"] == 2
        mgr.begin_step({})
        assert mgr._layer_counters["k"] == 0

    def test_fixed_slab_rwkv(self) -> None:
        """RWKV fixed-size slab allocation and I/O."""
        policy = _dict_to_proto_cache_policy({
            "slabs": [
                {
                    "slab_id": "state", "storage": "fixed",
                    "dims": {"layers": 4, "dim": 1024},
                    "layout": "RLD", "dtype": "float32",
                },
            ],
            "intercepts": [
                {
                    "slab_id": "state", "op_name": "state_evolve",
                    "direction": "read_write", "source": "output",
                    "layer": "sequential",
                },
            ],
            "block_size": 16,
            "max_requests": 256,
        })
        mgr = CacheManager(policy, num_blocks=1)
        assert mgr._slabs["state"].shape == (256, 4, 1024)

        data = torch.randn(1024)
        mgr.write_fixed("state", 0, 0, data)
        result = mgr.read_fixed("state", 0, 0)
        assert torch.allclose(result.float(), data)
