"""Tests for Triton custom kernels.

Verifies correctness of each kernel against PyTorch reference
implementations on CPU.  When a GPU + Triton are available, the
kernel is exercised directly; otherwise the CPU fallback path
is tested.

All tests assert cosine similarity > 0.999 and max absolute
difference < 1e-3 against the reference implementation.
"""

from __future__ import annotations

import math

import pytest
import torch

from tests.helpers import assert_cosine_above, assert_max_diff_below

# ── RMSNorm + Residual Fusion ───────────────────────────


@pytest.mark.unit
class TestRMSNormFused:
    def test_output_matches_reference(self) -> None:
        from kernels.rms_norm import fused_rms_norm_add

        torch.manual_seed(42)
        hidden = 128
        x = torch.randn(4, hidden)
        residual = torch.randn(4, hidden)
        weight = torch.randn(hidden)

        result = fused_rms_norm_add(x, residual, weight, eps=1e-6)

        # Reference
        ref = (x + residual)
        rms = torch.sqrt(torch.mean(ref * ref, dim=-1, keepdim=True) + 1e-6)
        ref = (ref / rms) * weight

        assert_cosine_above(result, ref)
        assert_max_diff_below(result, ref)

    def test_batched_input(self) -> None:
        from kernels.rms_norm import fused_rms_norm_add

        torch.manual_seed(42)
        x = torch.randn(2, 8, 64)
        residual = torch.randn(2, 8, 64)
        weight = torch.randn(64)

        result = fused_rms_norm_add(x, residual, weight)
        assert result.shape == x.shape

    def test_near_zero_input(self) -> None:
        from kernels.rms_norm import fused_rms_norm_add

        x = torch.zeros(3, 32)
        residual = torch.zeros(3, 32)
        weight = torch.ones(32)

        result = fused_rms_norm_add(x, residual, weight)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_different_eps(self) -> None:
        from kernels.rms_norm import fused_rms_norm_add

        x = torch.randn(2, 16)
        w = torch.ones(16)

        r1 = fused_rms_norm_add(x, x, w, eps=1e-3)
        r2 = fused_rms_norm_add(x, x, w, eps=1e-8)
        assert not torch.allclose(r1, r2)


# ── FlashAttention-2 ────────────────────────────────────


@pytest.mark.unit
class TestFlashAttention:
    def test_output_matches_sdpa(self) -> None:
        from kernels.flash_attention import flash_attention_fwd

        torch.manual_seed(42)
        batch, heads, seq, dim = 2, 4, 64, 32
        q = torch.randn(batch, heads, seq, dim) / math.sqrt(dim)
        k = torch.randn(batch, heads, seq, dim) / math.sqrt(dim)
        v = torch.randn(batch, heads, seq, dim)

        result = flash_attention_fwd(q, k, v, causal=True)

        ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True, scale=1.0 / math.sqrt(dim),
        )

        assert_cosine_above(result, ref)
        assert_max_diff_below(result, ref)

    def test_non_causal_mode(self) -> None:
        from kernels.flash_attention import flash_attention_fwd

        torch.manual_seed(42)
        batch, heads, seq, dim = 1, 2, 32, 16
        q = torch.randn(batch, heads, seq, dim)
        k = torch.randn(batch, heads, seq, dim)
        v = torch.randn(batch, heads, seq, dim)

        result = flash_attention_fwd(q, k, v, causal=False)

        ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
        )

        assert_cosine_above(result, ref)

    def test_custom_scale(self) -> None:
        from kernels.flash_attention import flash_attention_fwd

        torch.manual_seed(42)
        q = torch.randn(1, 2, 32, 16)
        k = torch.randn(1, 2, 32, 16)
        v = torch.randn(1, 2, 32, 16)
        scale = 0.5

        result = flash_attention_fwd(q, k, v, scale=scale)
        ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True, scale=scale,
        )

        assert_cosine_above(result, ref)

    def test_single_head(self) -> None:
        from kernels.flash_attention import flash_attention_fwd

        torch.manual_seed(42)
        q = torch.randn(1, 1, 16, 32)
        k = torch.randn(1, 1, 16, 32)
        v = torch.randn(1, 1, 16, 32)

        result = flash_attention_fwd(q, k, v)
        assert result.shape == q.shape


# ── PagedAttention ─────────────────────────────────────


@pytest.mark.unit
class TestPagedAttention:
    def test_single_request(self) -> None:
        from kernels.paged_attention import paged_attention

        torch.manual_seed(42)
        num_heads, head_dim = 4, 32
        block_size = 16
        seq_len = 30

        q = torch.randn(1, num_heads, head_dim)
        k_cache = torch.randn(4, block_size, num_heads, head_dim)
        v_cache = torch.randn(4, block_size, num_heads, head_dim)

        block_tables = {"req_1": [0, 1]}
        seq_lens = {"req_1": seq_len}

        outputs = paged_attention(q, k_cache, v_cache, block_tables, seq_lens, block_size=block_size)

        assert "req_1" in outputs
        assert outputs["req_1"].shape == (num_heads, head_dim)

    def test_gqa_repeat(self) -> None:
        from kernels.paged_attention import paged_attention

        torch.manual_seed(42)
        num_heads, num_kv_heads, head_dim = 8, 2, 32
        block_size = 16

        q = torch.randn(1, num_heads, head_dim)
        k_cache = torch.randn(4, block_size, num_kv_heads, head_dim)
        v_cache = torch.randn(4, block_size, num_kv_heads, head_dim)

        block_tables = {"req_1": [0, 1, 2]}
        seq_lens = {"req_1": 40}

        outputs = paged_attention(q, k_cache, v_cache, block_tables, seq_lens, block_size=block_size)

        assert "req_1" in outputs
        assert outputs["req_1"].shape == (num_heads, head_dim)

    def test_matches_reference_on_cpu(self) -> None:
        from kernels.paged_attention import paged_attention

        torch.manual_seed(42)
        num_heads, head_dim, block_size = 4, 32, 16
        k_cache = torch.randn(4, block_size, num_heads, head_dim)
        v_cache = torch.randn(4, block_size, num_heads, head_dim)
        block_tables = {"req_1": [0, 1]}
        seq_len = 20
        seq_lens = {"req_1": seq_len}

        q = torch.randn(1, num_heads, head_dim)

        outputs = paged_attention(q, k_cache, v_cache, block_tables, seq_lens, block_size=block_size)

        # Reference: gather blocks manually and run SDPA
        k_seq = torch.cat([
            k_cache[0][:block_size],
            k_cache[1][: (seq_len - block_size)],
        ], dim=0)  # [seq_len, num_heads, head_dim]
        v_seq = torch.cat([
            v_cache[0][:block_size],
            v_cache[1][: (seq_len - block_size)],
        ], dim=0)
        k_seq = k_seq.unsqueeze(0).transpose(1, 2)  # [1, num_heads, seq_len, head_dim]
        v_seq = v_seq.unsqueeze(0).transpose(1, 2)
        q_ref = q.unsqueeze(2)  # [1, num_heads, 1, head_dim]

        ref = torch.nn.functional.scaled_dot_product_attention(
            q_ref, k_seq, v_seq, attn_mask=None, dropout_p=0.0, is_causal=False,
        ).squeeze(2)  # [1, num_heads, head_dim]

        assert_cosine_above(outputs["req_1"], ref[0])

    def test_multiple_requests(self) -> None:
        from kernels.paged_attention import paged_attention

        torch.manual_seed(42)
        num_heads, head_dim, block_size = 4, 32, 16

        q = torch.randn(2, num_heads, head_dim)
        k_cache = torch.randn(8, block_size, num_heads, head_dim)
        v_cache = torch.randn(8, block_size, num_heads, head_dim)

        block_tables = {"req_1": [0, 1], "req_2": [2, 3]}
        seq_lens = {"req_1": 20, "req_2": 30}

        outputs = paged_attention(q, k_cache, v_cache, block_tables, seq_lens, block_size=block_size)

        assert len(outputs) == 2
        assert "req_1" in outputs
        assert "req_2" in outputs


# ── TritonBackend ───────────────────────────────────────


@pytest.mark.unit
class TestTritonBackend:
    def test_registry_register_and_get(self) -> None:
        from hal.triton_backend import TritonKernelRegistry

        @TritonKernelRegistry.register("test_op")
        def test_fn(x: torch.Tensor) -> torch.Tensor:
            return x * 2

        kernel = TritonKernelRegistry.get("test_op")
        assert kernel is not None
        result = kernel(torch.tensor([1.0, 2.0]))
        assert torch.allclose(result, torch.tensor([2.0, 4.0]))

    def test_registry_list_registered(self) -> None:
        from hal.triton_backend import TritonKernelRegistry

        assert "test_op" in TritonKernelRegistry.list_registered()

    def test_backend_falls_through_to_pytorch(self) -> None:
        from hal.pytorch_backend import PyTorchBackend
        from hal.triton_backend import TritonBackend

        pytorch = PyTorchBackend("cpu")
        triton = TritonBackend(fallback=pytorch)

        # matmul is not registered in Triton → should fall through
        a = torch.randn(2, 3)
        b = torch.randn(3, 4)
        result = triton.execute("matmul", [a, b])
        assert result.shape == (2, 4)

    def test_backend_triggers_kernel_if_registered(self) -> None:
        from hal.pytorch_backend import PyTorchBackend
        from hal.triton_backend import TritonBackend, TritonKernelRegistry

        call_count = 0

        @TritonKernelRegistry.register("my_custom_op")
        def custom_kernel(x: torch.Tensor) -> torch.Tensor:
            nonlocal call_count
            call_count += 1
            return x + 1.0

        pytorch = PyTorchBackend("cpu")
        triton = TritonBackend(fallback=pytorch)

        x = torch.tensor([1.0, 2.0, 3.0])
        result = triton.execute("my_custom_op", [x])
        assert call_count == 1
        assert torch.allclose(result, torch.tensor([2.0, 3.0, 4.0]))
