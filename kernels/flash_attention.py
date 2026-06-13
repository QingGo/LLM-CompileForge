# ruff: noqa: N803,N806  — Triton convention: uppercase constexpr arguments
"""FlashAttention-2 forward pass kernel.

Implements tiled attention with online softmax for numerical stability.
Avoids materializing the full N×N attention matrix in HBM.

Algorithm (per-head):
  1. Split Q into blocks along the sequence dimension.
  2. For each Q block:
     a. Load Q block into SRAM.
     b. Iterate over K,V blocks:
        - Load K block → compute QK^T → apply scale & causal mask.
        - Online softmax update (m, l running statistics).
        - Load V block → accumulate weighted sum.
     c. Final normalisation and write-back.

Reference: design-phase1.md §3.4.1; FlashAttention-2 paper.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F  # noqa: N812


def _flash_attention_fwd_cpu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    causal: bool = True,
) -> torch.Tensor:
    """CPU / fallback using PyTorch SDPA."""
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=causal,
        scale=scale,
    )


def _flash_attention_fwd_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    causal: bool = True,
) -> torch.Tensor:
    """Triton FlashAttention-2 kernel."""
    import triton  # noqa: F811
    import triton.language as tl

    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    @triton.jit
    def _kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        O_ptr,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_ob,
        stride_oh,
        stride_om,
        BATCH: tl.constexpr,
        N_HEADS: tl.constexpr,
        SEQ_LEN: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        CAUSAL: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_m_blocks = tl.cdiv(SEQ_LEN, BLOCK_M)
        batch_idx = pid // (N_HEADS * num_m_blocks)
        head_idx = (pid // num_m_blocks) % N_HEADS
        m_block_idx = pid % num_m_blocks

        offs_m = m_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)

        q = tl.load(
            Q_ptr + batch_idx * stride_qb + head_idx * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :],
            mask=offs_m[:, None] < SEQ_LEN,
        )

        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        for n_block_idx in range(0, tl.cdiv(SEQ_LEN, BLOCK_N)):
            offs_n = n_block_idx * BLOCK_N + tl.arange(0, BLOCK_N)

            k_block = tl.load(
                K_ptr + batch_idx * stride_kb + head_idx * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :],
                mask=offs_n[:, None] < SEQ_LEN,
            )

            qk = tl.dot(q, tl.trans(k_block))
            qk *= SCALE

            if CAUSAL:
                qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))

            m_curr = tl.max(qk, 1)
            m_new = tl.maximum(m_i, m_curr)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])

            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_new
            acc = acc * alpha[:, None]

            v_block = tl.load(
                V_ptr + batch_idx * stride_vb + head_idx * stride_vh + offs_n[None, :] * stride_vn + offs_d[:, None],
                mask=offs_n[None, :] < SEQ_LEN,
            )
            acc += tl.dot(p, v_block)

        acc = acc / l_i[:, None]
        tl.store(
            O_ptr + batch_idx * stride_ob + head_idx * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :],
            acc,
            mask=offs_m[:, None] < SEQ_LEN,
        )

    B, H, S, D = q.shape
    o = torch.empty_like(q)
    BLOCK_M = min(128, triton.next_power_of_2(S))
    BLOCK_N = min(64, triton.next_power_of_2(S))
    grid = (B * H * triton.cdiv(S, BLOCK_M),)

    _kernel[grid](
        q,
        k,
        v,
        o,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        BATCH=B,
        N_HEADS=H,
        SEQ_LEN=S,
        HEAD_DIM=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        CAUSAL=causal,
        SCALE=scale,
    )
    return o


def flash_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    causal: bool = True,
) -> torch.Tensor:
    """FlashAttention-2 forward pass.

    Args:
        q: Query tensor  [batch, heads, seq_len, head_dim].
        k: Key tensor    [batch, heads, seq_len, head_dim].
        v: Value tensor  [batch, heads, seq_len, head_dim].
        scale:  Attention scale (default 1/sqrt(head_dim)).
        causal: Apply causal mask (upper triangular -inf).

    Returns:
        Attention output [batch, heads, seq_len, head_dim].
    """
    from kernels import HAS_TRITON

    if HAS_TRITON and q.is_cuda:
        return _flash_attention_fwd_triton(q, k, v, scale, causal)
    return _flash_attention_fwd_cpu(q, k, v, scale, causal)
