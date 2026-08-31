# ruff: noqa: N803,N806  — Triton convention: uppercase constexpr arguments
"""PagedAttention kernel with block table indirection.

Wraps attention computation for paged KV cache: each request's K/V
data is stored in physical blocks scattered in memory; a block table
maps logical token positions → physical block IDs.

The kernel performs online softmax across blocks, using the block
table to resolve physical addresses for each logical token range.

Reference: design-phase1.md §3.4.2; design-phase2.md §2.3.1
"""

from __future__ import annotations

import math

import torch


def _paged_attention_cpu(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: dict[str, list[int]],
    seq_lens: dict[str, int],
    block_size: int = 16,
    scale: float | None = None,
) -> dict[str, torch.Tensor]:
    """CPU fallback: gather K/V from block tables, then SDPA."""
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    import torch.nn.functional as F  # noqa: N812

    outputs: dict[str, torch.Tensor] = {}
    for idx, (rid, blocks) in enumerate(block_tables.items()):
        seq_len = seq_lens.get(rid, len(blocks) * block_size)
        n_blocks = min(len(blocks), (seq_len + block_size - 1) // block_size)

        k_parts: list[torch.Tensor] = []
        v_parts: list[torch.Tensor] = []
        for block_idx in range(n_blocks):
            bid = blocks[block_idx]
            k_parts.append(k_cache[bid])
            v_parts.append(v_cache[bid])

        k_seq = torch.cat(k_parts, dim=0)[:seq_len]
        v_seq = torch.cat(v_parts, dim=0)[:seq_len]

        k_seq = k_seq.unsqueeze(0).transpose(1, 2)  # [1, kv_heads, seq, hd]
        v_seq = v_seq.unsqueeze(0).transpose(1, 2)

        # Q may be 3D [batch, heads, head_dim] (decode) or 4D (prefill)
        q_i = q[idx]  # [heads, head_dim]
        q_i = q_i.unsqueeze(0).unsqueeze(2)  # [1, heads, 1, head_dim]

        if q_i.shape[1] != k_seq.shape[1]:
            ratio = q_i.shape[1] // k_seq.shape[1]
            k_seq = k_seq.repeat_interleave(ratio, dim=1)
            v_seq = v_seq.repeat_interleave(ratio, dim=1)

        out = F.scaled_dot_product_attention(
            q_i,
            k_seq,
            v_seq,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
        )
        outputs[rid] = out.squeeze(0).squeeze(1)  # remove batch=1, seq=1 → [heads, head_dim]

    return outputs


def _paged_attention_triton(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: dict[str, list[int]],
    seq_lens: dict[str, int],
    block_size: int = 16,
    scale: float | None = None,
) -> dict[str, torch.Tensor]:
    """Triton PagedAttention kernel."""
    import triton  # noqa: F811
    import triton.language as tl

    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    @triton.jit  # type: ignore[untyped-decorator]
    def _kernel(
        Q_ptr: torch.Tensor,
        K_cache_ptr: torch.Tensor,
        V_cache_ptr: torch.Tensor,
        block_table_ptr: torch.Tensor,
        seq_lens_ptr: torch.Tensor,
        O_ptr: torch.Tensor,
        scale_val: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        N_HEADS: tl.constexpr,
        MAX_BLOCKS: tl.constexpr,
    ) -> None:
        pid = tl.program_id(0)
        batch_idx = pid // N_HEADS
        head_idx = pid % N_HEADS

        seq_len = tl.load(seq_lens_ptr + batch_idx).to(tl.int32)
        n_blocks = tl.cdiv(seq_len, BLOCK_SIZE).to(tl.int32)

        offs_d = tl.arange(0, HEAD_DIM)
        q = tl.load(Q_ptr + batch_idx * N_HEADS * HEAD_DIM + head_idx * HEAD_DIM + offs_d)

        m_i = float("-inf")
        l_i = 0.0
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

        for block_idx in range(MAX_BLOCKS):
            if block_idx >= n_blocks:
                break
            phys_block = tl.load(block_table_ptr + batch_idx * MAX_BLOCKS + block_idx)

            offs_n = tl.arange(0, BLOCK_SIZE)
            k = tl.load(
                K_cache_ptr
                + phys_block.to(tl.int64) * BLOCK_SIZE * N_HEADS * HEAD_DIM
                + head_idx * HEAD_DIM
                + offs_n[:, None] * HEAD_DIM
                + offs_d[None, :],
                mask=offs_n[:, None] < (seq_len - block_idx * BLOCK_SIZE),
            )

            scores = tl.sum(q[None, :] * k, axis=1) * scale_val
            scores = tl.where(
                tl.arange(0, BLOCK_SIZE)[:, None] < (seq_len - block_idx * BLOCK_SIZE),
                scores,
                float("-inf"),
            )

            m_curr = tl.max(scores)
            m_new = tl.maximum(m_i, m_curr)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new)

            l_i = l_i * alpha + tl.sum(p)
            m_i = m_new
            acc = acc * alpha

            v = tl.load(
                V_cache_ptr
                + phys_block.to(tl.int64) * BLOCK_SIZE * N_HEADS * HEAD_DIM
                + head_idx * HEAD_DIM
                + offs_n[None, :] * HEAD_DIM
                + offs_d[:, None],
                mask=offs_n[None, :] < (seq_len - block_idx * BLOCK_SIZE),
            )
            acc += tl.sum(p[:, None] * v, axis=0)

        tl.store(
            O_ptr + batch_idx * N_HEADS * HEAD_DIM + head_idx * HEAD_DIM + offs_d,
            acc,
        )

    B = len(block_tables)
    H, D = q.shape[1], q.shape[2]
    max_blocks = max(len(bt) for bt in block_tables.values())

    # Build flat tensors for the kernel
    bt_flat = torch.zeros((B, max_blocks), dtype=torch.long)
    sl_flat = torch.zeros(B, dtype=torch.long)
    for idx, (rid, bt) in enumerate(block_tables.items()):
        bt_flat[idx, : len(bt)] = torch.tensor(bt, dtype=torch.long)
        sl_flat[idx] = seq_lens.get(rid, len(bt) * block_size)

    o = torch.empty(B, H, D, device=q.device, dtype=q.dtype)
    grid = (B * H,)

    _kernel[grid](
        q,
        k_cache,
        v_cache,
        bt_flat,
        sl_flat,
        o,
        scale_val=scale,
        BLOCK_SIZE=block_size,
        HEAD_DIM=D,
        N_HEADS=H,
        MAX_BLOCKS=max_blocks,
    )

    outputs: dict[str, torch.Tensor] = {}
    for idx, rid in enumerate(block_tables):
        outputs[rid] = o[idx]
    return outputs


def paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: dict[str, list[int]],
    seq_lens: dict[str, int],
    block_size: int = 16,
    scale: float | None = None,
) -> dict[str, torch.Tensor]:
    """PagedAttention with block table indirection.

    Args:
        q:  Query tensor [batch, num_heads, head_dim] (decode: one token per request).
        k_cache: Paged K cache [num_blocks, block_size, num_kv_heads, head_dim].
        v_cache: Paged V cache [num_blocks, block_size, num_kv_heads, head_dim].
        block_tables: request_id → list[physical block IDs].
        seq_lens: request_id → current sequence length.
        block_size: Tokens per KV block (default 16).
        scale: Attention scale (default 1/sqrt(head_dim)).

    Returns:
        dict mapping request_id → output tensor [num_heads, head_dim].
    """
    from kernels import HAS_TRITON

    if HAS_TRITON and q.is_cuda:
        return _paged_attention_triton(q, k_cache, v_cache, block_tables, seq_lens, block_size, scale)
    return _paged_attention_cpu(q, k_cache, v_cache, block_tables, seq_lens, block_size, scale)
