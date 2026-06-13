"""Shared KV cache logic for paged attention executors.

Both Executor (IrModule) and MlirExecutor (MlirModule) share identical
KV cache read/write methods.  This mixin eliminates the ~250 lines of
near-duplicate code that previously lived in both files.

Usage::

    class MyExecutor(_KVCacheMixin):
        def __init__(self, ...):
            self._kv_cache: torch.Tensor | None = None
            self._block_tables: dict[str, list[int]] = {}
            self._block_size: int = 16
            self._num_kv_heads: int = 0
            self._head_dim: int = 0
"""

from __future__ import annotations

import torch


class _KVCacheMixin:
    """Shared paged KV cache read/write methods.

    Requires these attributes on the inheriting class:
        _kv_cache, _block_tables, _block_size, _num_kv_heads, _head_dim
    """

    _kv_cache: torch.Tensor | None
    _block_tables: dict[str, list[int]]
    _block_size: int
    _num_kv_heads: int
    _head_dim: int

    def set_kv_cache(
        self,
        kv_cache: torch.Tensor | None,
        block_tables: dict[str, list[int]] | None = None,
        block_size: int = 16,
        num_kv_heads: int = 0,
        head_dim: int = 0,
    ) -> None:
        self._kv_cache = kv_cache
        self._block_tables = block_tables or {}
        self._block_size = block_size
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim

    def prepare_kv_blocks(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        num_blocks: int,
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        shape = (num_blocks, num_layers, 2, block_size, num_kv_heads, head_dim)
        return torch.zeros(shape, dtype=dtype)

    def _max_seq_from_tables(self, block_tables: dict[str, list[int]]) -> int:
        total = 0
        for blocks in block_tables.values():
            total = max(total, len(blocks))
        return total * self._block_size

    def _write_kv_flat(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        block_tables: dict[str, list[int]],
        layer_idx: int,
    ) -> None:
        if self._kv_cache is None:
            return
        bs = self._block_size
        pos_list = positions.tolist() if positions.dim() >= 1 else [int(positions.item())]
        if not isinstance(pos_list, list):
            pos_list = [int(positions.item())]
        for i, pos in enumerate(pos_list):
            block_idx = pos // bs
            offset = pos % bs
            written = False
            for blocks in block_tables.values():
                if block_idx < len(blocks):
                    phys_id = blocks[block_idx]
                    self._kv_cache[phys_id, layer_idx, 0, offset] = key[i]
                    self._kv_cache[phys_id, layer_idx, 1, offset] = value[i]
                    written = True
                    break
            if not written and block_tables:
                import logging

                _log = logging.getLogger("engine._kv_cache")
                _log.warning(
                    "KV write skipped: position %d (block_idx=%d, offset=%d) "
                    "not found in any request's block table. This may indicate "
                    "a block allocation gap or stale position data.",
                    pos,
                    block_idx,
                    offset,
                )

    def _gather_kv_flat(
        self,
        block_tables: dict[str, list[int]],
        max_seq_len: int,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._kv_cache is None or not block_tables:
            raise RuntimeError("KV cache not initialized")

        bs = self._block_size
        num_requests = len(block_tables)
        nkh = self._num_kv_heads
        hd = self._head_dim
        dtype = self._kv_cache.dtype

        key = torch.zeros(num_requests, max_seq_len, nkh, hd, dtype=dtype)
        value = torch.zeros(num_requests, max_seq_len, nkh, hd, dtype=dtype)

        for req_idx, (_, blocks) in enumerate(sorted(block_tables.items())):
            for blk_offset, phys_id in enumerate(blocks):
                start = blk_offset * bs
                end = min(start + bs, max_seq_len)
                if start >= max_seq_len:
                    break
                key[req_idx, start:end] = self._kv_cache[phys_id, layer_idx, 0, : end - start]
                value[req_idx, start:end] = self._kv_cache[phys_id, layer_idx, 1, : end - start]

        return key, value

    def write_kv_to_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        block_tables: dict[str, list[int]],
        layer_idx: int = 0,
    ) -> None:
        self._write_kv_flat(key, value, positions, block_tables, layer_idx)

    def gather_kv_from_cache(
        self,
        block_tables: dict[str, list[int]],
        max_seq_len: int,
        layer_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._gather_kv_flat(block_tables, max_seq_len, layer_idx)


def _normalize_kv_for_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize K/V tensor shapes for paged cache storage.

    Handles various input formats:
      [batch, heads, seq, head_dim] → squeeze batch
      [heads, seq, head_dim] → [seq, heads, head_dim]
      [seq, hidden] → [seq, heads, head_dim]
    """
    if k.dim() >= 4 and k.shape[0] == 1:
        k = k.squeeze(0)
        v = v.squeeze(0)

    nkh = num_kv_heads
    hd = head_dim

    if k.dim() == 3 and k.shape[0] == nkh and k.shape[-1] == hd:
        k = k.permute(1, 0, 2)
        v = v.permute(1, 0, 2)
    elif k.dim() == 3 and k.shape[1] == nkh and k.shape[-1] == hd:
        pass  # Already [seq, heads, dim]
    elif k.dim() == 2 and k.shape[-1] == nkh * hd:
        k = k.reshape(-1, nkh, hd)
        v = v.reshape(-1, nkh, hd)
    elif k.dim() == 3 and k.shape[-1] == nkh * hd:
        k = k.reshape(-1, nkh, hd)
        v = v.reshape(-1, nkh, hd)

    return k, v
