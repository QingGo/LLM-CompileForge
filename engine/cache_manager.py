"""CacheManager — runtime slab/indexer for KV cache.

Translates a declarative CachePolicy into pre-allocated tensor storage
and provides positional read/write with paged addressing.

Architecture:
    CachePolicy  →  Slab allocation  →  per-step read/write via block_tables

Slab types:
  - paged  [num_blocks, layers, block_size, heads, dim]  — Llama K/V
  - fixed  [max_requests, layers, dim]                    — RWKV state
"""

from __future__ import annotations

import torch

from compiler.cache_policy import CachePolicy, _SlabSpec


def _dtype_from_str(name: str) -> torch.dtype:
    mapping: dict[str, torch.dtype] = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float64": torch.float64,
    }
    return mapping.get(name, torch.float32)


class CacheManager:
    """Manages cache slabs and positional I/O at runtime."""

    def __init__(
        self,
        policy: CachePolicy,
        num_blocks: int,
    ) -> None:
        self._policy = policy
        self._num_blocks = num_blocks
        self._block_size = policy.block_size
        self._max_requests = policy.max_requests

        self._slabs: dict[str, torch.Tensor] = {}
        self._specs: dict[str, _SlabSpec] = {}
        self._layer_counters: dict[str, int] = {}
        self._block_tables: dict[str, list[int]] = {}

        for spec in policy.slabs:
            tensor = self._allocate_slab(spec)
            self._slabs[spec.slab_id] = tensor
            self._specs[spec.slab_id] = spec
            self._layer_counters[spec.slab_id] = 0

    def _allocate_slab(self, spec: _SlabSpec) -> torch.Tensor:
        dtype = _dtype_from_str(spec.dtype)
        shape: tuple[int, ...]
        if spec.storage == "paged":
            shape = (
                self._num_blocks,
                int(spec.dims["layers"]),
                self._block_size,
                int(spec.dims["heads"]),
                int(spec.dims["dim"]),
            )
        elif spec.storage == "fixed":
            shape = (
                self._max_requests,
                int(spec.dims["layers"]),
                int(spec.dims["dim"]),
            )
        else:
            raise ValueError(f"Unknown storage type: {spec.storage}")
        return torch.zeros(shape, dtype=dtype)

    # ── Per-step state ────────────────────────────────────

    def begin_step(self, block_tables: dict[str, list[int]]) -> None:
        self._block_tables = block_tables
        for key in self._layer_counters:
            self._layer_counters[key] = 0

    def resolve_layer(self, slab_id: str) -> int:
        """Return and increment the sequential layer counter for *slab_id*."""
        layer = self._layer_counters[slab_id]
        self._layer_counters[slab_id] += 1
        return layer

    # ── Paged KV I/O ──────────────────────────────────────

    def write_paged(
        self,
        slab_id: str,
        layer_idx: int,
        data: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        slab = self._slabs[slab_id]
        bt = self._block_tables
        if not bt or slab is None:
            return
        bs = self._block_size
        data_norm = self._normalize_kv(data, slab_id)
        pos_list = self._pos_to_list(positions)
        for i, pos in enumerate(pos_list):
            block_idx = pos // bs
            offset = pos % bs
            for blocks in bt.values():
                if block_idx < len(blocks):
                    phys_id = blocks[block_idx]
                    slab[phys_id, layer_idx, offset] = data_norm[i]
                    break

    def read_paged(
        self,
        slab_id: str,
        layer_idx: int,
        max_seq_len: int,
    ) -> torch.Tensor:
        slab = self._slabs[slab_id]
        bt = self._block_tables
        if slab is None or not bt:
            raise RuntimeError("Paged slab read requested without active block tables")
        bs = self._block_size
        spec = self._specs[slab_id]
        nkh = int(spec.dims["heads"])
        hd = int(spec.dims["dim"])
        dtype = slab.dtype
        num_reqs = len(bt)
        result = torch.zeros(num_reqs, max_seq_len, nkh, hd, dtype=dtype)
        for req_idx, (_, blocks) in enumerate(sorted(bt.items())):
            for blk_offset, phys_id in enumerate(blocks):
                start = blk_offset * bs
                end = min(start + bs, max_seq_len)
                if start >= max_seq_len:
                    break
                result[req_idx, start:end] = slab[phys_id, layer_idx, : end - start]
        return result

    # ── Fixed state I/O ──────────────────────────────────

    def write_fixed(
        self,
        slab_id: str,
        slot: int,
        layer_idx: int,
        data: torch.Tensor,
    ) -> None:
        slab = self._slabs[slab_id]
        slab[slot, layer_idx] = data.float()

    def read_fixed(
        self,
        slab_id: str,
        slot: int,
        layer_idx: int,
    ) -> torch.Tensor:
        slab = self._slabs[slab_id]
        return slab[slot, layer_idx]

    # ── Helpers ───────────────────────────────────────────

    def _normalize_kv(self, data: torch.Tensor, slab_id: str) -> torch.Tensor:
        t = data
        if t.dim() >= 4 and t.shape[0] == 1:
            t = t.squeeze(0)
        spec = self._specs[slab_id]
        nkh = int(spec.dims["heads"])
        hd = int(spec.dims["dim"])
        if t.dim() == 3 and t.shape[0] == nkh and t.shape[-1] == hd:
            t = t.permute(1, 0, 2)
        elif t.dim() == 3 and t.shape[1] == nkh and t.shape[-1] == hd:
            pass  # Already [seq, heads, dim]
        elif t.dim() == 2 and t.shape[-1] == nkh * hd:
            t = t.reshape(-1, nkh, hd)
        elif t.dim() == 3 and t.shape[-1] == nkh * hd:
            t = t.reshape(-1, nkh, hd)
        return t

    @staticmethod
    def _pos_to_list(positions: torch.Tensor) -> list[int]:
        if positions.dim() >= 1:
            return [int(p) for p in positions.flatten().tolist()]
        return [int(positions.item())]

    # ── Slab info ─────────────────────────────────────────

    @property
    def has_paged_cache(self) -> bool:
        return any(
            s.storage == "paged" for s in self._policy.slabs
        )

    def configured_kv_heads(self) -> int:
        for spec in self._policy.slabs:
            if spec.storage == "paged" and "heads" in spec.dims:
                return int(spec.dims["heads"])
        return 0

    def configured_head_dim(self) -> int:
        for spec in self._policy.slabs:
            if spec.storage == "paged" and "dim" in spec.dims:
                return int(spec.dims["dim"])
        return 0

    def configured_num_layers(self) -> int:
        for spec in self._policy.slabs:
            if "layers" in spec.dims:
                return int(spec.dims["layers"])
        return 0
