"""PagedAttention KV Cache block management.

Implements the block-based KV cache allocation strategy inspired by
vLLM's PagedAttention. Physical KV cache is partitioned into fixed-size
blocks and managed via a logical-to-physical page table per request.

Key properties:
  - Zero-waste allocation — blocks allocated on demand, not pre-reserved.
  - Block sharing — prefix cache via reference counting (Beam Search, etc.).
  - External fragmentation minimised via shared block pool.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from utils.errors import OutOfMemoryError


@dataclass
class Block:
    """A physical KV cache block on device.

    The actual tensor data is owned by the Executor. BlockManager
    tracks only the logical ownership and reference counting.
    """

    block_id: int
    ref_count: int = 0


class BlockManager:
    """Manages the KV cache block pool.

    Responsibilities:
      - Allocate/free blocks for inference requests.
      - Maintain logical-to-physical block table per request.
      - Support prefix-cache sharing via ref-counted blocks.

    Example:
        bm = BlockManager(num_blocks=1000, block_size=16)
        blocks = bm.allocate("req_1", num_tokens=500)  # → 32 blocks
        bm.free("req_1")
    """

    def __init__(self, num_blocks: int, block_size: int = 16) -> None:
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")

        self.block_size = block_size
        self.num_blocks = num_blocks

        self.blocks: dict[int, Block] = {i: Block(i) for i in range(num_blocks)}
        self.free_blocks: list[int] = list(range(num_blocks))
        self.block_tables: dict[str, list[int]] = {}

        # Track which requests share each block (for prefix cache eviction)
        self._shared_owners: dict[int, set[str]] = {}

    # ── Allocation ──────────────────────────────────────────

    def allocate(self, request_id: str, num_tokens: int) -> list[int]:
        """Allocate physical blocks for a request.

        Args:
            request_id: Unique request identifier.
            num_tokens: Number of tokens that need KV cache space.

        Returns:
            List of physical block IDs allocated to this request.

        Raises:
            OutOfMemoryError: If not enough free blocks are available.
        """
        if request_id in self.block_tables:
            raise ValueError(f"request_id '{request_id}' already has allocated blocks")

        needed = math.ceil(num_tokens / self.block_size)
        if needed > len(self.free_blocks):
            raise OutOfMemoryError(needed, len(self.free_blocks), self.num_blocks)

        allocated: list[int] = []
        for _ in range(needed):
            bid = self.free_blocks.pop()
            self.blocks[bid].ref_count += 1
            allocated.append(bid)

        self.block_tables[request_id] = allocated
        return allocated

    def free(self, request_id: str) -> None:
        """Release all blocks allocated to a request.

        If a block is shared (ref_count > 1), only decrements the
        reference count without returning it to the free pool.
        """
        if request_id not in self.block_tables:
            return

        for bid in self.block_tables[request_id]:
            block = self.blocks[bid]
            block.ref_count -= 1
            if request_id in self._shared_owners.get(bid, set()):
                self._shared_owners[bid].discard(request_id)
            if block.ref_count == 0:
                self.free_blocks.append(bid)
                self._shared_owners.pop(bid, None)

        del self.block_tables[request_id]

    def free_block(self, block_id: int) -> None:
        """Release a single physical block (for LRU eviction).

        Decrements the reference count on the block.  If it reaches
        zero the block is returned to the free pool.  This is a
        low-level operation used by the RadixCache eviction path.
        """
        block = self.blocks[block_id]
        block.ref_count -= 1
        self._shared_owners.pop(block_id, None)
        if block.ref_count == 0:
            self.free_blocks.append(block_id)

    def increment_ref_count(self, block_id: int) -> None:
        """Increment reference count on a block (used by RadixCache on insert)."""
        if block_id in self.blocks:
            self.blocks[block_id].ref_count += 1

    def assign_cached_blocks(self, request_id: str, block_ids: list[int]) -> None:
        """Prepend pre-existing cached blocks to a request's block table.

        Increments reference counts on each block.  The blocks are
        appended to the front of the request's existing table (if any).
        This is the primary integration point for RadixCache — it lets
        a new request share KV blocks from cached tree nodes.
        """
        for bid in block_ids:
            self.blocks[bid].ref_count += 1
            self._shared_owners.setdefault(bid, set()).add(request_id)
        if request_id in self.block_tables:
            self.block_tables[request_id] = list(block_ids) + self.block_tables[request_id]
        else:
            self.block_tables[request_id] = list(block_ids)

    # ── Prefix Cache via Block Sharing ──────────────────────

    def share_prefix(self, src_request_id: str, dst_request_id: str, prefix_len: int) -> list[int]:
        """Share prefix KV cache blocks between two requests.

        The first n_blocks = ceil(prefix_len / block_size) blocks from the
        source request's table are shared (ref-counted) with the destination.

        Args:
            src_request_id: Source request that already has allocated blocks.
            dst_request_id: Destination request to share blocks with.
            prefix_len: Number of prefix tokens to share.

        Returns:
            List of shared physical block IDs.

        Raises:
            ValueError: If source doesn't have enough blocks for the prefix.
        """
        if src_request_id not in self.block_tables:
            raise ValueError(f"Source request '{src_request_id}' not found")
        if dst_request_id in self.block_tables:
            raise ValueError(f"Destination request '{dst_request_id}' already has blocks")

        n_blocks = math.ceil(prefix_len / self.block_size)
        src_blocks = self.block_tables[src_request_id]

        if n_blocks > len(src_blocks):
            raise ValueError(
                f"Source has {len(src_blocks)} blocks, cannot share {n_blocks} "
                f"(prefix_len={prefix_len}, block_size={self.block_size})"
            )

        shared = src_blocks[:n_blocks]
        for bid in shared:
            self.blocks[bid].ref_count += 1
            self._shared_owners.setdefault(bid, set()).add(dst_request_id)

        self.block_tables[dst_request_id] = list(shared)
        return shared

    # ── Query ───────────────────────────────────────────────

    def get_blocks(self, request_id: str) -> list[int]:
        """Return physical block IDs for a request.

        Raises KeyError if the request is unknown.
        """
        if request_id not in self.block_tables:
            raise KeyError(f"Unknown request_id: {request_id}")
        return list(self.block_tables[request_id])

    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def num_allocated_blocks(self) -> int:
        return self.num_blocks - len(self.free_blocks)

    def get_block_table(self, request_id: str) -> list[int]:
        """Alias for get_blocks — returns a copy of the block table."""
        return self.get_blocks(request_id)

    @property
    def utilization(self) -> float:
        """Fraction of the block pool currently in use."""
        if self.num_blocks == 0:
            return 0.0
        return self.num_allocated_blocks() / self.num_blocks
