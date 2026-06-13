"""Radix Tree Prefix Cache.

Organises KV cache blocks into a Radix Tree keyed by token sequences.
When a new request shares a prefix with a cached sequence, the
corresponding KV blocks are reused, skipping prefill for those tokens.

Based on SGLang's RadixAttention design (§2.6.2 of design-phase2.md).
"""

from __future__ import annotations

from typing import Any

from compiler.utils.logging import get_logger

_log = get_logger("cache.radix")


class RadixTreeNode:
    """A node in the Radix Tree representing a contiguous token subsequence.

    Each node stores the token sequence it represents and the physical
    KV block IDs that hold the computed KV cache for those tokens.
    """

    __slots__ = ("token_ids", "children", "kv_blocks", "ref_count")

    def __init__(self, token_ids: tuple[int, ...]) -> None:
        self.token_ids: tuple[int, ...] = token_ids
        self.children: dict[int, RadixTreeNode] = {}  # keyed by first token
        self.kv_blocks: list[int] = []
        self.ref_count: int = 0

    def __repr__(self) -> str:
        return (
            f"RadixTreeNode(tokens={self.token_ids[:8]}..., "
            f"children={len(self.children)}, blocks={len(self.kv_blocks)}, "
            f"ref={self.ref_count})"
        )


class RadixCache:
    """Prefix cache based on a Radix Tree of token sequences.

    The tree maps token sequences to physical KV block IDs.  Multiple
    requests can share the same prefix blocks via reference counting.

    Typical use in the scheduler:

        # On request admission:
        matched, matched_tokens = cache.match_prefix(req.prompt_tokens)
        if matched:
            req.prefill_pos = matched_tokens  # skip cached tokens

        # On request completion:
        cache.insert(req.prompt_tokens + req.output_tokens, kv_blocks)

        # Under memory pressure:
        cache.evict(target_blocks)
    """

    def __init__(self, block_manager: Any) -> None:
        self._bm = block_manager
        self.root = RadixTreeNode(())

    # ── Core API ─────────────────────────────────────────────

    def match_prefix(self, token_ids: list[int]) -> tuple[list[int], int]:
        """Find the longest prefix of *token_ids* present in the tree.

        Returns:
            (matched_kv_blocks, matched_token_count).
            Both are empty/zero when no prefix matches.
        """
        node = self.root
        matched_blocks: list[int] = []
        consumed = 0
        remaining = token_ids

        while remaining:
            first = remaining[0]
            child = node.children.get(first)
            if child is None:
                break

            # child.token_ids is the sequence stored at this child
            child_tokens = child.token_ids
            common = _common_prefix_len(remaining, child_tokens)

            if common < len(child_tokens):
                # Query ends inside this child — return proportional blocks
                n_blocks = _ceil_div(common, self._bm.block_size)
                matched_blocks.extend(child.kv_blocks[:n_blocks])
                consumed += common
                break

            # Full match of this child node
            matched_blocks.extend(child.kv_blocks)
            consumed += len(child_tokens)
            remaining = remaining[len(child_tokens) :]
            node = child

        return matched_blocks, consumed

    def insert(self, token_ids: list[int], kv_blocks: list[int]) -> None:
        """Insert a token sequence and its KV blocks into the tree.

        Splits existing nodes on partial prefix match.  kv_blocks
        are distributed across newly created leaf nodes proportionally
        to token count (one block per block_size tokens).
        """
        if not token_ids:
            return

        _log.debug("insert | tokens=%d blocks=%d", len(token_ids), len(kv_blocks))
        block_size = self._bm.block_size
        node = self.root
        remaining = list(token_ids)
        block_offset = 0

        while remaining:
            first = remaining[0]
            child = node.children.get(first)

            if child is None:
                # No matching child — create a new leaf
                new_blocks = kv_blocks[block_offset:]
                self._add_node(node, tuple(remaining), new_blocks)
                return

            child_tokens = child.token_ids
            common = _common_prefix_len(remaining, child_tokens)

            if common == 0:
                # Should not happen — first token matched but no common prefix
                new_blocks = kv_blocks[block_offset:]
                self._add_node(node, tuple(remaining), new_blocks)
                return

            if common < len(child_tokens):
                # Partial match — split the child node
                shared_tokens = child_tokens[:common]
                remaining_child_tokens = child_tokens[common:]

                # Split blocks proportionally.  The shared prefix gets
                # its share of the child's original blocks.  The new
                # sequence's blocks (kv_blocks) are NOT consumed here —
                # they apply only to the new tokens being inserted.
                shared_tokens_count = common
                n_shared_blocks = _ceil_div(shared_tokens_count, block_size)

                # The shared portion keeps the first n_shared_blocks
                shared_blocks = child.kv_blocks[:n_shared_blocks]
                child_remaining_blocks = child.kv_blocks[n_shared_blocks:]

                # Replace child with a new shared-prefix interior node ...
                split_node = RadixTreeNode(shared_tokens)
                split_node.kv_blocks = shared_blocks
                split_node.ref_count = child.ref_count
                node.children[first] = split_node

                # ... and attach the remaining child tokens as its child
                first_of_remainder = remaining_child_tokens[0]
                child.token_ids = remaining_child_tokens
                child.kv_blocks = child_remaining_blocks
                split_node.children[first_of_remainder] = child

                # Consume the shared prefix and continue inserting our tokens.
                # block_offset is NOT advanced here because we haven't consumed
                # any of the new kv_blocks yet (shared blocks came from the
                # old child).
                remaining = remaining[common:]
                node = split_node
            else:
                # Full match of this child node
                remaining = remaining[len(child_tokens) :]
                block_offset += len(child.kv_blocks)
                node = child

    def evict(self, target_blocks: int) -> int:
        """Evict subtrees whose ref_count is zero.

        Frees the underlying KV blocks via the BlockManager until
        *target_blocks* have been freed, or no more evictable nodes exist.

        Returns:
            Number of blocks actually freed.
        """
        freed = 0
        if target_blocks <= 0:
            return 0

        def _collect_evictable(node: RadixTreeNode, path: list[int]) -> list[tuple[list[int], RadixTreeNode]]:
            """BFS-style collection of evictable nodes (ref_count == 0).
            Returns list of (parent_path_tokens, node) for eviction.
            Not used — we do DFS instead."""
            result: list[tuple[list[int], RadixTreeNode]] = []
            for first_token, child in list(node.children.items()):
                if child.ref_count == 0:
                    result.append(([first_token], child))
                _collect_evictable(child, path + [first_token])
            return result

        # DFS to find and evict zero-ref leaves first (LRU-like)
        def _dfs_evict(node: RadixTreeNode, parent: RadixTreeNode, first_token: int) -> int:
            nonlocal freed
            local_freed = 0

            # Evict children first (leaves before parents)
            for ft, child in list(node.children.items()):
                local_freed += _dfs_evict(child, node, ft)

            if local_freed >= target_blocks - freed:
                return local_freed

            # Try to evict this node itself
            if node is not self.root and node.ref_count == 0:
                for bid in node.kv_blocks:
                    self._bm.free_block(bid)
                    freed += 1
                del parent.children[first_token]
                local_freed += len(node.kv_blocks)

            return local_freed

        for ft, child in list(self.root.children.items()):
            _dfs_evict(child, self.root, ft)
            if freed >= target_blocks:
                break

        return freed

    # ── Internals ────────────────────────────────────────────

    def _add_node(self, parent: RadixTreeNode, token_ids: tuple[int, ...], kv_blocks: list[int]) -> None:
        """Create a new leaf node under *parent*.

        Increments the block_manager reference count for each cached
        block so the blocks persist after the originating request is freed.
        The caller must ensure *kv_blocks* were previously allocated
        (via block_manager.allocate) and are therefore absent from the
        free_blocks pool.
        """
        node = RadixTreeNode(token_ids)
        node.kv_blocks = kv_blocks
        node.ref_count = 1
        for bid in kv_blocks:
            self._bm.increment_ref_count(bid)
        parent.children[token_ids[0]] = node


# ── helpers ─────────────────────────────────────────────────


def _common_prefix_len(a: list[int], b: tuple[int, ...]) -> int:
    """Return the number of leading elements shared by *a* and *b*."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b
