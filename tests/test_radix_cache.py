"""Unit tests for RadixTree prefix cache (cache/radix_cache.py).

Uses Rust PyBlockManager for block management (production code path).
"""

from __future__ import annotations

import llm_serveforge_runtime as _rt
import pytest

from cache.radix_cache import RadixCache

_counter = 0


def _alloc(bm: _rt.PyBlockManager, n_tokens: int) -> tuple[list[int], str]:
    global _counter
    _counter += 1
    req_id = f"__test_{_counter}"
    return bm.allocate(req_id, n_tokens), req_id


def _setup(num_blocks: int = 100, block_size: int = 4) -> tuple[_rt.PyBlockManager, RadixCache]:
    bm = _rt.PyBlockManager(num_blocks, block_size)
    cache = RadixCache(bm)
    return bm, cache


@pytest.mark.unit
class TestRadixCacheMatchPrefix:
    def test_empty_tree_returns_no_match(self):
        bm, cache = _setup()
        blocks, tokens = cache.match_prefix([1, 2, 3])
        assert blocks == []
        assert tokens == 0

    def test_exact_match_after_insert(self):
        bm, cache = _setup(block_size=4)
        blks, _ = _alloc(bm, 3)
        cache.insert([1, 2, 3], blks)
        blocks, tokens = cache.match_prefix([1, 2, 3])
        assert blocks == blks
        assert tokens == 3

    def test_partial_prefix_match(self):
        bm, cache = _setup(block_size=4)
        blks, _ = _alloc(bm, 8)
        cache.insert([1, 2, 3, 4, 5, 6, 7, 8], blks)
        blocks, tokens = cache.match_prefix([1, 2, 3, 4, 5, 6])
        assert blocks == blks
        assert tokens == 6

    def test_no_match_different_first_token(self):
        bm, cache = _setup()
        blks, _ = _alloc(bm, 3)
        cache.insert([1, 2, 3], blks)
        blocks, tokens = cache.match_prefix([9, 2, 3])
        assert blocks == []
        assert tokens == 0

    def test_match_shorter_than_stored(self):
        bm, cache = _setup(block_size=4)
        blks, _ = _alloc(bm, 8)
        cache.insert([1, 2, 3, 4, 5, 6, 7, 8], blks)
        blocks, tokens = cache.match_prefix([1, 2])
        assert blocks == blks[:1]
        assert tokens == 2

    def test_match_multiple_children(self):
        bm, cache = _setup(block_size=8)
        blks1, _ = _alloc(bm, 3)
        cache.insert([1, 2, 3], blks1)
        blks2, _ = _alloc(bm, 3)
        cache.insert([1, 4, 5], blks2)

        blocks, tokens = cache.match_prefix([1, 2, 3])
        assert blocks == blks1
        assert tokens == 3

        blocks, tokens = cache.match_prefix([1, 4, 5])
        assert blocks == [blks1[0], blks2[0]]
        assert tokens == 3


@pytest.mark.unit
class TestRadixCacheInsert:
    def test_insert_single_sequence(self):
        bm, cache = _setup(block_size=4)
        blks, _ = _alloc(bm, 8)
        cache.insert([1, 2, 3, 4, 5, 6, 7, 8], blks)
        assert 1 in cache.root.children
        child = cache.root.children[1]
        assert child.token_ids == (1, 2, 3, 4, 5, 6, 7, 8)
        assert child.kv_blocks == blks
        assert child.ref_count == 1

    def test_insert_empty_tokens_noop(self):
        bm, cache = _setup()
        cache.insert([], [0])
        assert len(cache.root.children) == 0

    def test_insert_overlapping_splits_node(self):
        bm, cache = _setup(block_size=4)
        blks1, _ = _alloc(bm, 8)
        cache.insert([1, 2, 3, 4, 5, 6, 7, 8], blks1)
        blks2, _ = _alloc(bm, 6)
        cache.insert([1, 2, 3, 4, 9, 0], blks2)

        child = cache.root.children[1]
        assert child.token_ids == (1, 2, 3, 4)
        assert child.kv_blocks == blks1[:1]
        assert len(child.children) == 2

    def test_insert_diverges_at_first_token(self):
        bm, cache = _setup()
        blks1, _ = _alloc(bm, 3)
        cache.insert([1, 2, 3], blks1)
        blks2, _ = _alloc(bm, 3)
        cache.insert([7, 8, 9], blks2)
        assert len(cache.root.children) == 2

    def test_insert_extends_existing_prefix(self):
        bm, cache = _setup(block_size=4)
        blks1, _ = _alloc(bm, 3)
        cache.insert([1, 2, 3], blks1)
        blks2, _ = _alloc(bm, 5)
        cache.insert([1, 2, 3, 4, 5], blks2)
        blocks, tokens = cache.match_prefix([1, 2, 3, 4, 5])
        assert tokens == 5

    def test_insert_increments_block_ref_count(self):
        bm, cache = _setup(block_size=4)
        blks, req_id = _alloc(bm, 4)
        assert bm.num_free_blocks() == 99
        cache.insert([1, 2, 3, 4], blks)
        # Block still held by cache after request freed
        bm.free(req_id)
        assert bm.num_free_blocks() == 99  # cache ref_count prevents release


@pytest.mark.unit
class TestRadixCacheEvict:
    def test_evict_frees_unreferenced_blocks(self):
        bm, cache = _setup(block_size=4)
        blks, req_id = _alloc(bm, 4)
        cache.insert([1, 2, 3, 4], blks)
        assert bm.num_free_blocks() == 99

        # Simulate: originating request finishes → free blocks
        bm.free(req_id)
        assert bm.num_free_blocks() == 99  # still held by cache ref_count

        # Mark cache node as unreferenced
        cache.root.children[1].ref_count = 0
        freed = cache.evict(2)
        assert freed == 1
        assert bm.num_free_blocks() == 100

    def test_evict_skips_referenced_nodes(self):
        bm, cache = _setup(block_size=4)
        blks, _ = _alloc(bm, 4)
        cache.insert([1, 2, 3, 4], blks)
        freed = cache.evict(2)
        assert freed == 0
        assert bm.num_free_blocks() == 99

    def test_evict_zero_target_is_noop(self):
        bm, cache = _setup()
        freed = cache.evict(0)
        assert freed == 0

    def test_evict_only_frees_unreferenced(self):
        bm, cache = _setup(block_size=4)
        blks1, _ = _alloc(bm, 4)
        cache.insert([1, 2, 3, 4], blks1)
        blks2, _ = _alloc(bm, 4)
        cache.insert([5, 6, 7, 8], blks2)

        cache.root.children[1].ref_count = 0
        freed = cache.evict(10)
        assert freed == 1
        assert 1 not in cache.root.children
        assert 5 in cache.root.children


@pytest.mark.unit
class TestRadixCacheBlockManagerIntegration:
    def test_match_reflects_block_size(self):
        bm, cache = _setup(block_size=8)
        blks, _ = _alloc(bm, 12)
        cache.insert([1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2], blks)
        blocks, tokens = cache.match_prefix([1, 2, 3, 4])
        assert blocks == blks[:1]
        assert tokens == 4

    def test_split_blocks_distribution(self):
        bm, cache = _setup(block_size=4)
        blks1, _ = _alloc(bm, 8)
        cache.insert([1, 2, 3, 4, 5, 6, 7, 8], blks1)
        blks2, _ = _alloc(bm, 8)
        cache.insert([1, 2, 3, 4, 9, 0, 1, 2], blks2)

        child = cache.root.children[1]
        assert child.kv_blocks == blks1[:1]
        assert len(child.children) == 2

    def test_free_block_returns_to_pool(self):
        bm = _rt.PyBlockManager(10, 4)
        blks = bm.allocate("req_a", 8)
        assert bm.num_free_blocks() == 8
        bm.free_block(blks[0])
        assert bm.num_free_blocks() == 9
        bm.free_block(blks[1])
        assert bm.num_free_blocks() == 10
