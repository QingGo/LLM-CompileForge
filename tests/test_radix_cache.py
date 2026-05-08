"""Unit tests for RadixTree prefix cache (cache/radix_cache.py)."""

from __future__ import annotations

import pytest
import torch

from cache.radix_cache import RadixCache
from engine.batch import SamplingParams
from engine.block_manager import BlockManager
from engine.scheduler import Scheduler

_counter = 0


def _alloc(bm: BlockManager, n_tokens: int) -> tuple[list[int], str]:
    """Allocate blocks for *n_tokens* tokens. Returns (blocks, request_id)."""
    global _counter
    _counter += 1
    req_id = f"__test_{_counter}"
    return bm.allocate(req_id, n_tokens), req_id


def _setup(num_blocks: int = 100, block_size: int = 4) -> tuple[BlockManager, RadixCache]:
    bm = BlockManager(num_blocks=num_blocks, block_size=block_size)
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
        blks, _ = _alloc(bm, 4)
        assert bm.blocks[blks[0]].ref_count == 1
        cache.insert([1, 2, 3, 4], blks)
        assert bm.blocks[blks[0]].ref_count == 2


@pytest.mark.unit
class TestRadixCacheEvict:
    def test_evict_frees_unreferenced_blocks(self):
        bm, cache = _setup(block_size=4)
        blks, req_id = _alloc(bm, 4)  # ref_count: 1
        cache.insert([1, 2, 3, 4], blks)  # ref_count: 2
        assert bm.num_free_blocks() == 99

        # Simulate: originating request finishes → free
        bm.free(req_id)
        assert bm.num_free_blocks() == 99  # still held by cache
        assert bm.blocks[blks[0]].ref_count == 1

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
        bm = BlockManager(num_blocks=10, block_size=4)
        blks = bm.allocate("req_a", 8)
        assert bm.num_free_blocks() == 8
        bm.free_block(blks[0])
        assert bm.num_free_blocks() == 9
        bm.free_block(blks[1])
        assert bm.num_free_blocks() == 10

    def test_free_block_respects_ref_count(self):
        bm = BlockManager(num_blocks=10, block_size=4)
        blks = bm.allocate("req_a", 4)
        bm.blocks[blks[0]].ref_count += 1
        bm.free_block(blks[0])
        assert bm.num_free_blocks() == 9
        bm.free_block(blks[0])
        assert bm.num_free_blocks() == 10


# ═══════════════════════════════════════════════════════════
# Scheduler + RadixCache end-to-end integration tests
# ═══════════════════════════════════════════════════════════


@pytest.mark.integration
@pytest.mark.timeout(5)
class TestRadixCacheSchedulerIntegration:
    def test_cache_hit_reduces_prefill(self):
        bm = BlockManager(num_blocks=100, block_size=4)
        cache = RadixCache(bm)
        scheduler = Scheduler(
            max_batch_size=4,
            max_tokens_per_step=100,
            chunk_size=100,
            radix_cache=cache,
        )

        # Request A: 8 tokens, generates 1 token then stops
        scheduler.add_request(
            [1, 2, 3, 4, 5, 6, 7, 8],
            SamplingParams(temperature=0, max_tokens=1, stop_token_ids=[]),
            request_id="A",
        )

        # Step 1: prefill A (all 8 tokens in one chunk)
        batch = scheduler.schedule(bm)
        req_a = [r for r in batch.requests if r.request_id == "A"][0]
        assert req_a.state == "decode"
        assert req_a.prefill_pos == 8
        blocks_a = bm.get_blocks("A")
        assert len(blocks_a) == 2  # 8 tokens / 4 block_size

        # Simulate model forward + decode
        scheduler.process_outputs(torch.randn(1, 100), batch)
        assert req_a.is_finished

        # Step 2: reap A → cache insert + block free
        scheduler.schedule(bm)
        assert "A" not in bm.block_tables
        # Cache should hold the block references
        matched, matched_tokens = cache.match_prefix([1, 2, 3, 4, 5, 6, 7, 8])
        assert matched_tokens == 8
        assert matched == blocks_a

        # Request B: shares prefix [1,2,3,4], diverges at [9,0]
        scheduler.add_request(
            [1, 2, 3, 4, 9, 0],
            SamplingParams(temperature=0, max_tokens=1, stop_token_ids=[]),
            request_id="B",
        )

        # Step 3: prefill B (cache hit on first 4 tokens)
        batch3 = scheduler.schedule(bm)
        req_b = [r for r in batch3.requests if r.request_id == "B"][0]
        assert req_b.prefill_pos >= 4, f"expected >=4, got {req_b.prefill_pos}"
        blocks_b = bm.get_blocks("B")
        assert len(blocks_b) >= 2
        assert blocks_b[0] == blocks_a[0]

    def test_fully_cached_request_skips_prefill(self):
        bm = BlockManager(num_blocks=100, block_size=4)
        cache = RadixCache(bm)
        scheduler = Scheduler(
            max_batch_size=4,
            max_tokens_per_step=100,
            chunk_size=100,
            radix_cache=cache,
        )

        scheduler.add_request(
            [10, 20, 30, 40],
            SamplingParams(temperature=0, max_tokens=1, stop_token_ids=[]),
            request_id="A",
        )
        batch = scheduler.schedule(bm)
        scheduler.process_outputs(torch.randn(1, 100), batch)
        scheduler.schedule(bm)

        # Request B: exact same prompt → fully cached
        scheduler.add_request(
            [10, 20, 30, 40],
            SamplingParams(temperature=0, max_tokens=1, stop_token_ids=[]),
            request_id="B",
        )
        batch3 = scheduler.schedule(bm)
        req_b = [r for r in batch3.requests if r.request_id == "B"][0]
        assert req_b.state == "decode"
        assert req_b.prefill_pos >= 4

        blocks_b = bm.get_blocks("B")
        assert len(blocks_b) == 1

    def test_no_cache_when_disabled(self):
        bm = BlockManager(num_blocks=100, block_size=4)
        scheduler = Scheduler(
            max_batch_size=4,
            max_tokens_per_step=100,
            chunk_size=100,
        )

        scheduler.add_request(
            [1, 2, 3, 4],
            SamplingParams(temperature=0, max_tokens=1, stop_token_ids=[]),
            request_id="A",
        )
        batch = scheduler.schedule(bm)
        assert len(batch.requests) == 1

    def test_block_ref_counts_after_cache_sharing(self):
        bm = BlockManager(num_blocks=100, block_size=4)
        cache = RadixCache(bm)
        scheduler = Scheduler(
            max_batch_size=4,
            max_tokens_per_step=100,
            chunk_size=100,
            radix_cache=cache,
        )

        scheduler.add_request(
            [1, 2, 3, 4, 5, 6, 7, 8],
            SamplingParams(temperature=0, max_tokens=1, stop_token_ids=[]),
            request_id="A",
        )
        batch = scheduler.schedule(bm)
        scheduler.process_outputs(torch.randn(1, 100), batch)
        scheduler.schedule(bm)

        blocks_a = batch.block_tables.get("A", [])
        if blocks_a:
            assert bm.blocks[blocks_a[0]].ref_count >= 1

        scheduler.add_request(
            [1, 2, 3, 4, 9, 0],
            SamplingParams(temperature=0, max_tokens=1, stop_token_ids=[]),
            request_id="B",
        )
        scheduler.schedule(bm)

        blocks_b = bm.get_blocks("B")
        shared_block = blocks_b[0]
        assert bm.blocks[shared_block].ref_count >= 2

