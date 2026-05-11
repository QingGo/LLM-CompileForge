"""Tests for Rust RadixCache PyO3 binding."""

from __future__ import annotations

import pytest


@pytest.mark.unit
class TestRadixCacheRust:
    def test_import(self) -> None:
        from llm_serveforge_runtime import PyRadixCache
        cache = PyRadixCache(block_size=16)
        assert cache is not None

    def test_empty_match(self) -> None:
        from llm_serveforge_runtime import PyRadixCache
        cache = PyRadixCache(block_size=16)
        blocks, matched = cache.match_prefix([1, 2, 3])
        assert blocks == []
        assert matched == 0

    def test_insert_and_match_exact(self) -> None:
        from llm_serveforge_runtime import PyBlockManager, PyRadixCache

        bm = PyBlockManager(num_blocks=100, block_size=16)
        cache = PyRadixCache(block_size=16)

        tokens = list(range(32))
        blocks = bm.allocate("r1", len(tokens))
        assert len(blocks) == 2

        cache.insert(tokens, blocks, bm)
        matched, matched_len = cache.match_prefix(tokens)
        assert matched == blocks
        assert matched_len == 32

    def test_insert_and_match_partial(self) -> None:
        from llm_serveforge_runtime import PyBlockManager, PyRadixCache

        bm = PyBlockManager(num_blocks=100, block_size=16)
        cache = PyRadixCache(block_size=16)

        tokens = list(range(64))
        blocks = bm.allocate("r1", len(tokens))
        cache.insert(tokens, blocks, bm)

        query = list(range(32)) + [999]
        matched, matched_len = cache.match_prefix(query)
        assert matched_len == 32
        assert len(matched) == 2

    def test_evict_frees_blocks(self) -> None:
        from llm_serveforge_runtime import PyBlockManager, PyRadixCache

        bm = PyBlockManager(num_blocks=100, block_size=16)
        cache = PyRadixCache(block_size=16)
        initial_free = bm.num_free_blocks()

        tokens = list(range(32))
        blocks = bm.allocate("r1", len(tokens))
        cache.insert(tokens, blocks, bm)

        bm.free("r1")
        freed = cache.evict(10, bm)
        assert freed == 2
        assert bm.num_free_blocks() == initial_free

    def test_cached_blocks_count(self) -> None:
        from llm_serveforge_runtime import PyBlockManager, PyRadixCache

        bm = PyBlockManager(num_blocks=100, block_size=16)
        cache = PyRadixCache(block_size=16)
        assert cache.cached_blocks() == 0

        tokens = list(range(32))
        blocks = bm.allocate("r1", len(tokens))
        cache.insert(tokens, blocks, bm)
        assert cache.cached_blocks() == 2

    def test_node_count(self) -> None:
        from llm_serveforge_runtime import PyBlockManager, PyRadixCache

        bm = PyBlockManager(num_blocks=100, block_size=16)
        cache = PyRadixCache(block_size=16)
        assert cache.node_count() == 1  # root only

        tokens = list(range(16))
        blocks = bm.allocate("r1", len(tokens))
        cache.insert(tokens, blocks, bm)
        assert cache.node_count() == 2  # root + child
