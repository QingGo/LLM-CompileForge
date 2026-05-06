import pytest

from engine.block_manager import BlockManager, OutOfMemoryError

# ═══════════════════════════════════════════════════════════
# BlockManager
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
class TestBlockManager:
    def test_creation(self):
        bm = BlockManager(num_blocks=100, block_size=16)
        assert bm.block_size == 16
        assert bm.num_blocks == 100
        assert bm.num_free_blocks() == 100

    def test_creation_invalid_params(self):
        with pytest.raises(ValueError, match="num_blocks"):
            BlockManager(num_blocks=0)
        with pytest.raises(ValueError, match="block_size"):
            BlockManager(num_blocks=10, block_size=0)

    def test_allocate_basic(self):
        bm = BlockManager(num_blocks=100, block_size=16)
        blocks = bm.allocate("req_1", num_tokens=50)
        # ceil(50/16) = 4 blocks
        assert len(blocks) == 4
        assert bm.num_free_blocks() == 96

    def test_allocate_exact_block_boundary(self):
        bm = BlockManager(num_blocks=100, block_size=16)
        blocks = bm.allocate("req_1", num_tokens=16)
        assert len(blocks) == 1

    def test_allocate_one_token_needs_one_block(self):
        bm = BlockManager(num_blocks=100, block_size=16)
        blocks = bm.allocate("req_1", num_tokens=1)
        assert len(blocks) == 1

    def test_allocate_duplicate_request_raises(self):
        bm = BlockManager(num_blocks=100, block_size=16)
        bm.allocate("req_1", num_tokens=10)
        with pytest.raises(ValueError, match="already has allocated"):
            bm.allocate("req_1", num_tokens=5)

    def test_free(self):
        bm = BlockManager(num_blocks=100, block_size=16)
        bm.allocate("req_1", num_tokens=50)
        assert bm.num_free_blocks() == 96
        bm.free("req_1")
        assert bm.num_free_blocks() == 100

    def test_free_unknown_request_noop(self):
        bm = BlockManager(num_blocks=100, block_size=16)
        bm.free("nonexistent")  # Should not raise

    def test_get_blocks(self):
        bm = BlockManager(num_blocks=100, block_size=16)
        allocated = bm.allocate("req_1", num_tokens=30)
        assert bm.get_blocks("req_1") == allocated

    def test_get_blocks_unknown_raises(self):
        bm = BlockManager(num_blocks=100, block_size=16)
        with pytest.raises(KeyError, match="Unknown request_id"):
            bm.get_blocks("nonexistent")

    def test_utilization(self):
        bm = BlockManager(num_blocks=100, block_size=16)
        assert bm.utilization == 0.0
        bm.allocate("req_1", num_tokens=160)  # 10 blocks
        assert bm.utilization == 0.1

    def test_out_of_memory(self):
        bm = BlockManager(num_blocks=5, block_size=16)
        bm.allocate("req_1", num_tokens=80)  # 5 blocks
        assert bm.num_free_blocks() == 0
        with pytest.raises(OutOfMemoryError):
            bm.allocate("req_2", num_tokens=16)  # Need 1 more

    def test_share_prefix(self):
        bm = BlockManager(num_blocks=100, block_size=16)
        bm.allocate("src", num_tokens=80)  # 5 blocks
        shared = bm.share_prefix("src", "dst", prefix_len=32)
        # prefix_len=32 → ceil(32/16) = 2 blocks
        assert len(shared) == 2
        # dst should have 2 blocks now
        assert bm.get_blocks("dst") == shared[:2]

    def test_share_prefix_from_unknown_src_raises(self):
        bm = BlockManager(num_blocks=100, block_size=16)
        with pytest.raises(ValueError, match="Source request"):
            bm.share_prefix("unknown", "dst", prefix_len=16)

    def test_share_prefix_too_large_raises(self):
        bm = BlockManager(num_blocks=100, block_size=16)
        bm.allocate("src", num_tokens=16)  # 1 block
        with pytest.raises(ValueError, match="cannot share"):
            bm.share_prefix("src", "dst", prefix_len=32)  # Need 2 blocks

    def test_free_shared_block_decrements_ref(self):
        bm = BlockManager(num_blocks=100, block_size=16)
        bm.allocate("src", num_tokens=80)  # 5 blocks
        bm.share_prefix("src", "dst", prefix_len=16)  # Share 1 block
        # Free src — 4 unshared blocks returned to pool, 1 shared stays
        bm.free("src")
        assert bm.num_free_blocks() == 99  # 95 + 4 freed, 1 still shared
        bm.free("dst")
        assert bm.num_free_blocks() == 100  # All freed

    def test_multiple_allocations(self):
        bm = BlockManager(num_blocks=100, block_size=16)
        b1 = bm.allocate("r1", num_tokens=32)  # 2 blocks
        b2 = bm.allocate("r2", num_tokens=48)  # 3 blocks
        assert len(b1) == 2
        assert len(b2) == 3
        assert bm.num_free_blocks() == 95
        bm.free("r1")
        assert bm.num_free_blocks() == 97
