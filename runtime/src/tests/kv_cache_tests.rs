//! Integration tests for KV cache block lifecycle.
//!
//! Tests the BlockManager's KV cache operations end-to-end:
//! allocation → write → read across block boundaries, multi-request
//! isolation, flush/zeroing, and pool-exhaustion error handling.
//!
//! These tests use `#[cfg(test)] mod kv_cache_integration` so they can
//! be invoked with `cargo test --lib kv_cache_integration`.

#[cfg(test)]
mod kv_cache_integration {
    use crate::block_manager::{BlockEntry, BlockManager};

    // ── Helpers ──────────────────────────────────────────────────

    /// Create a BlockManager backed by KVCacheBlocks with small
    /// dimensions for fast tests.
    fn make_cache_bm() -> BlockManager {
        BlockManager::new_with_cache(10, 4, 2, 3).unwrap()
    }

    const fn hidden_dim() -> usize {
        2 * 3 // num_kv_heads * head_dim
    }

    // ── Test 1: Allocation + write + read lifecycle ──────────────

    #[test]
    fn test_lifecycle_allocate_write_read() {
        let mut bm = make_cache_bm();
        let hd = hidden_dim();
        let n_tokens = 3;

        bm.allocate("req", n_tokens).unwrap();

        // Create deterministic key/value data where every element is
        // distinct so we can verify position-by-position.
        let key_data: Vec<f32> = (0..n_tokens * hd).map(|i| i as f32).collect();
        let val_data: Vec<f32> = (0..n_tokens * hd).map(|i| (i + 100) as f32).collect();

        bm.write_kv("req", 0, &key_data, hd, true).unwrap();
        bm.write_kv("req", 0, &val_data, hd, false).unwrap();

        // Read back all tokens in a single call.
        let (k, v) = bm.read_kv("req", n_tokens, hd).unwrap();

        assert_eq!(k.len(), n_tokens * hd, "key output length");
        assert_eq!(v.len(), n_tokens * hd, "value output length");
        assert_eq!(k, key_data, "Key data mismatch position-by-position");
        assert_eq!(v, val_data, "Value data mismatch position-by-position");
    }

    // ── Test 2: Cross-block boundary read ────────────────────────

    #[test]
    fn test_cross_block_boundary() {
        // block_size=4 → 4 slots per block.
        // Writing 6 tokens spans block 0 (slots 0–3) and block 1
        // (slots 0–1).  This exercises the intra-block → contiguous
        // → next-block data layout paths in read_kv.
        let mut bm = BlockManager::new_with_cache(10, 4, 2, 3).unwrap();
        let hd = hidden_dim();
        let n_tokens = 6;

        bm.allocate("req", n_tokens).unwrap();

        let key_data: Vec<f32> = (0..n_tokens * hd).map(|i| i as f32).collect();
        let val_data: Vec<f32> = (0..n_tokens * hd).map(|i| (i + 100) as f32).collect();

        bm.write_kv("req", 0, &key_data, hd, true).unwrap();
        bm.write_kv("req", 0, &val_data, hd, false).unwrap();

        // Read back all 6 tokens — this crosses the block boundary.
        let (k, v) = bm.read_kv("req", n_tokens, hd).unwrap();
        assert_eq!(k.len(), n_tokens * hd);
        assert_eq!(v.len(), n_tokens * hd);

        // Spot-check the boundary slot (slot 3 = last of block 0)
        // and the first slot of the next block (slot 4 = first of block 1).
        let slot3_off = 3 * hd;
        let slot4_off = 4 * hd;
        for i in 0..hd {
            let expected_k3 = (slot3_off + i) as f32;
            let expected_k4 = (slot4_off + i) as f32;
            assert!(
                (k[slot3_off + i] - expected_k3).abs() < 1e-6,
                "Block 0 slot 3 K[{}]: expected {}, got {}",
                i,
                expected_k3,
                k[slot3_off + i]
            );
            assert!(
                (k[slot4_off + i] - expected_k4).abs() < 1e-6,
                "Block 1 slot 0 K[{}]: expected {}, got {}",
                i,
                expected_k4,
                k[slot4_off + i]
            );
            let expected_v3 = (slot3_off + i + 100) as f32;
            let expected_v4 = (slot4_off + i + 100) as f32;
            assert!(
                (v[slot3_off + i] - expected_v3).abs() < 1e-6,
                "Block 0 slot 3 V[{}]: expected {}, got {}",
                i,
                expected_v3,
                v[slot3_off + i]
            );
            assert!(
                (v[slot4_off + i] - expected_v4).abs() < 1e-6,
                "Block 1 slot 0 V[{}]: expected {}, got {}",
                i,
                expected_v4,
                v[slot4_off + i]
            );
        }
    }

    // ── Test 3: Multi-request isolation ──────────────────────────

    #[test]
    fn test_multi_request_key_isolation() {
        let mut bm = make_cache_bm();
        let hd = hidden_dim();
        let n_tokens = 2;

        // Allocate disjoint physical blocks for each request.
        bm.allocate("req_a", n_tokens).unwrap();
        bm.allocate("req_b", n_tokens).unwrap();

        // Write distinct key data.
        let key_a = vec![1.0f32; n_tokens * hd];
        let key_b = vec![99.0f32; n_tokens * hd];
        bm.write_kv("req_a", 0, &key_a, hd, true).unwrap();
        bm.write_kv("req_b", 0, &key_b, hd, true).unwrap();

        // Read back each request's keys.
        let (k_a, _) = bm.read_kv("req_a", n_tokens, hd).unwrap();
        let (k_b, _) = bm.read_kv("req_b", n_tokens, hd).unwrap();

        // req_a contains only its own data.
        assert!(
            k_a.iter().all(|&x| (x - 1.0).abs() < 1e-6),
            "req_a key data leaked or corrupted"
        );
        // req_b contains only its own data.
        assert!(
            k_b.iter().all(|&x| (x - 99.0).abs() < 1e-6),
            "req_b key data leaked or corrupted"
        );
        // Cross-check: req_a does NOT contain req_b's data.
        assert!(
            k_a.iter().all(|&x| (x - 99.0).abs() > 1e-6),
            "req_a key data contaminated with req_b's values"
        );
    }

    #[test]
    fn test_multi_request_value_isolation() {
        let mut bm = make_cache_bm();
        let hd = hidden_dim();
        let n_tokens = 2;

        bm.allocate("req_x", n_tokens).unwrap();
        bm.allocate("req_y", n_tokens).unwrap();

        // Same keys, different values.
        let key = vec![5.0f32; n_tokens * hd];
        let val_x = vec![10.0f32; n_tokens * hd];
        let val_y = vec![20.0f32; n_tokens * hd];

        bm.write_kv("req_x", 0, &key, hd, true).unwrap();
        bm.write_kv("req_x", 0, &val_x, hd, false).unwrap();
        bm.write_kv("req_y", 0, &key, hd, true).unwrap();
        bm.write_kv("req_y", 0, &val_y, hd, false).unwrap();

        let (_, v_x) = bm.read_kv("req_x", n_tokens, hd).unwrap();
        let (_, v_y) = bm.read_kv("req_y", n_tokens, hd).unwrap();

        assert!(
            v_x.iter().all(|&x| (x - 10.0).abs() < 1e-6),
            "req_x value data leaked or corrupted"
        );
        assert!(
            v_y.iter().all(|&x| (x - 20.0).abs() < 1e-6),
            "req_y value data leaked or corrupted"
        );
    }

    // ── Test 4: Flush zeros cache data ───────────────────────────

    #[test]
    fn test_flush_zeros_cache_data() {
        let mut bm = make_cache_bm();
        let hd = hidden_dim();

        bm.allocate("req", 4).unwrap();
        let block_ids = bm.get_blocks("req").unwrap().to_vec();

        // Write non-zero data.
        let key_data = vec![42.0f32; 4 * hd];
        let val_data = vec![84.0f32; 4 * hd];
        bm.write_kv("req", 0, &key_data, hd, true).unwrap();
        bm.write_kv("req", 0, &val_data, hd, false).unwrap();

        // Confirm data was written.
        let (k_before, v_before) = bm.read_kv("req", 4, hd).unwrap();
        assert!(
            k_before.iter().any(|&x| x != 0.0),
            "key data should be non-zero before flush"
        );
        assert!(
            v_before.iter().any(|&x| x != 0.0),
            "value data should be non-zero before flush"
        );

        // Flush — this must zero underlying buffers then free blocks.
        bm.flush_request("req");

        // Request must be gone from block tables.
        assert!(
            bm.get_blocks("req").is_err(),
            "request should be unknown after flush"
        );

        // Each physical block's cache data must be zeroed.
        for &bid in &block_ids {
            match &bm.blocks[&bid] {
                BlockEntry::Cached(kv) => {
                    assert!(
                        kv.key_cache.iter().all(|&x| x == 0.0),
                        "block {} key_cache not zeroed after flush",
                        bid
                    );
                    assert!(
                        kv.value_cache.iter().all(|&x| x == 0.0),
                        "block {} value_cache not zeroed after flush",
                        bid
                    );
                }
                _ => panic!("block {} is not Cached variant", bid),
            }
        }

        // All blocks are back in the free pool.
        assert_eq!(
            bm.num_free_blocks(),
            bm.num_blocks,
            "all blocks should be free after flush"
        );
    }

    // ── Test 5: Eviction / pool exhaustion → graceful error ─────

    #[test]
    fn test_pool_exhaustion_graceful_error() {
        // Only 2 blocks in the pool.
        let mut bm = BlockManager::new_with_cache(2, 4, 2, 3).unwrap();

        // Use both blocks.
        bm.allocate("req_a", 4).unwrap(); // 1 block
        bm.allocate("req_b", 4).unwrap(); // 1 block

        assert_eq!(bm.num_free_blocks(), 0, "pool should be exhausted");

        // One more allocation must fail with OutOfMemoryError.
        let result = bm.allocate("req_c", 4);
        assert!(result.is_err(), "expected OutOfMemoryError");

        match result {
            Err(err) => {
                assert_eq!(err.needed, 1, "needed 1 block");
                assert_eq!(err.free, 0, "0 free blocks");
                assert_eq!(err.total, 2, "total pool is 2");
            }
            Ok(_) => unreachable!(),
        }

        // After freeing a block, allocation must succeed again.
        bm.free("req_a");
        assert_eq!(bm.num_free_blocks(), 1);

        let ok = bm.allocate("req_c", 4);
        assert!(ok.is_ok(), "allocation should succeed after freeing");

        // Verify the newly allocated block works for KV write/read.
        let hd = hidden_dim();
        let key_data = vec![7.0f32; 4 * hd];
        bm.write_kv("req_c", 0, &key_data, hd, true).unwrap();
        let (k, _) = bm.read_kv("req_c", 4, hd).unwrap();
        assert!(
            k.iter().all(|&x| (x - 7.0).abs() < 1e-6),
            "re-allocated block should accept writes"
        );
    }
}
