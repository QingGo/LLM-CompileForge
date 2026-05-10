//! Integration tests for the Rust scheduler and block manager.
//!
//! Tests the public API cross-module behavior: full prefill→decode→finish
//! lifecycle, chunked prefill, block allocation/deallocation, prefix
//! cache hit injection, and termination conditions.
//!
//! Placed inline (not in tests/) because pyo3 cdylib projects cannot
//! compile external integration tests without linking Python symbols.
//! These tests only exercise the pure-Rust types, not the pyo3 bindings.

#[cfg(test)]
mod integration_tests {
    use crate::block_manager::BlockManager;
    use crate::scheduler::Scheduler;
    use crate::types::{Batch, PrefixCacheHit, RequestState, ScheduledRequest};

    fn make_bm() -> BlockManager {
        BlockManager::new(1000, 16).unwrap()
    }

    fn make_scheduler(max_batch: usize, max_tokens: usize, chunk: usize) -> Scheduler {
        Scheduler::new(max_batch, max_tokens, chunk).unwrap()
    }

    // ── Scheduler integration tests ──────────────────────────

    #[test]
    fn test_full_prefill_decode_finish_cycle() {
        let mut s = make_scheduler(32, 512, 256);
        let mut bm = make_bm();

        let rid = s.add_request(vec![10, 20, 30], 0, 0.0, 2, vec![], Some("req_a".into()));
        assert_eq!(rid, "req_a");

        let batch = s.schedule(&mut bm, &[]);
        assert_eq!(batch.requests.len(), 1);
        assert_eq!(batch.requests[0].state, RequestState::Prefill);
        assert_eq!(batch.requests[0].input_ids, vec![10, 20, 30]);
        assert_eq!(batch.requests[0].positions, vec![0, 1, 2]);
        assert_eq!(batch.requests[0].n_tokens, 3);
        assert!(!batch.requests[0].block_table.is_empty());

        let batch = s.schedule(&mut bm, &[]);
        assert_eq!(batch.requests[0].state, RequestState::Decode);
        assert!(!s.record_output("req_a", 42));

        let _ = s.schedule(&mut bm, &[]);
        assert!(s.record_output("req_a", 99));

        let batch = s.schedule(&mut bm, &[]);
        assert!(batch.is_empty());
        assert!(!s.has_work());
    }

    #[test]
    fn test_stop_token_termination() {
        let mut s = make_scheduler(32, 512, 256);
        let mut bm = make_bm();

        s.add_request(vec![1, 2], 0, 0.0, 100, vec![13], Some("req".into()));
        let _ = s.schedule(&mut bm, &[]);
        let _ = s.schedule(&mut bm, &[]);
        assert!(s.record_output("req", 13));

        let batch = s.schedule(&mut bm, &[]);
        assert!(batch.is_empty());
    }

    #[test]
    fn test_chunked_prefill_sequence() {
        let mut s = make_scheduler(32, 512, 4);
        let mut bm = make_bm();
        s.add_request((0..10).collect(), 0, 0.0, 256, vec![], None);

        let b1 = s.schedule(&mut bm, &[]);
        assert_eq!(b1.requests[0].n_tokens, 4);
        assert_eq!(b1.requests[0].input_ids, vec![0, 1, 2, 3]);

        let b2 = s.schedule(&mut bm, &[]);
        assert_eq!(b2.requests[0].n_tokens, 4);
        assert_eq!(b2.requests[0].input_ids, vec![4, 5, 6, 7]);

        let b3 = s.schedule(&mut bm, &[]);
        assert_eq!(b3.requests[0].n_tokens, 2);
        assert_eq!(b3.requests[0].input_ids, vec![8, 9]);

        let b4 = s.schedule(&mut bm, &[]);
        assert_eq!(b4.requests[0].state, RequestState::Decode);
    }

    #[test]
    fn test_multiple_requests_mixed_prefill_decode() {
        let mut s = make_scheduler(32, 50, 256);
        let mut bm = make_bm();

        s.add_request((0..40).collect(), 0, 0.0, 256, vec![], Some("long".into()));
        s.add_request(vec![1, 2, 3], 0, 0.0, 256, vec![], Some("short".into()));

        let batch = s.schedule(&mut bm, &[]);
        assert_eq!(batch.requests.len(), 2);
        let total: usize = batch.requests.iter().map(|r| r.n_tokens).sum();
        assert!(total <= 50);

        let long = batch.requests.iter().find(|r| r.request_id == "long").unwrap();
        assert_eq!(long.state, RequestState::Prefill);

        let short = batch.requests.iter().find(|r| r.request_id == "short").unwrap();
        assert_eq!(short.n_tokens, 3);
    }

    #[test]
    fn test_block_table_persists_across_steps() {
        let mut s = make_scheduler(32, 512, 256);
        let mut bm = make_bm();

        s.add_request(vec![1, 2, 3, 4, 5], 0, 0.0, 256, vec![], Some("r".into()));

        let b1 = s.schedule(&mut bm, &[]);
        let blocks_1 = b1.requests[0].block_table.clone();

        let b2 = s.schedule(&mut bm, &[]);
        assert_eq!(blocks_1, b2.requests[0].block_table);
    }

    #[test]
    fn test_priority_ordering_with_batch_limit() {
        let mut s = make_scheduler(1, 512, 256);
        let mut bm = make_bm();

        s.add_request(vec![7], 10, 0.0, 256, vec![], Some("low".into()));
        s.add_request(vec![3], 0, 0.0, 256, vec![], Some("high".into()));
        s.add_request(vec![5], 5, 0.0, 256, vec![], Some("mid".into()));

        let b1 = s.schedule(&mut bm, &[]);
        assert_eq!(b1.requests[0].request_id, "high");
    }

    #[test]
    fn test_has_work_transitions() {
        let mut s = make_scheduler(32, 512, 256);
        assert!(!s.has_work());
        assert_eq!(s.waiting_count(), 0);
        assert_eq!(s.running_count(), 0);

        s.add_request(vec![1], 0, 0.0, 1, vec![], None);
        assert!(s.has_work());
        assert_eq!(s.waiting_count(), 1);
        assert_eq!(s.running_count(), 0);

        let mut bm = make_bm();
        let _ = s.schedule(&mut bm, &[]);
        assert_eq!(s.running_count(), 1);

        let _ = s.schedule(&mut bm, &[]);
        s.record_output("req_1", 42);
        let _ = s.schedule(&mut bm, &[]);
        assert!(!s.has_work());
        assert_eq!(s.running_count(), 0);
    }

    #[test]
    fn test_record_output_unknown_request_returns_false() {
        let mut s = make_scheduler(32, 512, 256);
        assert!(!s.record_output("nonexistent", 42));
    }

    #[test]
    fn test_custom_request_id_generation() {
        let mut s = make_scheduler(32, 512, 256);

        let r1 = s.add_request(vec![1], 0, 0.0, 256, vec![], None);
        let r2 = s.add_request(vec![2], 0, 0.0, 256, vec![], Some("my_id".into()));
        let r3 = s.add_request(vec![3], 0, 0.0, 256, vec![], None);

        assert_eq!(r1, "req_1");
        assert_eq!(r2, "my_id");
        assert_eq!(r3, "req_2"); // counter incremented twice (r1, r3), not three times
    }

    #[test]
    fn test_max_tokens_limit_termination() {
        let mut s = make_scheduler(32, 512, 256);
        let mut bm = make_bm();

        s.add_request(vec![1], 0, 0.0, 3, vec![], Some("t".into()));

        let _ = s.schedule(&mut bm, &[]);
        let _ = s.schedule(&mut bm, &[]);
        assert!(!s.record_output("t", 10));

        let _ = s.schedule(&mut bm, &[]);
        assert!(!s.record_output("t", 20));

        let _ = s.schedule(&mut bm, &[]);
        assert!(s.record_output("t", 30));

        let _ = s.schedule(&mut bm, &[]);
        assert!(!s.has_work());
    }

    #[test]
    fn test_batch_total_tokens_accurate() {
        let mut s = make_scheduler(32, 512, 256);
        let mut bm = make_bm();

        s.add_request(vec![1, 2, 3], 0, 0.0, 256, vec![], Some("a".into()));
        s.add_request(vec![4, 5, 6, 7, 8], 0, 0.0, 256, vec![], Some("b".into()));

        let batch = s.schedule(&mut bm, &[]);
        let manual_sum: usize = batch.requests.iter().map(|r| r.n_tokens).sum();
        assert_eq!(batch.total_tokens, manual_sum);
    }

    // ── BlockManager integration tests ───────────────────────

    #[test]
    fn test_allocate_free_cycle() {
        let mut bm = make_bm();
        let blks = bm.allocate("req", 50).unwrap();
        assert_eq!(blks.len(), 4);
        assert_eq!(bm.num_free_blocks(), 996);
        bm.free("req");
        assert_eq!(bm.num_free_blocks(), 1000);
    }

    #[test]
    fn test_block_sharing_ref_count() {
        let mut bm = make_bm();

        bm.allocate("src", 32).unwrap();
        bm.share_prefix("src", "dst", 16).unwrap();
        bm.free("src");
        assert_eq!(bm.num_free_blocks(), 999);
        bm.free("dst");
        assert_eq!(bm.num_free_blocks(), 1000);
    }

    #[test]
    fn test_out_of_memory_and_recovery() {
        let mut bm = BlockManager::new(10, 16).unwrap();

        assert!(bm.allocate("big", 200).is_err());

        let blks = bm.allocate("small", 32).unwrap();
        assert_eq!(blks.len(), 2);
        bm.free("small");
        assert_eq!(bm.num_free_blocks(), 10);
    }

    #[test]
    fn test_ensure_blocks_expansion() {
        let mut bm = make_bm();

        bm.ensure_blocks("req", 16).unwrap();
        assert_eq!(bm.get_blocks("req").unwrap().len(), 1);

        bm.ensure_blocks("req", 48).unwrap();
        assert_eq!(bm.get_blocks("req").unwrap().len(), 3);
    }

    #[test]
    fn test_assign_cached_blocks_then_ensure() {
        let mut bm = make_bm();

        bm.assign_cached_blocks("req", &[10, 11]);
        assert_eq!(bm.get_blocks("req").unwrap(), &[10, 11]);

        bm.ensure_blocks("req", 64).unwrap();
        assert_eq!(bm.get_blocks("req").unwrap().len(), 4);
        assert_eq!(bm.get_blocks("req").unwrap()[0], 10);
        assert_eq!(bm.get_blocks("req").unwrap()[1], 11);
    }

    #[test]
    fn test_increment_ref_count_then_free_block() {
        let mut bm = make_bm();

        bm.allocate("req", 16).unwrap();
        let bid = bm.get_blocks("req").unwrap()[0];

        bm.increment_ref_count(bid);
        bm.free("req");
        assert_eq!(bm.num_free_blocks(), 999);

        bm.free_block(bid);
        assert_eq!(bm.num_free_blocks(), 1000);
    }

    #[test]
    fn test_utilization_tracks_allocations() {
        let mut bm = BlockManager::new(100, 16).unwrap();
        assert!((bm.utilization() - 0.0).abs() < 0.001);

        bm.allocate("req", 160).unwrap();
        assert!((bm.utilization() - 0.10).abs() < 0.001);

        bm.allocate("req2", 320).unwrap();
        assert!((bm.utilization() - 0.30).abs() < 0.001);

        bm.free("req");
        assert!((bm.utilization() - 0.20).abs() < 0.001);
    }

    #[test]
    fn test_multiple_independent_requests_no_block_conflict() {
        let mut bm = make_bm();

        let a = bm.allocate("a", 32).unwrap();
        let b = bm.allocate("b", 32).unwrap();

        let a_set: std::collections::HashSet<usize> = a.into_iter().collect();
        let b_set: std::collections::HashSet<usize> = b.into_iter().collect();
        assert!(a_set.is_disjoint(&b_set));
    }

    // ── Cross-module integration tests ───────────────────────

    #[test]
    fn test_prefix_cache_hit_skips_prefill() {
        let mut s = make_scheduler(32, 512, 256);
        let mut bm = make_bm();

        let cache_block = bm.allocate("_cache_owner", 64).unwrap();
        let cached_bid = cache_block[0];

        s.add_request(vec![1, 2, 3, 4, 5, 6], 0, 0.0, 256, vec![], Some("hit".into()));

        let hits = vec![PrefixCacheHit {
            request_id: "hit".into(),
            matched_blocks: vec![cached_bid],
            matched_tokens: 4,
        }];

        let batch = s.schedule(&mut bm, &hits);
        assert_eq!(batch.requests[0].input_ids, vec![5, 6]);
        assert_eq!(batch.requests[0].positions, vec![4, 5]);
        assert_eq!(batch.requests[0].n_tokens, 2);
        assert_eq!(batch.requests[0].block_table[0], cached_bid);
    }

    #[test]
    fn test_fully_cached_request_goes_straight_to_decode() {
        let mut s = make_scheduler(32, 512, 256);
        let mut bm = make_bm();

        let cache_blocks = bm.allocate("_cache", 48).unwrap();

        s.add_request(vec![1, 2, 3], 0, 0.0, 256, vec![], Some("full".into()));

        let hits = vec![PrefixCacheHit {
            request_id: "full".into(),
            matched_blocks: cache_blocks.clone(),
            matched_tokens: 3,
        }];

        let batch = s.schedule(&mut bm, &hits);
        assert_eq!(batch.requests[0].state, RequestState::Decode);
        assert_eq!(batch.requests[0].n_tokens, 1);
    }
}
