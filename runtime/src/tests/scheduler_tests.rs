use super::*;

fn make_block_manager() -> BlockManager {
    BlockManager::new(1000, 16).unwrap()
}

#[test]
fn test_scheduler_creation() {
    let s = Scheduler::new(32, 512, 256, false).unwrap();
    assert_eq!(s.max_batch_size, 32);
    assert_eq!(s.chunk_size, 256);
    assert!(s.waiting.is_empty());
    assert!(s.running.is_empty());
}

#[test]
fn test_invalid_params() {
    assert!(Scheduler::new(0, 512, 256, false).is_err());
    assert!(Scheduler::new(32, 512, 0, false).is_err());
}

#[test]
fn test_add_request_returns_id() {
    let mut s = Scheduler::new(32, 512, 256, false).unwrap();
    let rid = s.add_request(vec![1, 2, 3], 0, 0.0, 256, vec![], None);
    assert!(rid.starts_with("req_"));
    assert_eq!(s.waiting_count(), 1);
}

#[test]
fn test_stop_token_termination() {
    let mut s = Scheduler::new(32, 512, 256, false).unwrap();
    let mut bm = BlockManager::new(1000, 16).unwrap();
    let rid = s.add_request(vec![1, 2, 3], 0, 0.0, 256, vec![7, 8], None);
    s.schedule(&mut bm, &[]);  // move to running
    assert!(!s.running_request(&rid).unwrap().is_finished());
    s.record_output(&rid, 5);  // not a stop token
    assert!(!s.running_request(&rid).unwrap().is_finished());
    s.record_output(&rid, 7);  // stop token
    assert!(s.running_request(&rid).unwrap().is_finished());
}

#[test]
fn test_add_request_with_custom_id() {
    let mut s = Scheduler::new(32, 512, 256, false).unwrap();
    let rid = s.add_request(vec![1, 2], 0, 0.0, 256, vec![], Some("my_id".into()));
    assert_eq!(rid, "my_id");
}

#[test]
fn test_empty_schedule_returns_empty_batch() {
    let mut s = Scheduler::new(32, 512, 256, false).unwrap();
    let mut bm = make_block_manager();
    let batch = s.schedule(&mut bm, &[]);
    assert!(batch.is_empty());
}

#[test]
fn test_single_request_prefill() {
    let mut s = Scheduler::new(32, 512, 256, false).unwrap();
    let mut bm = make_block_manager();
    s.add_request(vec![1, 2, 3, 4, 5], 0, 0.0, 256, vec![], None);
    let batch = s.schedule(&mut bm, &[]);
    assert_eq!(batch.requests.len(), 1);
    assert_eq!(batch.requests[0].state, RequestState::Prefill);
    assert_eq!(batch.requests[0].input_ids, vec![1, 2, 3, 4, 5]);
}

#[test]
fn test_request_transitions_to_decode() {
    let mut s = Scheduler::new(32, 512, 256, false).unwrap();
    let mut bm = make_block_manager();
    s.add_request(vec![1, 2, 3], 0, 0.0, 256, vec![], None);

    // First schedule: prefill all 3 tokens
    let batch = s.schedule(&mut bm, &[]);
    assert_eq!(batch.requests.len(), 1);
    assert_eq!(batch.requests[0].state, RequestState::Prefill);
    assert_eq!(batch.requests[0].n_tokens, 3);

    // Second schedule: now in decode
    let batch = s.schedule(&mut bm, &[]);
    assert_eq!(batch.requests.len(), 1);
    assert_eq!(batch.requests[0].state, RequestState::Decode);
}

#[test]
fn test_scheduler_decode_single_token() {
    let mut s = Scheduler::new(32, 512, 256, true).unwrap(); // use_kv_cache=true
    let mut bm = BlockManager::new(1000, 16).unwrap();
    s.add_request(vec![1, 2, 3, 4], 0, 0.0, 256, vec![], None);

    // First schedule: prefill all 4 tokens
    let batch = s.schedule(&mut bm, &[]);
    assert_eq!(batch.requests.len(), 1);
    assert_eq!(batch.requests[0].state, RequestState::Prefill);
    assert_eq!(batch.requests[0].input_ids, vec![1, 2, 3, 4]);

    // Second schedule: decode should send single token with position = current_seq_len
    let batch = s.schedule(&mut bm, &[]);
    assert_eq!(batch.requests.len(), 1);
    assert_eq!(batch.requests[0].state, RequestState::Decode);
    assert!(batch.requests[0].use_kv_cache);
    assert_eq!(batch.requests[0].input_ids.len(), 1);
    // prompt_tokens.len()=4 + output_tokens.len()=0 = current_seq_len=4
    assert_eq!(batch.requests[0].positions, vec![4]);
    assert!(!batch.requests[0].kv_cache_block_table.is_empty(),
        "kv_cache_block_table must be populated when use_kv_cache=true");

    // Record a generated token and schedule again
    s.record_output("req_1", 42);
    let batch2 = s.schedule(&mut bm, &[]);
    assert_eq!(batch2.requests.len(), 1);
    assert_eq!(batch2.requests[0].input_ids.len(), 1);
    // prompt_tokens.len()=4 + output_tokens.len()=1 = current_seq_len=5
    assert_eq!(batch2.requests[0].positions, vec![5]);
    // Last token should be the recorded output token
    assert_eq!(batch2.requests[0].input_ids[0], 42);
}

#[test]
fn test_chunked_prefill_splits_long_prompt() {
    let mut s = Scheduler::new(32, 512, 4, false).unwrap(); // chunk_size=4
    let mut bm = make_block_manager();
    let prompt: Vec<u32> = (0..10).collect();
    s.add_request(prompt, 0, 0.0, 256, vec![], None);

    // First step: 4 tokens (chunk_size)
    let batch = s.schedule(&mut bm, &[]);
    assert_eq!(batch.requests.len(), 1);
    assert_eq!(batch.requests[0].state, RequestState::Prefill);
    assert_eq!(batch.requests[0].n_tokens, 4);

    // Second step: next 4 tokens
    let batch = s.schedule(&mut bm, &[]);
    assert_eq!(batch.requests[0].n_tokens, 4);

    // Third step: last 2 tokens
    let batch = s.schedule(&mut bm, &[]);
    assert_eq!(batch.requests[0].n_tokens, 2); // remaining

    // Fourth step: decode
    let batch = s.schedule(&mut bm, &[]);
    assert_eq!(batch.requests[0].state, RequestState::Decode);
}

#[test]
fn test_priority_queue_order() {
    let mut s = Scheduler::new(1, 512, 256, false).unwrap(); // batch_size=1 to test ordering
    let mut bm = make_block_manager();

    s.add_request(vec![7], 0, 0.0, 256, vec![], None);
    s.add_request(vec![2], 5, 0.0, 256, vec![], None);
    s.add_request(vec![3], 10, 0.0, 256, vec![], None);

    // First admitted should be priority 0 (lowest value = highest priority)
    let batch = s.schedule(&mut bm, &[]);
    assert_eq!(batch.requests.len(), 1);
    assert_eq!(batch.requests[0].request_id, "req_1");
}

#[test]
fn test_finished_request_reaped() {
    let mut s = Scheduler::new(32, 512, 256, false).unwrap();
    let mut bm = make_block_manager();

    let rid = s.add_request(vec![1], 0, 0.0, 256, vec![], None);
    let _ = s.schedule(&mut bm, &[]); // prefill
    let _ = s.schedule(&mut bm, &[]); // decode starts

    // Mark as finished
    s.record_output(&rid, 42); // not a stop token, check max_tokens
    // Fast-forward past max_tokens
    for _ in 0..255 {
        if let Some(req) = s.running.iter_mut().find(|r| r.request_id == rid) {
            req.append_token(42);
        }
    }
    s.record_output(&rid, 99); // Should hit max_tokens=256

    // Next schedule should reap
    let batch = s.schedule(&mut bm, &[]);
    assert!(batch.is_empty());
}

#[test]
fn test_has_work() {
    let mut s = Scheduler::new(32, 512, 256, false).unwrap();
    assert!(!s.has_work());
    s.add_request(vec![1], 0, 0.0, 256, vec![], None);
    assert!(s.has_work());
}

// ── Property-based tests ───────────────────────────────────

/// Invariant: after schedule, running requests never exceed max_batch_size.
#[test]
fn prop_schedule_never_exceeds_batch_size() {
    use proptest::prelude::*;
    proptest!(|(n_requests in 0..50usize)| {
        let mut bm = BlockManager::new(10000, 16).unwrap();
        let mut s = Scheduler::new(8, 512, 64, false).unwrap();
        for i in 0..n_requests {
            let prompt: Vec<u32> = (0..((i % 10) + 1) as u32).collect();
            s.add_request(prompt, 0, 0.0, 64, vec![], None);
        }
        let batch = s.schedule(&mut bm, &[]);
        assert!(batch.requests.len() <= 8,
                "batch size {} exceeds max_batch_size=8", batch.requests.len());
        for req in &batch.requests {
            assert!(!req.block_table.is_empty(),
                    "scheduled request {} has empty block table", req.request_id);
        }
    });
}

/// Invariant: scheduling with no waiting requests returns empty batch.
#[test]
fn prop_schedule_empty_returns_empty() {
    use proptest::prelude::*;
    proptest!(|(max_batch in 1..32usize)| {
        let mut bm = BlockManager::new(100, 16).unwrap();
        let mut s = Scheduler::new(max_batch, 512, 64, false).unwrap();
        let batch = s.schedule(&mut bm, &[]);
        assert!(batch.requests.is_empty());
    });
}

/// Invariant: recorded output increments output_tokens.
#[test]
fn prop_record_output_increments_tokens() {
    use proptest::prelude::*;
    proptest!(|(n_tokens in 1..20usize)| {
        let mut bm = BlockManager::new(100, 16).unwrap();
        let mut s = Scheduler::new(4, 512, 64, false).unwrap();
        s.add_request(vec![1, 2, 3], 0, 0.0, 256, vec![], None);
        s.schedule(&mut bm, &[]);
        let rid = s.running[0].request_id.clone();
        let initial = s.running[0].output_tokens.len();
        for t in 0..n_tokens {
            s.record_output(&rid, t as u32);
        }
        assert_eq!(s.running[0].output_tokens.len() - initial, n_tokens,
                   "output_tokens must grow by the number of record_output calls");
    });
}
