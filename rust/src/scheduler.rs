//! Continuous Batching scheduler with Chunked Prefill.
//!
//! Port of `engine/scheduler.py`.  The scheduler is the runtime's
//! central decision-maker.  Each `schedule()` call:
//!
//!   1. Reaps finished requests (frees blocks, records prefix cache hints).
//!   2. Admits new requests from the priority-sorted waiting queue.
//!   3. Builds prefill chunks respecting `chunk_size` and `max_tokens_per_step`.
//!   4. Builds decode steps (one token per request).
//!   5. Returns a `Batch` with per-request token/position metadata.
//!
//! Key strategies: FCFS (default), priority queue, chunked prefill,
//! hybrid prefill+decode mixing, prefix cache integration.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::block_manager::BlockManager;
use crate::types::{Batch, PrefixCacheHit, Request, RequestState, ScheduledRequest};

// ── Priority queue entry for the waiting queue ──────────────

struct QueueEntry {
    priority: i32,
    request: Request,
}

impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse: smaller priority = higher priority (min-heap)
        other.priority.cmp(&self.priority)
    }
}

impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for QueueEntry {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for QueueEntry {}

// ── Scheduler ───────────────────────────────────────────────

pub struct Scheduler {
    max_batch_size: usize,
    max_tokens_per_step: usize,
    chunk_size: usize,
    waiting: BinaryHeap<QueueEntry>,
    running: Vec<Request>,
    request_counter: usize,
}

impl Scheduler {
    pub fn new(
        max_batch_size: usize,
        max_tokens_per_step: usize,
        chunk_size: usize,
    ) -> Result<Self, String> {
        if max_batch_size == 0 {
            return Err("max_batch_size must be positive".into());
        }
        if chunk_size == 0 {
            return Err("chunk_size must be positive".into());
        }
        Ok(Self {
            max_batch_size,
            max_tokens_per_step,
            chunk_size,
            waiting: BinaryHeap::new(),
            running: Vec::new(),
            request_counter: 0,
        })
    }

    // ── Public API ──────────────────────────────────────────

    /// Add a request to the waiting queue.  Returns the assigned request ID.
    pub fn add_request(
        &mut self,
        prompt_tokens: Vec<u32>,
        priority: i32,
        arrival_time: f64,
        max_tokens: usize,
        stop_token_ids: Vec<u32>,
        request_id: Option<String>,
    ) -> String {
        let rid = request_id.unwrap_or_else(|| {
            self.request_counter += 1;
            format!("req_{}", self.request_counter)
        });
        let req = Request::new(
            rid.clone(),
            prompt_tokens,
            priority,
            arrival_time,
            max_tokens,
            stop_token_ids,
        );
        self.waiting.push(QueueEntry { priority, request: req });
        rid
    }

    /// Run one scheduling step.  Consumes prefix cache hints from Python.
    ///
    /// Returns a `Batch` with per-request metadata.  Python uses this
    /// to build `SequenceGroup` tensors.
    pub fn schedule(
        &mut self,
        block_manager: &mut BlockManager,
        cache_hits: &[PrefixCacheHit],
    ) -> Batch {
        let finished_request_ids: Vec<String> = self.running.iter()
            .filter(|r| r.is_finished())
            .map(|r| r.request_id.clone())
            .collect();

        // Reap finished requests
        self.running.retain(|r| !r.is_finished());
        for rid in &finished_request_ids {
            block_manager.free(rid);
        }

        // Admit new requests
        while let Some(entry) = self.waiting.pop() {
            if self.running.len() >= self.max_batch_size {
                self.waiting.push(entry);
                break;
            }
            let mut req = entry.request;
            req.state = RequestState::Prefill;

            // Apply prefix cache hits
            if let Some(hit) = cache_hits.iter().find(|h| h.request_id == req.request_id) {
                if hit.matched_tokens > 0 {
                    req.prefill_pos = hit.matched_tokens;
                    block_manager.assign_cached_blocks(&req.request_id, &hit.matched_blocks);
                    if hit.matched_tokens >= req.prompt_tokens.len() {
                        req.state = RequestState::Decode;
                    }
                }
            }

            self.running.push(req);
        }

        if self.running.is_empty() {
            return Batch::empty();
        }

        self.build_batch(block_manager)
    }

    // ── Internal: batch construction ────────────────────────

    fn build_batch(&mut self, block_manager: &mut BlockManager) -> Batch {
        let mut scheduled: Vec<ScheduledRequest> = Vec::new();
        let mut total_prefill_tokens: usize = 0;
        let mut num_decode: usize = 0;

        for i in 0..self.running.len() {
            let req = &mut self.running[i];
            match req.state {
                RequestState::Prefill => {
                    let remaining = req.tokens_remaining();
                    if remaining == 0 {
                        req.state = RequestState::Decode;
                        continue;
                    }

                    let prefill_budget = self.max_tokens_per_step
                        .saturating_sub(total_prefill_tokens);
                    if prefill_budget == 0 {
                        continue;
                    }

                    let n_tokens = remaining.min(self.chunk_size).min(prefill_budget);
                    let start_pos = req.prefill_pos;
                    let end_pos = start_pos + n_tokens;

                    let chunk_ids: Vec<u32> = req.prompt_tokens[start_pos..end_pos].to_vec();
                    let positions: Vec<u32> = (start_pos as u32..end_pos as u32).collect();
                    req.prefill_pos = end_pos;

                    // Ensure blocks exist — skip request on OOM so it retries next step
                    if block_manager.block_tables.contains_key(&req.request_id) {
                        if block_manager.ensure_blocks(&req.request_id, req.prompt_tokens.len()).is_err() {
                            continue; // OOM, retry next step
                        }
                    } else if block_manager.allocate(&req.request_id, req.prompt_tokens.len()).is_err() {
                        continue; // OOM, retry next step
                    }

                    let blocks = match block_manager.get_blocks(&req.request_id) {
                        Ok(blks) => blks.to_vec(),
                        Err(_) => {
                            // Block table missing after successful allocation — logic error.
                            // Skip request to avoid passing empty block_table to Python.
                            continue;
                        }
                    };

                    if req.prefill_pos >= req.prompt_tokens.len() {
                        req.state = RequestState::Decode;
                    }

                    scheduled.push(ScheduledRequest {
                        request_id: req.request_id.clone(),
                        input_ids: chunk_ids,
                        positions,
                        state: RequestState::Prefill,
                        block_table: blocks,
                        n_tokens,
                    });

                    total_prefill_tokens += n_tokens;
                }

                RequestState::Decode => {
                    if total_prefill_tokens + num_decode + 1 > self.max_tokens_per_step {
                        continue;
                    }

                    num_decode += 1;

                    let (last_token, pos) = if let Some(last) = req.output_tokens.last() {
                        (*last, (req.prompt_tokens.len() + req.output_tokens.len() - 1) as u32)
                    } else {
                        let last = req.prompt_tokens[req.prompt_tokens.len() - 1];
                        (last, (req.prompt_tokens.len() - 1) as u32)
                    };

                    let blocks = match block_manager.get_blocks(&req.request_id) {
                        Ok(blks) => blks.to_vec(),
                        Err(_) => continue,
                    };

                    scheduled.push(ScheduledRequest {
                        request_id: req.request_id.clone(),
                        input_ids: vec![last_token],
                        positions: vec![pos],
                        state: RequestState::Decode,
                        block_table: blocks,
                        n_tokens: 1,
                    });
                }

                _ => {}
            }
        }

        let total_tokens: usize = scheduled.iter().map(|s| s.n_tokens).sum();
        Batch {
            requests: scheduled,
            total_tokens,
        }
    }

    // ── Query helpers ───────────────────────────────────────

    pub fn waiting_count(&self) -> usize {
        self.waiting.len()
    }

    pub fn running_count(&self) -> usize {
        self.running.len()
    }

    pub fn has_work(&self) -> bool {
        !self.waiting.is_empty() || !self.running.is_empty()
    }

    /// Get a reference to a running request by ID.
    pub fn running_request(&self, request_id: &str) -> Option<&Request> {
        self.running.iter().find(|r| r.request_id == request_id)
    }

    /// Get a finished request by ID (removes it from running list).
    pub fn get_finished_request(&mut self, request_id: &str) -> Option<Request> {
        let idx = self.running.iter().position(|r| r.request_id == request_id)?;
        Some(self.running.remove(idx))
    }

    /// Record output tokens for a request and check for termination.
    ///
    /// Returns `true` if the request should be marked finished.
    pub fn record_output(&mut self, request_id: &str, token_id: u32) -> bool {
        let Some(req) = self.running.iter_mut().find(|r| r.request_id == request_id) else {
            return false;
        };
        req.append_token(token_id);

        if req.output_tokens.len() >= req.max_tokens {
            req.mark_finished();
            return true;
        }
        if req.stop_token_ids.contains(&token_id) {
            req.mark_finished();
            return true;
        }
        false
    }
}

// ── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_block_manager() -> BlockManager {
        BlockManager::new(1000, 16).unwrap()
    }

    #[test]
    fn test_scheduler_creation() {
        let s = Scheduler::new(32, 512, 256).unwrap();
        assert_eq!(s.max_batch_size, 32);
        assert_eq!(s.chunk_size, 256);
        assert!(s.waiting.is_empty());
        assert!(s.running.is_empty());
    }

    #[test]
    fn test_invalid_params() {
        assert!(Scheduler::new(0, 512, 256).is_err());
        assert!(Scheduler::new(32, 512, 0).is_err());
    }

    #[test]
    fn test_add_request_returns_id() {
        let mut s = Scheduler::new(32, 512, 256).unwrap();
        let rid = s.add_request(vec![1, 2, 3], 0, 0.0, 256, vec![], None);
        assert!(rid.starts_with("req_"));
        assert_eq!(s.waiting_count(), 1);
    }

    #[test]
    fn test_add_request_with_custom_id() {
        let mut s = Scheduler::new(32, 512, 256).unwrap();
        let rid = s.add_request(vec![1, 2], 0, 0.0, 256, vec![], Some("my_id".into()));
        assert_eq!(rid, "my_id");
    }

    #[test]
    fn test_empty_schedule_returns_empty_batch() {
        let mut s = Scheduler::new(32, 512, 256).unwrap();
        let mut bm = make_block_manager();
        let batch = s.schedule(&mut bm, &[]);
        assert!(batch.is_empty());
    }

    #[test]
    fn test_single_request_prefill() {
        let mut s = Scheduler::new(32, 512, 256).unwrap();
        let mut bm = make_block_manager();
        s.add_request(vec![1, 2, 3, 4, 5], 0, 0.0, 256, vec![], None);
        let batch = s.schedule(&mut bm, &[]);
        assert_eq!(batch.requests.len(), 1);
        assert_eq!(batch.requests[0].state, RequestState::Prefill);
        assert_eq!(batch.requests[0].input_ids, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_request_transitions_to_decode() {
        let mut s = Scheduler::new(32, 512, 256).unwrap();
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
    fn test_chunked_prefill_splits_long_prompt() {
        let mut s = Scheduler::new(32, 512, 4).unwrap(); // chunk_size=4
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
        let mut s = Scheduler::new(1, 512, 256).unwrap(); // batch_size=1 to test ordering
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
        let mut s = Scheduler::new(32, 512, 256).unwrap();
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
        let mut s = Scheduler::new(32, 512, 256).unwrap();
        assert!(!s.has_work());
        s.add_request(vec![1], 0, 0.0, 256, vec![], None);
        assert!(s.has_work());
    }
}
