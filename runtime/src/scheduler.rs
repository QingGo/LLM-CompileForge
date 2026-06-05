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

use crate::cache::block::BlockManager;
use crate::types::{Batch, PrefixCacheHit, Request, RequestState, ScheduledRequest};

// ── Priority queue entry for the waiting queue ──────────────

pub(crate) struct QueueEntry {
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
    use_kv_cache: bool,
    pub(crate) waiting: BinaryHeap<QueueEntry>,
    pub(crate) running: Vec<Request>,
    request_counter: usize,
}

impl Scheduler {
    pub fn new(
        max_batch_size: usize,
        max_tokens_per_step: usize,
        chunk_size: usize,
        use_kv_cache: bool,
    ) -> Result<Self, anyhow::Error> {
        if max_batch_size == 0 {
            return Err(anyhow::anyhow!("max_batch_size must be positive"));
        }
        if chunk_size == 0 {
            return Err(anyhow::anyhow!("chunk_size must be positive"));
        }
        Ok(Self {
            max_batch_size,
            max_tokens_per_step,
            chunk_size,
            use_kv_cache,
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
            .map(|r| r.request_id.to_owned())
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
                        block_table: blocks,
                        use_kv_cache: false,
                        kv_cache_block_table: Vec::new(),
                        n_tokens,
                    });

                    total_prefill_tokens += n_tokens;
                }

                RequestState::Decode => {
                    if total_prefill_tokens + num_decode + 1 > self.max_tokens_per_step {
                        continue;
                    }

                    num_decode += 1;

                    let blocks = match block_manager.get_blocks(&req.request_id) {
                        Ok(blks) => blks.to_vec(),
                        Err(_) => continue,
                    };

                    if self.use_kv_cache {
                        // Decode with KV cache: send only the most recent token.
                        // The KV cache already has prompt + previous output tokens,
                        // so the model only needs the last token and its absolute position.
                        let last_token = req.output_tokens.last().copied()
                            .unwrap_or_else(|| req.prompt_tokens.last().copied().unwrap_or(0));
                        let current_seq_len = req.prompt_tokens.len() + req.output_tokens.len();

                        scheduled.push(ScheduledRequest {
                            request_id: req.request_id.clone(),
                            input_ids: vec![last_token],
                            positions: vec![current_seq_len as u32],
                            block_table: blocks.clone(),
                            use_kv_cache: true,
                            kv_cache_block_table: blocks,
                            n_tokens: 1,
                        });
                    } else {
                        // Decode without KV cache: send full sequence (prompt + output)
                        let mut full_input_ids = Vec::with_capacity(
                            req.prompt_tokens.len() + req.output_tokens.len(),
                        );
                        full_input_ids.extend_from_slice(&req.prompt_tokens);
                        full_input_ids.extend_from_slice(&req.output_tokens);
                        let total_len = full_input_ids.len();
                        let positions: Vec<u32> = (0..total_len as u32).collect();

                        scheduled.push(ScheduledRequest {
                            request_id: req.request_id.clone(),
                            input_ids: full_input_ids,
                            positions,
                            block_table: blocks,
                            use_kv_cache: false,
                            kv_cache_block_table: Vec::new(),
                            n_tokens: 1,
                        });
                    }
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
#[path = "tests/scheduler_tests.rs"]
mod tests;
