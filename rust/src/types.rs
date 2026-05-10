//! Shared data types for the Rust scheduler and block manager.
//!
//! These types are the FFI bridge between Rust and Python. They carry
//! only metadata (integers, strings) — no tensor references. Python
//! converts them into PyTorch tensors after receiving a Batch from
//! `Scheduler::schedule()`.

use std::fmt;

/// The lifecycle state of an inference request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestState {
    Waiting,
    Prefill,
    Decode,
    Finished,
}

impl fmt::Display for RequestState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RequestState::Waiting => write!(f, "waiting"),
            RequestState::Prefill => write!(f, "prefill"),
            RequestState::Decode => write!(f, "decode"),
            RequestState::Finished => write!(f, "finished"),
        }
    }
}

/// An inference request tracked by the scheduler.
#[derive(Debug, Clone)]
pub struct Request {
    pub request_id: String,
    pub prompt_tokens: Vec<u32>,
    pub output_tokens: Vec<u32>,
    pub state: RequestState,
    pub priority: i32,
    pub arrival_time: f64,
    pub prefill_pos: usize,
    pub max_tokens: usize,
    pub stop_token_ids: Vec<u32>,
}

impl Request {
    pub fn new(
        request_id: String,
        prompt_tokens: Vec<u32>,
        priority: i32,
        arrival_time: f64,
        max_tokens: usize,
        stop_token_ids: Vec<u32>,
    ) -> Self {
        Self {
            request_id,
            prompt_tokens,
            output_tokens: Vec::new(),
            state: RequestState::Waiting,
            priority,
            arrival_time,
            prefill_pos: 0,
            max_tokens,
            stop_token_ids,
        }
    }

    pub fn is_finished(&self) -> bool {
        self.state == RequestState::Finished
    }

    pub fn tokens_remaining(&self) -> usize {
        self.prompt_tokens.len().saturating_sub(self.prefill_pos)
    }

    pub fn mark_finished(&mut self) {
        self.state = RequestState::Finished;
    }

    pub fn append_token(&mut self, token_id: u32) {
        self.output_tokens.push(token_id);
    }
}

/// A single request as scheduled for one forward pass step.
///
/// Contains everything the Python executor needs to build input tensors
/// and manage KV cache blocks for this request.
#[derive(Debug, Clone)]
pub struct ScheduledRequest {
    pub request_id: String,
    pub input_ids: Vec<u32>,
    pub positions: Vec<u32>,
    pub state: RequestState,
    pub block_table: Vec<usize>,
    pub n_tokens: usize,
}

/// The output of one `schedule()` call.  All metadata — no tensors.
///
/// Python uses this to construct `SequenceGroup` tensors and pass them
/// to the executor.
#[derive(Debug, Clone)]
pub struct Batch {
    pub requests: Vec<ScheduledRequest>,
    pub total_tokens: usize,
}

impl Batch {
    pub fn empty() -> Self {
        Self {
            requests: Vec::new(),
            total_tokens: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }
}

/// Prefix cache hint passed from Python to Rust during scheduling.
///
/// When the Python-side `RadixCache` finds a prefix match for an
/// incoming request, it passes the matched block IDs and token count
/// to the scheduler so the BlockManager can assign the cached blocks.
#[derive(Debug, Clone)]
pub struct PrefixCacheHit {
    pub request_id: String,
    pub matched_blocks: Vec<usize>,
    pub matched_tokens: usize,
}
