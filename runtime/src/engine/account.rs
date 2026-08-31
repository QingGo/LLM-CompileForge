//! SERVEFORGE_ACCOUNT step ledger.
//!
//! The account mode is deliberately opt-in and aggregate-only: it adds a
//! handful of `Instant::now` calls per function/node and prints exactly one
//! stderr line per scheduling step.  It never enables per-node printing
//! (that is `SERVEFORGE_PROFILE`'s job), so the reported numbers do not
//! suffer from the un-buffered eprintln distortion documented as F1 in
//! `.omo/plans/p1-post-d3-replan.md`.
//!
//! Attribution contract:
//! * `compute`   — kernel arithmetic (dylib execute, plan kernel execute,
//!                 fast-path numeric work, fused-layer work)
//! * `cache`     — KV cache reads/writes and per-layer block allocation
//! * `sampler`   — logits slice + argmax + sampling decision
//! * `executor`  — everything inside the forward call that is not compute
//!                 or cache (input assembly, buffer management, extraction)
//! * `schedule`  — scheduler.schedule() inside runner step
//! * `runner`    — residual orchestration (bookkeeping, tokenizer decode)
//!
//! `forward_ms == compute + executor + cache` by construction; `step_ms`
//! decomposes into `schedule + forward + cache(allocation) + sampler +
//! runner`.  A small timing overhead may remain in `runner_ms`.

use serde::Serialize;

/// True when `SERVEFORGE_ACCOUNT=1` (the only enabled value).
pub(crate) fn accounting_enabled() -> bool {
    std::env::var("SERVEFORGE_ACCOUNT")
        .ok()
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// Breakdown of one `forward_with_kv_timed` call.
#[derive(Debug, Default, Clone, Copy, Serialize)]
pub struct ForwardAccount {
    pub total_ms: f64,
    pub compute_ms: f64,
    pub cache_ms: f64,
    pub executor_ms: f64,
}

impl ForwardAccount {
    /// Derive the executor residual from the measured total.  Clamped so a
    /// few timer-resolution artifacts cannot create a negative residual.
    pub fn finalize(&mut self) {
        let accounted = self.compute_ms + self.cache_ms;
        self.executor_ms = (self.total_ms - accounted).max(0.0);
    }

    pub fn add_compute_ms(&mut self, ms: f64) {
        self.compute_ms += ms.max(0.0);
    }

    pub fn add_cache_ms(&mut self, ms: f64) {
        self.cache_ms += ms.max(0.0);
    }
}

/// Breakdown of one runner scheduling step.
#[derive(Debug, Default, Clone, Copy, Serialize)]
pub struct StepAccount {
    pub step_ms: f64,
    pub schedule_ms: f64,
    pub forward_ms: f64,
    pub compute_ms: f64,
    pub executor_ms: f64,
    pub cache_ms: f64,
    pub sampler_ms: f64,
    pub runner_ms: f64,
}

impl StepAccount {
    pub fn finalize_runner_residual(&mut self) {
        let accounted = self.schedule_ms + self.forward_ms + self.cache_ms + self.sampler_ms;
        self.runner_ms = (self.step_ms - accounted).max(0.0);
    }
}

/// Component timings returned by one op-plan invocation.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct OpPlanAccount {
    pub exec_ms: f64,
    pub cache_ms: f64,
}

/// Aggregate account report returned by the CLI benchmark path.
#[derive(Debug, Default, Clone, Serialize)]
pub struct AccountSummary {
    pub prefill_steps: usize,
    pub decode_steps: usize,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub decode_avg_ms: f64,
    pub decode_compute_ms: f64,
    pub decode_executor_ms: f64,
    pub decode_cache_ms: f64,
    pub decode_sampler_ms: f64,
    pub decode_runner_ms: f64,
    pub decode_tokens_s: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_account_residual_is_non_negative() {
        let mut a = ForwardAccount {
            total_ms: 10.0,
            compute_ms: 6.0,
            cache_ms: 2.0,
            executor_ms: 0.0,
        };
        a.finalize();
        assert!((a.executor_ms - 2.0).abs() < 1e-9);
    }

    #[test]
    fn step_account_runner_residual_absorbs_timer_overhead() {
        let mut s = StepAccount {
            step_ms: 10.0,
            schedule_ms: 1.0,
            forward_ms: 5.0,
            compute_ms: 3.0,
            executor_ms: 1.5,
            cache_ms: 0.5,
            sampler_ms: 1.0,
            runner_ms: 0.0,
        };
        s.finalize_runner_residual();
        assert!((s.runner_ms - 2.5).abs() < 1e-9);
    }
}
