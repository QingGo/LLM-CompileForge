//! Inference runner — orchestrates the full autoregressive loop.
//!
//! Integrates:
//!   - ``Tokenizer`` for prompt encoding / output decoding
//!   - ``ModelExecutor`` for forward passes
//!   - ``Sampler`` for token selection
//!   - ``Scheduler`` + ``BlockManager`` + ``RadixCache`` for request management
//!
//! Usage (CLI):
//!   let mut runner = InferenceRunner::new(executor, tokenizer, seed, max_tokens)?;
//!   runner.add_request("Hello, who are you?", temperature=0.7)?;
//!   while runner.has_work() {
//!       let results = runner.step(&sampling)?;
//!       for r in &results { print!("{}", r.text); }
//!   }

use std::collections::HashMap;

use crate::cache::block::BlockManager;
use crate::cache::policy::CachePolicy;
use crate::cache::radix::RadixCache;
use crate::engine::account::{accounting_enabled, AccountSummary, StepAccount};
use crate::engine::executor::ModelExecutor;
use crate::engine::sampler::{Sampler, SamplerConfig};
use crate::engine::scheduler::Scheduler;
use crate::engine::tokenizer::{ChatMessage, Tokenizer};
use crate::engine::types::RequestState;

/// A single result from one step for one request.
#[derive(Debug, Clone)]
pub struct StepResult {
    pub request_id: String,
    pub token_id: u32,
    pub text: String,
    pub finished: bool,
}

/// Final generation result for a complete request.
#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub text: String,
    #[allow(dead_code)]
    pub tokens: Vec<u32>,
    /// Prefill wall time: add_request -> first output token (or the whole
    /// run when no output token is produced).
    pub prefill_ms: f64,
    /// Decode wall time across all steps that produced tokens 2..=N.
    pub decode_ms: f64,
    /// Step ledger when `SERVEFORGE_ACCOUNT=1`; `None` in normal runs.
    pub account: Option<AccountSummary>,
}

/// Configuration for the inference runner.
#[derive(Debug, Clone)]
pub struct RunnerConfig {
    pub max_batch_size: usize,
    pub max_tokens_per_step: usize,
    pub chunk_size: usize,
    pub num_blocks: usize,
    pub block_size: usize,
    pub max_tokens_per_request: usize,
    pub seed: u64,
    pub use_chat_template: bool,
    /// Enable KV cache block table for decode requests.
    pub use_kv_cache: bool,
    /// Number of KV heads (required when use_kv_cache=true).
    pub num_kv_heads: usize,
    /// Head dimension (required when use_kv_cache=true).
    pub head_dim: usize,
}

/// Tracks per-request KV cache allocation state.
pub struct KVCacheState {
    /// Map from request_id to cache slab identifier.
    /// Used to track which requests have KV cache blocks allocated.
    pub request_id_to_cache: HashMap<String, String>,
}

impl KVCacheState {
    pub fn new() -> Self {
        Self {
            request_id_to_cache: HashMap::new(),
        }
    }
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            max_tokens_per_step: 128,
            chunk_size: 64,
            num_blocks: 1024,
            block_size: 16,
            max_tokens_per_request: 512,
            seed: 42,
            use_chat_template: true,
            use_kv_cache: false,
            num_kv_heads: 0,
            head_dim: 0,
        }
    }
}

impl RunnerConfig {
    /// Apply a compiled model's cache policy to this runner configuration.
    ///
    /// KV mode is enabled only when the policy carries both bound intercepts
    /// and slabs with `heads`/`dim` dimensions.  Models compiled without a
    /// cache policy (or legacy JSON-only policy) keep `use_kv_cache=false`
    /// and the full-sequence recompute path.
    pub fn with_cache_policy(mut self, policy: &CachePolicy) -> Self {
        if !policy.intercepts.is_empty() && !policy.slabs.is_empty() {
            let heads = policy
                .slabs
                .first()
                .and_then(|s| s.dims.get("heads"))
                .copied()
                .unwrap_or(0);
            let dim = policy
                .slabs
                .first()
                .and_then(|s| s.dims.get("dim"))
                .copied()
                .unwrap_or(0);
            if heads > 0 && dim > 0 {
                self.use_kv_cache = true;
                self.num_kv_heads = heads;
                self.head_dim = dim;
                if policy.block_size > 0 {
                    self.block_size = policy.block_size;
                }
            } else {
                log::warn!(
                    "cache policy present but KV dimensions missing (heads={}, dim={}) — keeping legacy path",
                    heads,
                    dim,
                );
            }
        }
        self
    }
}

/// Inference runner — wires scheduler, executor, sampler, tokenizer.
pub struct InferenceRunner {
    executor: ModelExecutor,
    scheduler: Scheduler,
    block_manager: BlockManager,
    radix_cache: RadixCache,
    sampler: Sampler,
    tokenizer: Tokenizer,
    config: RunnerConfig,
    /// Cache prefix hits from the previous step (fed into scheduler).
    cache_hits: Vec<crate::engine::types::PrefixCacheHit>,
    /// Track per-request last logits position for logits→token extraction.
    #[allow(dead_code)]
    last_positions: HashMap<String, usize>,
    /// KV cache state (only used when config.use_kv_cache is true).
    kv_cache_state: Option<KVCacheState>,
    /// Function indices whose consumed outputs own a per-layer KV block
    /// table (`<request_id>_f<func_index>`).
    kv_layer_ids: Vec<usize>,
}

impl InferenceRunner {
    /// Create a new inference runner with default config.
    pub fn new(
        executor: ModelExecutor,
        tokenizer: Tokenizer,
        config: RunnerConfig,
    ) -> Result<Self, anyhow::Error> {
        let scheduler = Scheduler::new(
            config.max_batch_size,
            config.max_tokens_per_step,
            config.chunk_size,
            config.use_kv_cache,
        )
        .map_err(|e| anyhow::anyhow!("scheduler init: {}", e))?;

        let block_manager = if config.use_kv_cache && config.num_kv_heads > 0 && config.head_dim > 0
        {
            BlockManager::new_with_cache(
                config.num_blocks,
                config.block_size,
                config.num_kv_heads,
                config.head_dim,
            )
            .map_err(|e| anyhow::anyhow!("block_manager init (with cache): {}", e))?
        } else {
            BlockManager::new(config.num_blocks, config.block_size)
                .map_err(|e| anyhow::anyhow!("block_manager init: {}", e))?
        };

        let radix_cache = RadixCache::new(config.block_size);

        let mut kv_layer_ids: Vec<usize> = if config.use_kv_cache {
            if executor.cache_policy.intercepts.is_empty() {
                executor
                    .compute_graph
                    .functions
                    .iter()
                    .filter(|f| f.outputs.iter().any(|o| o.consumed_internally))
                    .map(|f| f.index)
                    .collect()
            } else {
                executor
                    .cache_policy
                    .intercepts
                    .iter()
                    .map(|i| i.func_index)
                    .collect()
            }
        } else {
            Vec::new()
        };
        kv_layer_ids.sort_unstable();
        kv_layer_ids.dedup();

        let kv_cache_state = if config.use_kv_cache {
            Some(KVCacheState::new())
        } else {
            None
        };

        Ok(Self {
            executor,
            scheduler,
            block_manager,
            radix_cache,
            sampler: Sampler::new(config.seed),
            tokenizer,
            config,
            cache_hits: Vec::new(),
            last_positions: HashMap::new(),
            kv_cache_state,
            kv_layer_ids,
        })
    }

    // ── Request management ──────────────────────────────────

    /// Add a prompt to the waiting queue.  Returns the assigned request ID.
    pub fn add_request(
        &mut self,
        prompt: &str,
        sampling: SamplerConfig,
    ) -> Result<String, anyhow::Error> {
        let formatted = if self.config.use_chat_template && self.tokenizer.has_chat_template() {
            let messages = vec![ChatMessage::user(prompt)];
            self.tokenizer.apply_chat_template(&messages, true)?
        } else {
            prompt.to_string()
        };

        let input_ids = self.tokenizer.encode(&formatted)?;
        log::debug!("encode prompt={:?} input_ids={:?}", formatted, input_ids);
        self.add_request_tokens(input_ids, sampling)
    }

    /// Add an already-tokenized prompt to the waiting queue.
    ///
    /// This is the reproducibility path used by ``--prompt-ids``: the same
    /// integer list can be passed to the Python engine and the Rust CLI
    /// without going through either tokenizer implementation.
    pub fn add_request_tokens(
        &mut self,
        input_ids: Vec<u32>,
        sampling: SamplerConfig,
    ) -> Result<String, anyhow::Error> {
        if input_ids.is_empty() {
            anyhow::bail!("empty prompt token ids");
        }

        let stop_ids = self.tokenizer.stop_token_ids();
        let rid = self.scheduler.add_request(
            input_ids,
            0, // priority
            sampling
                .max_tokens
                .unwrap_or(self.config.max_tokens_per_request),
            stop_ids,
            None,
        );

        // Try prefix cache match
        let prompt_tokens: &[u32] = self
            .scheduler
            .running_request(&rid)
            .map(|r| r.prompt_tokens.as_slice())
            .unwrap_or(&[]);
        let (matched_blocks, matched_tokens) = self.radix_cache.match_prefix(prompt_tokens);
        if !matched_blocks.is_empty() {
            // Assign cached blocks to the new request
            self.block_manager
                .assign_cached_blocks(&rid, &matched_blocks);
            self.cache_hits.push(crate::engine::types::PrefixCacheHit {
                request_id: rid.clone(),
                matched_blocks,
                matched_tokens,
            });
        }

        Ok(rid)
    }

    /// Number of token slots a scheduled KV write needs to cover.
    fn kv_target_tokens(req: &crate::engine::types::ScheduledRequest) -> usize {
        if req.n_tokens == 1 {
            // Decode: the fed token's position is current_seq_len-1 and the
            // scheduler already grows the main table to
            // prompt_len+output_len+1.  positions[0]+2 reproduces that same
            // target for the per-layer tables.
            req.positions.first().map(|&p| p as usize + 2).unwrap_or(1)
        } else {
            // Prefill chunk: positions are absolute and increasing.
            req.positions
                .last()
                .map(|&p| p as usize + 1)
                .unwrap_or(req.n_tokens)
        }
    }

    fn ensure_kv_layer_blocks(
        &mut self,
        request_id: &str,
        target_tokens: usize,
    ) -> Result<(), anyhow::Error> {
        if !self.config.use_kv_cache {
            return Ok(());
        }
        for &fi in &self.kv_layer_ids {
            let layer_rid = format!("{}_f{}", request_id, fi);
            if self.block_manager.block_tables.contains_key(&layer_rid) {
                self.block_manager
                    .ensure_blocks(&layer_rid, target_tokens)
                    .map_err(|e| anyhow::anyhow!("KV layer {} OOM: {}", layer_rid, e))?;
            } else {
                self.block_manager
                    .allocate(&layer_rid, target_tokens)
                    .map_err(|e| anyhow::anyhow!("KV layer {} OOM: {}", layer_rid, e))?;
            }
        }
        Ok(())
    }

    fn flush_kv_layer_blocks(&mut self, request_id: &str) {
        for &fi in &self.kv_layer_ids {
            let layer_rid = format!("{}_f{}", request_id, fi);
            self.block_manager.flush_request(&layer_rid);
        }
    }

    /// Run one scheduling step.  Returns results for all requests in the batch.
    pub fn step(&mut self, sampling: &SamplerConfig) -> Result<Vec<StepResult>, anyhow::Error> {
        self.step_inner(sampling, false).map(|(results, _)| results)
    }

    /// Accounted variant used by `SERVEFORGE_ACCOUNT=1`.
    pub fn step_timed(
        &mut self,
        sampling: &SamplerConfig,
    ) -> Result<(Vec<StepResult>, StepAccount), anyhow::Error> {
        let (results, account) = self.step_inner(sampling, true)?;
        let account = account.expect("collect=true always returns StepAccount");
        Ok((results, account))
    }

    fn step_inner(
        &mut self,
        sampling: &SamplerConfig,
        collect: bool,
    ) -> Result<(Vec<StepResult>, Option<StepAccount>), anyhow::Error> {
        let step_t0 = collect.then(std::time::Instant::now);
        let mut account = collect.then(StepAccount::default);

        // 1. Schedule: get the batch of requests to process
        let schedule_t0 = collect.then(std::time::Instant::now);
        let cache_hits = std::mem::take(&mut self.cache_hits);
        let batch = self
            .scheduler
            .schedule(&mut self.block_manager, &cache_hits);
        if let Some(account) = account.as_mut() {
            account.schedule_ms = schedule_t0
                .expect("collect=true schedules a timer")
                .elapsed()
                .as_secs_f64()
                * 1e3;
        }

        if batch.requests.is_empty() {
            if let Some(mut account) = account {
                account.step_ms = step_t0
                    .expect("collect=true always has a step timer")
                    .elapsed()
                    .as_secs_f64()
                    * 1e3;
                account.finalize_runner_residual();
                return Ok((Vec::new(), Some(account)));
            }
            return Ok((Vec::new(), None));
        }

        let mut results: Vec<StepResult> = Vec::with_capacity(batch.requests.len());

        // 2. Process each request individually (single-request forward)
        for req in &batch.requests {
            let input_ids = &req.input_ids;
            if input_ids.is_empty() {
                continue;
            }

            // Each consumed-output function has its own per-layer block
            // table (`<rid>_f<fi>`).  Grow it together with the main table
            // before prefill writes / decode read+write.
            if self.config.use_kv_cache {
                let cache_t0 = collect.then(std::time::Instant::now);
                let target_tokens = Self::kv_target_tokens(req);
                self.ensure_kv_layer_blocks(&req.request_id, target_tokens)?;
                if let Some(account) = account.as_mut() {
                    account.cache_ms += cache_t0
                        .expect("collect=true schedules a cache timer")
                        .elapsed()
                        .as_secs_f64()
                        * 1e3;
                }
            }

            // Forward pass — KV cache path or standard path.  The runner
            // decision is global (`config.use_kv_cache`): prefill chunks
            // must also go through `forward_with_kv` so their K/V outputs
            // are written to the block manager before the first decode.
            let logits_tensor = if self.config.use_kv_cache {
                // Track request in KVCacheState
                if let Some(ref mut kv_state) = self.kv_cache_state {
                    kv_state
                        .request_id_to_cache
                        .entry(req.request_id.clone())
                        .or_insert_with(|| "default".to_string());
                }
                if collect {
                    let (tensor, forward_account) = self.executor.forward_with_kv_accounted(
                        input_ids,
                        &req.positions,
                        Some(&mut self.block_manager),
                        Some(&req.request_id),
                    )?;
                    let account = account
                        .as_mut()
                        .expect("collect=true always has StepAccount");
                    account.forward_ms = forward_account.total_ms;
                    account.compute_ms = forward_account.compute_ms;
                    account.executor_ms = forward_account.executor_ms;
                    account.cache_ms += forward_account.cache_ms;
                    tensor
                } else {
                    self.executor.forward_with_kv(
                        input_ids,
                        &req.positions,
                        Some(&mut self.block_manager),
                        Some(&req.request_id),
                    )?
                }
            } else if collect {
                let (tensor, forward_account) = self
                    .executor
                    .forward_with_positions_accounted(input_ids, &req.positions)?;
                let account = account
                    .as_mut()
                    .expect("collect=true always has StepAccount");
                account.forward_ms = forward_account.total_ms;
                account.compute_ms = forward_account.compute_ms;
                account.executor_ms = forward_account.executor_ms;
                tensor
            } else {
                self.executor
                    .forward_with_positions(input_ids, &req.positions)?
            };
            let all_logits = logits_tensor.as_slice();
            if all_logits.is_empty() {
                continue;
            }

            // Intermediate prefill chunks must not sample: their
            // last-position logits come from partial context (the rest of
            // the prompt has not been seen yet).  The final chunk carries
            // RequestState::Decode (the scheduler flips the state once the
            // whole prompt has been scheduled), so sampling there is
            // correct and yields the first output token.
            if req.state == RequestState::Prefill {
                continue;
            }

            // Extract last-position logits
            let vocab = self.vocab_size();
            let logits = if all_logits.len() >= req.n_tokens * vocab {
                &all_logits[all_logits.len() - vocab..]
            } else {
                &all_logits[all_logits.len().saturating_sub(vocab)..]
            };

            log::debug!("logits all_logits.len={} req.n_tokens={} vocab={} extracted_len={} logits[..5]={:?}",
                all_logits.len(), req.n_tokens, self.vocab_size(), logits.len(), &logits[..5.min(logits.len())]);

            // Sample token.  The greedy argmax is computed inside the
            // sampler; the separate argmax below is debug-only and must not
            // pay for a second 50k scan on the hot decode path.
            let sampler_t0 = collect.then(std::time::Instant::now);
            let token_id = self.sampler.sample(logits, sampling);
            if let Some(account) = account.as_mut() {
                account.sampler_ms = sampler_t0
                    .expect("collect=true schedules a sampler timer")
                    .elapsed()
                    .as_secs_f64()
                    * 1e3;
            }
            if log::log_enabled!(log::Level::Debug) {
                let argmax_idx = logits
                    .iter()
                    .enumerate()
                    .fold((0, f32::NEG_INFINITY), |(mi, mv), (i, &v)| {
                        if v > mv {
                            (i, v)
                        } else {
                            (mi, mv)
                        }
                    })
                    .0;
                log::debug!(
                    "logits argmax_idx={} argmax_val={}",
                    argmax_idx,
                    logits[argmax_idx]
                );
            }

            log::debug!(
                "step input_ids.len={} positions.len={} n_tokens={} token_id={} text={:?}",
                input_ids.len(),
                req.positions.len(),
                req.n_tokens,
                token_id,
                self.tokenizer.decode_token(token_id)
            );

            // Record output in scheduler
            let finished = self.scheduler.record_output(&req.request_id, token_id);

            // Update prefix cache if finished
            if finished {
                if let Some(request) = self.scheduler.get_finished_request(&req.request_id) {
                    let block_table = self
                        .block_manager
                        .get_blocks(&req.request_id)
                        .unwrap_or_default()
                        .to_vec();
                    let all_tokens: Vec<u32> = request
                        .prompt_tokens
                        .iter()
                        .chain(request.output_tokens.iter())
                        .copied()
                        .collect();
                    if !all_tokens.is_empty() {
                        self.radix_cache
                            .insert(&all_tokens, &block_table, &mut self.block_manager);
                    }
                }

                // Flush KV cache for finished request (after radix cache uses block table)
                if self.config.use_kv_cache {
                    if let Some(ref mut kv_state) = self.kv_cache_state {
                        kv_state.request_id_to_cache.remove(&req.request_id);
                    }
                    self.block_manager.flush_request(&req.request_id);
                    self.flush_kv_layer_blocks(&req.request_id);
                }
            }

            // Decode token
            let text = self.tokenizer.decode_token(token_id);

            results.push(StepResult {
                request_id: req.request_id.clone(),
                token_id,
                text,
                finished,
            });
        }

        if let Some(mut account) = account {
            account.step_ms = step_t0
                .expect("collect=true always has a step timer")
                .elapsed()
                .as_secs_f64()
                * 1e3;
            account.finalize_runner_residual();
            Ok((results, Some(account)))
        } else {
            Ok((results, None))
        }
    }

    // ─── Convenience: full generate() ────────────────────────

    /// Generate a response for a text prompt using the autoregressive loop.
    pub fn generate(
        &mut self,
        prompt: &str,
        temperature: f32,
        top_p: f32,
        top_k: usize,
    ) -> Result<GenerationResult, anyhow::Error> {
        let sampling = SamplerConfig {
            temperature,
            top_p,
            top_k,
            ..Default::default()
        };
        if let Err(e) = sampling.validate() {
            anyhow::bail!("invalid sampling config: {}", e);
        }
        let rid = self.add_request(prompt, sampling.clone())?;
        self.generate_request(rid, &sampling)
    }

    /// Generate a response from explicit prompt token ids.
    ///
    /// Token ids are interpreted literally (no chat template, no tokenizer
    /// encode), which makes this path comparable to the Python engine's
    /// ``add_request(token_ids, ...)`` contract.
    pub fn generate_from_tokens(
        &mut self,
        prompt_ids: &[u32],
        temperature: f32,
        top_p: f32,
        top_k: usize,
    ) -> Result<GenerationResult, anyhow::Error> {
        let sampling = SamplerConfig {
            temperature,
            top_p,
            top_k,
            ..Default::default()
        };
        if let Err(e) = sampling.validate() {
            anyhow::bail!("invalid sampling config: {}", e);
        }
        let rid = self.add_request_tokens(prompt_ids.to_vec(), sampling.clone())?;
        self.generate_request(rid, &sampling)
    }

    fn generate_request(
        &mut self,
        rid: String,
        sampling: &SamplerConfig,
    ) -> Result<GenerationResult, anyhow::Error> {
        let mut output_tokens: Vec<u32> = Vec::new();
        let mut prefill_ms = 0.0f64;
        let mut decode_ms = 0.0f64;
        let mut first_token_seen = false;
        let account_enabled = accounting_enabled();
        let mut account_summary = if account_enabled {
            Some(AccountSummary::default())
        } else {
            None
        };
        let mut step_no = 0usize;

        while self.scheduler.has_work() {
            let step_t0 = if account_enabled {
                None
            } else {
                Some(std::time::Instant::now())
            };
            let (results, step_account) = if account_enabled {
                let (results, account) = self.step_timed(sampling)?;
                (results, Some(account))
            } else {
                (self.step(sampling)?, None)
            };
            let elapsed_ms = if let Some(account) = step_account.as_ref() {
                account.step_ms
            } else {
                step_t0
                    .expect("non-accounted generate always has a step timer")
                    .elapsed()
                    .as_secs_f64()
                    * 1e3
            };

            let produced_for_rid = results.iter().any(|r| r.request_id == rid);
            let is_decode_step = produced_for_rid && first_token_seen;
            if produced_for_rid {
                if !first_token_seen {
                    // The step that produced the first token still contains
                    // the prompt prefill forward pass.
                    prefill_ms += elapsed_ms;
                    first_token_seen = true;
                } else {
                    decode_ms += elapsed_ms;
                }
            } else {
                // Intermediate prefill chunks produce no sampled token.
                prefill_ms += elapsed_ms;
            }

            if let (Some(summary), Some(account)) =
                (account_summary.as_mut(), step_account.as_ref())
            {
                if is_decode_step {
                    summary.decode_steps += 1;
                    summary.decode_ms += account.step_ms;
                    summary.decode_compute_ms += account.compute_ms;
                    summary.decode_executor_ms += account.executor_ms;
                    summary.decode_cache_ms += account.cache_ms;
                    summary.decode_sampler_ms += account.sampler_ms;
                    summary.decode_runner_ms += account.runner_ms;
                } else {
                    summary.prefill_steps += 1;
                    summary.prefill_ms += account.step_ms;
                }
                let kind = if is_decode_step { "decode" } else { "prefill" };
                eprintln!(
                    "[account] step={} kind={} tokens={} total={:.3}ms schedule={:.3} forward={:.3} compute={:.3} executor={:.3} cache={:.3} sampler={:.3} runner={:.3}",
                    step_no,
                    kind,
                    results.len(),
                    account.step_ms,
                    account.schedule_ms,
                    account.forward_ms,
                    account.compute_ms,
                    account.executor_ms,
                    account.cache_ms,
                    account.sampler_ms,
                    account.runner_ms,
                );
            }
            step_no += 1;

            for r in &results {
                if r.request_id == rid {
                    output_tokens.push(r.token_id);
                    if r.finished {
                        break;
                    }
                }
            }
        }

        if !first_token_seen {
            prefill_ms = prefill_ms.max(0.0);
        }

        if let Some(summary) = account_summary.as_mut() {
            summary.prefill_ms = prefill_ms;
            summary.decode_ms = decode_ms;
            summary.decode_avg_ms = if summary.decode_steps > 0 {
                summary.decode_ms / summary.decode_steps as f64
            } else {
                0.0
            };
            let decode_tokens = output_tokens.len().saturating_sub(1);
            summary.decode_tokens_s = if decode_ms > 0.0 {
                decode_tokens as f64 / (decode_ms / 1000.0)
            } else {
                0.0
            };
        }

        let text = self
            .tokenizer
            .decode(&output_tokens)
            .map_err(|e| anyhow::anyhow!("decode error: {}", e))?;

        Ok(GenerationResult {
            text,
            tokens: output_tokens,
            prefill_ms,
            decode_ms,
            account: account_summary,
        })
    }

    /// True if there are any pending or running requests.
    pub fn has_work(&self) -> bool {
        self.scheduler.has_work()
    }

    /// Return the vocabulary size from the compute graph's global output shape,
    /// falling back to the tokenizer vocab size if unavailable.
    fn vocab_size(&self) -> usize {
        let (g_func, g_idx) = self.executor.compute_graph.global_output;
        let output_def = &self.executor.compute_graph.functions[g_func].outputs[g_idx];
        let vocab = output_def.shape.last().copied().unwrap_or(0) as usize;
        if vocab > 0 {
            vocab
        } else {
            self.tokenizer.vocab_size()
        }
    }
}

impl SamplerConfig {
    fn validate(&self) -> Result<(), String> {
        if self.temperature < 0.0 {
            return Err("temperature must be >= 0".into());
        }
        if self.top_p < 0.0 || self.top_p > 1.0 {
            return Err("top_p must be in [0, 1]".into());
        }
        Ok(())
    }
}
#[cfg(test)]
#[path = "../tests/runner_tests.rs"]
mod tests;
