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
use crate::executor::ModelExecutor;
use crate::cache::radix::RadixCache;
use crate::sampler::{Sampler, SamplerConfig};
use crate::scheduler::Scheduler;
use crate::tokenizer::{ChatMessage, Tokenizer};

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
    cache_hits: Vec<crate::types::PrefixCacheHit>,
    /// Track per-request last logits position for logits→token extraction.
    #[allow(dead_code)]
    last_positions: HashMap<String, usize>,
    /// KV cache state (only used when config.use_kv_cache is true).
    kv_cache_state: Option<KVCacheState>,
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

        let block_manager = if config.use_kv_cache && config.num_kv_heads > 0 && config.head_dim > 0 {
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
        if input_ids.is_empty() {
            anyhow::bail!("empty prompt after encoding");
        }

        let stop_ids = self.tokenizer.stop_token_ids();
        let rid = self.scheduler.add_request(
            input_ids,
            0,                          // priority
            sampling.max_tokens.unwrap_or(self.config.max_tokens_per_request),
            stop_ids,
            None,
        );

        // Try prefix cache match
        let prompt_tokens: &[u32] = self.scheduler.running_request(&rid)
            .map(|r| r.prompt_tokens.as_slice())
            .unwrap_or(&[]);
        let (matched_blocks, matched_tokens) =
            self.radix_cache.match_prefix(prompt_tokens);
        if !matched_blocks.is_empty() {
            // Assign cached blocks to the new request
            self.block_manager.assign_cached_blocks(&rid, &matched_blocks);
            self.cache_hits.push(crate::types::PrefixCacheHit {
                request_id: rid.clone(),
                matched_blocks,
                matched_tokens,
            });
        }

        Ok(rid)
    }

    /// Run one scheduling step.  Returns results for all requests in the batch.
    pub fn step(&mut self, sampling: &SamplerConfig) -> Result<Vec<StepResult>, anyhow::Error> {
        // 1. Schedule: get the batch of requests to process
        let cache_hits = std::mem::take(&mut self.cache_hits);
        let batch = self.scheduler.schedule(&mut self.block_manager, &cache_hits);

        if batch.requests.is_empty() {
            return Ok(Vec::new());
        }

        let mut results: Vec<StepResult> = Vec::with_capacity(batch.requests.len());

        // 2. Process each request individually (single-request forward)
        for req in &batch.requests {
            let input_ids = &req.input_ids;
            if input_ids.is_empty() {
                continue;
            }

            // Forward pass — KV cache path or standard path
            let logits_tensor = if req.use_kv_cache {
                // Track request in KVCacheState
                if let Some(ref mut kv_state) = self.kv_cache_state {
                    kv_state.request_id_to_cache.entry(req.request_id.clone())
                        .or_insert_with(|| "default".to_string());
                }
                self.executor.forward_with_kv(
                    input_ids,
                    &req.positions,
                    Some(&mut self.block_manager),
                    Some(&req.request_id),
                )?
            } else {
                self.executor.forward_with_positions(input_ids, &req.positions)?
            };
            let all_logits = logits_tensor.as_slice();
            if all_logits.is_empty() {
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
            let argmax_idx = logits.iter().enumerate().fold((0, f32::NEG_INFINITY), |(mi, mv), (i, &v)| if v > mv { (i, v) } else { (mi, mv) }).0;
            log::debug!("logits argmax_idx={} argmax_val={}", argmax_idx, logits[argmax_idx]);

            // Sample token
            let token_id = self.sampler.sample(logits, sampling);

            log::debug!("step input_ids.len={} positions.len={} n_tokens={} token_id={} text={:?}",
                input_ids.len(), req.positions.len(), req.n_tokens, token_id,
                self.tokenizer.decode_token(token_id));

            // Record output in scheduler
            let finished = self.scheduler.record_output(&req.request_id, token_id);

            // Update prefix cache if finished
            if finished {
                if let Some(request) = self.scheduler.get_finished_request(&req.request_id) {
                    let block_table = self.block_manager.get_blocks(&req.request_id)
                        .unwrap_or_default()
                        .to_vec();
                    let all_tokens: Vec<u32> = request.prompt_tokens
                        .iter()
                        .chain(request.output_tokens.iter())
                        .copied()
                        .collect();
                    if !all_tokens.is_empty() {
                        self.radix_cache.insert(
                            &all_tokens,
                            &block_table,
                            &mut self.block_manager,
                        );
                    }
                }

                // Flush KV cache for finished request (after radix cache uses block table)
                if self.config.use_kv_cache {
                    if let Some(ref mut kv_state) = self.kv_cache_state {
                        kv_state.request_id_to_cache.remove(&req.request_id);
                    }
                    self.block_manager.flush_request(&req.request_id);
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

        Ok(results)
    }

    // ─── Convenience: full generate() ────────────────────────

    /// Generate a response for a prompt using the autoregressive loop.
    /// This is a convenience wrapper around add_request + step loop.
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
        let mut output_tokens: Vec<u32> = Vec::new();

        while self.scheduler.has_work() {
            let results = self.step(&sampling)?;
            for r in &results {
                if r.request_id == rid {
                    if !r.text.is_empty() {
                        print!("{}", r.text);
                    }
                    output_tokens.push(r.token_id);
                    if r.finished {
                        break;
                    }
                }
            }
        }

        let text = self
            .tokenizer
            .decode(&output_tokens)
            .map_err(|e| anyhow::anyhow!("decode error: {}", e))?;

        Ok(GenerationResult {
            text,
            tokens: output_tokens,
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
        if vocab > 0 { vocab } else { self.tokenizer.vocab_size() }
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
#[path = "tests/runner_tests.rs"]
mod tests;
