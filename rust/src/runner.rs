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
//!       let results = runner.step()?;
//!       for r in &results { print!("{}", r.text); }
//!   }

use std::collections::HashMap;

use crate::block_manager::BlockManager;
use crate::executor::ModelExecutor;
use crate::radix_cache::RadixCache;
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
    last_positions: HashMap<String, usize>,
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
        )
        .map_err(|e| anyhow::anyhow!("scheduler init: {}", e))?;

        let block_manager = BlockManager::new(config.num_blocks, config.block_size)
            .map_err(|e| anyhow::anyhow!("block_manager init: {}", e))?;

        let radix_cache = RadixCache::new(config.block_size);

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
        if input_ids.is_empty() {
            anyhow::bail!("empty prompt after encoding");
        }

        let stop_ids = self.tokenizer.stop_token_ids();
        let rid = self.scheduler.add_request(
            input_ids,
            0,                          // priority
            0.0,                        // arrival_time (wall clock TBD)
            sampling.max_tokens.unwrap_or(self.config.max_tokens_per_request),
            stop_ids,
            None,
        );

        // Try prefix cache match
        let prompt_tokens = &self.scheduler.running_request(&rid)
            .map(|r| r.prompt_tokens.clone())
            .unwrap_or_default();
        let (matched_blocks, matched_tokens) =
            self.radix_cache.match_prefix(&prompt_tokens);
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
    pub fn step(&mut self) -> Result<Vec<StepResult>, anyhow::Error> {
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

            // Forward pass with positions (enables future KV cache integration)
            let logits_tensor = self.executor.forward_with_positions(input_ids, &req.positions)?;
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

            // Sample token
            let sampler_config = SamplerConfig {
                temperature: 1.0,
                top_p: 1.0,
                top_k: 0,
                max_tokens: None,
            };
            let token_id = self.sampler.sample(logits, &sampler_config);

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
        let rid = self.add_request(prompt, sampling)?;
        let mut output_tokens: Vec<u32> = Vec::new();

        while self.scheduler.has_work() {
            let results = self.step()?;
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
            .unwrap_or_else(|_| "[decode error]".to_string());

        Ok(GenerationResult {
            text,
            tokens: output_tokens,
        })
    }

    /// True if there are any pending or running requests.
    pub fn has_work(&self) -> bool {
        self.scheduler.has_work()
    }

    /// Return the vocabulary size (constant 50272 for OPT-125m).
    fn vocab_size(&self) -> usize {
        // FIXME: extract from model metadata when available
        50272
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

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::ModelExecutor;
    use crate::tokenizer::Tokenizer;

    /// Create a ModelExecutor that loads from the opt_125m_v8 compiled model.
    /// This requires the model to have been compiled (CI must run setup.sh).
    fn compiled_executor() -> Option<ModelExecutor> {
        let dylib = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../compiled/opt_125m_v8/libopt_125m.dylib"
        );
        if !std::path::Path::new(dylib).exists() {
            return None;
        }
        ModelExecutor::load(dylib, None).ok()
    }

    fn dummy_tokenizer() -> Option<Tokenizer> {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tests/data/test_tokenizer.json"
        );
        if !std::path::Path::new(path).exists() {
            return None;
        }
        Tokenizer::from_file(path).ok()
    }

    #[test]
    fn test_runner_create() {
        let exec = match compiled_executor() {
            Some(e) => e,
            None => return,
        };
        let tokenizer = match dummy_tokenizer() {
            Some(t) => t,
            None => return,
        };
        let config = RunnerConfig::default();
        let mut runner =
            InferenceRunner::new(exec, tokenizer, config).expect("create runner");
        assert!(!runner.has_work());
    }

    #[test]
    fn test_runner_add_request() {
        let exec = match compiled_executor() {
            Some(e) => e,
            None => return,
        };
        let tokenizer = match dummy_tokenizer() {
            Some(t) => t,
            None => return,
        };
        let config = RunnerConfig::default();
        let mut runner =
            InferenceRunner::new(exec, tokenizer, config).expect("create runner");
        let rid = runner.add_request(
            "Hello",
            SamplerConfig { temperature: 0.0, top_p: 1.0, top_k: 0, max_tokens: None },
        ).expect("add request");
        assert!(!rid.is_empty());
        assert!(runner.has_work());
    }

    #[test]
    fn test_runner_step_empty() {
        let exec = match compiled_executor() {
            Some(e) => e,
            None => return,
        };
        let tokenizer = match dummy_tokenizer() {
            Some(t) => t,
            None => return,
        };
        let config = RunnerConfig::default();
        let mut runner =
            InferenceRunner::new(exec, tokenizer, config).expect("create runner");
        let results = runner.step().expect("step with no requests");
        assert!(results.is_empty());
    }

    #[test]
    fn test_runner_config_default() {
        let config = RunnerConfig::default();
        assert_eq!(config.max_batch_size, 8);
        assert_eq!(config.block_size, 16);
    }
}
