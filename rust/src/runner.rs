//! Inference runner — orchestrates the full autoregressive loop.
//!
//! Wires together:
//!   - ``Tokenizer`` for prompt encoding / output decoding
//!   - ``ModelExecutor`` for forward passes
//!   - ``Sampler`` for token selection
//!   - ``Scheduler`` + ``BlockManager`` for request management

use crate::block_manager::BlockManager;
use crate::executor::ModelExecutor;
use crate::sampler::{Sampler, SamplerConfig};
use crate::scheduler::Scheduler;
use crate::tokenizer::Tokenizer;

pub struct GenerationResult {
    pub text: String,
    pub tokens: Vec<u32>,
}

pub struct InferenceRunner<'a> {
    executor: &'a ModelExecutor,
    sampler: Sampler,
    tokenizer: Tokenizer,
    max_tokens: usize,
}

impl<'a> InferenceRunner<'a> {
    pub fn new(
        executor: &'a ModelExecutor,
        tokenizer: Tokenizer,
        seed: u64,
        max_tokens: usize,
    ) -> Self {
        Self {
            executor,
            sampler: Sampler::new(seed),
            tokenizer,
            max_tokens,
        }
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        temperature: f32,
        top_p: f32,
        top_k: usize,
    ) -> Result<GenerationResult, anyhow::Error> {
        let input_ids = self.tokenizer.encode(prompt)?;
        if input_ids.is_empty() {
            return Ok(GenerationResult {
                text: String::new(),
                tokens: vec![],
            });
        }

        let mut output_tokens: Vec<u32> = Vec::new();
        let mut current_ids: Vec<u32> = input_ids;
        let eos_id = self.tokenizer.eos_token_id();
        let sampler_config = SamplerConfig {
            temperature,
            top_p,
            top_k,
        };

        for _step in 0..self.max_tokens {
            let logits_tensor = self.executor.forward(&current_ids)?;
            let logits = logits_tensor.as_slice();

            if logits.is_empty() {
                anyhow::bail!("forward pass returned empty logits");
            }

            let token_id = self.sampler.sample(logits, &sampler_config);

            if Some(token_id) == eos_id {
                break;
            }

            output_tokens.push(token_id);

            // Single-token decode step
            current_ids = vec![token_id];
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
}
