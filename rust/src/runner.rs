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
use crate::tokenizer::{ChatMessage, Tokenizer};

pub struct GenerationResult {
    pub text: String,
    pub tokens: Vec<u32>,
}

pub struct InferenceRunner<'a> {
    executor: &'a ModelExecutor,
    sampler: Sampler,
    tokenizer: Tokenizer,
    max_tokens: usize,
    use_chat_template: bool,
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
            use_chat_template: true,
        }
    }

    pub fn with_chat_template(mut self, enabled: bool) -> Self {
        self.use_chat_template = enabled;
        self
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        temperature: f32,
        top_p: f32,
        top_k: usize,
    ) -> Result<GenerationResult, anyhow::Error> {
        let formatted_prompt = if self.use_chat_template && self.tokenizer.has_chat_template() {
            let messages = vec![ChatMessage::user(prompt)];
            self.tokenizer.apply_chat_template(&messages, true)?
        } else {
            prompt.to_string()
        };

        let input_ids = self.tokenizer.encode(&formatted_prompt)?;
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
            let all_logits = logits_tensor.as_slice();
            if all_logits.is_empty() {
                anyhow::bail!("forward pass returned empty logits");
            }
            // The forward returns logits shape [2, 4, 50272] (exported static shape).
            // Sample from the LAST position (batch=1, seq=3, all vocab = 50272).
            const VOCAB_SIZE: usize = 50272;
            let last_start = all_logits.len() - VOCAB_SIZE;
            let logits = &all_logits[last_start..];

            let token_id = self.sampler.sample(logits, &sampler_config);

            if Some(token_id) == eos_id {
                break;
            }

            output_tokens.push(token_id);

            // Append new token to history (model has fixed input of 8 tokens)
            current_ids.push(token_id);
            if current_ids.len() > 8 {
                current_ids.remove(0);
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
}
