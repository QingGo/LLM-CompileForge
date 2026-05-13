//! Tokenizer wrapper using the ``tokenizers`` crate (HuggingFace format).
//!
//! Loads ``tokenizer.json`` from a HuggingFace model directory and provides
//! ``encode`` / ``decode`` for prompt processing and output generation.

use tokenizers::Tokenizer as HfTokenizer;

pub struct Tokenizer {
    inner: HfTokenizer,
}

impl Tokenizer {
    pub fn from_file(path: &str) -> Result<Self, anyhow::Error> {
        let inner = HfTokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {}", e))?;
        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, anyhow::Error> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("tokenizer encode error: {}", e))?;
        Ok(encoding.get_ids().iter().map(|&id| id).collect())
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String, anyhow::Error> {
        self.inner
            .decode(tokens, true)
            .map_err(|e| anyhow::anyhow!("tokenizer decode error: {}", e))
    }

    pub fn eos_token_id(&self) -> Option<u32> {
        for key in &["eos_token", "</s>", "<|endoftext|>", "<|im_end|>"] {
            if let Some(id) = self.inner.get_vocab(true).get(*key).copied() {
                return Some(id);
            }
        }
        None
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    pub fn bos_token_id(&self) -> Option<u32> {
        self.inner.get_vocab(true).get("bos_token").copied()
            .or_else(|| self.inner.token_to_id("<s>"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_tokenizer_from_file_not_found() {
        let result = Tokenizer::from_file("/nonexistent/tokenizer.json");
        assert!(result.is_err());
    }
}
