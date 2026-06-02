//! Tokenizer wrapper using the ``tokenizers`` crate (HuggingFace format).
//!
//! Loads ``tokenizer.json`` from a HuggingFace model directory and provides
//! ``encode`` / ``decode`` for prompt processing and output generation.
//!
//! Chat template support via ``minijinja``:
//! - Loads ``chat_template`` from ``tokenizer_config.json`` (HuggingFace standard).
//! - Falls back to a hardcoded Qwen chat template if not found.

use tokenizers::Tokenizer as HfTokenizer;

#[derive(Debug, Clone, serde::Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn user(content: &str) -> Self {
        Self {
            role: "user".to_string(),
            content: content.to_string(),
        }
    }

    #[allow(dead_code)]
    pub fn system(content: &str) -> Self {
        Self {
            role: "system".to_string(),
            content: content.to_string(),
        }
    }

    #[allow(dead_code)]
    pub fn assistant(content: &str) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.to_string(),
        }
    }
}

const QWEN_CHAT_TEMPLATE: &str = r#"
{%- if messages|length > 0 and messages[0].role == 'system' %}
    {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {%- set loop_messages = messages[1:] %}
{%- else %}
    {%- set loop_messages = messages %}
{%- endif %}
{%- for message in loop_messages %}
    {%- if message.role == 'user' %}
        {{- '<|im_start|>user\n' + message.content + '<|im_end|>\n' }}
    {%- elif message.role == 'assistant' %}
        {{- '<|im_start|>assistant\n' + message.content + '<|im_end|>\n' }}
    {%- elif message.role == 'tool' %}
        {{- '<|im_start|>user\n' + message.content + '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
"#;

pub fn render_chat_template(
    template: &str,
    messages: &[ChatMessage],
    add_generation_prompt: bool,
) -> Result<String, anyhow::Error> {
    let mut env = minijinja::Environment::new();
    env.set_auto_escape_callback(|_| minijinja::AutoEscape::None);
    env.add_template("chat", template)
        .map_err(|e| anyhow::anyhow!("failed to parse chat template: {}", e))?;
    let tmpl = env
        .get_template("chat")
        .map_err(|e| anyhow::anyhow!("failed to load chat template: {}", e))?;

    let ctx = minijinja::context! {
        messages => messages,
        add_generation_prompt => add_generation_prompt,
    };

    // Render with disabled auto-escape by wrapping in a safe-marked string
    let result = tmpl
        .render(&ctx)
        .map_err(|e| anyhow::anyhow!("failed to render chat template: {}", e))?;

    // minijinja may auto-escape < > etc. — strip backslash escapes
    Ok(result.replace("\\<", "<"))
}

pub struct Tokenizer {
    inner: HfTokenizer,
    chat_template: Option<String>,
}

impl Tokenizer {
    pub fn from_file(path: &str) -> Result<Self, anyhow::Error> {
        let inner = HfTokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {}", e))?;
        Ok(Self {
            inner,
            chat_template: None,
        })
    }

    pub fn from_file_with_chat_template(
        tokenizer_path: &str,
        config_path: Option<&str>,
    ) -> Result<Self, anyhow::Error> {
        let inner = HfTokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {}", e))?;

        let chat_template = config_path
            .and_then(|p| {
                std::fs::read_to_string(p).ok().and_then(|s| {
                    serde_json::from_str::<serde_json::Value>(&s).ok().and_then(|v| {
                        v.get("chat_template")
                            .and_then(|ct| ct.as_str().map(|s| s.to_string()))
                    })
                })
            })
            .or_else(|| Some(QWEN_CHAT_TEMPLATE.to_string()));

        Ok(Self {
            inner,
            chat_template,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, anyhow::Error> {
        let encoding = self
            .inner
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("tokenizer encode error: {}", e))?;
        let ids: Vec<u32> = encoding.get_ids().to_vec();
        log::debug!("encode text={:?} input_ids={:?}", text, ids);
        Ok(ids)
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

    #[allow(dead_code)]
    pub fn bos_token_id(&self) -> Option<u32> {
        self.inner
            .get_vocab(true)
            .get("bos_token")
            .copied()
            .or_else(|| self.inner.token_to_id("<s>"))
    }

    pub fn has_chat_template(&self) -> bool {
        self.chat_template.is_some()
    }

    pub fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String, anyhow::Error> {
        let template = self
            .chat_template
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no chat template configured"))?;

        render_chat_template(template, messages, add_generation_prompt)
    }

    /// Get stop token IDs (EOS + common stop tokens).
    pub fn stop_token_ids(&self) -> Vec<u32> {
        let mut ids = Vec::new();
        if let Some(eos) = self.eos_token_id() {
            ids.push(eos);
        }
        for key in &["<|im_end|>", "</s>", "<|endoftext|>"] {
            if let Some(id) = self.inner.get_vocab(true).get(*key).copied() {
                if !ids.contains(&id) {
                    ids.push(id);
                }
            }
        }
        ids
    }

    /// Decode a single token ID to a string.
    pub fn decode_token(&self, token_id: u32) -> String {
        self.inner
            .decode(&[token_id], true)
            .unwrap_or_else(|_| "\u{FFFD}".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_from_file_not_found() {
        let result = Tokenizer::from_file("/nonexistent/tokenizer.json");
        assert!(result.is_err());
    }

    #[test]
    fn test_qwen_chat_template_simple() {
        let messages = vec![ChatMessage::user("hello")];
        let result =
            render_chat_template(QWEN_CHAT_TEMPLATE, &messages, true).unwrap();
        assert_eq!(
            result,
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn test_qwen_chat_template_with_system() {
        let messages = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("hello"),
        ];
        let result =
            render_chat_template(QWEN_CHAT_TEMPLATE, &messages, true).unwrap();
        assert_eq!(
            result,
            "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn test_qwen_chat_template_no_generation_prompt() {
        let messages = vec![ChatMessage::user("hello")];
        let result =
            render_chat_template(QWEN_CHAT_TEMPLATE, &messages, false).unwrap();
        assert_eq!(
            result,
            "<|im_start|>user\nhello<|im_end|>\n"
        );
    }

    #[test]
    fn test_chat_template_empty_messages() {
        let result = render_chat_template(QWEN_CHAT_TEMPLATE, &[], true).unwrap();
        assert_eq!(result, "<|im_start|>assistant\n");
    }
}
