//! Typed errors for the LLM-ServeForge runtime.
//!
//! The library layer uses these domain-specific error types so callers
//! (including Python via pyo3) can distinguish transient vs permanent
//! failures. The CLI binary wraps them in ``anyhow`` for user display.

#[derive(Debug, thiserror::Error)]
pub enum ExecutorError {
    #[allow(dead_code)]
    #[error("weight not found: {0}")]
    WeightNotFound(String),

    #[allow(dead_code)]
    #[error("kernel function not found: {0}")]
    KernelNotFound(String),

    #[error("kernel arity mismatch: expected {expected}, got {actual}")]
    KernelArityMismatch { expected: usize, actual: usize },

    #[error("SFCF parse error: {0}")]
    SfcfParse(String),

    #[allow(dead_code)]
    #[error("compute graph error: {0}")]
    ComputeGraph(String),

    #[error("missing compute graph section in SFCF data")]
    MissingComputeGraph,

    #[allow(dead_code)]
    #[error("forward pass produced empty logits")]
    EmptyLogits,

    #[allow(dead_code)]
    #[error("logits contain non-finite values")]
    NonFiniteLogits,

    #[allow(dead_code)]
    #[error("global output index out of bounds: func {func_idx}, output {output_idx}")]
    GlobalOutputOutOfBounds { func_idx: usize, output_idx: usize },
}

#[derive(Debug, thiserror::Error)]
#[allow(dead_code)]
pub enum MemoryError {
    #[error("out of memory: {0}")]
    OutOfMemory(String),

    #[error("buffer not initialized")]
    UninitializedBuffer,
}

#[derive(Debug, thiserror::Error)]
#[allow(dead_code)]
pub enum TokenizerError {
    #[error("tokenizer load failed: {0}")]
    LoadFailed(String),

    #[error("encode failed: {0}")]
    EncodeFailed(String),

    #[error("decode failed: {0}")]
    DecodeFailed(String),

    #[error("EOS token not found")]
    EosNotFound,
}
