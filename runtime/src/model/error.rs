//! Typed errors for the LLM-ServeForge runtime.
//!
//! The library layer uses these domain-specific error types so callers
//! (including Python via pyo3) can distinguish transient vs permanent
//! failures. The CLI binary wraps them in ``anyhow`` for user display.

#[derive(Debug, thiserror::Error)]
pub enum ExecutorError {
    #[error("kernel arity mismatch: expected {expected}, got {actual}")]
    KernelArityMismatch { expected: usize, actual: usize },

    #[error("SFCF parse error: {0}")]
    SfcfParse(String),

}

#[derive(Debug, thiserror::Error)]
pub enum HalExecutionError {
    #[error("HAL op execution failed: func[{func_idx}] op[{op_idx}] '{op_name}' — {message}")]
    OpFailed {
        func_idx: usize,
        op_idx: usize,
        op_name: String,
        message: String,
    },

    #[error("HAL op PANIC: func[{func_idx}] op[{op_idx}] '{op_name}' — {panic_msg}")]
    OpPanic {
        func_idx: usize,
        op_idx: usize,
        op_name: String,
        panic_msg: String,
    },

    #[error("HAL runner: {0}")]
    Runner(String),
}
