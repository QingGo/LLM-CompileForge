//! CLI argument parsing — Clap-based subcommand definitions.

pub mod model_loader;

use clap::{Parser, Subcommand, ValueEnum};

/// `--exec-plan` selector for the Phase 5 op-plan path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ExecPlanArg {
    /// Use sfa_op_plan when present, otherwise the func path (default).
    Auto,
    /// Always use the func-level `_mlir_ciface_*` path.
    Func,
    /// Require sfa_op_plan; fail when the dylib has none.
    Op,
}

/// `--weight-dtype` A/B selector for dtype-aware projection kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum WeightDtypeArg {
    /// Preserve safetensors source dtype (F16 source → F16 kernels).
    Auto,
    /// Force f32-promoted weights everywhere (A/B control).
    F32,
    /// Require F16 source weights and use raw F16 kernels.
    F16,
    /// Require BF16 source weights and use raw BF16 kernels.
    Bf16,
}

#[derive(Parser)]
#[command(name = "serveforge", about = "AOT-compiled LLM inference runtime")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    Run {
        model: String,
        #[arg(short, long)]
        prompt: String,
        #[arg(short, long, default_value = "outputs/compiled")]
        compiled_dir: String,
        #[arg(long, default_value = "64")]
        max_tokens: usize,
        #[arg(long, default_value = "1.0")]
        temperature: f32,
        #[arg(long, default_value = "1.0")]
        top_p: f32,
        #[arg(long, default_value = "0")]
        top_k: usize,
        #[arg(long)]
        tokenizer: Option<String>,
        #[arg(long)]
        tokenizer_config: Option<String>,
        #[arg(long)]
        safetensors: Option<String>,
        #[arg(long, default_value = "42")]
        seed: u64,
        #[arg(long)]
        no_chat_template: bool,
        /// Interpret `--prompt` as whitespace-separated token ids.
        #[arg(long)]
        prompt_ids: bool,
        /// Print the generated token ids to stderr.
        #[arg(long)]
        print_token_ids: bool,
        /// Emit one JSON benchmark object on stdout instead of generated text.
        #[arg(long)]
        bench: bool,
        /// Number of measured benchmark generations in this process
        /// (median is reported when greater than 1).
        #[arg(long, default_value_t = 1)]
        bench_runs: usize,
        /// Number of discarded warm-up generations before the measured runs.
        /// The first warm-up prefill is reported as `cold_prefill_ms`.
        #[arg(long, default_value_t = 0)]
        bench_warmup_runs: usize,
        /// Phase 4 prototype: use the OPT fused Rust decoder layer instead of
        /// `_mlir_ciface_main_*a/*b` dylib functions.
        #[arg(long)]
        opt_fused_fastpath: bool,
        /// Phase 5: choose between the HAL op plan and the func path.
        #[arg(long, value_enum, default_value_t = ExecPlanArg::Auto)]
        exec_plan: ExecPlanArg,
        /// Phase 5: projection weight storage dtype (source-preserving by
        /// default; explicit values are A/B switches).
        #[arg(long, value_enum, default_value_t = WeightDtypeArg::Auto)]
        weight_dtype: WeightDtypeArg,
    },
    Info {
        model: String,
        #[arg(short, long, default_value = "outputs/compiled")]
        compiled_dir: String,
    },
    #[command(name = "serve", about = "Start HTTP server with OpenAI-compatible API")]
    Serve {
        model: String,
        #[arg(short, long, default_value = "outputs/compiled")]
        compiled_dir: String,
        #[arg(short, long, default_value_t = 8000)]
        port: u16,
    },
}
