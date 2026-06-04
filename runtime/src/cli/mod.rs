//! CLI argument parsing — Clap-based subcommand definitions.

pub mod model_loader;

use clap::{Parser, Subcommand};

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
