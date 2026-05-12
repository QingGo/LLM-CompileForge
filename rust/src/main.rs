//! ServeForge — CLI for AOT-compiled LLM inference.
//!
//! Usage:
//!   serveforge run <model> --prompt <text>
//!   serveforge info <model>

mod block_manager;
mod executor;
mod hal_cpu;
mod radix_cache;
mod scheduler;
mod types;
mod weight_loader;

use clap::{Parser, Subcommand};

#[cfg(test)]
mod m1_tests;

#[derive(Parser)]
#[command(name = "serveforge", about = "AOT-compiled LLM inference runtime")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run inference on a compiled model.
    Run {
        /// Model name (e.g. qwen3.5-0.8b).
        model: String,
        /// The input prompt text (UTF-8).
        #[arg(short, long)]
        prompt: String,
        /// Path to compiled artifacts directory.
        #[arg(short, long, default_value = "compiled")]
        compiled_dir: String,
        /// Maximum number of tokens to generate.
        #[arg(long, default_value = "64")]
        max_tokens: usize,
    },
    /// Show compiled model info.
    Info {
        /// Model name.
        model: String,
        /// Path to compiled artifacts directory.
        #[arg(short, long, default_value = "compiled")]
        compiled_dir: String,
    },
}

fn main() -> Result<(), anyhow::Error> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Run {
            model,
            prompt,
            compiled_dir,
            max_tokens,
        } => {
            let artifact_path = format!("{}/{}", compiled_dir, model);
            println!("Loading model from: {}", artifact_path);
            println!("Prompt: {}", prompt);

            let executor = executor::ModelExecutor::load(
                &format!("{}/lib{}.dylib", artifact_path, model),
                Some(&format!("{}/weights.safetensors", artifact_path)),
            )
            .map_err(|e| {
                anyhow::anyhow!(
                    "Failed to load model (have you compiled it?): {}",
                    e
                )
            })?;

            println!(
                "Model loaded: {} weight tensors, {} alloc",
                executor
                    .weight_provider
                    .as_ref()
                    .map(|_w| "ok")
                    .unwrap_or("0"),
                executor.allocated_bytes(),
            );

            // Stub: run inference loop
            // In the next phase, this will:
            //  1. Tokenize the prompt
            //  2. Embed tokens → input tensor
            //  3. For each function: build input memrefs, call func, collect output
            //  4. Sample next token
            //  5. Loop until max_tokens or EOS

            println!("[stub] Would generate up to {} tokens", max_tokens);
            println!("[stub] Done — run the full compiler pipeline first.");
        }
        Commands::Info { model, compiled_dir } => {
            let artifact_path = format!("{}/{}", compiled_dir, model);
            println!("Model: {}", model);
            println!("Artifact dir: {}", artifact_path);

            if std::path::Path::new(&format!(
                "{}/lib{}.dylib",
                artifact_path, model
            ))
            .exists()
            {
                println!("Status: compiled (.dylib present)");
            } else {
                println!("Status: not yet compiled");
            }
        }
    }

    Ok(())
}
