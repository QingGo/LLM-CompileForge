//! ServeForge — CLI for AOT-compiled LLM inference.
//!
//! Usage:
//!   serveforge run <model> --prompt <text> [--tokenizer <path>] [--safetensors <path>]
//!   serveforge info <model>

mod block_manager;
mod compute_graph;
mod error;
mod executor;
mod hal_cpu;
mod radix_cache;
mod runner;
mod sampler;
mod scheduler;
mod sfcf;
mod tensor;
mod tokenizer;
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
    Run {
        model: String,
        #[arg(short, long)]
        prompt: String,
        #[arg(short, long, default_value = "compiled")]
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
        safetensors: Option<String>,
        #[arg(long, default_value = "42")]
        seed: u64,
    },
    Info {
        model: String,
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
            temperature,
            top_p,
            top_k,
            tokenizer,
            safetensors,
            seed,
        } => {
            let artifact_path = format!("{}/{}", compiled_dir, model);
            let dylib_path = format!("{}/lib{}.dylib", artifact_path, model);
            let st_path = safetensors
                .clone()
                .unwrap_or_else(|| format!("{}/weights.safetensors", artifact_path));
            let tok_path = tokenizer
                .clone()
                .unwrap_or_else(|| format!("{}/tokenizer.json", artifact_path));

            eprintln!("Loading model from: {}", artifact_path);
            eprintln!("Prompt: {}", prompt);

            let executor = executor::ModelExecutor::load(
                &dylib_path,
                Some(&st_path),
            )
            .map_err(|e| {
                anyhow::anyhow!(
                    "Failed to load model (have you compiled it?): {}",
                    e
                )
            })?;

            eprintln!(
                "Model loaded: {} functions, {} weight mappings, {} constants",
                executor.compute_graph.functions.len(),
                executor.weight_provider.name_mapping().len(),
                executor.weight_provider.constants().len(),
            );

            let tok = tokenizer::Tokenizer::from_file(&tok_path)
                .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

            let mut runner = runner::InferenceRunner::new(
                &executor,
                tok,
                seed,
                max_tokens,
            );

            let result = runner.generate(&prompt, temperature, top_p, top_k)?;
            print!("{}", result.text);
        }
        Commands::Info { model, compiled_dir } => {
            let artifact_path = format!("{}/{}", compiled_dir, model);
            println!("Model: {}", model);
            println!("Artifact dir: {}", artifact_path);

            let dylib = format!("{}/lib{}.dylib", artifact_path, model);
            if std::path::Path::new(&dylib).exists() {
                println!("Status: compiled (.dylib present)");
                let meta = std::fs::metadata(&dylib).map(|m| m.len()).unwrap_or(0);
                println!("Size: {} bytes", meta);
            } else {
                println!("Status: not yet compiled");
            }
        }
    }

    Ok(())
}
