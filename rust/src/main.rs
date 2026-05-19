//! ServeForge — CLI for AOT-compiled LLM inference.
//!
//! Usage:
//!   serveforge run <model> --prompt <text> [--tokenizer <path>] [--safetensors <path>]
//!   serveforge info <model>

mod ciface_high;
mod block_manager;
mod compute_graph;
mod error;
mod executor;
mod hal;
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
mod e2e_tests;

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
            tokenizer_config,
            safetensors,
            seed,
            no_chat_template,
        } => {
            let artifact_path = std::path::Path::new(&compiled_dir).join(&model);
            // Scan for any .dylib in the model directory — the actual name
            // depends on --model-name passed to compile_dylib.py, which may
            // differ from the directory name.
            let dylib_path = std::fs::read_dir(&artifact_path)
                .ok()
                .and_then(|entries| {
                    entries.filter_map(|e| e.ok()).find(|e| {
                        e.path().extension().map(|ext| ext == "dylib").unwrap_or(false)
                    })
                })
                .map(|e| e.path().to_string_lossy().to_string())
                .unwrap_or_else(|| format!("{}/lib{}.dylib", artifact_path.to_string_lossy(), model));
            let st_path = safetensors
                .clone()
                .unwrap_or_else(|| format!("{}/weights.safetensors", artifact_path.to_string_lossy()));
            let st_path_opt: Option<&str> = if std::path::Path::new(&st_path).exists() {
                Some(&st_path)
            } else {
                None
            };
            let tok_path = tokenizer
                .clone()
                .unwrap_or_else(|| format!("{}/tokenizer.json", artifact_path.display()));

            eprintln!("Loading model from: {}", artifact_path.display());
            eprintln!("Prompt: {}", prompt);

            let executor = executor::ModelExecutor::load(
                &dylib_path,
                st_path_opt,
            )
            .map_err(|e| {
                // 三论: 信息论 — error must include WHAT failed and WHERE
                let ap = artifact_path.display();
                anyhow::anyhow!(
                    "Failed to load model '{}': {}\n\
                     Tried dylib: {}\n\
                     Suggestions:\n\
                     - Run: python scripts/compile_dylib.py {} --model-name <name>\n\
                     - Check compiled/{} exists and contains a .dylib file\n\
                     - Use --safetensors to point to the weights file",
                    model, e, dylib_path, ap, model,
                )
            })?;

            eprintln!(
                "Model loaded: {} functions, {} weight mappings, {} constants",
                executor.compute_graph.functions.len(),
                executor.weight_provider.name_mapping().len(),
                executor.weight_provider.constants().len(),
            );

            let tok = if no_chat_template {
                tokenizer::Tokenizer::from_file(&tok_path)
                    .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?
            } else {
                let cfg_path = tokenizer_config.clone().or_else(|| {
                    let alt = format!("{}/tokenizer_config.json", artifact_path.display());
                    if std::path::Path::new(&alt).exists() {
                        Some(alt)
                    } else {
                        None
                    }
                });
                let cfg_ref = cfg_path.as_deref();
                tokenizer::Tokenizer::from_file_with_chat_template(&tok_path, cfg_ref)
                    .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?
            };

            let runner_config = runner::RunnerConfig {
                max_tokens_per_request: max_tokens,
                seed,
                use_chat_template: !no_chat_template,
                ..Default::default()
            };
            let mut runner = runner::InferenceRunner::new(
                executor,
                tok,
                runner_config,
            )
            .map_err(|e| anyhow::anyhow!("Failed to create runner: {}", e))?;

            let result = runner.generate(&prompt, temperature, top_p, top_k)?;
            print!("{}", result.text);
        }
        Commands::Info { model, compiled_dir } => {
            let artifact_path = std::path::Path::new(&compiled_dir).join(&model);
            println!("Model: {}", model);
            println!("Artifact dir: {}", artifact_path.display());
            if !artifact_path.is_dir() {
                println!("Status: not found");
                return Ok(());
            }
            if !artifact_path.is_dir() {
                println!("Status: not found");
                return Ok(());
            }
            let dylib = std::fs::read_dir(&artifact_path)
                .ok()
                .and_then(|entries| {
                    entries.filter_map(|e| e.ok()).find(|e| {
                        e.path().extension().map(|ext| ext == "dylib").unwrap_or(false)
                    })
                });
            if let Some(dylib) = dylib {
                let meta = std::fs::metadata(dylib.path()).map(|m| m.len()).unwrap_or(0);
                println!("Status: compiled (.dylib present, {} bytes)", meta);
            } else {
                println!("Status: not compiled (no .dylib found)");
            }
        }
    }

    Ok(())
}
