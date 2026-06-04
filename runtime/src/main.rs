//! ServeForge — CLI for AOT-compiled LLM inference.
//!
//! Usage:
//!   serveforge run <model> --prompt <text> [--tokenizer <path>] [--safetensors <path>]
//!   serveforge info <model>

mod abi;
mod cache_policy;
mod ciface_high;
mod block_manager;
mod cli;
mod compute_graph;
mod error;
mod compute_graph_runner;
mod executor;
mod hal;
mod kv_cache;
mod kv_cache_intercept;
mod radix_cache;
mod runner;
mod sampler;
mod scheduler;
mod server;
mod sfcf;
mod sfa_tensor;
mod tensor;
mod tokenizer;
mod types;
mod weight_loader;
mod global_input;

use std::sync::Arc;

use clap::Parser;
use cli::{Cli, Commands};
use tokio::sync::Mutex;



#[cfg(test)]
#[path = "tests/m1_tests.rs"]
mod m1_tests;
#[path = "tests/e2e_tests.rs"]
mod e2e_tests;

fn main() -> Result<(), anyhow::Error> {
    env_logger::Builder::from_env(env_logger::Env::default().filter_or("RUST_LOG", "warn")).init();
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
            let dylib_path = cli::model_loader::resolve_dylib_path(&artifact_path, &model);
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

            log::info!("Loading model from: {}", artifact_path.display());
            log::info!("Prompt: {}", prompt);

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
                     - Run: python compiler/compile_dylib.py {} --model-name <name>\n\
                     - Check outputs/compiled/{} exists and contains a .dylib file\n\
                     - Use --safetensors to point to the weights file",
                    model, e, dylib_path, ap, model,
                )
            })?;

            log::info!(
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
        Commands::Serve { model, compiled_dir, port } => {
            load_model_and_run_serve(&model, &compiled_dir, port)?;
        }
    }

    Ok(())
}

fn load_model_and_run_serve(model: &str, compiled_dir: &str, port: u16) -> Result<(), anyhow::Error> {
    let artifact_path = std::path::Path::new(&compiled_dir).join(model);

    let dylib_path = cli::model_loader::resolve_dylib_path(&artifact_path, model);
    let st_path_opt = cli::model_loader::resolve_safetensors_path(&artifact_path);
    let st_path_ref: Option<&str> = st_path_opt.as_deref();

    log::info!("Loading model from: {}", artifact_path.display());

    let executor = executor::ModelExecutor::load(&dylib_path, st_path_ref).map_err(|e| {
        let ap = artifact_path.display();
        anyhow::anyhow!(
            "Failed to load model '{}': {}\n\
             Tried dylib: {}\n\
             Suggestions:\n\
             - Run: python compiler/compile_dylib.py {} --model-name <name>\n\
             - Check outputs/compiled/{} exists and contains a .dylib file\n\
             - Use --safetensors to point to the weights file",
            model, e, dylib_path, ap, model,
        )
    })?;

    log::info!(
        "Model loaded: {} functions, {} weight mappings, {} constants",
        executor.compute_graph.functions.len(),
        executor.weight_provider.name_mapping().len(),
        executor.weight_provider.constants().len(),
    );

    // Load tokenizer with chat template support
    let tok_path = format!("{}/tokenizer.json", artifact_path.display());
    let cfg_path = format!("{}/tokenizer_config.json", artifact_path.display());
    let cfg_ref = if std::path::Path::new(&cfg_path).exists() {
        Some(cfg_path.as_str())
    } else {
        None
    };
    let tok = tokenizer::Tokenizer::from_file_with_chat_template(&tok_path, cfg_ref)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    let runner_config = runner::RunnerConfig {
        use_chat_template: false,
        ..Default::default()
    };
    let runner = runner::InferenceRunner::new(executor, tok, runner_config)
        .map_err(|e| anyhow::anyhow!("Failed to create runner: {}", e))?;

    let runner = Arc::new(Mutex::new(runner));

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async move { server::run(runner, port).await })
}
