//! ServeForge — CLI for AOT-compiled LLM inference.
//!
//! Usage:
//!   serveforge run <model> --prompt <text> [--tokenizer <path>] [--safetensors <path>]
//!   serveforge info <model>

mod cache;
mod cli;
mod debug;
mod engine;
mod hal;
mod kv_cache;
mod model;
mod server;

use std::sync::Arc;

use clap::Parser;
use cli::{Cli, Commands};
use tokio::sync::Mutex;

#[path = "tests/e2e_tests.rs"]
mod e2e_tests;
#[cfg(test)]
#[path = "tests/m1_tests.rs"]
mod m1_tests;

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
            prompt_ids,
            print_token_ids,
            bench,
            bench_runs,
            bench_warmup_runs,
            opt_fused_fastpath,
            exec_plan,
            weight_dtype,
        } => {
            let artifact_path = std::path::Path::new(&compiled_dir).join(&model);
            let dylib_path = cli::model_loader::resolve_dylib_path(&artifact_path, &model);
            let st_path = safetensors.clone().unwrap_or_else(|| {
                format!("{}/weights.safetensors", artifact_path.to_string_lossy())
            });
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

            let model_load_t0 = std::time::Instant::now();
            let mut executor = engine::executor::ModelExecutor::load(&dylib_path, st_path_opt)
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
                        model,
                        e,
                        dylib_path,
                        ap,
                        model,
                    )
                })?;
            let model_load_ms = model_load_t0.elapsed().as_secs_f64() * 1e3;
            if opt_fused_fastpath {
                executor.set_opt_fused_fastpath(true);
                log::info!("OPT fused decoder-layer fast path enabled");
            }
            {
                use engine::executor::{ExecPlanMode, WeightDtypeMode};
                let mode = match exec_plan {
                    cli::ExecPlanArg::Auto => ExecPlanMode::Auto,
                    cli::ExecPlanArg::Func => ExecPlanMode::Func,
                    cli::ExecPlanArg::Op => ExecPlanMode::Op,
                };
                executor.set_exec_plan_mode(mode);
                log::info!("Execution plan mode: {mode:?}");
                let weight_mode = match weight_dtype {
                    cli::WeightDtypeArg::Auto => WeightDtypeMode::Auto,
                    cli::WeightDtypeArg::F32 => WeightDtypeMode::F32,
                    cli::WeightDtypeArg::F16 => WeightDtypeMode::F16,
                    cli::WeightDtypeArg::Bf16 => WeightDtypeMode::Bf16,
                };
                executor.set_weight_dtype_mode(weight_mode);
                log::info!("Weight dtype mode: {weight_mode:?}");
            }

            log::info!(
                "Model loaded: {} functions, {} weight mappings, {} constants",
                executor.compute_graph.functions.len(),
                executor.weight_provider.name_mapping().len(),
                executor.weight_provider.constants().len(),
            );

            let tok = if no_chat_template {
                engine::tokenizer::Tokenizer::from_file(&tok_path)
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
                engine::tokenizer::Tokenizer::from_file_with_chat_template(&tok_path, cfg_ref)
                    .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?
            };

            let runner_config = engine::runner::RunnerConfig {
                max_tokens_per_request: max_tokens,
                seed,
                use_chat_template: !no_chat_template,
                ..Default::default()
            }
            .with_cache_policy(&executor.cache_policy);
            let mut runner = engine::runner::InferenceRunner::new(executor, tok, runner_config)
                .map_err(|e| anyhow::anyhow!("Failed to create runner: {}", e))?;

            let explicit_prompt_ids: Option<Vec<u32>> = if prompt_ids {
                Some(
                    prompt
                        .split_whitespace()
                        .map(|tok| {
                            tok.parse::<u32>().map_err(|e| {
                                anyhow::anyhow!("invalid prompt token id {:?}: {}", tok, e)
                            })
                        })
                        .collect::<Result<Vec<u32>, _>>()?,
                )
            } else {
                None
            };

            anyhow::ensure!(bench_runs > 0, "--bench-runs must be >= 1");
            anyhow::ensure!(
                bench || (bench_runs == 1 && bench_warmup_runs == 0),
                "--bench-runs/--bench-warmup-runs require --bench"
            );

            if !bench {
                anyhow::ensure!(
                    bench_runs == 1 && bench_warmup_runs == 0,
                    "--bench-runs/--bench-warmup-runs require --bench"
                );
                let result = if let Some(ids) = explicit_prompt_ids.as_deref() {
                    runner.generate_from_tokens(ids, temperature, top_p, top_k)?
                } else {
                    runner.generate(&prompt, temperature, top_p, top_k)?
                };
                if print_token_ids {
                    eprintln!("[token_ids] {:?}", result.tokens);
                }
                print!("{}", result.text);
            } else {
            let total_runs = bench_warmup_runs
                .checked_add(bench_runs)
                .ok_or_else(|| anyhow::anyhow!("benchmark run count overflow"))?;
            let mut warmup_rows: Vec<serde_json::Value> = Vec::with_capacity(bench_warmup_runs);
            let mut measured_rows: Vec<serde_json::Value> = Vec::with_capacity(bench_runs);

            for run_idx in 0..total_runs {
                let result = if let Some(ids) = explicit_prompt_ids.as_deref() {
                    runner.generate_from_tokens(ids, temperature, top_p, top_k)?
                } else {
                    runner.generate(&prompt, temperature, top_p, top_k)?
                };
                let decode_tokens = result.tokens.len().saturating_sub(1);
                let decode_tokens_s = if result.decode_ms > 0.0 {
                    decode_tokens as f64 / (result.decode_ms / 1000.0)
                } else {
                    0.0
                };
                let round_ms = |v: f64| (v * 1000.0).round() / 1000.0;
                let mut row = serde_json::json!({
                    "prefill_ms": round_ms(result.prefill_ms),
                    "decode_ms": round_ms(result.decode_ms),
                    "decode_tokens": decode_tokens,
                    "decode_tokens_s": (decode_tokens_s * 1000.0).round() / 1000.0,
                    "token_ids": result.tokens,
                });
                if let Some(account) = result.account.as_ref() {
                    row["account"] = serde_json::to_value(account)?;
                }
                if run_idx < bench_warmup_runs {
                    warmup_rows.push(row);
                } else {
                    measured_rows.push(row);
                }
            }

            let first_token_ids = measured_rows
                .first()
                .and_then(|row| row.get("token_ids"))
                .cloned();
            for row in &measured_rows {
                anyhow::ensure!(
                    row.get("token_ids") == first_token_ids.as_ref(),
                    "benchmark runs produced different token ids: {:?} vs {:?}",
                    row.get("token_ids"),
                    first_token_ids,
                );
            }

            let mut decode_order: Vec<usize> = (0..measured_rows.len()).collect();
            decode_order.sort_by(|&a, &b| {
                measured_rows[a]["decode_ms"]
                    .as_f64()
                    .unwrap_or(0.0)
                    .total_cmp(
                        &measured_rows[b]["decode_ms"]
                            .as_f64()
                            .unwrap_or(0.0),
                    )
            });
            let median_idx = decode_order[decode_order.len() / 2];
            let median_row = &measured_rows[median_idx];
            let median = |key: &str| -> f64 {
                let mut values: Vec<f64> = measured_rows
                    .iter()
                    .map(|row| row[key].as_f64().unwrap_or(0.0))
                    .collect();
                values.sort_by(f64::total_cmp);
                values[values.len() / 2]
            };
            let decode_ms_median = median("decode_ms");
            let prefill_ms_median = median("prefill_ms");
            let decode_tokens = median_row["decode_tokens"].as_u64().unwrap_or(0) as usize;
            let decode_tokens_s = if decode_ms_median > 0.0 {
                decode_tokens as f64 / (decode_ms_median / 1000.0)
            } else {
                0.0
            };
            let cold_prefill_ms = warmup_rows
                .first()
                .or_else(|| measured_rows.first())
                .and_then(|row| row["prefill_ms"].as_f64())
                .unwrap_or(prefill_ms_median);
            let warm_prefill_ms = if bench_warmup_runs > 0 {
                Some(prefill_ms_median)
            } else {
                None
            };
            let min_max = |key: &str| -> (f64, f64) {
                let values: Vec<f64> = measured_rows
                    .iter()
                    .map(|row| row[key].as_f64().unwrap_or(0.0))
                    .collect();
                (
                    values.iter().copied().fold(f64::INFINITY, f64::min),
                    values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                )
            };
            let (decode_ms_min, decode_ms_max) = min_max("decode_ms");
            let (prefill_ms_min, prefill_ms_max) = min_max("prefill_ms");

            let mut bench_json = serde_json::json!({
                "model_load_ms": (model_load_ms * 1000.0).round() / 1000.0,
                "prefill_ms": (prefill_ms_median * 1000.0).round() / 1000.0,
                "cold_prefill_ms": (cold_prefill_ms * 1000.0).round() / 1000.0,
                "warm_prefill_ms": warm_prefill_ms
                    .map(|v| (v * 1000.0).round() / 1000.0),
                "prefill_ms_min": (prefill_ms_min * 1000.0).round() / 1000.0,
                "prefill_ms_max": (prefill_ms_max * 1000.0).round() / 1000.0,
                "decode_ms": (decode_ms_median * 1000.0).round() / 1000.0,
                "decode_ms_min": (decode_ms_min * 1000.0).round() / 1000.0,
                "decode_ms_max": (decode_ms_max * 1000.0).round() / 1000.0,
                "decode_tokens": decode_tokens,
                "decode_tokens_s": (decode_tokens_s * 1000.0).round() / 1000.0,
                "token_ids": median_row["token_ids"],
                "runs": measured_rows,
            });
            if !warmup_rows.is_empty() {
                bench_json["warmup_runs"] = serde_json::Value::Array(warmup_rows);
            }
            if let Some(account) = median_row.get("account") {
                bench_json["account"] = account.clone();
            }

            if print_token_ids {
                eprintln!("[token_ids] {:?}", bench_json["token_ids"]);
            }
            println!("{}", serde_json::to_string(&bench_json)?);
            }
        }
        Commands::Info {
            model,
            compiled_dir,
        } => {
            let artifact_path = std::path::Path::new(&compiled_dir).join(&model);
            println!("Model: {}", model);
            println!("Artifact dir: {}", artifact_path.display());
            if !artifact_path.is_dir() {
                println!("Status: not found");
                return Ok(());
            }
            let dylib = std::fs::read_dir(&artifact_path).ok().and_then(|entries| {
                entries.filter_map(|e| e.ok()).find(|e| {
                    e.path()
                        .extension()
                        .map(|ext| ext == "dylib")
                        .unwrap_or(false)
                })
            });
            if let Some(dylib) = dylib {
                let meta = std::fs::metadata(dylib.path())
                    .map(|m| m.len())
                    .unwrap_or(0);
                println!("Status: compiled (.dylib present, {} bytes)", meta);
            } else {
                println!("Status: not compiled (no .dylib found)");
            }
        }
        Commands::Serve {
            model,
            compiled_dir,
            port,
        } => {
            load_model_and_run_serve(&model, &compiled_dir, port)?;
        }
    }

    Ok(())
}

fn load_model_and_run_serve(
    model: &str,
    compiled_dir: &str,
    port: u16,
) -> Result<(), anyhow::Error> {
    let artifact_path = std::path::Path::new(&compiled_dir).join(model);

    let dylib_path = cli::model_loader::resolve_dylib_path(&artifact_path, model);
    let st_path_opt = cli::model_loader::resolve_safetensors_path(&artifact_path);
    let st_path_ref: Option<&str> = st_path_opt.as_deref();

    log::info!("Loading model from: {}", artifact_path.display());

    let executor =
        engine::executor::ModelExecutor::load(&dylib_path, st_path_ref).map_err(|e| {
            let ap = artifact_path.display();
            anyhow::anyhow!(
                "Failed to load model '{}': {}\n\
             Tried dylib: {}\n\
             Suggestions:\n\
             - Run: python compiler/compile_dylib.py {} --model-name <name>\n\
             - Check outputs/compiled/{} exists and contains a .dylib file\n\
             - Use --safetensors to point to the weights file",
                model,
                e,
                dylib_path,
                ap,
                model,
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
    let tok = engine::tokenizer::Tokenizer::from_file_with_chat_template(&tok_path, cfg_ref)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    let runner_config = engine::runner::RunnerConfig {
        use_chat_template: false,
        ..Default::default()
    }
    .with_cache_policy(&executor.cache_policy);
    let runner = engine::runner::InferenceRunner::new(executor, tok, runner_config)
        .map_err(|e| anyhow::anyhow!("Failed to create runner: {}", e))?;

    let runner = Arc::new(Mutex::new(runner));

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async move { server::run(runner, port).await })
}
