//! ServeForge — CLI for AOT-compiled LLM inference.
//!
//! Usage:
//!   serveforge run <model> --prompt <text> [--tokenizer <path>] [--safetensors <path>]
//!   serveforge info <model>

mod ciface_high;
mod block_manager;
mod compute_graph;
mod error;
mod compute_graph_runner;
mod executor;
mod kernel_catalog;
mod hal;
mod kv_cache;
mod kv_cache_intercept;
mod radix_cache;
mod runner;
mod sampler;
mod scheduler;
mod sfcf;
mod tensor;
mod tokenizer;
mod types;
mod weight_loader;
mod global_input;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::stream::{self, Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;
use clap::{Parser, Subcommand};

use crate::runner::InferenceRunner;
use crate::sampler::SamplerConfig;

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
    #[command(name = "serve", about = "Start HTTP server with OpenAI-compatible API")]
    Serve {
        model: String,
        #[arg(short, long, default_value = "compiled")]
        compiled_dir: String,
        #[arg(short, long, default_value_t = 8000)]
        port: u16,
    },
}

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
                     - Run: python scripts/compile_dylib.py {} --model-name <name>\n\
                     - Check compiled/{} exists and contains a .dylib file\n\
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
            run_serve(&model, &compiled_dir, port)?;
        }
    }

    Ok(())
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
}

#[derive(Deserialize)]
struct CompletionRequest {
    #[serde(default)]
    model: String,
    prompt: Value,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_top_p")]
    top_p: f32,
    #[serde(default)]
    top_k: usize,
    #[serde(default)]
    #[allow(dead_code)]
    stream: bool,
}

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatCompletionRequest {
    #[serde(default)]
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_top_p")]
    top_p: f32,
    #[serde(default)]
    top_k: usize,
    #[serde(default)]
    stream: bool,
}

fn default_max_tokens() -> usize {
    256
}
fn default_temperature() -> f32 {
    1.0
}
fn default_top_p() -> f32 {
    1.0
}

/// Build a chat prompt from messages using a simple fallback format.
/// Since OPT-125m has no chat template, use "role: content\n" format.
fn build_chat_prompt(req: &ChatCompletionRequest) -> String {
    let mut parts = Vec::new();
    for msg in &req.messages {
        parts.push(format!("{}: {}", msg.role, msg.content));
    }
    parts.push("Assistant: ".to_string());
    parts.join("\n")
}

async fn health(
    State(_state): State<Arc<Mutex<InferenceRunner>>>,
) -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok".to_string() })
}

#[allow(dead_code)]
async fn v1_completions(
    State(runner): State<Arc<Mutex<InferenceRunner>>>,
    Json(req): Json<CompletionRequest>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let prompt_str: String = match &req.prompt {
        Value::String(s) => s.clone(),
        Value::Array(arr) => {
            let prompt_tokens: Vec<u32> = arr
                .iter()
                .filter_map(|v| v.as_u64().map(|n| n as u32))
                .collect();
            prompt_tokens
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        }
        _ => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(json!({"error": "prompt must be string or integer array"})),
            ))
        }
    };

    if prompt_str.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "prompt must not be empty"})),
        ));
    }

    let mut runner = runner.lock().await;
    let sampling = SamplerConfig {
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        max_tokens: Some(req.max_tokens),
    };
    let rid = runner.add_request(&prompt_str, sampling).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("add_request: {}", e)})),
        )
    })?;

    let mut all_tokens: Vec<u32> = Vec::new();
    let mut output_text = String::new();
    let mut finished = false;

    while runner.has_work() {
        let step_sampling = SamplerConfig {
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k,
            max_tokens: Some(req.max_tokens),
        };
        let results = runner.step(&step_sampling).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": format!("step: {}", e)})),
            )
        })?;
        for r in &results {
            if r.request_id == rid {
                all_tokens.push(r.token_id);
                output_text.push_str(&r.text);
                if r.finished {
                    finished = true;
                    break;
                }
            }
        }
        if finished {
            break;
        }
    }

    let prompt_token_count = prompt_str.split_whitespace().count();
    let completion_token_count = all_tokens.len();
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let finish_reason: Value = if finished {
        Value::String("stop".to_string())
    } else {
        Value::Null
    };

    Ok(Json(json!({
        "id": rid,
        "object": "text_completion",
        "created": created,
        "model": req.model,
        "choices": [{
            "text": output_text,
            "index": 0,
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": prompt_token_count,
            "completion_tokens": completion_token_count,
            "total_tokens": prompt_token_count + completion_token_count
        }
    })))
}

/// SSE streaming for chat completions — yields one SSE event per token.
fn v1_chat_completions_stream(
    runner: Arc<Mutex<InferenceRunner>>,
    prompt: String,
    sampling: SamplerConfig,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let state = (runner, prompt, sampling, None::<String>);

    let token_stream = stream::unfold(
        state,
        |(runner, prompt, sampling, rid)| async move {
            // Clone Arc before locking so the original runner can be moved freely
            let guard_runner = runner.clone();
            let mut guard = guard_runner.lock().await;

            // First poll: add the request and capture the rid for filtering
            let rid = match rid {
                Some(id) => id,
                None => match guard.add_request(&prompt, sampling.clone()) {
                    Ok(id) => id,
                    Err(e) => {
                        return Some((
                            Ok(Event::default().data(format!("error: {}", e))),
                            (runner, prompt, sampling, None),
                        ));
                    }
                },
            };

            if !guard.has_work() {
                return None;
            }

            match guard.step(&sampling) {
                Ok(results) => {
                    for r in &results {
                        if r.request_id != rid {
                            continue;
                        }
                        let event_data = json!({
                            "choices": [{
                                "delta": {"content": &r.text},
                                "index": 0,
                                "finish_reason": if r.finished {
                                    Value::String("stop".to_string())
                                } else {
                                    Value::Null
                                }
                            }]
                        })
                        .to_string();
                        return Some((
                            Ok(Event::default().data(event_data)),
                            (runner, prompt, sampling, Some(rid)),
                        ));
                    }
                    // No matching result found — keep polling
                    Some((
                        Ok(Event::default().data("")),
                        (runner, prompt, sampling, Some(rid)),
                    ))
                }
                Err(e) => Some((
                    Ok(Event::default().data(format!("error: {}", e))),
                    (runner, prompt, sampling, Some(rid)),
                )),
            }
        },
    );

    // Append the SSE closing event
    token_stream.chain(stream::once(async { Ok(Event::default().data("[DONE]")) }))
}

/// POST /v1/chat/completions — streaming and non-streaming chat completions.
async fn v1_chat_completions(
    State(runner): State<Arc<Mutex<InferenceRunner>>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, (StatusCode, Json<Value>)> {
    let prompt = build_chat_prompt(&req);

    if prompt.trim().is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "messages must not be empty"})),
        ));
    }

    let sampling = SamplerConfig {
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        max_tokens: Some(req.max_tokens),
    };

    if req.stream {
        let stream = v1_chat_completions_stream(runner, prompt, sampling);
        return Ok(Sse::new(stream).into_response());
    }

    // Non-streaming: same step-loop pattern as v1_completions
    let mut guard = runner.lock().await;
    let rid = guard.add_request(&prompt, sampling).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("add_request: {}", e)})),
        )
    })?;

    let mut all_tokens: Vec<u32> = Vec::new();
    let mut output_text = String::new();
    let mut finished = false;

    while guard.has_work() {
        let step_sampling = SamplerConfig {
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k,
            max_tokens: Some(req.max_tokens),
        };
        let results = guard.step(&step_sampling).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": format!("step: {}", e)})),
            )
        })?;
        for r in &results {
            if r.request_id == rid {
                all_tokens.push(r.token_id);
                output_text.push_str(&r.text);
                if r.finished {
                    finished = true;
                    break;
                }
            }
        }
        if finished {
            break;
        }
    }

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let finish_reason: Value = if finished {
        Value::String("stop".to_string())
    } else {
        Value::Null
    };

    Ok(Json(json!({
        "id": rid,
        "object": "chat.completion",
        "created": created,
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": output_text
            },
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": prompt.split_whitespace().count(),
            "completion_tokens": all_tokens.len(),
            "total_tokens": prompt.split_whitespace().count() + all_tokens.len()
        }
    }))
    .into_response())
}

fn run_serve(model: &str, compiled_dir: &str, port: u16) -> Result<(), anyhow::Error> {
    let artifact_path = std::path::Path::new(&compiled_dir).join(model);

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
    let st_path = format!("{}/weights.safetensors", artifact_path.to_string_lossy());
    let st_path_opt: Option<&str> = if std::path::Path::new(&st_path).exists() {
        Some(&st_path)
    } else {
        None
    };

    log::info!("Loading model from: {}", artifact_path.display());

    let executor = executor::ModelExecutor::load(&dylib_path, st_path_opt).map_err(|e| {
        let ap = artifact_path.display();
        anyhow::anyhow!(
            "Failed to load model '{}': {}\n\
             Tried dylib: {}\n\
             Suggestions:\n\
             - Run: python scripts/compile_dylib.py {} --model-name <name>\n\
             - Check compiled/{} exists and contains a .dylib file\n\
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
    rt.block_on(async move {
        let app = Router::new()
            .route("/health", get(health))
            .route("/v1/completions", post(v1_completions))
            .route("/v1/chat/completions", post(v1_chat_completions))
            .layer(CorsLayer::permissive())
            .with_state(runner);

        let addr = format!("0.0.0.0:{}", port);
        println!("[serve] Listening on http://{}", addr);

        let listener = TcpListener::bind(&addr)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to bind to {}: {}", addr, e))?;
        axum::serve(listener, app)
            .await
            .map_err(|e| anyhow::anyhow!("Server error: {}", e))?;

        Ok::<_, anyhow::Error>(())
    })?;

    Ok(())
}
