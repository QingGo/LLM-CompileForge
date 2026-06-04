//! HTTP request handlers for OpenAI-compatible API.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};
use tokio::sync::Mutex;

use crate::runner::InferenceRunner;
use crate::sampler::SamplerConfig;

use super::streaming::{build_chat_prompt, v1_chat_completions_stream};
use super::types::{ChatCompletionRequest, CompletionRequest, HealthResponse};

pub async fn health(
    State(_state): State<Arc<Mutex<InferenceRunner>>>,
) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
    })
}

/// POST /v1/completions
pub async fn v1_completions(
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
    let rid = runner
        .add_request(&prompt_str, sampling)
        .map_err(|e| {
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

/// POST /v1/chat/completions — streaming and non-streaming chat completions.
pub async fn v1_chat_completions(
    State(runner): State<Arc<Mutex<InferenceRunner>>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, (StatusCode, Json<Value>)> {
    let prompt = build_chat_prompt(&req.messages);

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
        return Ok(axum::response::sse::Sse::new(stream).into_response());
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
