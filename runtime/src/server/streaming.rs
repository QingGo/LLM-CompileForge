//! SSE streaming for chat completions — yields one SSE event per token.

use std::convert::Infallible;
use std::sync::Arc;

use axum::response::sse::Event;
use futures::stream::{self, Stream, StreamExt};
use serde_json::{json, Value};
use tokio::sync::Mutex;

use crate::runner::InferenceRunner;
use crate::sampler::SamplerConfig;

/// Build a chat prompt from messages using a simple fallback format.
/// Since OPT-125m has no chat template, use "role: content\n" format.
pub fn build_chat_prompt(messages: &[super::types::ChatMessage]) -> String {
    let mut parts = Vec::new();
    for msg in messages {
        parts.push(format!("{}: {}", msg.role, msg.content));
    }
    parts.push("Assistant: ".to_string());
    parts.join("\n")
}

/// SSE streaming for chat completions — yields one SSE event per token.
pub fn v1_chat_completions_stream(
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
