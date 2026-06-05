//! HTTP server module — OpenAI-compatible API built on axum.
//!
//! Provides health, /v1/completions, and /v1/chat/completions endpoints.

pub mod handlers;
pub mod streaming;
pub mod types;

use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;

use crate::engine::runner::InferenceRunner;

/// Start the HTTP server on the given port.
pub async fn run(
    runner: Arc<Mutex<InferenceRunner>>,
    port: u16,
) -> Result<(), anyhow::Error> {
    let app = Router::new()
        .route("/health", get(handlers::health))
        .route("/v1/completions", post(handlers::v1_completions))
        .route("/v1/chat/completions", post(handlers::v1_chat_completions))
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

    Ok(())
}
