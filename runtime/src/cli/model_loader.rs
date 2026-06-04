//! Model loading utilities — dylib path resolution, safetensors discovery.

use std::path::Path;

/// Scan a directory for the first .dylib file.
pub fn resolve_dylib_path(artifact_path: &Path, model: &str) -> String {
    std::fs::read_dir(artifact_path)
        .ok()
        .and_then(|entries| {
            entries.filter_map(|e| e.ok()).find(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "dylib")
                    .unwrap_or(false)
            })
        })
        .map(|e| e.path().to_string_lossy().to_string())
        .unwrap_or_else(|| format!("{}/lib{}.dylib", artifact_path.display(), model))
}

/// Resolve safetensors path: check compiled dir first, then HF cache.
pub fn resolve_safetensors_path(artifact_path: &Path) -> Option<String> {
    let try_names = [
        "model.safetensors",
        "weights.safetensors",
        "pytorch_model.bin",
    ];
    for name in &try_names {
        let p = artifact_path.join(name);
        if p.exists() {
            return Some(p.to_string_lossy().to_string());
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        let hub_dir = std::path::PathBuf::from(&home)
            .join(".cache/huggingface/hub/models--facebook--opt-125m");
        let snapshots_dir = hub_dir.join("snapshots");
        if let Ok(entries) = std::fs::read_dir(&snapshots_dir) {
            for entry in entries.flatten() {
                let safetensors = entry.path().join("model.safetensors");
                if safetensors.exists() {
                    return Some(safetensors.to_string_lossy().to_string());
                }
            }
        }
    }
    None
}
