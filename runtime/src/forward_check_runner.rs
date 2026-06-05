//! Minimal forward-check logic for Issue #45 diagnosis.
//!
//! Loads the compiled opt-125m .dylib, runs forward pass with fixed input,
//! and dumps logits to /tmp/rust_logits.csv for comparison with Python.

use std::path::PathBuf;

fn find_safetensors(compiled_dir: &PathBuf) -> Option<String> {
    let try_names = ["model.safetensors", "weights.safetensors", "pytorch_model.bin"];
    for name in &try_names {
        let p = compiled_dir.join(name);
        if p.exists() {
            return Some(p.to_string_lossy().to_string());
        }
    }
    let home = std::env::var("HOME").ok()?;
    let hub_dir = PathBuf::from(&home).join(".cache/huggingface/hub/models--facebook--opt-125m");
    let snapshots_dir = hub_dir.join("snapshots");
    let entries = std::fs::read_dir(&snapshots_dir).ok()?;
    for entry in entries.flatten() {
        let safetensors = entry.path().join("model.safetensors");
        if safetensors.exists() {
            return Some(safetensors.to_string_lossy().to_string());
        }
    }
    None
}

/// Entry point for the forward_check binary.
pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let compiled_dir = PathBuf::from("outputs/compiled/opt_125m_fresh");
    let dylib_path = compiled_dir.join("libopt_125m_fresh.dylib");

    if !dylib_path.exists() {
        let found = std::fs::read_dir(&compiled_dir)
            .map_err(|e| format!("Cannot read compiled dir '{}': {}", compiled_dir.display(), e))?
            .filter_map(|e| e.ok())
            .find(|e| {
                e.path().extension().map(|ext| ext == "dylib").unwrap_or(false)
            })
            .map(|e| e.path());
        if let Some(p) = found {
            return run_forward(&p, &compiled_dir);
        }
        return Err(format!(
            "No .dylib found in '{}'. Compile the model first.",
            compiled_dir.display()
        )
        .into());
    }

    run_forward(&dylib_path, &compiled_dir)
}

fn run_forward(
    dylib_path: &PathBuf,
    compiled_dir: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let safetensors_path = find_safetensors(compiled_dir).ok_or_else(|| {
        format!(
            "Cannot find model.safetensors/weights.safetensors/pytorch_model.bin in '{}' or HF cache",
            compiled_dir.display()
        )
    })?;

    println!("[forward_check] dylib: {}", dylib_path.display());
    println!("[forward_check] safetensors: {}", safetensors_path);

    let executor = crate::executor::ModelExecutor::load(
        &dylib_path.to_string_lossy(),
        Some(&safetensors_path),
    )
    .map_err(|e| format!("Failed to create executor: {}", e))?;

    let input_ids: Vec<u32> = if let Ok(val) = std::env::var("FORWARD_CHECK_TOKENS") {
        let tokens: Vec<u32> = val
            .split(',')
            .filter_map(|s| s.trim().parse::<u32>().ok())
            .collect();
        eprintln!(
            "[forward_check] Using tokens from FORWARD_CHECK_TOKENS={:?}",
            tokens
        );
        tokens
    } else {
        vec![2u32, 32826, 85, 4129]
    };
    println!(
        "[forward_check] Running forward with {} input tokens: {:?}",
        input_ids.len(),
        input_ids
    );

    let output = executor
        .forward(&input_ids)
        .map_err(|e| format!("Forward failed: {}", e))?;

    let logits = output.as_slice();
    let csv_path = "/tmp/rust_logits.csv";
    let mut wtr = csv::Writer::from_path(csv_path)?;
    for &v in logits {
        wtr.write_record(&[format!("{:.8}", v)])?;
    }
    wtr.flush()?;

    println!(
        "[forward_check] Logits written to {}",
        csv_path
    );
    println!(
        "[forward_check] Shape: {:?}, numel: {}",
        output.shape,
        logits.len()
    );
    println!(
        "[forward_check] First 5 logits: {:?}",
        &logits[..5.min(logits.len())]
    );
    println!(
        "[forward_check] Last 3 logits: {:?}",
        &logits[logits.len().saturating_sub(3)..]
    );
    println!("[forward_check] Done ✓");

    Ok(())
}
