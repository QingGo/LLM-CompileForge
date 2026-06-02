//! HAL IR forward-check binary.
//!
//! Loads the compiled opt-125m HAL IR (`generated/hal_ir.json`), runs
//! the full 28-function forward pass through `HalRustExecutable`, and
//! dumps logits to `/tmp/rust_hal_logits.csv`.
//!
//! Uses `ModelExecutor` (Path A, dylib-based) only for access to the
//! `WeightProvider` — the actual forward pass runs via `HalRustRunner`
//! (Path B, pure-Rust).  WeightProvider integration is reserved for
//! Task 5; currently all weights are zero-filled.
//!
//! Usage:
//!   cargo run --bin forward_check_hal --features hal-rust
//!
//! Modules use `#[path]` to reference source files in src/ since this
//! binary lives under src/bin/.

#[path = "../block_manager.rs"]
mod block_manager;
#[path = "../abi.rs"]
mod abi;
#[path = "../ciface_high.rs"]
mod ciface_high;
#[path = "../compute_graph.rs"]
mod compute_graph;
#[path = "../compute_graph_runner.rs"]
mod compute_graph_runner;
#[path = "../error.rs"]
mod error;
#[path = "../executor.rs"]
mod executor;
#[path = "../global_input.rs"]
mod global_input;
#[path = "../hal/mod.rs"]
mod hal;
#[path = "../hal_runner/mod.rs"]
mod hal_runner;
#[path = "../kernel_catalog.rs"]
mod kernel_catalog;
#[path = "../kv_cache.rs"]
mod kv_cache;
#[path = "../kv_cache_intercept.rs"]
mod kv_cache_intercept;
#[path = "../sfcf.rs"]
mod sfcf;
#[path = "../sfa_tensor.rs"]
mod sfa_tensor;
#[path = "../tensor.rs"]
mod tensor;
#[path = "../weight_loader.rs"]
mod weight_loader;

use std::path::PathBuf;

#[cfg(not(feature = "hal-rust"))]
fn main() {
    eprintln!(
        "forward_check_hal requires --features hal-rust.  \
         Build with: cargo build --bin forward_check_hal --features hal-rust"
    );
    std::process::exit(1);
}

#[cfg(feature = "hal-rust")]
/// Find a safetensors file for WeightProvider loading.
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

#[cfg(feature = "hal-rust")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging.  Use RUST_LOG=info for moderate output.
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    let compiled_dir = PathBuf::from("compiled/opt_125m_fresh");
    let hal_ir_dir = compiled_dir.join("generated");
    let hal_ir_path = hal_ir_dir.join("hal_ir.json");
    let dylib_path = compiled_dir.join("libopt_125m.dylib");

    // Validate paths.
    if !hal_ir_path.exists() {
        return Err(format!(
            "HAL IR not found at '{}'. Re-compile the model with HAL IR support.",
            hal_ir_path.display()
        )
        .into());
    }

    // Load ModelExecutor for WeightProvider access (weights loaded
    // from safetensors).  The executor itself is not used for forward —
    // it only provides weight data for the HAL IR runner.
    let safetensors_path = find_safetensors(&compiled_dir)
        .ok_or_else(|| format!(
            "Cannot find model.safetensors/weights.safetensors/pytorch_model.bin in '{}' or HF cache",
            compiled_dir.display()
        ))?;

    // Resolve dylib: try the expected name first, then search.
    let dylib_path = if dylib_path.exists() {
        dylib_path
    } else {
        let found = std::fs::read_dir(&compiled_dir)
            .map_err(|e| format!("Cannot read compiled dir '{}': {}", compiled_dir.display(), e))?
            .filter_map(|e| e.ok())
            .find(|e| {
                e.path().extension().map(|ext| ext == "dylib").unwrap_or(false)
            })
            .map(|e| e.path())
            .ok_or_else(|| format!(
                "No .dylib found in '{}'. Compile the model first.",
                compiled_dir.display()
            ))?;
        found
    };

    println!("[forward_check_hal] HAL IR: {}", hal_ir_path.display());
    println!("[forward_check_hal] dylib (for weight access): {}", dylib_path.display());
    println!("[forward_check_hal] safetensors: {}", safetensors_path);

    // Load ModelExecutor (Path A) for WeightProvider access.
    // The WeightProvider loads real weights from safetensors and is
    // passed to the HAL IR runner for weight injection.
    let executor = crate::executor::ModelExecutor::load(
        &dylib_path.to_string_lossy(),
        Some(&safetensors_path),
    )
    .map_err(|e| format!("Failed to create executor: {}", e))?;

    // Load HAL IR.
    let hal_ir_content = std::fs::read_to_string(&hal_ir_path)
        .map_err(|e| format!("Failed to read HAL IR '{}': {}", hal_ir_path.display(), e))?;
    let runner = crate::hal_runner::HalRustRunner::from_json(&hal_ir_content)
        .map_err(|e| format!("Failed to parse HAL IR: {}", e))?;

    println!(
        "[forward_check_hal] HAL IR: {} functions, {} model",
        runner.hal_ir.num_functions,
        runner.hal_ir.model_name,
    );

    // Create HalRustExecutable and CPU stream.
    let hal_exe = crate::hal::rust::executable::HalRustExecutable::new(
        runner.hal_ir.num_functions,
    );
    let stream = crate::hal::cpu::CpuStream;

    // Default input tokens (same as forward_check.rs).
    let input_ids: Vec<u32> = if let Ok(val) = std::env::var("FORWARD_CHECK_TOKENS") {
        let tokens: Vec<u32> = val
            .split(',')
            .filter_map(|s| s.trim().parse::<u32>().ok())
            .collect();
        eprintln!("[forward_check_hal] Using tokens from FORWARD_CHECK_TOKENS={:?}", tokens);
        tokens
    } else {
        vec![2u32, 32826, 85, 4129]
    };
    let positions: Vec<u32> = (0..input_ids.len() as u32).collect();
    println!(
        "[forward_check_hal] Running forward with {} input tokens: {:?}",
        input_ids.len(),
        input_ids
    );

    // Run the full forward pass through the HAL IR runner with real weights.
    let output = crate::hal_runner::run_hal_function_graph(
        &hal_exe,
        &runner.hal_ir,
        Some(&executor.weight_provider),
        &stream,
        &input_ids,
        &positions,
    )
    .map_err(|e| format!("HAL forward failed: {}", e))?;

    let logits = output.as_slice();
    let csv_path = "/tmp/rust_hal_logits.csv";
    let mut wtr = csv::Writer::from_path(csv_path)?;
    for &v in logits {
        wtr.write_record(&[format!("{:.8}", v)])?;
    }
    wtr.flush()?;

    println!("[forward_check_hal] Logits written to {}", csv_path);
    println!("[forward_check_hal] Shape: {:?}, numel: {}", output.shape, logits.len());
    println!("[forward_check_hal] First 5 logits: {:?}", &logits[..5.min(logits.len())]);
    println!("[forward_check_hal] Last 3 logits: {:?}", &logits[logits.len().saturating_sub(3)..]);

    // Basic sanity: check that logits are all finite (no NaN, no inf).
    let has_nan = logits.iter().any(|&v| v.is_nan());
    let has_inf = logits.iter().any(|&v| v.is_infinite());
    if has_nan || has_inf {
        eprintln!(
            "[forward_check_hal] WARNING: output contains {} NaN, {} Inf",
            if has_nan { "some" } else { "no" },
            if has_inf { "some" } else { "no" },
        );
    } else {
        println!("[forward_check_hal] All logits are finite ✓");
    }

    println!("[forward_check_hal] Done ✓");

    Ok(())
}
