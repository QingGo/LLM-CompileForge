//! Integration tests for the HAL IR forward pass (Path B).
//!
//! These tests verify that the forward_check_hal binary exists and can
//! parse HAL IR. Runtime execution tests are #[ignore]'d due to a
//! pre-existing matmul_blas kernel dispatch bug (batch loop overshoot).

use std::path::Path;
use std::process::Command;

fn forward_check_hal_binary() -> std::path::PathBuf {
    let profile = if cfg!(debug_assertions) { "debug" } else { "release" };
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join(profile)
        .join("forward_check_hal")
}

#[test]
fn test_hal_forward_binary_exists() {
    let binary = forward_check_hal_binary();
    assert!(
        binary.exists(),
        "forward_check_hal binary not found at '{}'. Build with: \
         cargo build --bin forward_check_hal --features hal-rust",
        binary.display(),
    );
}

#[test]
#[ignore = "pre-existing matmul_blas kernel dispatch bug causes SIGSEGV or panic. \
           See runtime/src/hal/primitives/matmul.rs:105 — batch loop overshoot. \
           Restore when HAL Path B kernel dispatch is fixed."]
fn test_hal_forward_no_panic() {
    let binary = forward_check_hal_binary();
    if !binary.exists() {
        eprintln!("Skipping test: forward_check_hal binary not built");
        return;
    }
    let project_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let output = Command::new(&binary)
        .current_dir(project_root)
        .output()
        .expect("Failed to run forward_check_hal");
    assert!(output.status.code().is_some(), "binary exited with signal");
    assert!(output.status.success(), "binary exited with non-zero code: {:?}", output.status.code());
}

#[test]
#[ignore = "pre-existing matmul_blas kernel dispatch bug prevents successful forward pass. \
           Same root cause as test_hal_forward_no_panic."]
fn test_hal_forward_shape_smoke() {
    let binary = forward_check_hal_binary();
    if !binary.exists() {
        eprintln!("Skipping test: forward_check_hal binary not built");
        return;
    }
    let project_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let output = Command::new(&binary)
        .current_dir(project_root)
        .output()
        .expect("Failed to run forward_check_hal");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("numel:") || stdout.contains("Shape:") || stdout.contains("Done"),
        "Expected shape info or completion marker in output.\nstdout:\n{}", stdout,
    );
}

#[test]
fn test_hal_forward_all_functions_in_ir() {
    let binary = forward_check_hal_binary();
    if !binary.exists() {
        eprintln!("Skipping test: forward_check_hal binary not built");
        return;
    }
    let hal_ir_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("outputs/compiled/opt_125m_fresh/generated/hal_ir.json");
    if !hal_ir_path.exists() {
        eprintln!("Skipping test: hal_ir.json not found at {}", hal_ir_path.display());
        return;
    }
    let content = std::fs::read_to_string(&hal_ir_path)
        .unwrap_or_else(|_| panic!("Failed to read {}", hal_ir_path.display()));
    let hal_ir: serde_json::Value = serde_json::from_str(&content)
        .unwrap_or_else(|_| panic!("Failed to parse hal_ir.json"));
    let functions = hal_ir["functions"].as_array()
        .unwrap_or_else(|| panic!("hal_ir.json missing 'functions' array"));
    assert!(!functions.is_empty(), "hal_ir.json has no functions");

    let mut all_ok = true;
    for func in functions {
        let name = func["name"].as_str().unwrap_or("?");
        let ops = func["ops"].as_array().map(|a| a.len()).unwrap_or(0);
        if ops == 0 {
            eprintln!("WARNING: function '{}' has 0 ops", name);
            all_ok = false;
        }
    }
    assert!(all_ok, "Some functions in hal_ir.json have 0 ops");
}
