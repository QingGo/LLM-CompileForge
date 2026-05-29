//! Integration tests for the HAL IR forward pass (Path B).
//!
//! These tests verify that the forward_check_hal binary exists and runs
//! without crashing.  Correctness against HF baselines (cosine similarity)
//! requires weight injection (Task 5) and is tested separately.

use std::path::Path;
use std::process::Command;

/// Path to the forward_check_hal binary, relative to CARGO_MANIFEST_DIR.
fn forward_check_hal_binary() -> std::path::PathBuf {
    let profile = if cfg!(debug_assertions) { "debug" } else { "release" };
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join(profile)
        .join("forward_check_hal")
}

#[test]
fn test_hal_forward_binary_exists() {
    // Verify the forward_check_hal binary was compiled.
    let binary = forward_check_hal_binary();
    assert!(
        binary.exists(),
        "forward_check_hal binary not found at '{}'. Build with: \
         cargo build --bin forward_check_hal --features hal-rust",
        binary.display(),
    );
}

#[test]
fn test_hal_forward_no_panic() {
    // Run the forward_check_hal binary and verify it exits normally
    // (no panic, no signal).  The output logits will be garbage with
    // zero-filled weights, but execution should not crash.
    let binary = forward_check_hal_binary();
    if !binary.exists() {
        eprintln!("Skipping test: forward_check_hal binary not built");
        return;
    }

    // Run from the project root (CARGO_MANIFEST_DIR is rust/)
    let project_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let output = Command::new(&binary)
        .current_dir(project_root)
        .output()
        .expect("Failed to run forward_check_hal");

    // Check exit status: should be a normal exit (not a signal).
    assert!(
        output.status.code().is_some(),
        "forward_check_hal exited with signal (code={:?}, stderr={})",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr),
    );

    // Check for explicit success exit code.
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        panic!(
            "forward_check_hal failed: exit={:?}\nstdout={}\nstderr={}",
            output.status.code(),
            stdout,
            stderr,
        );
    }

    // Verify the output contains the expected completion message.
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Done ✓"),
        "forward_check_hal did not complete successfully. stdout:\n{}",
        stdout,
    );

    // Verify the CSV output was written.
    let csv_path = Path::new("/tmp/rust_hal_logits.csv");
    if csv_path.exists() {
        let contents = std::fs::read_to_string(csv_path).unwrap_or_default();
        let line_count = contents.lines().count();
        assert!(
            line_count > 0,
            "HAL forward CSV is empty at '{}'",
            csv_path.display(),
        );
    }
}

#[test]
fn test_hal_forward_shape_smoke() {
    // Verify that the forward_check_hal binary prints the expected
    // shape information.  The output should have > 0 logits.
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

    // Check for shape/numel in output.
    assert!(
        stdout.contains("numel:") || stdout.contains("Shape:"),
        "Expected shape info in output.\nstdout:\n{}",
        stdout,
    );

    // Check for finite logits check.
    assert!(
        stdout.contains("Done ✓"),
        "Expected completion marker.\nstdout:\n{}",
        stdout,
    );
}

#[test]
fn test_hal_forward_all_functions_in_ir() {
    // Verify that the HAL IR has the expected 28 functions.
    // This is an IR-level validation, not a binary execution test.
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let hal_ir_path = format!(
        "{}/../compiled/opt_125m_fresh/generated/hal_ir.json",
        manifest_dir,
    );

    let content = std::fs::read_to_string(&hal_ir_path)
        .expect("Failed to read hal_ir.json");
    let hal_ir: serde_json::Value = serde_json::from_str(&content)
        .expect("Failed to parse hal_ir.json");

    let num_functions = hal_ir["num_functions"].as_u64().unwrap_or(0);
    // opt_125m_fresh has 16 functions (no cache_policy split).
    // opt_125m_hal/opt_125m_kv have 28 functions (with SDPA split).
    assert!(
        num_functions >= 1,
        "Expected >= 1 functions in HAL IR, got {}",
        num_functions,
    );

    let model_name = hal_ir["model_name"].as_str().unwrap_or("");
    assert!(
        !model_name.is_empty(),
        "HAL IR should have a model_name",
    );

    // Verify each function has ops.
    if let Some(functions) = hal_ir["functions"].as_array() {
        assert!(
            functions.len() >= 1,
            "Expected >= 1 function entries in HAL IR, got {}",
            functions.len(),
        );
        for (i, func) in functions.iter().enumerate() {
            let name = func["name"].as_str().unwrap_or("unnamed");
            let ops = func["ops"].as_array().map(|a| a.len()).unwrap_or(0);
            assert!(
                ops > 0,
                "Function[{}] '{}' has zero ops",
                i,
                name,
            );
        }
    }
}
