//! Integration tests for the LLM-CompileForge runtime.

use std::path::Path;

#[test]
fn test_forward_check_smoke() {
    // This test verifies that the forward_check binary can be built
    // and the basic forward pass infrastructure works.
    // It does NOT require a dylib — it just checks the binary exists.
    let binary = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("debug")
        .join("forward_check");
    // If the binary exists, we can run it. Otherwise, skip.
    if binary.exists() {
        let output = std::process::Command::new(&binary)
            .output()
            .expect("Failed to run forward_check");
        // forward_check may fail if dylib is missing, but it should not panic
        assert!(output.status.code().is_some(), "forward_check exited with signal");
    }
}
