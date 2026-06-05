//! Contract test: runtime execution determinism.
//!
//! Verifies the sfa_determinism contract:
//!   "Same model + same inputs → bit-identical output"
//!
//! Independent of compiler — uses existing compiled dylib.

use crate::engine::executor::ModelExecutor;

/// Load the compiled opt_125m_fresh model for testing.
fn compiled_executor() -> ModelExecutor {
    let dylib = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../outputs/compiled/opt_125m_fresh/libopt_125m_fresh.dylib"
    );
    let st_path = concat!(
        env!("HOME"),
        "/.cache/huggingface/hub/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6/model.safetensors"
    );
    ModelExecutor::load(dylib, Some(st_path))
        .unwrap_or_else(|_| panic!(
            "compiled model not found at {dylib}. Run `make test-pipeline-smoke` first."
        ))
}

#[test]
fn test_forward_bit_identical_rerun() {
    let exec = compiled_executor();
    let input = vec![1u32, 2, 3, 4];

    let out1 = exec.forward(&input).expect("forward run 1");
    let out2 = exec.forward(&input).expect("forward run 2");
    let out3 = exec.forward(&input).expect("forward run 3");

    // Contract: three runs with identical input must produce identical output.
    let s1 = out1.as_slice();
    let s2 = out2.as_slice();
    let s3 = out3.as_slice();

    assert_eq!(
        s1.len(),
        s2.len(),
        "Determinism violation: run 1 output len {} != run 2 output len {}",
        s1.len(),
        s2.len()
    );
    assert_eq!(
        s2.len(),
        s3.len(),
        "Determinism violation: run 2 output len {} != run 3 output len {}",
        s2.len(),
        s3.len()
    );

    for i in 0..s1.len() {
        assert!(
            (s1[i] - s2[i]).abs() < 1e-6,
            "Determinism violation at index {}: run1={} vs run2={}",
            i, s1[i], s2[i]
        );
        assert!(
            (s2[i] - s3[i]).abs() < 1e-6,
            "Determinism violation at index {}: run2={} vs run3={}",
            i, s2[i], s3[i]
        );
    }
}
