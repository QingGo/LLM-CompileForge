//! M1 integration test: load a compiled .dylib and call a compute function.
//!
//! Verifies the Rust → compiled MLIR kernel FFI path works end-to-end:
//!   1. ``libloading`` opens the .dylib
//!   2. ``_mlir_ciface_*`` is looked up
//!   3. Input memref descriptors are constructed
//!   4. The function is called via the C ABI
//!   5. Output is read back from the result descriptor

use std::path::Path;

#[cfg(test)]
mod m1_tests {
    use crate::hal::cpu::kernel::CifaceFn3;
    use crate::hal::cpu::memref::MemRefDesc1;

    const DYLIB_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../tests/data/test_m1.dylib");

    #[test]
    fn test_m1_add_two_via_dylib() {
        let lib = unsafe {
            libloading::Library::new(DYLIB_PATH).expect("failed to load test_m1.dylib")
        };

        let func: libloading::Symbol<CifaceFn3> = unsafe {
            lib.get(b"_mlir_ciface_add_two")
                .expect("symbol _mlir_ciface_add_two not found")
        };

        // The ciface wrapper takes struct-based descriptors.
        // add_two operates on memref<2xf32> (rank 1 with 2 elements)
        const N: usize = 2;
        let a: Vec<f32> = vec![1.0; N];    // [1.0, 1.0]
        let b: Vec<f32> = vec![2.0; N];    // [2.0, 2.0]

        let a_desc = MemRefDesc1::from_f32_slice(&a, [N]);
        let b_desc = MemRefDesc1::from_f32_slice(&b, [N]);
        let mut out_desc = MemRefDesc1::zeroed([N]);

        unsafe {
            func(
                &mut out_desc as *mut MemRefDesc1 as *mut std::ffi::c_void,
                &a_desc as *const MemRefDesc1 as *const std::ffi::c_void,
                &b_desc as *const MemRefDesc1 as *const std::ffi::c_void,
            )
        };

        let out = unsafe { out_desc.read_output_f32() };
        assert_eq!(out.len(), N, "output size mismatch");
        for (i, &val) in out.iter().enumerate() {
            assert!(
                (val - 3.0).abs() < 1e-5,
                "mismatch at index {}: expected 3.0, got {}",
                i,
                val
            );
        }
    }
}

/// End-to-end test: load the full opt_125m model, run forward, dump logits.
/// Run with: cargo test -- --nocapture test_opt_125m_forward
/// Then compare with HF reference using Python.
#[cfg(test)]
mod integration_tests {
    use crate::executor::ModelExecutor;

    fn find_safetensors() -> Option<String> {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        let candidates = [
            format!("{}/.cache/huggingface/hub/models--facebook--opt-125m/snapshots", home),
            format!("{}/.cache/huggingface/hub/models--facebook--opt-125m/blobs", home),
            "compiled/opt_125m_v8/model.safetensors".to_string(),
            "compiled/opt_125m_v8/weights.safetensors".to_string(),
        ];
        for dir_path in &candidates {
            let path = std::path::Path::new(dir_path);
            if path.is_dir() {
                if let Ok(entries) = std::fs::read_dir(path) {
                    for entry in entries.flatten() {
                        let p = entry.path();
                        if p.extension().map(|e| e == "safetensors").unwrap_or(false) {
                            return Some(p.to_string_lossy().to_string());
                        }
                        if p.is_dir() {
                            let model_st = p.join("model.safetensors");
                            if model_st.exists() {
                                return Some(model_st.to_string_lossy().to_string());
                            }
                        }
                    }
                }
            } else if path.is_file() {
                return Some(dir_path.clone());
            }
        }
        None
    }

    #[test]
    fn test_opt_125m_forward_runs() {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let dylib = format!("{}/../compiled/opt_125m_fresh/libopt_125m.dylib", manifest_dir);
        let st = find_safetensors();

        eprintln!("dylib: {}", dylib);
        eprintln!("safetensors: {:?}", st);

        let executor = ModelExecutor::load(&dylib, st.as_deref())
            .expect("Failed to load ModelExecutor. Has the model been compiled?");

        eprintln!(
            "Model loaded: {} functions, {} weight mappings, {} constants",
            executor.compute_graph.functions.len(),
            executor.weight_provider.name_mapping().len(),
            executor.weight_provider.constants().len(),
        );

        // Use input_ids that match the compiled model's GlobalInput shape (1×4=4).
        let input_ids: Vec<u32> = vec![2, 32826, 85, 4129];

        let result = executor.forward(&input_ids).expect("forward failed");
        let logits_slice = result.as_slice();
        eprintln!(
            "Forward OK: shape={:?}, numel={}, first={:.4}, last={:.4}, mean={:.4}",
            result.shape,
            result.numel(),
            logits_slice[0],
            logits_slice[logits_slice.len() - 1],
            logits_slice.iter().sum::<f32>() / logits_slice.len() as f32,
        );

        use std::io::Write;
        let csv_path = "/tmp/rust_logits.csv";
        if let Ok(mut f) = std::fs::File::create(csv_path) {
            for (i, &v) in logits_slice.iter().enumerate() {
                if i > 0 { write!(f, ",").ok(); }
                write!(f, "{:.8}", v).ok();
            }
            eprintln!("Logits saved to {}", csv_path);
        }

        assert_eq!(result.shape.len(), 3, "expected 3D output");
        assert!(result.shape[0] >= 1, "batch should be >= 1");
        assert_eq!(result.shape[1], 4, "seq=4");
        assert_eq!(result.shape[2], 50272, "vocab=50272");
        assert!(logits_slice[0].is_finite(), "logits should be finite");
        assert!(result.numel() > 0, "output should not be empty");
    }

    /// End-to-end test: full auto-regressive loop via InferenceRunner.
    /// Loads compiled model, creates runner, feeds a short prompt,
    /// runs step() multiple times, verifies tokens are produced.
    #[test]
    fn test_runner_generate_end_to_end() {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let dylib = format!("{}/../compiled/opt_125m_fresh/libopt_125m.dylib", manifest_dir);
        let st = find_safetensors();

        if !std::path::Path::new(&dylib).exists() {
            eprintln!("SKIP: compiled model not found at {}", dylib);
            return;
        }

        let executor = match ModelExecutor::load(&dylib, st.as_deref()) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("SKIP: failed to load model: {}", e);
                return;
            }
        };

        // Use test tokenizer (minimal BPE for unit tests)
        let tok_path = format!("{}/../tests/data/test_tokenizer.json", manifest_dir);
        let tokenizer = match crate::tokenizer::Tokenizer::from_file(&tok_path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("SKIP: no test tokenizer: {}", e);
                return;
            }
        };

        let config = crate::runner::RunnerConfig {
            max_batch_size: 4,
            max_tokens_per_step: 128,
            chunk_size: 64,
            num_blocks: 1024,
            block_size: 16,
            max_tokens_per_request: 5,
            seed: 42,
            use_chat_template: false,
            use_kernel_catalog: false,
        };

        let mut runner = match crate::runner::InferenceRunner::new(executor, tokenizer, config) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("SKIP: failed to create runner: {}", e);
                return;
            }
        };

        // Add a simple prompt — "a b" — which the test tokenizer can encode
        // (test tokenizer has a=4, b=5, <s>=0, </s>=2)
        let sampling = crate::sampler::SamplerConfig {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            max_tokens: Some(5),
        };
        let rid = runner
            .add_request("a b", sampling.clone())
            .expect("add request");

        eprintln!("Added request: {}", rid);
        let mut total_tokens = 0usize;

        // Run up to 10 steps
        for step in 0..10 {
            let results = runner.step(&sampling).expect("step failed");
            if results.is_empty() {
                eprintln!("No more work after step {}", step);
                break;
            }
            for r in &results {
                eprintln!("step[{}] request={} token={} text='{}' finished={}",
                           step, r.request_id, r.token_id, r.text, r.finished);
                total_tokens += 1;
                if r.finished {
                    eprintln!("Request finished at step {}", step);
                }
            }
        }

        assert!(total_tokens >= 1, "should produce at least 1 token");
        eprintln!(
            "E2E OK: produced {} tokens across {} steps",
            total_tokens, 10.min(total_tokens)
        );
    }
}
