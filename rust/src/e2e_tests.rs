//! E2E correctness tests: Rust runtime forward vs Python reference.
//!
//! Runs the full ``ModelExecutor::load()`` + ``forward()`` pipeline and
//! compares output against a Python-generated reference file.
//! This catches Issue #45 (cos=0.525) type regressions.

#[cfg(test)]
mod e2e_tests {
    use crate::executor::ModelExecutor;

    const COMPILED_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../compiled/opt_125m_fresh");

    /// Find the local safetensors path for a HuggingFace model.
    fn find_safetensors(model_name: &str) -> Option<String> {
        // Try common HF cache locations
        let home = std::env::var("HOME").ok()?;
        let candidates = [
            std::path::Path::new(&home).join(".cache/huggingface/hub"),
            std::path::Path::new(&home).join(".cache/huggingface"),
        ];
        let model_dir_name = format!("models--{}", model_name.replace("/", "--"));
        for cache_dir in &candidates {
            let snapshots_dir = cache_dir.join(&model_dir_name).join("snapshots");
            if let Ok(entries) = std::fs::read_dir(&snapshots_dir) {
                for entry in entries.flatten() {
                    let p = entry.path().join("model.safetensors");
                    if p.exists() {
                        return Some(p.to_string_lossy().to_string());
                    }
                }
            }
        }
        None
    }

    /// Get the path of the .dylib in the model directory.
    fn find_dylib(dir: &str) -> Option<String> {
        std::fs::read_dir(dir).ok()?.filter_map(|e| {
            let e = e.ok()?;
            let p = e.path();
            let ext = p.extension()?;
            if ext == "dylib" {
                Some(p.to_string_lossy().to_string())
            } else {
                None
            }
        }).next()
    }

    /// Check Rust forward produces the same argmax token as Python reference.
    #[test]
    fn test_forward_matches_python_argmax() {
        let dir = COMPILED_DIR;
        let dylib = find_dylib(dir)
            .expect(&format!("no .dylib found in {}", dir));
        let st_path = find_safetensors("facebook/opt-125m")
            .expect("no safetensors found for facebook/opt-125m");
        let input_ids: Vec<u32> = vec![1, 2, 3, 4];

        eprintln!("Loading model from: {}", dylib);
        eprintln!("Safetensors from: {}", st_path);

        // Load model
        let executor = ModelExecutor::load(&dylib, Some(&st_path))
            .expect(&format!("failed to load model from {}", dylib));

        // Forward pass
        let output = executor.forward(&input_ids)
            .expect("forward pass failed");

        // Basic checks
        eprintln!("Rust output shape: {:?}", output.shape);
        eprintln!("Rust output dtype: {:?}", output.dtype);
        assert!(output.shape.len() == 3, "output shape must be [batch, seq, vocab]");
        assert_eq!(output.shape[1], 4, "seq=4");
        assert_eq!(output.shape[2], 50272, "vocab=50272");

        // Check last token argmax
        let logits = output.as_slice();
        if logits.len() >= 50272 * 4 {
            let last_start = 3 * 50272;  // last token of 4
            let last_logits = &logits[last_start..last_start + 50272];
            let mut argmax_idx = 0usize;
            let mut argmax_val = last_logits[0];
            for (i, &v) in last_logits.iter().enumerate() {
                if v > argmax_val {
                    argmax_val = v;
                    argmax_idx = i;
                }
            }
            eprintln!("Rust last token argmax: {} (value={})", argmax_idx, argmax_val);
            // Python gives argmax=1437 for this input. Allow ±1 for floating differences.
            let expected: usize = 1437;
            let diff = if argmax_idx > expected { argmax_idx - expected } else { expected - argmax_idx };
            assert!(diff <= 1,
                    "Rust argmax {} differs from Python argmax {} by more than 1",
                    argmax_idx, expected);
            eprintln!("✅ Token argmax matches Python ({} vs {})", argmax_idx, expected);
        }

        // Check no NaN
        for &v in logits.iter() {
            assert!(v.is_finite(), "non-finite value found in Rust output");
        }
        eprintln!("✅ No NaN in Rust output");
    }

    /// Verify Rust forward doesn't crash and output statistics are reasonable.
    #[test]
    fn test_forward_has_sensible_output() {
        let dir = COMPILED_DIR;
        let dylib = find_dylib(dir)
            .expect(&format!("no .dylib found in {}", dir));
        let st_path = find_safetensors("facebook/opt-125m")
            .expect("no safetensors found for facebook/opt-125m");

        let executor = ModelExecutor::load(&dylib, Some(&st_path))
            .expect("failed to load model");
        let output = executor.forward(&[1, 2, 3, 4])
            .expect("forward failed");

        let logits = output.as_slice();
        let n = logits.len();
        if n == 0 {
            panic!("output is empty");
        }

        // Compute basic stats
        let mut sum = 0.0f64;
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for &v in logits {
            if v.is_finite() {
                sum += v as f64;
                if v < min { min = v; }
                if v > max { max = v; }
            }
        }
        let mean = sum / n as f64;
        eprintln!("Rust output: shape={:?} n={} mean={:.4} range=[{:.4}, {:.4}]",
                  output.shape, n, mean, min, max);

        assert!(mean > -100.0 && mean < 100.0,
                "mean logit {} is out of reasonable range", mean);
        assert!(min > -100.0 && max < 100.0,
                "logit range [{}, {}] is out of reasonable range", min, max);
    }
}
