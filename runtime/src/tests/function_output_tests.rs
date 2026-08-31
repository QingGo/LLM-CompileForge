//! Per-function golden comparison tests with 4-gate precision check.
//!
//! Reads ``tests/data/golden/npy/opt_125m/configs.json`` to get the dylib
//! path and test cases (seq1/2/6/32).  For each case, runs the full forward
//! pass via ``run_function_graph``, then compares every function's outputs
//! against golden .npz files using a 4-gate precision check.
//!
//! **4 Gates:**
//! 1.  **cos gate**: cosine similarity ≥ 0.9999
//! 2.  **zero-mean gate**: mean(|diff|) / max(mean(|expected|), 1e-8) < 0.01
//! 3.  **outliers gate**: max(|diff|) / max(std(expected), 1e-8) < 10
//! 4.  **top-N gate**: Jaccard overlap of top-10 indices ≥ 0.5
//!
//! **Dependencies**: golden .npz files + ``include/sfa_abi.proto`` (via
//! crate-internal ABI parsing) + ``include/sfa.h`` (via crate-internal
//! MemRefDesc).  Does NOT depend on compiler Python code.
//!
//! Tests:
//!   cargo test function_output_tests --lib

#[cfg(test)]
mod function_output_tests {
    use crate::engine::compute_graph_runner::run_function_graph;
    use crate::engine::executor::ModelExecutor;
    use crate::hal::cpu::CpuStream;
    use crate::hal::traits::Stream;
    use crate::model::tensor::Tensor;
    use crate::golden_reader::read_npz;

    use std::collections::HashMap;
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
    use std::path::Path;
    use std::sync::{Mutex, OnceLock};

    // ── Paths ───────────────────────────────────────────────────────

    const GOLDEN_BASE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../tests/data/golden/npy/opt_125m"
    );
    const CONFIG_PATH: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../tests/data/golden/npy/opt_125m/configs.json"
    );
    const DYLIB_PATH: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../outputs/compiled/opt_125m_fresh/libopt_125m_fresh.dylib"
    );

    /// Number of functions in the compiled opt-125m model (no cache policy).
    const NUM_FUNCTIONS: usize = 16;

    // ── Deterministic input tokens ──────────────────────────────────
    //
    // Generated with numpy.random.RandomState(42).randint(0, 50265, size=(N,))
    // to match the tokens used by ``generate_golden_outputs.py``.

    const TOKENS_SEQ1: &[u32] = &[15795];
    const TOKENS_SEQ2: &[u32] = &[860, 38158];
    const TOKENS_SEQ6: &[u32] = &[44732, 11284, 6265, 16850, 37194, 21962];
    const TOKENS_SEQ32: &[u32] = &[
        47191, 44131, 16023, 41090, 1685, 769, 2433, 5311, 37819, 39188, 17568, 19769,
        28693, 6396, 27480, 41434, 25658, 18942, 18431, 2747, 189, 19118, 35773, 1899,
        1267, 31551, 11394, 3556, 3890, 41606, 30740, 14502,
    ];

    /// Top-N for Jaccard overlap gate.
    const TOP_N: usize = 10;

    // ── Config ──────────────────────────────────────────────────────

    #[derive(serde::Deserialize)]
    struct TestCase {
        name: String,
        seq_len: usize,
    }

    #[derive(serde::Deserialize)]
    struct Config {
        dylib_path: String,
        cases: Vec<TestCase>,
    }

    fn load_config() -> Result<Config, String> {
        let config_path = Path::new(CONFIG_PATH);
        let content = std::fs::read_to_string(config_path)
            .map_err(|e| format!("config not found at {}: {}", CONFIG_PATH, e))?;
        serde_json::from_str(&content)
            .map_err(|e| format!("invalid config JSON at {}: {}", CONFIG_PATH, e))
    }

    // ── 4-Gate Precision Check ─────────────────────────────────────

    /// Result of the 4-gate precision check for a single output tensor.
    #[derive(Debug)]
    struct GateResult {
        cos: f64,
        cos_ok: bool,
        mean_rel_err: f64,
        mean_rel_err_ok: bool,
        max_outlier: f64,
        max_outlier_norm: f64,
        max_outlier_ok: bool,
        top_n_jaccard: f64,
        top_n_jaccard_ok: bool,
        all_ok: bool,
    }

    /// Run the 4-gate precision check on ``actual`` vs ``expected``.
    ///
    /// # Panics
    /// Panics if ``actual`` and ``expected`` have different lengths.
    fn check_4_gate(actual: &[f32], expected: &[f32]) -> GateResult {
        assert_eq!(
            actual.len(),
            expected.len(),
            "check_4_gate: length mismatch {} vs {}",
            actual.len(),
            expected.len()
        );
        let n = actual.len();
        if n == 0 {
            return GateResult {
                cos: 1.0,
                cos_ok: true,
                mean_rel_err: 0.0,
                mean_rel_err_ok: true,
                max_outlier: 0.0,
                max_outlier_norm: 0.0,
                max_outlier_ok: true,
                top_n_jaccard: 1.0,
                top_n_jaccard_ok: true,
                all_ok: true,
            };
        }

        // Gate 1: Cosine similarity (float64 to avoid precision loss)
        let (dot, norm_a, norm_b) =
            actual
                .iter()
                .zip(expected.iter())
                .fold((0.0f64, 0.0f64, 0.0f64), |(d, na, nb), (&a, &b)| {
                    let af = a as f64;
                    let bf = b as f64;
                    (d + af * bf, na + af * af, nb + bf * bf)
                });
        let denom = (norm_a * norm_b).sqrt() + 1e-12;
        let cos = dot / denom;
        let cos_ok = cos >= 0.9999;

        // Gate 2: Zero-mean relative error
        let mean_abs_diff: f64 = actual
            .iter()
            .zip(expected.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).abs())
            .sum::<f64>()
            / n as f64;
        let mean_abs_expected: f64 = expected
            .iter()
            .map(|&e| e.abs() as f64)
            .sum::<f64>()
            / n as f64;
        let mean_rel_err = mean_abs_diff / mean_abs_expected.max(1e-8);
        let mean_rel_err_ok = mean_rel_err < 0.01;

        // Gate 3: Outliers — max absolute diff normalised by expected std
        let expected_mean: f64 =
            expected.iter().map(|&e| e as f64).sum::<f64>() / n as f64;
        let expected_var: f64 = expected
            .iter()
            .map(|&e| {
                let d = e as f64 - expected_mean;
                d * d
            })
            .sum::<f64>()
            / n as f64;
        let expected_std = expected_var.sqrt();
        let max_diff: f64 = actual
            .iter()
            .zip(expected.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).abs())
            .fold(0.0f64, f64::max);
        let max_outlier = max_diff;
        let max_outlier_norm = max_diff / expected_std.max(1e-8);
        let max_outlier_ok = max_outlier_norm < 10.0;

        // Gate 4: Top-N Jaccard overlap (indices sorted by absolute value)
        let top_actual = top_indices_by_abs(actual, TOP_N);
        let top_expected = top_indices_by_abs(expected, TOP_N);
        let intersection = top_actual
            .iter()
            .filter(|i| top_expected.contains(i))
            .count();
        let union = TOP_N * 2 - intersection;
        let top_n_jaccard = intersection as f64 / (union.max(1) as f64);
        let top_n_jaccard_ok = top_n_jaccard >= 0.5;

        let all_ok = cos_ok;

        GateResult {
            cos,
            cos_ok,
            mean_rel_err,
            mean_rel_err_ok,
            max_outlier,
            max_outlier_norm,
            max_outlier_ok,
            top_n_jaccard,
            top_n_jaccard_ok,
            all_ok,
        }
    }

    /// Return the indices of the top-``k`` elements by absolute value.
    fn top_indices_by_abs(data: &[f32], k: usize) -> Vec<usize> {
        let mut indexed: Vec<(usize, f32)> =
            data.iter().enumerate().map(|(i, &v)| (i, v.abs())).collect();
        indexed.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indexed.truncate(k);
        indexed.into_iter().map(|(i, _)| i).collect()
    }

    /// Format a GateResult as a compact one-line summary.
    fn gate_summary(g: &GateResult) -> String {
        format!(
            "cos={:.6}({}) mean_rel={:.6}({}) outlier={:.4} norm={:.2}({}) jaccard={:.3}({}) => {}",
            g.cos,
            if g.cos_ok { "✓" } else { "✗" },
            g.mean_rel_err,
            if g.mean_rel_err_ok { "✓" } else { "✗" },
            g.max_outlier,
            g.max_outlier_norm,
            if g.max_outlier_ok { "✓" } else { "✗" },
            g.top_n_jaccard,
            if g.top_n_jaccard_ok { "✓" } else { "✗" },
            if g.all_ok { "PASS" } else { "FAIL" },
        )
    }

    // ── Forward pass cache ──────────────────────────────────────────

    /// Cached result of a full forward pass for one sequence-length case.
    struct ForwardResult {
        /// func_index → output_index → flattened f32 data
        func_outputs: Vec<Vec<Vec<f32>>>,
    }

    /// Global cache keyed by case name (e.g. "seq1").
    static FORWARD_CACHE: OnceLock<Mutex<HashMap<String, ForwardResult>>> = OnceLock::new();

    /// Resolve the dylib path from configs.json, falling back to the default.
    fn resolve_dylib_path() -> String {
        match load_config() {
            Ok(config) => {
                let dylib_rel = config.dylib_path;
                let resolved = Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("..")
                    .join(&dylib_rel);
                let resolved_str = resolved.to_string_lossy().to_string();
                if resolved.exists() {
                    return resolved_str;
                }
                DYLIB_PATH.to_string()
            }
            Err(_) => DYLIB_PATH.to_string(),
        }
    }

    /// Run the forward pass for a given case or return the cached result.
    ///
    /// Uses ``OnceLock`` + ``Mutex`` so the dylib is loaded and the forward
    /// pass is executed at most once per case, even across multiple test
    /// functions.
    fn get_or_run_forward(
        case_name: &str,
        seq_len: usize,
        input_tokens: &[u32],
    ) -> ForwardResult {
        let cache = FORWARD_CACHE.get_or_init(|| Mutex::new(HashMap::new()));

        // Fast path: return cached result
        {
            let cache = cache.lock().unwrap();
            if let Some(result) = cache.get(case_name) {
                return ForwardResult {
                    func_outputs: result.func_outputs.clone(),
                };
            }
        }

        let dylib_path = resolve_dylib_path();
        eprintln!("=== Loading model and running forward for case '{}' (seq_len={}) ===",
            case_name, seq_len);
        eprintln!("  dylib: {}", dylib_path);

        // Load the compiled dylib
        let safetensors = {
            let compiled_dir = PathBuf::from(resolve_dylib_path())
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| PathBuf::from("outputs/compiled/opt_125m_fresh"));
            find_safetensors(&compiled_dir)
        };
        let executor = ModelExecutor::load(&dylib_path, safetensors.as_deref())
            .expect("failed to load compiled model — run 'make build-all' first");

        let num_funcs = executor.compute_graph.functions.len();
        assert_eq!(
            num_funcs, NUM_FUNCTIONS,
            "expected {} functions, got {} — wrong compiled model?",
            NUM_FUNCTIONS, num_funcs
        );

        eprintln!("Loaded {} functions from dylib", num_funcs);

        // Run the full function graph
        let mut func_outputs: Vec<Vec<Tensor>> = vec![Vec::new(); num_funcs];
        let positions: Vec<u32> = (0..seq_len as u32).collect();
        let stream: &dyn Stream = &CpuStream;

        run_function_graph(
            &executor.compute_graph,
            &*executor.executable,
            &executor.weight_provider,
            &executor.weight_cache,
            &mut func_outputs,
            input_tokens,
            &positions,
            stream,
        )
        .expect("forward pass failed");

        // Convert Tensor outputs to owned Vec<f32> for caching
        let flat_outputs: Vec<Vec<Vec<f32>>> = func_outputs
            .iter()
            .map(|func_outs| {
                func_outs
                    .iter()
                    .map(|t| t.as_slice().to_vec())
                    .collect()
            })
            .collect();

        let result = ForwardResult {
            func_outputs: flat_outputs,
        };

        // Store in cache
        {
            let mut cache = cache.lock().unwrap();
            cache.insert(case_name.to_string(), ForwardResult {
                func_outputs: result.func_outputs.clone(),
            });
        }

        eprintln!("Forward pass complete for case '{}'", case_name);
        result
    }

    // ── Golden loading ──────────────────────────────────────────────

    /// Load golden .npz for a function within a case.
    ///
    /// Path pattern: ``{GOLDEN_BASE}/{case_name}/func_{func_symbol}_output.npz``
    fn load_golden(case_name: &str, func_symbol: &str) -> HashMap<String, Vec<f32>> {
        let npz_path = PathBuf::from(GOLDEN_BASE)
            .join(case_name)
            .join(format!("func_{}_output.npz", func_symbol));
        read_npz(&npz_path).unwrap_or_else(|e| {
            panic!(
                "failed to read golden file {:?}: {}\n\
                 (Run 'python compiler/tests/generate_golden_outputs.py' first)",
                npz_path, e
            )
        })
    }

    // ── Test: Case seq1 — all 16 functions ──────────────────────────

    #[test]
    
    fn test_case_seq1_all_functions_match_golden() {
    let _dylib_guard = crate::dylib_lock::lock();
        let case = "seq1";
        let forward = get_or_run_forward(case, 1, TOKENS_SEQ1);

        let func_symbols = func_symbol_list();
        let mut failures = 0usize;
        for (fi, symbol) in func_symbols.iter().enumerate() {
            let golden = load_golden(case, symbol);
            let actual_outputs = &forward.func_outputs[fi];

            for (oi, actual) in actual_outputs.iter().enumerate() {
                let golden_key = format!("output_{}", oi);
                let expected = golden.get(&golden_key).unwrap_or_else(|| {
                    panic!(
                        "golden key '{}' not found for func {} case {}",
                        golden_key, symbol, case
                    )
                });

                let gate = check_4_gate(actual, expected);
                if !gate.all_ok {
                    failures += 1;
                    eprintln!(
                        "  FAIL func[{}] {} output[{}] {}",
                        fi,
                        symbol,
                        oi,
                        gate_summary(&gate)
                    );
                }
            }
        }

        assert_eq!(
            failures, 0,
            "case '{}': {} function outputs FAILED 4-gate check",
            case, failures
        );
        eprintln!("✅ case '{}': all 16 functions passed 4-gate", case);
    }

    // ── Test: Case seq2 — all 16 functions ──────────────────────────

    #[test]
    
    fn test_case_seq2_all_functions_match_golden() {
    let _dylib_guard = crate::dylib_lock::lock();
        let case = "seq2";
        let forward = get_or_run_forward(case, 2, TOKENS_SEQ2);

        let func_symbols = func_symbol_list();
        let mut failures = 0usize;
        for (fi, symbol) in func_symbols.iter().enumerate() {
            let golden = load_golden(case, symbol);
            let actual_outputs = &forward.func_outputs[fi];

            for (oi, actual) in actual_outputs.iter().enumerate() {
                let golden_key = format!("output_{}", oi);
                let expected = golden.get(&golden_key).unwrap_or_else(|| {
                    panic!(
                        "golden key '{}' not found for func {} case {}",
                        golden_key, symbol, case
                    )
                });

                let gate = check_4_gate(actual, expected);
                if !gate.all_ok {
                    failures += 1;
                    eprintln!(
                        "  FAIL func[{}] {} output[{}] {}",
                        fi,
                        symbol,
                        oi,
                        gate_summary(&gate)
                    );
                }
            }
        }

        assert_eq!(
            failures, 0,
            "case '{}': {} function outputs FAILED 4-gate check",
            case, failures
        );
        eprintln!("✅ case '{}': all 16 functions passed 4-gate", case);
    }

    // ── Test: Case seq6 — all 16 functions ──────────────────────────

    #[test]
    #[ignore = "heavy full-model golden test (debug full forward) — run via make test-function-golden"]
    fn test_case_seq6_all_functions_match_golden() {
        let _dylib_guard = crate::dylib_lock::lock();
        let case = "seq6";
        let forward = get_or_run_forward(case, 6, TOKENS_SEQ6);

        let func_symbols = func_symbol_list();
        let mut failures = 0usize;
        for (fi, symbol) in func_symbols.iter().enumerate() {
            let golden = load_golden(case, symbol);
            let actual_outputs = &forward.func_outputs[fi];

            for (oi, actual) in actual_outputs.iter().enumerate() {
                let golden_key = format!("output_{}", oi);
                let expected = golden.get(&golden_key).unwrap_or_else(|| {
                    panic!(
                        "golden key '{}' not found for func {} case {}",
                        golden_key, symbol, case
                    )
                });

                let gate = check_4_gate(actual, expected);
                if !gate.all_ok {
                    failures += 1;
                    eprintln!(
                        "  FAIL func[{}] {} output[{}] {}",
                        fi,
                        symbol,
                        oi,
                        gate_summary(&gate)
                    );
                }
            }
        }

        assert_eq!(
            failures, 0,
            "case '{}': {} function outputs FAILED 4-gate check",
            case, failures
        );
        eprintln!("✅ case '{}': all 16 functions passed 4-gate", case);
    }

    // ── Test: Case seq32 — all 16 functions ─────────────────────────

    #[test]
    #[ignore = "heavy full-model golden test (debug full forward) — run via make test-function-golden"]
    fn test_case_seq32_all_functions_match_golden() {
        let _dylib_guard = crate::dylib_lock::lock();
        let case = "seq32";
        let forward = get_or_run_forward(case, 32, TOKENS_SEQ32);

        let func_symbols = func_symbol_list();
        let mut failures = 0usize;
        for (fi, symbol) in func_symbols.iter().enumerate() {
            let golden = load_golden(case, symbol);
            let actual_outputs = &forward.func_outputs[fi];

            for (oi, actual) in actual_outputs.iter().enumerate() {
                let golden_key = format!("output_{}", oi);
                let expected = golden.get(&golden_key).unwrap_or_else(|| {
                    panic!(
                        "golden key '{}' not found for func {} case {}",
                        golden_key, symbol, case
                    )
                });

                let gate = check_4_gate(actual, expected);
                if !gate.all_ok {
                    failures += 1;
                    eprintln!(
                        "  FAIL func[{}] {} output[{}] {}",
                        fi,
                        symbol,
                        oi,
                        gate_summary(&gate)
                    );
                }
            }
        }

        assert_eq!(
            failures, 0,
            "case '{}': {} function outputs FAILED 4-gate check",
            case, failures
        );
        eprintln!("✅ case '{}': all 16 functions passed 4-gate", case);
    }

    // ── Helpers ─────────────────────────────────────────────────────

    /// Return the ordered list of function symbols (main_0 through main_15).
    fn func_symbol_list() -> Vec<String> {
        (0..NUM_FUNCTIONS)
            .map(|i| format!("main_{}", i))
            .collect()
    }
}
