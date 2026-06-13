//! Diagnostic tests to isolate non-KV argmax bug and KV sret-null bug.
//!
//! These tests load real compiled dylibs and exercise individual functions
//! to determine WHERE the execution diverges from expected results.

#[cfg(test)]
mod diagnostic_tests {
    use crate::hal::cpu::executable::CpuExecutable;
    use crate::hal::cpu::kernel::KernelFn;
    use crate::hal::cpu::memref::{MemRefDesc2, MemRefDesc3};
    use crate::model::abi::{self, proto::SfaAbiHeader, SfaWeightProvider};
    use crate::model::compute_graph::{ComputeGraph, FuncDef, IOTensorDef, InputBinding};
    use crate::model::weight_loader::{WeightProvider, WeightRegistry};
    use std::collections::HashMap;

    const FRESH_DYLIB: &str =
        concat!(env!("CARGO_MANIFEST_DIR"), "/../outputs/compiled/opt_125m_fresh/libopt_125m.dylib");
    const KV_DYLIB: &str =
        concat!(env!("CARGO_MANIFEST_DIR"), "/../outputs/compiled/opt_125m_kv/libopt_125m.dylib");

    fn find_safetensors() -> String {
        let home = std::env::var("HOME").expect("HOME not set");
        let base = std::path::Path::new(&home)
            .join(".cache/huggingface/hub/models--facebook--opt-125m/snapshots");
        for entry in std::fs::read_dir(&base).expect("read snapshots dir") {
            let entry = entry.expect("read entry");
            let p = entry.path().join("model.safetensors");
            if p.exists() {
                return p.to_string_lossy().to_string();
            }
        }
        panic!("safetensors not found");
    }

    /// Load ABI proto from dylib (same pattern as abi.rs::load_from_dylib)
    fn load_abi(dylib_path: &str) -> (SfaAbiHeader, SfaWeightProvider) {
        unsafe {
            let lib = libloading::Library::new(dylib_path).expect("load dylib");
            let abi = abi::load_sfa_abi(&lib).expect("load abi");
            let weights = abi::load_sfa_weights(&lib).expect("load weights");
            // Leak the lib to keep symbols alive
            std::mem::forget(lib);
            (abi, weights)
        }
    }

    // ── Test: single-function ciface call on real dylib ─────────

    /// Call a single ciface function from the real dylib with zeroed inputs.
    /// Verifies the ciface dispatch and sret parsing work for the actual model.
    #[test]
    fn test_diagnose_single_func_ciface() {
        let dylib = FRESH_DYLIB;
        unsafe {
            let lib = libloading::Library::new(dylib).expect("load dylib");
            let abi = abi::load_sfa_abi(&lib).expect("load abi");
            let f0 = &abi.funcs[0];
            eprintln!("func_0: {} inputs={} symbol={}", f0.num_inputs, f0.input_fields.len(), f0.symbol);

            // Build a CpuExecutable (uses RawCpuExecutable internally)
            let raw = crate::hal::cpu::executable::RawCpuExecutable::load(dylib)
                .expect("load raw executable");

            // Call the last small function (func_13 or func_14 - output/lm_head) with zeroed inputs
            // These have fewer inputs so we can construct them manually
            for fi in [13usize, 14, 15].iter() {
                let f = &abi.funcs[*fi];
                eprintln!("Testing func_{}: symbol={} inputs={}",
                    fi, f.symbol, f.num_inputs);

                // Build zeroed input descriptors
                let rank = if *fi == 15 { 3usize } else { 3usize }; // lm_head takes rank-3
                let input_data = vec![0.0f32; 2 * 4 * 768];
                let input_slice = input_data.leak(); // leak to keep alive during FFI call
                let input_desc = MemRefDesc3::from_f32_slice(input_slice, [2, 4, 768]);

                // Look up the symbol
                let arity = 1 + f.num_inputs as usize;
                let kernel = match raw.lookup_typed(&f.symbol, arity) {
                    Ok(k) => k,
                    Err(e) => {
                        eprintln!("  SKIP: lookup_typed failed: {:?}", e);
                        continue;
                    }
                };

                // Allocate sret buffer for 1 output
                let out_rank = if *fi == 15 { 3usize } else { 3usize };
                let desc_size = 24 + 16 * out_rank;
                let data_size = 2 * 4 * 50272 * 4; // batch * seq * vocab * sizeof(f32)
                let sret_size = (desc_size + data_size).max(4096);
                let mut sret_buf: Vec<u8> = vec![0u8; sret_size];
                let sret_ptr = sret_buf.as_mut_ptr() as *mut std::ffi::c_void;

                // Build arg list: [sret_ptr, &input_desc, ...]
                let input_ptrs: Vec<*const std::ffi::c_void> = (0..f.num_inputs as usize)
                    .map(|_| &input_desc as *const MemRefDesc3 as *const std::ffi::c_void)
                    .collect();

                // Call via high_arity
                let mut all_args: Vec<*const std::ffi::c_void> = vec![sret_ptr as *const std::ffi::c_void];
                all_args.extend(input_ptrs);

                let raw_ptr = kernel.as_raw_ptr();
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    crate::model::ciface_high::call_high_arity(raw_ptr, &all_args);
                }));

                match result {
                    Ok(()) => {
                        // Read sret descriptor
                        let sret_slice = &sret_buf[..];
                        match crate::hal::cpu::sret::read_sret_descriptor(sret_slice, out_rank) {
                            Ok((_alloc, aligned, sizes)) => {
                                if aligned.is_null() {
                                    eprintln!("  FAIL: func_{} sret aligned ptr is NULL! (rank={})", fi, out_rank);
                                } else {
                                    let numel: usize = sizes.iter().map(|&s| s as usize).product();
                                    let data = std::slice::from_raw_parts(aligned as *const f32, numel.min(10));
                                    eprintln!("  OK: func_{} sret ok, sizes={:?}, first={:?}",
                                        fi, sizes, &data[..data.len().min(10)]);
                                }
                            }
                            Err(e) => {
                                eprintln!("  FAIL: func_{} sret read error: {:?}", fi, e);
                            }
                        }
                    }
                    Err(_) => {
                        eprintln!("  FAIL: func_{} panicked during ciface call", fi);
                    }
                }
            }
            std::mem::forget(lib);
        }
    }

    // ── Test: forward_with_kv sret null diagnosis ─────────────────

    /// Load the KV dylib, run forward_with_kv with minimal input,
    /// check if any function produces a null sret aligned pointer.
    #[test]
    fn test_diagnose_kv_sret_null() {
        let dylib = KV_DYLIB;
        let st_path = find_safetensors();

        let executor = crate::engine::executor::ModelExecutor::load(dylib, Some(&st_path))
            .expect("load KV model");

        eprintln!("KV model: {} functions", executor.compute_graph.functions.len());
        for (fi, func) in executor.compute_graph.functions.iter().enumerate() {
            let ci = func.outputs.iter().filter(|o| o.consumed_internally).count();
            if ci > 0 {
                eprintln!("  func_{}: {} consumed_internally outputs", fi, ci);
            }
        }

        // Try forward_with_positions (no KV cache) first — should not trigger sret null
        let prompt: Vec<u32> = vec![2, 133, 812, 9, 1470, 16];
        let positions: Vec<u32> = (0..prompt.len() as u32).collect();

        eprintln!("Testing forward_with_positions (no cache)...");
        match executor.forward_with_positions(&prompt, &positions) {
            Ok(out) => {
                eprintln!("  OK: shape={:?}, finite={}",
                    out.shape,
                    out.as_slice().iter().all(|v| v.is_finite()));
            }
            Err(e) => {
                eprintln!("  FAIL: {:?}", e);
            }
        }

        // Now try forward_with_kv with BlockManager
        eprintln!("Testing forward_with_kv (with cache)...");
        let num_kv_heads = 12usize;
        let head_dim = 64usize;
        let block_size = 16usize;
        let num_blocks = 64usize;

        let mut bm = crate::cache::block::BlockManager::new_with_cache(
            num_blocks, block_size, num_kv_heads, head_dim)
            .expect("block manager");

        bm.allocate("diag_req", prompt.len() + 10)
            .expect("allocate blocks");

        // Allocate per-layer cache blocks for consumed_internally functions
        let n_tokens = prompt.len() + 10;
        for func in &executor.compute_graph.functions {
            let has_ci = func.outputs.iter().any(|o| o.consumed_internally);
            if has_ci {
                let layer_rid = format!("diag_req_f{}", func.index);
                bm.allocate(&layer_rid, n_tokens)
                    .unwrap_or_else(|e| panic!("alloc {}: {e}", layer_rid));
            }
        }

        match executor.forward_with_kv(&prompt, &positions, Some(&mut bm), Some("diag_req")) {
            Ok(out) => {
                eprintln!("  OK: shape={:?}, finite={}",
                    out.shape,
                    out.as_slice().iter().all(|v| v.is_finite()));
            }
            Err(e) => {
                eprintln!("  FAIL: {:?}", e);
            }
        }
    }

    // ── Test: compare per-function argmax between KV and fresh ──

    /// Run the first function (embed_prefix) on both fresh and KV dylibs,
    /// check if outputs are identical.
    #[test]
    fn test_diagnose_func0_kv_vs_fresh() {
        let prompt: Vec<u32> = vec![2, 133, 812, 9, 1470, 16];
        let positions: Vec<u32> = (0..prompt.len() as u32).collect();

        let st_path = find_safetensors();

        let fresh = crate::engine::executor::ModelExecutor::load(FRESH_DYLIB, Some(&st_path))
            .expect("load fresh");
        let kv = crate::engine::executor::ModelExecutor::load(KV_DYLIB, Some(&st_path))
            .expect("load kv");

        let fresh_out = fresh.forward_with_positions(&prompt, &positions)
            .expect("fresh forward");
        let kv_out = kv.forward_with_positions(&prompt, &positions)
            .expect("kv forward");

        let fresh_logits = fresh_out.as_slice();
        let kv_logits = kv_out.as_slice();

        eprintln!("Fresh shape={:?} KV shape={:?}", fresh_out.shape, kv_out.shape);
        eprintln!("Fresh first 5: {:?}", &fresh_logits[..5.min(fresh_logits.len())]);
        eprintln!("KV first 5: {:?}", &kv_logits[..5.min(kv_logits.len())]);

        // Same prompt, same weights - outputs should match closely
        if fresh_logits.len() == kv_logits.len() && fresh_logits.len() > 0 {
            let mut max_diff = 0.0f32;
            for i in 0..fresh_logits.len() {
                let diff = (fresh_logits[i] - kv_logits[i]).abs();
                if diff > max_diff { max_diff = diff; }
            }
            eprintln!("Max abs diff between fresh and KV: {:.6}", max_diff);
        }

        // Get last token argmax for both
        let vocab = 50272;
        let last = fresh_out.shape[1] - 1;
        let fresh_last = &fresh_logits[last * vocab..(last + 1) * vocab];
        let kv_last = &kv_logits[last * vocab..(last + 1) * vocab];

        let fresh_argmax = fresh_last.iter().enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(mi, mv), (i, &v)| if v > mv { (i, v) } else { (mi, mv) });
        let kv_argmax = kv_last.iter().enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(mi, mv), (i, &v)| if v > mv { (i, v) } else { (mi, mv) });

        eprintln!("Fresh argmax: {} (value={})", fresh_argmax.0, fresh_argmax.1);
        eprintln!("KV argmax: {} (value={})", kv_argmax.0, kv_argmax.1);
        eprintln!("Python expected argmax: 1437");
    }

    // ── TDD: Test that catches Bug 2 (KV sret null for rank-4 outputs) ──

    /// Call a single rank-4 function from the KV dylib with zeroed inputs.
    /// FAILING test — demonstrates the sret null pointer bug.
    #[test]
    fn test_bug2_kv_rank4_sret_not_null() {
        let dylib = KV_DYLIB;
        unsafe {
            let lib = libloading::Library::new(dylib).expect("load kv dylib");
            let abi = abi::load_sfa_abi(&lib).expect("load abi");
            let f = &abi.funcs[1]; // func_1a: 3 rank-4 outputs
            eprintln!("Testing {} (inputs={} outputs={})",
                f.symbol, f.num_inputs, f.outputs.len());

            let raw = crate::hal::cpu::executable::RawCpuExecutable::load(dylib)
                .expect("load raw");

            // Build zeroed inputs matching func_1's ranks
            let mut input_descs: Vec<crate::hal::cpu::memref::MemRefDescAny> = Vec::new();
            for inp in &f.input_fields {
                let rank = inp.rank.max(1) as usize;
                let data: Vec<f32> = match rank {
                    1 => vec![0.0f32; 64],
                    2 => vec![0.0f32; 2 * 64],
                    _ => vec![0.0f32; 2 * 4 * 768],
                };
                let leaked = data.leak();
                match rank {
                    1 => input_descs.push(crate::hal::cpu::memref::MemRefDescAny::R1(
                        crate::hal::cpu::memref::MemRefDesc1::from_f32_slice(leaked, [64]))),
                    2 => input_descs.push(crate::hal::cpu::memref::MemRefDescAny::R2(
                        crate::hal::cpu::memref::MemRefDesc2::from_f32_slice(leaked, [2, 64]))),
                    _ => input_descs.push(crate::hal::cpu::memref::MemRefDescAny::R3(
                        crate::hal::cpu::memref::MemRefDesc3::from_f32_slice(leaked, [2, 4, 768]))),
                }
            }

            let arity = 1 + f.num_inputs as usize;
            let kernel = raw.lookup_typed(&f.symbol, arity).expect("lookup");

            let desc_per_out = 24 + 16 * 4; // 88 bytes per rank-4 desc
            let total_desc = desc_per_out * f.outputs.len() as usize;
            let sret_size = (total_desc + 4096).max(4096);
            let mut sret_buf: Vec<u8> = vec![0u8; sret_size];
            let sret_ptr = sret_buf.as_mut_ptr() as *mut std::ffi::c_void;

            let mut all_args: Vec<*const std::ffi::c_void> = vec![sret_ptr as *const std::ffi::c_void];
            for desc in &input_descs {
                all_args.push(desc.as_input_ptr());
            }
            crate::model::ciface_high::call_high_arity(kernel.as_raw_ptr(), &all_args);

            let mut offset = 0usize;
            for (oi, out) in f.outputs.iter().enumerate() {
                let rank = out.rank as usize;
                let desc_slice = &sret_buf[offset..offset + 24 + 16 * rank];
                match crate::hal::cpu::sret::read_sret_descriptor(desc_slice, rank) {
                    Ok((_alloc, aligned, sizes)) => {
                        eprintln!("  out[{}]: aligned={:?} sizes={:?}", oi, aligned, sizes);
                        assert!(!aligned.is_null(),
                            "BUG: func_1 output[{}] sret aligned pointer is null (rank {})",
                            oi, rank);
                    }
                    Err(e) => panic!("out[{}] sret read error: {:?}", oi, e),
                }
                offset += 24 + 16 * rank;
            }
            std::mem::forget(lib);
        }
    }

    // ── TDD: Test that catches Bug 1 (non-KV argmax error) ──

    #[test]
    fn test_bug1_forward_argmax_matches_python() {
        let st_path = find_safetensors();
        let executor = crate::engine::executor::ModelExecutor::load(FRESH_DYLIB, Some(&st_path))
            .expect("load fresh model");

        let input_ids: Vec<u32> = vec![2, 31414, 6, 232, 328];
        let positions: Vec<u32> = (0..input_ids.len() as u32).collect();

        let output = executor.forward_with_positions(&input_ids, &positions)
            .expect("forward");

        let logits = output.as_slice();
        let vocab = 50272;
        let last_start = (input_ids.len() - 1) * vocab;
        let last_logits = &logits[last_start..last_start + vocab];

        let top5: Vec<usize> = {
            let mut indexed: Vec<(usize, f32)> = last_logits.iter()
                .enumerate().map(|(i, &v)| (i, v)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            indexed.iter().take(5).map(|(i, _)| *i).collect()
        };
        eprintln!("Rust top-5: {:?}", top5);
        eprintln!("Expected:   [50118, 38, 1437, 653, 1308]");

        let expected: std::collections::HashSet<usize> =
            [50118, 38, 1437, 653, 1308].iter().cloned().collect();
        let overlap = top5.iter().filter(|i| expected.contains(i)).count();
        assert!(overlap >= 3,
            "BUG: top-5 overlap={} < 3. Rust top-5: {:?}", overlap, top5);
    }

    // ── TDD Layer 3: weight loading correctness (reproduces Bug 1) ──

    #[test]
    fn test_bug1_weight_loading_matches_hf() {
        let st_path = find_safetensors();
        // Load weight mapping from dylib's sfa_weights proto
        let lib = unsafe { libloading::Library::new(FRESH_DYLIB).expect("load dylib") };
        let weight_provider = unsafe { crate::model::abi::load_sfa_weights(&lib) }
            .expect("load weight proto");
        // Build WeightRegistry from proto
        let registry = crate::model::weight_loader::WeightRegistry {
            name_mapping: weight_provider.name_mapping.clone(),
            constants: weight_provider.constants.clone(),
        };
        unsafe { lib.close().ok(); }
        let st_path_ref = std::path::Path::new(&st_path);
        let provider = crate::model::weight_loader::WeightProvider::new(registry, Some(st_path_ref))
            .expect("load weight provider");

        let (memref, dtype) = provider.get_weight_memref("lm_head_weight")
            .expect("lm_head_weight not found");
        eprintln!("lm_head_weight: dtype={:?}", dtype);

        // Convert F16→F32 (same as load_weight_tensor does)
        let n = memref.numel();
        let data_vec: Vec<f32> = unsafe {
            crate::model::weight_loader::convert_weight_to_f32(memref.aligned, n, dtype)
        };
        let data = &data_vec[..8];
        eprintln!("First 8 (F32): {:?}", data);

        let expected: [f32; 8] = [
            0.114990234375, -0.143798828125, 0.055450439453125, 0.03125,
            0.063720703125, 0.06597900390625, 0.07763671875, 0.0946044921875,
        ];
        for (i, (&actual, &expected)) in data.iter().zip(expected.iter()).enumerate() {
            assert!((actual - expected).abs() < 0.001,
                "BUG: lm_head[{}]: actual={} expected={}", i, actual, expected);
        }

        // Weight tying: embed_tokens == lm_head
        let (memref2, dtype2) = provider.get_weight_memref("model_decoder_embed_tokens_weight")
            .expect("embed_tokens not found");
        let data2_vec = unsafe {
            crate::model::weight_loader::convert_weight_to_f32(memref2.aligned, memref2.numel(), dtype2)
        };
        let data2 = &data2_vec[..8];
        for (i, (&a, &b)) in data.iter().zip(data2.iter()).enumerate() {
            assert!((a - b).abs() < 0.001,
                "BUG: weight tying broken at [{}]: {} vs {}", i, a, b);
        }

        let (memref3, dtype3) = provider.get_weight_memref("model_decoder_final_layer_norm_weight")
            .expect("final_layer_norm not found");
        let data3_vec = unsafe {
            crate::model::weight_loader::convert_weight_to_f32(memref3.aligned, memref3.numel(), dtype3)
        };
        let data3 = &data3_vec[..5];
        let expected3: [f32; 5] = [
            1.033203125, 1.001953125, 1.015625, 1.029296875, 1.0400390625,
        ];
        for (i, (&actual, &expected)) in data3.iter().zip(expected3.iter()).enumerate() {
            assert!((actual - expected).abs() < 0.001,
                "BUG: final_layer_norm[{}]: actual={} expected={}", i, actual, expected);
        }
        eprintln!("PASS: weight loading verified against HF reference");
    }

    // ── TDD Layer 4: per-token logit comparison (reproduces Bug 1) ──

    #[test]
    fn test_bug1_logits_match_hf_reference() {
        let st_path = find_safetensors();
        let executor = crate::engine::executor::ModelExecutor::load(FRESH_DYLIB, Some(&st_path))
            .expect("load fresh model");

        let input_ids: Vec<u32> = vec![2, 133, 812, 9, 1470, 16];
        let positions: Vec<u32> = (0..input_ids.len() as u32).collect();
        let output = executor.forward_with_positions(&input_ids, &positions)
            .expect("forward");
        let logits = output.as_slice();
        let last_start = (input_ids.len() - 1) * 50272;
        let last = &logits[last_start..last_start + 50272];

        let refs: &[(usize, f32)] = &[
            (0, -9.704624), (1, -9.700383), (2, -1.434589), (5, 7.657997),
            (38, 0.561057), (85, -1.716756), (812, 3.779548),
            (1437, 2.172346), (1515, 3.739946), (50118, 3.144424),
        ];

        let mut max_err = 0.0f32;
        let mut err_count = 0usize;
        for &(tok, hf_val) in refs {
            let err = (last[tok] - hf_val).abs();
            if err > max_err { max_err = err; }
            if err > 1.0 { err_count += 1; }
        }
        eprintln!("Max logit error vs HF: {:.6}", max_err);
        for &(tok, hf_val) in refs {
            eprintln!("  token {tok}: rust={:.6} hf={:.6} diff={:.6}",
                last[tok], hf_val, (last[tok] - hf_val).abs());
        }
        assert!(err_count <= 2,
            "BUG: {} tokens have logit error > 1.0 vs HF. Max err={:.6}",
            err_count, max_err);
    }

    /// Trace per-function execution to find where divergence starts.
    #[test]
    fn test_trace_per_function_divergence() {
        let st_path = find_safetensors();
        let executor = crate::engine::executor::ModelExecutor::load(FRESH_DYLIB, Some(&st_path))
            .expect("load fresh model");

        let input_ids: Vec<u32> = vec![2, 133, 812, 9, 1470, 16];
        let positions: Vec<u32> = (0..input_ids.len() as u32).collect();
        let output = executor.forward_with_positions(&input_ids, &positions)
            .expect("forward");

        let logits = output.as_slice();
        let vocab = 50272;
        let last_start = (input_ids.len() - 1) * vocab;

        let mut indexed: Vec<(usize, f32)> = logits[last_start..last_start + vocab]
            .iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        eprintln!("Rust top-5: {:?}", &indexed[..5]);
        eprintln!("Expected (HF): top-1=5 (7.66), then 812 (3.78), 9, 1515, 2");

        for (tok, hf_val) in &[(5u32, 7.658f32), (812, 3.780), (1515, 3.740)] {
            eprintln!("  token {}: rust={:.4} hf={:.4} diff={:.4}",
                tok, logits[last_start + *tok as usize], hf_val,
                (logits[last_start + *tok as usize] - hf_val).abs());
        }
    }
}
