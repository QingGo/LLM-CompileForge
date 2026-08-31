//! Dylib-level seam tests: `main_1b` (SDPA + FFN) must be shape-invariant.
//!
//! Regression (2026-08): the KV-cache decode step (Q with q_len=1 over
//! cached K/V with k_len=7) diverged from a full recompute of the same
//! 7 tokens even though Q, K, V, mask, and residual inputs were
//! verified identical at the runner level. This test calls the compiled
//! `_mlir_ciface_main_1b` directly with synthetic inputs to isolate the
//! dylib from the KV-cache machinery.

#[cfg(test)]
mod sdpa_decode_shape_tests {
    use crate::engine::executor::ModelExecutor;
    use crate::hal::cpu::CpuStream;
    use crate::model::compute_graph::InputBinding;
    use crate::model::sfa_tensor::SFATensor;
    use crate::model::tensor::{Dtype, Tensor};

    const KV_DYLIB: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../outputs/compiled/opt_125m_kv/libopt_125m_kv.dylib"
    );

    fn find_safetensors() -> String {
        let home = std::env::var("HOME").expect("HOME not set");
        let snapshots = std::path::Path::new(&home)
            .join(".cache/huggingface/hub/models--facebook--opt-125m/snapshots");
        for entry in std::fs::read_dir(&snapshots).unwrap().flatten() {
            let p = entry.path().join("model.safetensors");
            if p.exists() {
                return p.to_string_lossy().to_string();
            }
        }
        panic!("no safetensors found in HF cache");
    }

    /// Load a weight tensor with the same f16→f32 semantics as the runner.
    fn load_weight(exec: &ModelExecutor, key: &str) -> Tensor {
        let (desc, dtype) = exec
            .weight_provider
            .get_weight_memref(key)
            .unwrap_or_else(|| panic!("weight {key} not found"));
        let n = desc.numel();
        let shape: Vec<usize> = desc.sizes.iter().map(|&d| d as usize).collect();
        // SAFETY: desc.aligned points at safetensors mmap data valid for n
        // elements; convert_weight_to_f32 reads exactly that many.
        let data: Vec<f32> = unsafe {
            crate::model::weight_loader::convert_weight_to_f32(desc.aligned, n, dtype)
        };
        Tensor::new_owned(shape, data, Dtype::F32)
    }

    /// main_0 re-exports layer-0 weight tensors at fixed output indices;
    /// map them to compiled weight names for the runner-agnostic SSA case.
    fn weight_name_for_main0_output(oi: usize) -> Option<&'static str> {
        Some(match oi {
            18 => "model_decoder_layers_0_fc1_bias",
            19 => "model_decoder_layers_0_fc1_weight",
            20 => "model_decoder_layers_0_fc2_bias",
            21 => "model_decoder_layers_0_fc2_weight",
            22 => "model_decoder_layers_0_final_layer_norm_bias",
            23 => "model_decoder_layers_0_final_layer_norm_weight",
            28 => "model_decoder_layers_0_self_attn_out_proj_bias",
            29 => "model_decoder_layers_0_self_attn_out_proj_weight",
            _ => return None,
        })
    }

    /// Run `_mlir_ciface_main_1b` with synthetic Q/K/V/mask/hidden.
    /// Returns the [1, q_len, 768] output.
    fn run_main_1b(
        exec: &ModelExecutor,
        q_len: usize,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        mask: &[f32],
        mask_shape: &[usize],
        hidden: &[f32],
        seq_scalar: f32,
    ) -> Vec<f32> {
        let func = &exec.compute_graph.functions[2];
        assert_eq!(func.symbol, "_mlir_ciface_main_1b");

        let nh = 12usize;
        let hd = 64usize;
        let k_len = k.len() / (nh * hd);

        let mut kept: Vec<SFATensor> = Vec::new();
        let mut inputs: Vec<crate::hal::sfa::SfaMemRef> = Vec::new();

        for (bi, (binding, _io_def)) in func.inputs.iter().enumerate() {
            let tensor: Tensor = match binding {
                InputBinding::Weight(key) => load_weight(exec, key),
                InputBinding::GlobalInput => unreachable!("main_1b has no global inputs"),
                InputBinding::Ssa { producer_func, output_idx } => match (*producer_func, *output_idx) {
                    (0, 14) => Tensor::new_owned(vec![1], vec![seq_scalar], Dtype::F32),
                    (0, 210) => Tensor::new_owned(vec![1], vec![1.0f32], Dtype::F32),
                    (1, 0) => Tensor::new_owned(vec![1, nh, q_len, hd], q.to_vec(), Dtype::F32),
                    (1, 1) => Tensor::new_owned(vec![1, nh, k_len, hd], k.to_vec(), Dtype::F32),
                    (1, 2) => Tensor::new_owned(vec![1, nh, k_len, hd], v.to_vec(), Dtype::F32),
                    (0, 13) => Tensor::new_owned(mask_shape.to_vec(), mask.to_vec(), Dtype::F32),
                    (0, 12) => Tensor::new_owned(vec![1, q_len, 768], hidden.to_vec(), Dtype::F32),
                    (0, oi) => {
                        let name = weight_name_for_main0_output(oi)
                            .unwrap_or_else(|| panic!("unexpected ssa (0,{oi}) at input {bi}"));
                        load_weight(exec, name)
                    }
                    other => panic!("unexpected ssa binding {other:?} at input {bi}"),
                },
            };
            kept.push(SFATensor::from_vec_f32(
                tensor.as_slice().to_vec(),
                tensor.shape.clone(),
            ));
        }

        for t in &kept {
            inputs.push(t.as_sfa_memref());
        }
        let out = SFATensor::from_vec_f32(vec![0.0f32; q_len * 768], vec![1, q_len, 768]);
        let mut output_sfa = vec![out.as_sfa_memref()];
        exec.executable
            .execute(&func.symbol, &CpuStream, &inputs, &mut output_sfa)
            .expect("execute main_1b");
        let sfa = out.as_sfa_memref();
        // SAFETY: the dylib wrote q_len*768 f32 elements into `out`'s
        // buffer; execute() copies them back per the sret descriptors.
        unsafe { std::slice::from_raw_parts(sfa.data_ptr() as *const f32, q_len * 768).to_vec() }
    }

    #[test]
    fn main_1b_decode_shape_matches_recompute_last_row() {
        let _dylib_guard = crate::dylib_lock::lock();
        let exec = ModelExecutor::load(KV_DYLIB, Some(&find_safetensors()))
            .expect("load kv dylib");

        let nh = 12usize;
        let hd = 64usize;
        let k_len = 7usize;

        // Deterministic synthetic data.
        let mut rng_state: u32 = 0x12345678;
        let mut rnd = || {
            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((rng_state >> 8) & 0xFFFF) as f32 / 65535.0 - 0.5
        };

        let k: Vec<f32> = (0..nh * k_len * hd).map(|_| rnd()).collect();
        let v: Vec<f32> = (0..nh * k_len * hd).map(|_| rnd()).collect();
        let hidden: Vec<f32> = (0..7 * 768).map(|_| rnd()).collect();

        // ── Case A: decode shape — q_len=1, k_len=7, mask [1,1,1,1] ──
        // (The compiled model builds the mask from the CURRENT seq dims,
        // so during decode it is a 1×1 scalar that broadcasts to the
        // scores; mirror that exact shape here.)
        let q_a: Vec<f32> = (0..nh * hd).map(|_| rnd()).collect();
        let mask_a = vec![1.0f32; 1];
        let hidden_a = hidden[6 * 768..7 * 768].to_vec();
        let out_a = run_main_1b(
            &exec, 1, &q_a, &k, &v, &mask_a, &[1, 1, 1, 1], &hidden_a, 1.0,
        );

        // ── Case B: recompute shape — q_len=7, row 6 == case A ──
        let mut q_b: Vec<f32> = vec![0.0; nh * 7 * hd];
        for h in 0..nh {
            let src = h * hd;
            let dst = h * 7 * hd + 6 * hd;
            q_b[dst..dst + hd].copy_from_slice(&q_a[src..src + hd]);
        }
        // Rows 0..5 arbitrary (causally masked from row 6's perspective).
        for h in 0..nh {
            for p in 0..6 {
                let base = h * 7 * hd + p * hd;
                for d in 0..hd {
                    q_b[base + d] = rnd();
                }
            }
        }
        // Causal mask: row 6 all-ones.
        let mut mask_b = vec![0.0f32; 1 * 7 * 7];
        for i in 0..7 {
            for j in 0..7 {
                if j <= i {
                    mask_b[i * 7 + j] = 1.0;
                }
            }
        }
        let out_b = run_main_1b(
            &exec, 7, &q_b, &k, &v, &mask_b, &[1, 1, 7, 7], &hidden, 7.0,
        );

        let row6 = &out_b[6 * 768..7 * 768];
        let max_diff = out_a
            .iter()
            .zip(row6.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        eprintln!(
            "main_1b decode-vs-recompute: max_diff={} first={:?} vs {:?}",
            max_diff,
            &out_a[..4],
            &row6[..4]
        );
        assert!(
            max_diff < 1e-3,
            "main_1b is NOT shape-invariant: decode (q=1,k=7) output diverges \
             from recompute (q=7,k=7) last row (max_diff={})",
            max_diff
        );
    }
}
