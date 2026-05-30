//! Per-op correctness tests for HAL primitives.
//!
//! Each test generates deterministic input data, runs the Rust HAL primitive,
//! computes a manual reference output, and asserts max absolute error within
//! tolerance. No Python or external dependencies required.
//!
//! Run with: `cargo test hal_op_correctness --features hal-rust`

#[cfg(feature = "hal-rust")]
mod tests {
    use llm_serveforge_runtime::hal::primitives::*;

    // --------------------------------------------------------------------------
    // Deterministic pseudo-random generator (Lehmer LCG)
    // --------------------------------------------------------------------------

    /// Simple LCG producing f32 values in [-1, 1].
    fn rng(seed: u32) -> impl FnMut() -> f32 {
        let mut state = seed;
        move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state as f32) / (u32::MAX as f32) * 2.0 - 1.0
        }
    }

    // --------------------------------------------------------------------------
    // Reference implementations
    // --------------------------------------------------------------------------

    /// Reference layer norm: same algorithm as fused_layer_norm.
    fn ref_layer_norm(inp: &[f32], gamma: &[f32], beta: &[f32], hidden_dim: usize, eps: f32) -> Vec<f32> {
        let n = inp.len();
        let num_chunks = n / hidden_dim;
        let mut out = vec![0.0f32; n];
        for c in 0..num_chunks {
            let start = c * hidden_dim;
            let chunk = &inp[start..start + hidden_dim];
            let mean = chunk.iter().sum::<f32>() / hidden_dim as f32;
            let var = chunk.iter().map(|&x| { let d = x - mean; d * d }).sum::<f32>() / hidden_dim as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            for i in 0..hidden_dim {
                let normalized = (chunk[i] - mean) * inv_std;
                let g = if i < gamma.len() { gamma[i] } else { 1.0 };
                let b = if i < beta.len() { beta[i] } else { 0.0 };
                out[start + i] = g * normalized + b;
            }
        }
        out
    }

    /// Reference softmax: numerically stable softmax along last dim.
    fn ref_softmax(inp: &[f32], last_dim: usize) -> Vec<f32> {
        let n = inp.len();
        let mut out = vec![0.0f32; n];
        for chunk_start in (0..n).step_by(last_dim) {
            let chunk_end = (chunk_start + last_dim).min(n);
            let chunk = &inp[chunk_start..chunk_end];
            let max_val = chunk.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for (i, &v) in chunk.iter().enumerate() {
                let e = (v - max_val).exp();
                out[chunk_start + i] = e;
                sum += e;
            }
            let inv_sum = 1.0 / sum;
            for i in 0..(chunk_end - chunk_start) {
                out[chunk_start + i] *= inv_sum;
            }
        }
        out
    }

    /// Reference 3D softmax: same as fused_softmax_3d.
    fn ref_softmax_3d(inp: &[f32], batch: usize, seq: usize, vocab: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; inp.len()];
        for b in 0..batch {
            for s in 0..seq {
                let offset = (b * seq + s) * vocab;
                let chunk = &inp[offset..offset + vocab];
                let max_val = chunk.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for (i, &v) in chunk.iter().enumerate() {
                    let e = (v - max_val).exp();
                    out[offset + i] = e;
                    sum += e;
                }
                let inv_sum = 1.0 / sum;
                for i in 0..vocab {
                    out[offset + i] *= inv_sum;
                }
            }
        }
        out
    }

    /// Reference 2D transpose.
    fn ref_transpose_2d(inp: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                out[j * rows + i] = inp[i * cols + j];
            }
        }
        out
    }

    /// Reference gather: copies weight rows by index.
    fn ref_gather_f32(weight_table: &[f32], indices: &[f32], embed_dim: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; indices.len() * embed_dim];
        for (i, &idx_f32) in indices.iter().enumerate() {
            let idx = idx_f32 as usize;
            let src_start = idx * embed_dim;
            let dst_start = i * embed_dim;
            out[dst_start..dst_start + embed_dim]
                .copy_from_slice(&weight_table[src_start..src_start + embed_dim]);
        }
        out
    }

    /// Reference 2D matmul: same algorithm as matmul_naive.
    fn ref_matmul_2d(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, trans_b: bool) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    let b_val = if trans_b {
                        b[j * k + p]
                    } else {
                        b[p * n + j]
                    };
                    sum += a[i * k + p] * b_val;
                }
                out[i * n + j] = sum;
            }
        }
        out
    }

    // --------------------------------------------------------------------------
    // Helpers
    // --------------------------------------------------------------------------

    /// Max absolute difference between two slices.
    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "slice length mismatch");
        a.iter().zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    /// Create a vector of `n` random f32 values seeded by `seed`.
    fn rand_vec(n: usize, seed: u32) -> Vec<f32> {
        let mut gen = rng(seed);
        (0..n).map(|_| gen()).collect()
    }

    // --------------------------------------------------------------------------
    // 1. matmul_blas
    // --------------------------------------------------------------------------

    #[test]
    fn test_matmul_blas_small_2x2() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut out = [0.0f32; 4];
        matmul_blas(&a, &b, &mut out, &[2, 2], &[2, 2], false).unwrap();
        let expected = [19.0, 22.0, 43.0, 50.0];
        let diff = max_abs_diff(&out, &expected);
        assert!(diff < 1e-4, "max diff = {}", diff);
    }

    #[test]
    fn test_matmul_blas_random_no_trans() {
        let m = 8; let k = 4; let n = 6;
        let a = rand_vec(m * k, 100);
        let b = rand_vec(k * n, 200);
        let mut out = vec![0.0f32; m * n];
        matmul_blas(&a, &b, &mut out, &[m as i64, k as i64], &[k as i64, n as i64], false).unwrap();
        let expected = ref_matmul_2d(&a, &b, m, k, n, false);
        let diff = max_abs_diff(&out, &expected);
        assert!(diff < 1e-4, "max diff = {}", diff);
    }

    #[test]
    fn test_matmul_blas_random_transpose_b() {
        let m = 8; let k = 4; let n = 6;
        // A: [m, k], B stored as [n, k] for transpose_b=true
        let a = rand_vec(m * k, 300);
        let b = rand_vec(n * k, 400);
        let mut out = vec![0.0f32; m * n];
        matmul_blas(&a, &b, &mut out, &[m as i64, k as i64], &[n as i64, k as i64], true).unwrap();
        // Reference: matmul_naive with transpose_b=true expects B stored as [n, k]
        let expected = ref_matmul_2d(&a, &b, m, k, n, true);
        let diff = max_abs_diff(&out, &expected);
        assert!(diff < 1e-4, "max diff = {}", diff);
    }

    #[test]
    fn test_matmul_blas_narrow_transpose_b_attention() {
        // Simulate attention: Q[4, 64] @ K^T[4, 64] -> [4, 4]
        let m = 4; let k = 64; let n = 4;
        let a = rand_vec(m * k, 500);
        let b = rand_vec(n * k, 600);
        let mut out = vec![0.0f32; m * n];
        matmul_blas(&a, &b, &mut out, &[m as i64, k as i64], &[n as i64, k as i64], true).unwrap();
        // For narrow matrices, matmul_blas falls back to matmul_naive internally.
        // Compare against our reference to verify correctness.
        let expected = ref_matmul_2d(&a, &b, m, k, n, true);
        let diff = max_abs_diff(&out, &expected);
        assert!(diff < 1e-4, "max diff = {} (narrow attention matmul)", diff);
    }

    #[test]
    fn test_matmul_blas_1xk_times_kx1() {
        let k = 16;
        let a = rand_vec(k, 700);
        let b = rand_vec(k, 800);
        let mut out = [0.0f32; 1];
        matmul_blas(&a, &b, &mut out, &[1, k as i64], &[k as i64, 1], false).unwrap();
        let expected = ref_matmul_2d(&a, &b, 1, k, 1, false);
        let diff = max_abs_diff(&out, &expected);
        assert!(diff < 1e-4, "max diff = {}", diff);
    }

    // --------------------------------------------------------------------------
    // 2. fused_layer_norm
    // --------------------------------------------------------------------------

    #[test]
    fn test_layer_norm_identity_affine() {
        let hidden = 16; let batch = 2;
        let inp = rand_vec(batch * hidden, 1000);
        let gamma = vec![1.0f32; hidden];
        let beta = vec![0.0f32; hidden];
        let mut out = vec![0.0f32; batch * hidden];
        fused_layer_norm(&inp, &mut out, &gamma, &beta, hidden, 1e-5);
        let expected = ref_layer_norm(&inp, &gamma, &beta, hidden, 1e-5);
        let diff = max_abs_diff(&out, &expected);
        assert!(diff < 1e-4, "max diff = {}", diff);
    }

    #[test]
    fn test_layer_norm_custom_affine() {
        let hidden = 16; let batch = 3;
        let inp = rand_vec(batch * hidden, 1100);
        let gamma = rand_vec(hidden, 1200);
        let beta = rand_vec(hidden, 1300);
        let mut out = vec![0.0f32; batch * hidden];
        fused_layer_norm(&inp, &mut out, &gamma, &beta, hidden, 1e-5);
        let expected = ref_layer_norm(&inp, &gamma, &beta, hidden, 1e-5);
        let diff = max_abs_diff(&out, &expected);
        assert!(diff < 1e-4, "max diff = {}", diff);
    }

    #[test]
    fn test_layer_norm_varied_eps() {
        let hidden = 8; let batch = 4;
        let inp = rand_vec(batch * hidden, 1400);
        let gamma = vec![1.0f32; hidden];
        let beta = vec![0.0f32; hidden];
        for &eps in &[1e-3, 1e-5, 1e-8] {
            let mut out = vec![0.0f32; batch * hidden];
            fused_layer_norm(&inp, &mut out, &gamma, &beta, hidden, eps);
            let expected = ref_layer_norm(&inp, &gamma, &beta, hidden, eps);
            let diff = max_abs_diff(&out, &expected);
            assert!(diff < 1e-4, "eps={}: max diff = {}", eps, diff);
        }
    }

    #[test]
    fn test_layer_norm_single_element() {
        // Layer norm on single-element chunks: should be zero (mean = value, var = 0)
        let hidden = 1;
        let inp = [3.5f32, -2.0];
        let gamma = [1.0];
        let beta = [0.0];
        let mut out = [0.0f32; 2];
        fused_layer_norm(&inp, &mut out, &gamma, &beta, hidden, 1e-5);
        // With hidden_dim=1: mean=value, var=0, normalized=0/eps.sqrt() ≈ 0
        let expected = ref_layer_norm(&inp, &gamma, &beta, hidden, 1e-5);
        let diff = max_abs_diff(&out, &expected);
        assert!(diff < 1e-4, "max diff = {}", diff);
    }

    // --------------------------------------------------------------------------
    // 3. Element-wise ops: vec_add, vec_mul, vec_relu
    // --------------------------------------------------------------------------

    #[test]
    fn test_vec_add_random() {
        let n = 256;
        let a = rand_vec(n, 1500);
        let b = rand_vec(n, 1600);
        let mut out = vec![0.0f32; n];
        vec_add(&a, &b, &mut out);
        let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
        let diff = max_abs_diff(&out, &expected);
        assert!(diff < 1e-6, "max diff = {}", diff);
    }

    #[test]
    fn test_vec_mul_random() {
        let n = 256;
        let a = rand_vec(n, 1700);
        let b = rand_vec(n, 1800);
        let mut out = vec![0.0f32; n];
        vec_mul(&a, &b, &mut out);
        let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();
        let diff = max_abs_diff(&out, &expected);
        assert!(diff < 1e-6, "max diff = {}", diff);
    }

    #[test]
    fn test_vec_relu_random() {
        let n = 256;
        let a = rand_vec(n, 1900);
        let mut out = vec![0.0f32; n];
        vec_relu(&a, &mut out);
        let expected: Vec<f32> = a.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect();
        let diff = max_abs_diff(&out, &expected);
        assert!(diff < 1e-6, "max diff = {}", diff);
    }

    #[test]
    fn test_vec_relu_zero_and_negative() {
        let a = [0.0, -0.5, -2.0, 0.0, -0.001];
        let mut out = [0.0f32; 5];
        vec_relu(&a, &mut out);
        assert_eq!(out, [0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    // --------------------------------------------------------------------------
    // 4. Softmax
    // --------------------------------------------------------------------------

    #[test]
    fn test_softmax_random_2d() {
        let batch = 3; let seq = 4;
        let last_dim = seq;
        let inp = rand_vec(batch * last_dim, 2000);
        let mut out = vec![0.0f32; batch * last_dim];
        fused_softmax(&inp, &mut out, last_dim);
        let expected = ref_softmax(&inp, last_dim);
        let diff = max_abs_diff(&out, &expected);
        assert!(diff < 1e-4, "max diff = {}", diff);
        // Each chunk must sum to 1.0
        for c in 0..batch {
            let sum: f32 = out[c * last_dim..(c + 1) * last_dim].iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "chunk {} sum = {}", c, sum);
        }
    }

    #[test]
    fn test_softmax_uniform_input() {
        let last_dim = 8;
        let inp = vec![3.0f32; last_dim * 2]; // 2 chunks of 8
        let mut out = vec![0.0f32; last_dim * 2];
        fused_softmax(&inp, &mut out, last_dim);
        for chunk in out.chunks(last_dim) {
            for &v in chunk {
                assert!((v - 1.0 / last_dim as f32).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_softmax_large_values_stable() {
        let inp = [1000.0f32, 1001.0, 1002.0, 1003.0];
        let mut out = [0.0f32; 4];
        fused_softmax(&inp, &mut out, 4);
        assert!(out.iter().all(|&v| v.is_finite()));
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_3d_random() {
        let batch = 2; let seq = 3; let vocab = 8;
        let inp = rand_vec(batch * seq * vocab, 2100);
        let mut out = vec![0.0f32; batch * seq * vocab];
        fused_softmax_3d(&inp, &mut out, batch, seq, vocab);
        let expected = ref_softmax_3d(&inp, batch, seq, vocab);
        let diff = max_abs_diff(&out, &expected);
        assert!(diff < 1e-4, "max diff = {}", diff);
    }

    // --------------------------------------------------------------------------
    // 5. Transpose
    // --------------------------------------------------------------------------

    #[test]
    fn test_transpose_2d_random() {
        let rows = 6; let cols = 10;
        let inp = rand_vec(rows * cols, 2200);
        let mut out = vec![0.0f32; rows * cols];
        transpose_2d(&inp, &mut out, rows, cols);
        let expected = ref_transpose_2d(&inp, rows, cols);
        let diff = max_abs_diff(&out, &expected);
        assert!(diff < 1e-6, "max diff = {}", diff);
    }

    #[test]
    fn test_transpose_nd_swap_last_two() {
        let shape = [2i64, 3, 4];
        let total: usize = (2 * 3 * 4) as usize;
        let inp = rand_vec(total, 2300);
        let mut out = vec![0.0f32; total];
        transpose_nd(&inp, &mut out, &shape, &[2, 4, 3], &[0, 2, 1]).unwrap();
        // Verify specific element: out[b,h,s] = inp[b,s,h]
        // out[0,0,0] = inp[0,0,0]
        assert!((out[0] - inp[0]).abs() < 1e-6);
        // out[0,0,1] = inp[0,1,0] = inp[4]
        assert!((out[1] - inp[4]).abs() < 1e-6);
        // out[0,1,0] = inp[0,0,1] = inp[1]
        assert!((out[3] - inp[1]).abs() < 1e-6);
    }

    #[test]
    fn test_transpose_2d_square() {
        let size = 5;
        let inp: Vec<f32> = (0..size * size).map(|i| i as f32).collect();
        let mut out = vec![0.0f32; size * size];
        transpose_2d(&inp, &mut out, size, size);
        let expected = ref_transpose_2d(&inp, size, size);
        assert_eq!(out, expected);
    }

    // --------------------------------------------------------------------------
    // 6. Gather
    // --------------------------------------------------------------------------

    #[test]
    fn test_gather_f32_random() {
        let vocab = 32; let embed_dim = 8; let num_indices = 5;
        let weight_table = rand_vec(vocab * embed_dim, 2400);
        let indices: Vec<f32> = (0..num_indices).map(|i| (i * 3 % vocab) as f32).collect();
        let mut out = vec![0.0f32; num_indices * embed_dim];
        gather_f32(&weight_table, &indices, &mut out, embed_dim).unwrap();
        let expected = ref_gather_f32(&weight_table, &indices, embed_dim);
        assert_eq!(out, expected);
    }

    #[test]
    fn test_gather_f32_single_index() {
        let weight_table = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [3, 2]
        let indices = [1.0f32]; // grab row 1
        let mut out = [0.0f32; 2];
        gather_f32(&weight_table, &indices, &mut out, 2).unwrap();
        assert_eq!(out, [3.0, 4.0]);
    }

    // --------------------------------------------------------------------------
    // 7. Fused RMS Norm
    // --------------------------------------------------------------------------

    #[test]
    fn test_rms_norm_basic() {
        let inp = [1.0f32, 2.0, 3.0, 4.0];
        let gamma = [1.0f32; 4];
        let mut out = [0.0f32; 4];
        fused_rms_norm(&inp, &mut out, &gamma, 4, 1e-5);
        // mean(x^2) = (1+4+9+16)/4 = 7.5, inv_rms = 1/sqrt(7.5) ≈ 0.365148
        let inv_rms = 1.0 / ((1.0 + 4.0 + 9.0 + 16.0) / 4.0_f32).sqrt();
        assert!((out[0] - 1.0 * inv_rms).abs() < 1e-5);
        assert!((out[1] - 2.0 * inv_rms).abs() < 1e-5);
        assert!((out[2] - 3.0 * inv_rms).abs() < 1e-5);
        assert!((out[3] - 4.0 * inv_rms).abs() < 1e-5);
    }

    #[test]
    fn test_rms_norm_with_gamma() {
        let hidden = 8; let batch = 2;
        let inp = rand_vec(batch * hidden, 2500);
        let gamma = rand_vec(hidden, 2600);
        let mut out = vec![0.0f32; batch * hidden];
        fused_rms_norm(&inp, &mut out, &gamma, hidden, 1e-5);
        // Manual reference
        let mut expected = vec![0.0f32; batch * hidden];
        for c in 0..batch {
            let start = c * hidden;
            let chunk = &inp[start..start + hidden];
            let mean_sq: f32 = chunk.iter().map(|&x| x * x).sum::<f32>() / hidden as f32;
            let inv_rms = 1.0 / (mean_sq + 1e-5).sqrt();
            for i in 0..hidden {
                expected[start + i] = gamma[i] * chunk[i] * inv_rms;
            }
        }
        let diff = max_abs_diff(&out, &expected);
        assert!(diff < 1e-4, "max diff = {}", diff);
    }

    // --------------------------------------------------------------------------
    // 8. Reduce ops
    // --------------------------------------------------------------------------

    #[test]
    fn test_reduce_mean_last_dim_random() {
        let outer = 5; let last_dim = 8;
        let inp = rand_vec(outer * last_dim, 2700);
        let mut out = vec![0.0f32; outer];
        reduce_mean_last_dim(&inp, &mut out, last_dim);
        let expected: Vec<f32> = inp.chunks(last_dim)
            .map(|chunk| chunk.iter().sum::<f32>() / last_dim as f32)
            .collect();
        let diff = max_abs_diff(&out, &expected);
        assert!(diff < 1e-5, "max diff = {}", diff);
    }

    #[test]
    fn test_reduce_sum_last_dim_random() {
        let outer = 5; let last_dim = 8;
        let inp = rand_vec(outer * last_dim, 2800);
        let mut out = vec![0.0f32; outer];
        reduce_sum_last_dim(&inp, &mut out, last_dim);
        let expected: Vec<f32> = inp.chunks(last_dim)
            .map(|chunk| chunk.iter().sum())
            .collect();
        let diff = max_abs_diff(&out, &expected);
        assert!(diff < 1e-5, "max diff = {}", diff);
    }

    // --------------------------------------------------------------------------
    // 9. Edge cases
    // --------------------------------------------------------------------------

    #[test]
    fn test_matmul_blas_zero_dim() {
        // M=0: no output to compute
        let a = [1.0f32; 4];
        let b = [2.0f32; 4];
        let mut out = [0.0f32; 0];
        assert!(matmul_blas(&a, &b, &mut out, &[0, 2], &[2, 2], false).is_ok());
    }

    #[test]
    fn test_softmax_zero_dim() {
        let inp = [1.0f32; 4];
        let mut out = [0.0f32; 4];
        fused_softmax(&inp, &mut out, 0);
        // With last_dim=0, no chunks processed; out remains zero
        assert_eq!(out, [0.0f32; 4]);
    }

    #[test]
    fn test_layer_norm_zero_hidden_dim() {
        let inp = [1.0f32; 4];
        let gamma = [1.0f32; 4];
        let beta = [0.0f32; 4];
        let mut out = [0.0f32; 4];
        fused_layer_norm(&inp, &mut out, &gamma, &beta, 0, 1e-5);
        // With hidden_dim=0, no processing; out unchanged
        assert_eq!(out, [0.0f32; 4]);
    }
}
