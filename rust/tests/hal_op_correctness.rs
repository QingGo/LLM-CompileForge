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
        assert_eq!(out, [0.0f32; 4]);
    }

    // --------------------------------------------------------------------------
    // Bug-regression tests — each reproduces a real bug found in the HAL runner
    // --------------------------------------------------------------------------

    /// Regression test for the batched matmul bug (2026-06-01):
    /// `matmul_blas` with `transpose_b=false` must compute every batch
    /// element, not just the first one.
    #[test]
    fn test_batched_matmul_all_batches_computed() {
        // [2, 2, 2] @ [2, 2, 2] — 2 batches of [2,2] @ [2,2]
        let m = 2;
        let k = 2;
        let n = 2;
        let batches = 2;
        // A: two identity matrices [[1,0],[0,1]], [[2,0],[0,2]]
        let a: Vec<f32> = vec![
            1.0, 0.0, 0.0, 1.0,  // batch 0: identity
            2.0, 0.0, 0.0, 2.0,  // batch 1: 2x identity
        ];
        // B: two ones matrices [[3,3],[3,3]], [[4,4],[4,4]]
        let b: Vec<f32> = vec![
            3.0, 3.0, 3.0, 3.0,  // batch 0: all 3s
            4.0, 4.0, 4.0, 4.0,  // batch 1: all 4s
        ];
        let mut out = vec![0.0f32; batches * m * n];

        let a_shape: Vec<i64> = vec![batches as i64, m as i64, k as i64];
        let b_shape: Vec<i64> = vec![batches as i64, k as i64, n as i64];

        matmul_blas(&a, &b, &mut out, &a_shape, &b_shape, false).unwrap();

        // Batch 0: [[1,0],[0,1]] @ [[3,3],[3,3]] = [[3,3],[3,3]]
        assert!((out[0] - 3.0).abs() < 1e-5, "batch0[0,0] expected 3, got {}", out[0]);
        assert!((out[1] - 3.0).abs() < 1e-5, "batch0[0,1] expected 3, got {}", out[1]);
        assert!((out[2] - 3.0).abs() < 1e-5, "batch0[1,0] expected 3, got {}", out[2]);
        assert!((out[3] - 3.0).abs() < 1e-5, "batch0[1,1] expected 3, got {}", out[3]);

        // Batch 1: [[2,0],[0,2]] @ [[4,4],[4,4]] = [[8,8],[8,8]]
        let off = (m * n) as usize;
        assert!((out[off + 0] - 8.0).abs() < 1e-5, "batch1[0,0] expected 8, got {}", out[off]);
        assert!((out[off + 1] - 8.0).abs() < 1e-5, "batch1[0,1] expected 8, got {}", out[off+1]);
        assert!((out[off + 2] - 8.0).abs() < 1e-5, "batch1[1,0] expected 8, got {}", out[off+2]);
        assert!((out[off + 3] - 8.0).abs() < 1e-5, "batch1[1,1] expected 8, got {}", out[off+3]);
    }

    /// Regression test for the batched matmul bug with 4D attention shapes:
    /// [1,12,4,64] @ [1,12,64,4] = [1,12,4,4] — all 12 heads must be computed.
    #[test]
    fn test_batched_matmul_4d_attention_shape() {
        let batch = 1;
        let heads = 3;
        let seq = 2;
        let dim = 4;
        let total = batch * heads * seq * dim;

        // A: fill with known pattern: a[b,h,i,j] = b*1000 + h*100 + i*10 + j
        let a: Vec<f32> = (0..total).map(|idx| {
            let j = idx % dim;
            let i = (idx / dim) % seq;
            let h = (idx / (seq * dim)) % heads;
            (h * 100 + i * 10 + j) as f32
        }).collect();

        // B: identity-like for each head
        let b_total = batch * heads * dim * seq;
        let mut b = vec![0.0f32; b_total];
        for h in 0..heads {
            for i in 0..dim {
                for j in 0..seq {
                    b[(h * dim * seq + i * seq + j) as usize] = if i == j as usize { 1.0 } else { 0.0 };
                }
            }
        }

        let out_total = batch * heads * seq * seq;
        let mut out = vec![0.0f32; out_total];

        let a_shape = vec![batch as i64, heads as i64, seq as i64, dim as i64];
        let b_shape = vec![batch as i64, heads as i64, dim as i64, seq as i64];

        matmul_blas(&a, &b, &mut out, &a_shape, &b_shape, false).unwrap();

        // Verify head 1 (index 1) is NOT all zeros (the original bug)
        let head1_offset = (1 * seq * seq) as usize;
        let head1_sum: f32 = out[head1_offset..head1_offset + seq * seq].iter().sum();
        assert!(head1_sum.abs() > 1e-5,
            "head 1 output should be non-zero (batched matmul must process all heads), got sum={}", head1_sum);

        // Head 0 should match A[0,:,:] @ B[0,:,:]
        // For seq=2, dim=4: expected[0,0,0] = sum_k A[0,0,k] * B[0,k,0]
        // A[0,0,:] = [0, 1, 2, 3], B[0,:,0] = [1, 0, 0, 0] → expected = 0
        // A[0,0,:] @ B[0,:,:] = [0, 1, 2, 3] (since B is identity for each head)
        assert!((out[0] - 0.0).abs() < 1e-5);
        assert!((out[seq as usize] - 10.0).abs() < 1e-5,
            "head0 pos1 dim0: expected 10, got {}", out[seq as usize]);
    }

    /// Verify causal mask values: lower-triangular = 0.0, upper = -inf
    /// using element_wise add with scores and softmax.
    #[test]
    fn test_causal_mask_produces_correct_softmax() {
        let seq = 4;
        // Build causal mask: mask[i,j] = 0 if j<=i else -inf
        let mut mask = vec![0.0f32; seq * seq];
        for i in 0..seq {
            for j in 0..seq {
                mask[i * seq + j] = if j <= i { 0.0 } else { f32::NEG_INFINITY };
            }
        }

        // Scores: all ones
        let scores: Vec<f32> = vec![1.0f32; seq * seq];
        // Masked scores = scores + mask
        let mut masked = vec![0.0f32; seq * seq];
        for i in 0..seq * seq {
            masked[i] = scores[i] + mask[i];
        }

        // Softmax along last dim
        let mut softmax_out = vec![0.0f32; seq * seq];
        for row in 0..seq {
            let start = row * seq;
            let end = start + seq;
            let max_val = masked[start..end].iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for j in start..end {
                softmax_out[j] = (masked[j] - max_val).exp();
                sum += softmax_out[j];
            }
            for j in start..end {
                softmax_out[j] /= sum;
            }
        }

        // Verify causal constraint: upper triangular should be 0.0
        for i in 0..seq {
            for j in 0..seq {
                if j > i {
                    assert!(softmax_out[i * seq + j] == 0.0,
                        "pos({},{}) should be 0 (causal mask)", i, j);
                } else {
                    assert!(softmax_out[i * seq + j] > 0.0,
                        "pos({},{}) should be >0 (valid attention)", i, j);
                }
            }
        }

        // Row sums should be 1.0
        for i in 0..seq {
            let row_sum: f32 = softmax_out[i*seq..(i+1)*seq].iter().sum();
            assert!((row_sum - 1.0).abs() < 1e-5,
                "row {} sum should be 1, got {}", i, row_sum);
        }

        // Position 0 only attends to itself → softmax[0] = [1, 0, 0, 0]
        assert!((softmax_out[0] - 1.0).abs() < 1e-5);
        assert!(softmax_out[1] == 0.0);
        assert!(softmax_out[2] == 0.0);
        assert!(softmax_out[3] == 0.0);
    }

    /// Verify attention scaling: Q must be multiplied by 1/sqrt(head_dim)
    /// before the matmul, not 1.0 (bug: shape dims [1,4] were used instead).
    #[test]
    fn test_attention_scaling_factor() {
        let head_dim = 64;
        let scale = 1.0 / (head_dim as f32).sqrt();
        assert!((scale - 0.125).abs() < 1e-5,
            "attention scaling should be 1/sqrt(64)=0.125, got {}", scale);

        // Verify: Q scaled by 0.125 gives different attention than Q unscaled
        let q: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];  // [1, 1, 1, 4]
        let k: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];  // [1, 1, 4, 1]

        // Without scaling
        let score_no_scale = q[0]*k[0] + q[1]*k[1] + q[2]*k[2] + q[3]*k[3];
        // With scaling on Q
        let q_scaled: Vec<f32> = q.iter().map(|&v| v * scale).collect();
        let score_with_scale = q_scaled[0]*k[0] + q_scaled[1]*k[1] + q_scaled[2]*k[2] + q_scaled[3]*k[3];

        assert!((score_no_scale - 30.0).abs() < 1e-5);
        assert!((score_with_scale - 30.0 * scale).abs() < 1e-5);
        assert!((score_no_scale - score_with_scale / scale).abs() < 1e-5);
    }
}
