//! Fused softmax — numerically stable softmax in a single pass.
//!
//! Computes: `out[i] = exp(inp[i] - max(inp)) / sum(exp(inp - max(inp)))`
//!
//! This is a fused kernel that combines max → exp → sum → div into a single
//! pass, avoiding intermediate buffers and reducing memory bandwidth.

/// Fused softmax over the last dimension.
///
/// `inp` and `out` must have the same length. `last_dim` is the size of
/// the softmax dimension (typically vocab_size or hidden_dim).
pub fn fused_softmax(inp: &[f32], out: &mut [f32], last_dim: usize) {
    if last_dim == 0 {
        return;
    }
    let n = inp.len();
    for chunk_start in (0..n).step_by(last_dim) {
        let chunk_end = (chunk_start + last_dim).min(n);
        let chunk = &inp[chunk_start..chunk_end];
        let out_chunk = &mut out[chunk_start..chunk_end];

        // Pass 1: find max for numerical stability
        let max_val = chunk.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Pass 2: compute exp and sum
        let mut sum = 0.0f32;
        for (i, &v) in chunk.iter().enumerate() {
            let e = (v - max_val).exp();
            out_chunk[i] = e;
            sum += e;
        }

        // Pass 3: normalize
        let inv_sum = 1.0 / sum;
        for v in out_chunk.iter_mut() {
            *v *= inv_sum;
        }
    }
}

/// Fused softmax with explicit output shape (for 3D tensors).
/// Softmax is applied along the last dimension.
pub fn fused_softmax_3d(inp: &[f32], out: &mut [f32], batch: usize, seq: usize, vocab: usize) {
    let last_dim = vocab;
    for b in 0..batch {
        for s in 0..seq {
            let offset = (b * seq + s) * vocab;
            let chunk = &inp[offset..offset + vocab];
            let out_chunk = &mut out[offset..offset + vocab];

            let max_val = chunk.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for (i, &v) in chunk.iter().enumerate() {
                let e = (v - max_val).exp();
                out_chunk[i] = e;
                sum += e;
            }
            let inv_sum = 1.0 / sum;
            for v in out_chunk.iter_mut() {
                *v *= inv_sum;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_softmax_basic() {
        let inp = [1.0, 2.0, 3.0];
        let mut out = [0.0; 3];
        fused_softmax(&inp, &mut out, 3);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(out[0] < out[1]);
        assert!(out[1] < out[2]);
    }

    #[test]
    fn test_fused_softmax_numerical_stability() {
        // Large values should not overflow
        let inp = [1000.0, 1001.0, 1002.0];
        let mut out = [0.0; 3];
        fused_softmax(&inp, &mut out, 3);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(out.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_fused_softmax_multiple_chunks() {
        // Two chunks of size 3
        let inp = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = [0.0; 6];
        fused_softmax(&inp, &mut out, 3);
        // First chunk
        let sum1: f32 = out[0..3].iter().sum();
        assert!((sum1 - 1.0).abs() < 1e-6);
        // Second chunk
        let sum2: f32 = out[3..6].iter().sum();
        assert!((sum2 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_fused_softmax_3d() {
        let inp = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = [0.0; 6];
        fused_softmax_3d(&inp, &mut out, 1, 2, 3);
        let sum1: f32 = out[0..3].iter().sum();
        let sum2: f32 = out[3..6].iter().sum();
        assert!((sum1 - 1.0).abs() < 1e-6);
        assert!((sum2 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_fused_softmax_uniform() {
        // Uniform input should give uniform output
        let inp = [5.0, 5.0, 5.0, 5.0];
        let mut out = [0.0; 4];
        fused_softmax(&inp, &mut out, 4);
        for &v in &out {
            assert!((v - 0.25).abs() < 1e-6);
        }
    }
}
