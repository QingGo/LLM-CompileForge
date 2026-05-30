//! Fused Scaled Dot-Product Attention.
//!
//! Computes: softmax(Q @ K.T / sqrt(d) + mask) @ V
//! with row-major BLAS matmuls and numerically stable softmax.

use super::matmul::matmul_blas;

/// Run fused SDPA on row-major tensors.
///
/// All tensors are row-major flat `&[f32]` slices.
/// `q_shape`, `k_shape`, `v_shape` are full shapes of Q, K, V.
/// Expected layout after reshape/transpose: `[batch, heads, seq, head_dim]`.
/// `mask_shape` is the shape of the attention mask (may be broadcast).
pub fn fused_sdpa(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: &[f32],
    out: &mut [f32],
    q_shape: &[i64],
    k_shape: &[i64],
    v_shape: &[i64],
    mask_shape: &[i64],
) -> Result<(), String> {
    let rank = q_shape.len();
    if rank < 4 {
        return Err(format!("fused_sdpa: expected rank >= 4, got {:?}", q_shape));
    }
    if k_shape.len() != rank || v_shape.len() != rank {
        return Err(format!(
            "fused_sdpa: rank mismatch: q={:?}, k={:?}, v={:?}",
            q_shape, k_shape, v_shape,
        ));
    }

    let batch = q_shape[0] as usize;
    let heads = q_shape[1] as usize;
    let seq = q_shape[2] as usize;
    let head_dim = q_shape[3] as usize;

    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let total = batch * heads;

    // Allocate scratch space for scores: [batch, heads, seq, seq]
    let scores_len = total * seq * seq;
    let mut scores = vec![0.0f32; scores_len];

    // Q shape for BLAS: [total, seq, head_dim]
    let q_blas_shape = [total as i64, seq as i64, head_dim as i64];
    // K shape for BLAS: [total, seq, head_dim] → transposed to [total, head_dim, seq]
    let k_blas_shape = [total as i64, seq as i64, head_dim as i64];
    // Scores shape for BLAS: [total, seq, seq]
    let scores_blas_shape = [total as i64, seq as i64, seq as i64];

    let q_size = total * seq * head_dim;
    let k_size = total * seq * head_dim;

    // ── Step 1: scores = Q @ K.T ──────────────────────────────────────
    let chunk_q = q_size / total;
    let chunk_k = k_size / total;
    let chunk_scores = scores_len / total;

    for h in 0..total {
        let q_slice = &q[h * chunk_q..(h + 1) * chunk_q];
        let k_slice = &k[h * chunk_k..(h + 1) * chunk_k];
        let scores_slice = &mut scores[h * chunk_scores..(h + 1) * chunk_scores];

        matmul_blas(
            q_slice,
            k_slice,
            scores_slice,
            &[seq as i64, head_dim as i64],
            &[seq as i64, head_dim as i64],
            true,
        )?;
    }

    // ── Step 2: scale and apply mask ──────────────────────────────────
    // If the mask is degenerate (too few elements for a full [seq, seq]
    // causal mask), generate a lower-triangular causal mask on the fly.
    let use_builtin_causal = mask.len() < seq * seq;
    for h in 0..total {
        let scores_slice = &mut scores[h * chunk_scores..(h + 1) * chunk_scores];
        for i in 0..seq {
            for j in 0..seq {
                let idx = i * seq + j;
                let mut val = scores_slice[idx] * scale;
                if use_builtin_causal {
                    // Built-in causal: position i attends to positions j ≤ i
                    val = if i >= j { val } else { f32::NEG_INFINITY };
                } else {
                    let mask_idx = if mask_shape[mask_shape.len() - 1] as usize == seq
                        && mask_shape[mask_shape.len() - 2] as usize == seq
                    {
                        i * seq + j
                    } else if mask_shape.len() >= 3 && mask_shape[1] as usize == 1 {
                        i * seq + j
                    } else {
                        0
                    };
                    if mask_idx < mask.len() {
                        val += mask[mask_idx];
                    }
                }
                scores_slice[idx] = val;
            }
        }
    }

    // ── Step 3: softmax along last dim (stable) ───────────────────────
    for h in 0..total {
        let scores_slice = &mut scores[h * chunk_scores..(h + 1) * chunk_scores];
        for i in 0..seq {
            let row = &mut scores_slice[i * seq..(i + 1) * seq];
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for v in row.iter_mut() {
                *v = (*v - max_val).exp();
                sum += *v;
            }
            if sum > 0.0 {
                for v in row.iter_mut() {
                    *v /= sum;
                }
            }
        }
    }

    // ── Step 4: output = scores @ V ──────────────────────────────────
    let v_size = total * seq * head_dim;
    let chunk_v = v_size / total;
    let chunk_out = (total * seq * head_dim) / total;

    for h in 0..total {
        let scores_slice = &scores[h * chunk_scores..(h + 1) * chunk_scores];
        let v_slice = &v[h * chunk_v..(h + 1) * chunk_v];
        let out_slice = &mut out[h * chunk_out..(h + 1) * chunk_out];

        matmul_blas(
            scores_slice,
            v_slice,
            out_slice,
            &[seq as i64, seq as i64],
            &[seq as i64, head_dim as i64],
            false,
        )?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_sdpa_single_head_no_mask() {
        // Q, K, V: [1, 1, 2, 4] — batch=1, heads=1, seq=2, head_dim=4
        let q: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
        ];
        let k: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
        ];
        let v: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
        ];
        let mask: Vec<f32> = vec![0.0f32; 4];
        let mut out = vec![0.0f32; 8];

        fused_sdpa(
            &q, &k, &v, &mask, &mut out,
            &[1, 1, 2, 4], &[1, 1, 2, 4], &[1, 1, 2, 4], &[1, 1, 2, 2],
        )
        .unwrap();

        // scores = Q@K.T/2 = diag(0.5), softmax gives [0.6225, 0.3775] per row.
        // pos0 out = 0.6225*[1,2,3,4] + 0.3775*[5,6,7,8] ≈ [2.51, 3.51, 4.51, 5.51]
        // pos1 out = 0.3775*[1,2,3,4] + 0.6225*[5,6,7,8] ≈ [3.49, 4.49, 5.49, 6.49]
        assert!((out[0] - 2.51).abs() < 0.02, "out[0]={}", out[0]);
        assert!((out[4] - 3.49).abs() < 0.02, "out[4]={}", out[4]);
    }

    #[test]
    fn test_fused_sdpa_causal_mask() {
        let q: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
        ];
        let k: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
        ];
        let v: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
        ];
        // Causal mask: future positions get -inf (-1e10).
        let mask: Vec<f32> = vec![
            0.0, -1e10,
            0.0, 0.0,
        ];
        let mut out = vec![0.0f32; 8];

        fused_sdpa(
            &q, &k, &v, &mask, &mut out,
            &[1, 1, 2, 4], &[1, 1, 2, 4], &[1, 1, 2, 4], &[1, 1, 2, 2],
        )
        .unwrap();

        // Pos 0: scores=[0.5, -inf] → softmax=[1.0, 0.0] → V[0]=[1,2,3,4]
        assert!((out[0] - 1.0).abs() < 0.01, "out[0]={}", out[0]);
        // Pos 1: scores=[0.5, 0] → softmax=[0.6225, 0.3775] → ≈[3.49, 4.49, 5.49, 6.49]
        assert!((out[4] - 3.49).abs() < 0.02, "out[4]={}", out[4]);
    }
}
