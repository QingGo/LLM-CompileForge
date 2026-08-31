//! Numeric kernels shared by the fused prototype (Phase 4) and the
//! op-plan executor (Phase 5).
//!
//! The dominant BLAS arithmetic comes from [`crate::engine::blas`]; these
//! functions implement the lowered-MLIR element-wise/reduction order that
//! is too small to delegate to a vendor library.

use crate::engine::blas;
use crate::model::tensor::Tensor;

pub(crate) const LN_EPS: f32 = 1.0e-5;
pub(crate) const NEG_MASK: f32 = -1.0e20;

/// `y = layer_norm(x) * weight + bias` in the lowered MLIR order:
/// mean -> center -> square -> variance -> rstd -> normalize -> affine.
pub(crate) fn layer_norm_into(
    x: &[f32],
    rows: usize,
    cols: usize,
    weight: &[f32],
    bias: &[f32],
    eps: f32,
    out: &mut Vec<f32>,
) -> Result<(), anyhow::Error> {
    anyhow::ensure!(
        x.len() == rows * cols && weight.len() == cols && bias.len() == cols,
        "layer_norm_into: shape mismatch (x={}, rows={}, cols={}, w={}, b={})",
        x.len(),
        rows,
        cols,
        weight.len(),
        bias.len()
    );
    out.clear();
    out.reserve(rows * cols);
    let inv_cols = 1.0f32 / cols as f32;
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let mut sum = 0.0f32;
        for &v in row {
            sum += v;
        }
        let mean = sum * inv_cols;
        let mut sq_sum = 0.0f32;
        for &v in row {
            let centered = v - mean;
            sq_sum += centered * centered;
            out.push(centered);
        }
        let rstd = (sq_sum * inv_cols + eps).sqrt();
        let start = r * cols;
        for d in 0..cols {
            let centered = out[start + d];
            out[start + d] = (centered / rstd) * weight[d] + bias[d];
        }
    }
    Ok(())
}

/// `data[r, c] += bias[c]` for `[rows, cols]` row-major data.
pub(crate) fn add_row_bias(data: &mut [f32], rows: usize, cols: usize, bias: &[f32]) {
    debug_assert_eq!(data.len(), rows * cols);
    debug_assert_eq!(bias.len(), cols);
    for r in 0..rows {
        let row = &mut data[r * cols..(r + 1) * cols];
        for (v, &b) in row.iter_mut().zip(bias.iter()) {
            *v += b;
        }
    }
}

/// Position-major `[seq, heads*dim]` -> head-major BNSD
/// `[heads, seq, dim]` (the layout the SDPA consumer expects).
pub(crate) fn pmajor_to_bnsd(src: &[f32], seq: usize, heads: usize, dim: usize) -> Vec<f32> {
    let mut dst = vec![0.0f32; src.len()];
    for h in 0..heads {
        for p in 0..seq {
            let src_off = p * heads * dim + h * dim;
            let dst_off = h * seq * dim + p * dim;
            dst[dst_off..dst_off + dim].copy_from_slice(&src[src_off..src_off + dim]);
        }
    }
    dst
}

/// Causal scaled-dot-product attention.
///
/// `q` is BNSD (head-major, rows within a head are contiguous).  `k`/`v`
/// may be either:
/// * BNLD position-major cache (`kv_bnsd_layout=false`, `row_stride`
///   = hidden_dim), accessed per-head through the head offset, or
/// * BNSD head-major (`kv_bnsd_layout=true`, `row_stride` = head_dim).
///
/// The decode path uses direct dot products (the same reduction order as
/// the dylib's linalg.generic); the prefill path uses per-head SGEMM.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_forward(
    q: &[f32],
    seq: usize,
    k: &[f32],
    v: &[f32],
    kv_len: usize,
    kv_row_stride: usize,
    kv_bnsd_layout: bool,
    heads: usize,
    dim: usize,
    is_decode: bool,
    mask: Option<&Tensor>,
    scores: &mut Vec<f32>,
    probs: &mut Vec<f32>,
    ctx_head: &mut Vec<f32>,
    ctx_flat: &mut Vec<f32>,
) -> Result<(), anyhow::Error> {
    anyhow::ensure!(
        q.len() == heads * seq * dim,
        "attention_forward: q len {} != heads*seq*dim={}",
        q.len(),
        heads * seq * dim
    );
    let expected_kv = if kv_bnsd_layout {
        heads * kv_len * dim
    } else {
        kv_len * heads * dim
    };
    anyhow::ensure!(
        k.len() == expected_kv && v.len() == expected_kv,
        "attention_forward: KV len {} != expected {}",
        k.len(),
        expected_kv
    );

    scores.clear();
    scores.resize(heads * seq * kv_len, 0.0);
    probs.clear();
    probs.resize(heads * seq * kv_len, 0.0);
    ctx_head.clear();
    ctx_head.resize(seq * heads * dim, 0.0);
    ctx_flat.clear();
    ctx_flat.resize(seq * heads * dim, 0.0);

    // Decode (query length 1) is the hot path.
    if is_decode && seq == 1 {
        for h in 0..heads {
            for ki in 0..kv_len {
                let mut dot = 0.0f32;
                if kv_bnsd_layout {
                    let k_row = h * kv_len * dim + ki * dim;
                    for d in 0..dim {
                        dot += q[h * dim + d] * k[k_row + d];
                    }
                } else {
                    let k_row = ki * kv_row_stride + h * dim;
                    for d in 0..dim {
                        dot += q[h * dim + d] * k[k_row + d];
                    }
                }
                scores[h * kv_len + ki] = dot;
            }

            let row = &scores[h * kv_len..(h + 1) * kv_len];
            let mut max = NEG_MASK;
            for &s in row {
                max = max.max(s);
            }
            let mut sum = 0.0f32;
            for ki in 0..kv_len {
                let e = (row[ki] - max).exp();
                probs[h * kv_len + ki] = e;
                sum += e;
            }
            for ki in 0..kv_len {
                probs[h * kv_len + ki] /= sum;
            }

            for d in 0..dim {
                let mut acc = 0.0f32;
                if kv_bnsd_layout {
                    let v_off = h * kv_len * dim + d;
                    for ki in 0..kv_len {
                        acc += probs[h * kv_len + ki] * v[v_off + ki * dim];
                    }
                } else {
                    let v_off = h * dim + d;
                    for ki in 0..kv_len {
                        acc += probs[h * kv_len + ki] * v[ki * kv_row_stride + v_off];
                    }
                }
                ctx_head[h * dim + d] = acc;
            }
            ctx_flat[h * dim..(h + 1) * dim]
                .copy_from_slice(&ctx_head[h * dim..(h + 1) * dim]);
        }
        return Ok(());
    }

    for h in 0..heads {
        let q_off = h * seq * dim;
        let kv_off = if kv_bnsd_layout { h * kv_len * dim } else { h * dim };
        let ldb = kv_row_stride;
        let score_off = h * seq * kv_len;
        let head_ctx_off = h * seq * dim;

        // scores = Q @ K^T (Q is already pre-scaled by 0.125).
        blas::sgemm(
            blas::CBLAS_ROW_MAJOR,
            blas::CBLAS_NO_TRANS,
            blas::CBLAS_TRANS,
            seq,
            kv_len,
            dim,
            1.0,
            &q[q_off..q_off + seq * dim],
            dim,
            &k[kv_off..],
            ldb,
            0.0,
            &mut scores[score_off..score_off + seq * kv_len],
            kv_len,
        );

        if !is_decode {
            if let Some(mask_tensor) = mask {
                // Causal mask [1, 1, seq, kv_len]; the M1 compiler contract
                // always emits the full prefix shape.
                anyhow::ensure!(
                    mask_tensor.shape == vec![1, 1, seq, kv_len],
                    "attention mask shape {:?} does not match [1, 1, {}, {}]",
                    mask_tensor.shape,
                    seq,
                    kv_len
                );
                let mask = mask_tensor.as_slice();
                let mask_min = mask.iter().copied().fold(f32::INFINITY, f32::min);
                for qi in 0..seq {
                    let row_off = score_off + qi * kv_len;
                    for ki in 0..kv_len {
                        if mask_min < -100.0 {
                            // Additive mask contract (already contains -inf).
                            scores[row_off + ki] += mask[qi * kv_len + ki];
                        } else if mask[qi * kv_len + ki] == 0.0 {
                            // Indicator mask contract emitted by the OPT
                            // split compiler: 0 = masked position.
                            scores[row_off + ki] += NEG_MASK;
                        }
                    }
                }
            } else {
                for qi in 0..seq {
                    let row_off = score_off + qi * kv_len;
                    for ki in (qi + 1)..kv_len {
                        scores[row_off + ki] += NEG_MASK;
                    }
                }
            }
        }

        // max -> exp -> sum -> divide, matching the lowered SDPA order.
        for qi in 0..seq {
            let row_off = score_off + qi * kv_len;
            let mut max = NEG_MASK;
            for ki in 0..kv_len {
                max = max.max(scores[row_off + ki]);
            }
            let mut sum = 0.0f32;
            for ki in 0..kv_len {
                let e = (scores[row_off + ki] - max).exp();
                probs[row_off + ki] = e;
                sum += e;
            }
            for ki in 0..kv_len {
                probs[row_off + ki] /= sum;
            }
        }

        // ctx_head = probs @ V.
        blas::sgemm(
            blas::CBLAS_ROW_MAJOR,
            blas::CBLAS_NO_TRANS,
            blas::CBLAS_NO_TRANS,
            seq,
            dim,
            kv_len,
            1.0,
            &probs[score_off..score_off + seq * kv_len],
            kv_len,
            &v[kv_off..],
            ldb,
            0.0,
            &mut ctx_head[head_ctx_off..head_ctx_off + seq * dim],
            dim,
        );

        // Interleave head-major [heads, seq, dim] back to position-major
        // [seq, heads*dim] for the out projection.
        for qi in 0..seq {
            let dst_off = qi * heads * dim + h * dim;
            let src_off = head_ctx_off + qi * dim;
            ctx_flat[dst_off..dst_off + dim].copy_from_slice(&ctx_head[src_off..src_off + dim]);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_norm_matches_reference_order() {
        let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let weight: Vec<f32> = vec![0.5, 1.5];
        let bias: Vec<f32> = vec![0.25, -0.25];
        let rows = 4usize;
        let cols = 2usize;
        let inv = 1.0f32 / cols as f32;

        let mut out = Vec::new();
        layer_norm_into(&x, rows, cols, &weight, &bias, LN_EPS, &mut out).unwrap();

        for r in 0..rows {
            let row = &x[r * cols..(r + 1) * cols];
            let mean = row.iter().sum::<f32>() * inv;
            let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() * inv;
            let rstd = (var + LN_EPS).sqrt();
            for d in 0..cols {
                let expected = ((row[d] - mean) / rstd) * weight[d] + bias[d];
                assert_eq!(out[r * cols + d], expected);
            }
        }
    }

    #[test]
    fn pmajor_to_bnsd_layout() {
        let src: Vec<f32> = (0..2 * 2 * 3).map(|i| i as f32).collect();
        let out = pmajor_to_bnsd(&src, 2, 2, 3);
        for h in 0..2 {
            for p in 0..2 {
                for d in 0..3 {
                    assert_eq!(out[h * 2 * 3 + p * 3 + d], src[p * 2 * 3 + h * 3 + d]);
                }
            }
        }
    }

    #[test]
    fn attention_decode_is_unmasked_dot_product() {
        let q = vec![1.0f32, 2.0];
        let k = vec![3.0f32, 0.0, 0.0, 4.0];
        let v = vec![5.0f32, 0.0, 0.0, 6.0];
        let mut scores = Vec::new();
        let mut probs = Vec::new();
        let mut ctx_head = Vec::new();
        let mut ctx_flat = Vec::new();
        attention_forward(
            &q,
            1,
            &k,
            &v,
            2,
            2,
            true,
            1,
            2,
            true,
            None,
            &mut scores,
            &mut probs,
            &mut ctx_head,
            &mut ctx_flat,
        )
        .unwrap();
        let e5 = (-5.0f32).exp();
        let w0 = e5 / (e5 + 1.0);
        let w1 = 1.0 / (e5 + 1.0);
        let expected = [w0 * 5.0, w1 * 6.0];
        assert!((ctx_flat[0] - expected[0]).abs() < 1e-5);
        assert!((ctx_flat[1] - expected[1]).abs() < 1e-5);
    }

    #[test]
    fn attention_prefill_applies_causal_mask() {
        // seq=2, kv_len=2, dim=1, one head.  The future score must have
        // zero probability after the mask is added.
        let q = vec![1.0f32, 1.0];
        let k = vec![1.0f32, 1.0];
        let v = vec![10.0f32, 20.0];
        let mut scores = Vec::new();
        let mut probs = Vec::new();
        let mut ctx_head = Vec::new();
        let mut ctx_flat = Vec::new();
        attention_forward(
            &q,
            2,
            &k,
            &v,
            2,
            1,
            true,
            1,
            1,
            false,
            None,
            &mut scores,
            &mut probs,
            &mut ctx_head,
            &mut ctx_flat,
        )
        .unwrap();
        assert_eq!(probs[0], 1.0, "row 0 must only attend to key 0");
        assert_eq!(probs[1], 0.0, "masked key 1 must have zero probability");
        assert!(
            (probs[2] - probs[3]).abs() < 1e-6,
            "row 1 attends to both keys with equal probability"
        );
        assert_eq!(ctx_flat[0], 10.0);
        assert!((ctx_flat[1] - 15.0).abs() < 1e-6);
    }
}
