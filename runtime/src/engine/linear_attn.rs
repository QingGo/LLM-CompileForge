//! Gated DeltaNet linear-attention primitive.
//!
//! This module implements the numerical core of Qwen3.5 / Qwen3-Next
//! `GatedDeltaNet` as a pure f32 recurrence. It is intentionally separate
//! from the func-level dylib path: a correct small kernel is the first step
//! toward `sf.linear_attn` and the recurrent-state cache (E11 / P1).
//!
//! Layouts (all row-major):
//! - q/k: `[batch, seq, heads, key_dim]`
//! - v:   `[batch, seq, heads, value_dim]`
//! - g/beta: `[batch, seq, heads]`
//! - state: `[batch, heads, key_dim, value_dim]`
//! - output: `[batch, seq, heads, value_dim]`

use anyhow::{anyhow, ensure};

use crate::engine::blas;

pub(crate) const L2_EPS: f32 = 1.0e-6;

fn l2norm_row(row: &[f32]) -> Vec<f32> {
    let norm = row.iter().map(|&x| x * x).sum::<f32>().sqrt() + L2_EPS;
    row.iter().map(|&x| x / norm).collect()
}

/// Checked Gated DeltaNet recurrence.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gated_delta_rule_checked(
    batch: usize,
    seq: usize,
    heads: usize,
    key_dim: usize,
    value_dim: usize,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    initial_state: Option<&[f32]>,
    output_final_state: bool,
    use_qk_l2norm: bool,
) -> Result<(Vec<f32>, Option<Vec<f32>>), anyhow::Error> {
    let q_len = batch
        .checked_mul(seq)
        .and_then(|n| n.checked_mul(heads))
        .and_then(|n| n.checked_mul(key_dim))
        .ok_or_else(|| anyhow!("q shape overflow"))?;
    let k_len = q_len;
    let v_len = batch
        .checked_mul(seq)
        .and_then(|n| n.checked_mul(heads))
        .and_then(|n| n.checked_mul(value_dim))
        .ok_or_else(|| anyhow!("v shape overflow"))?;
    let g_len = batch
        .checked_mul(seq)
        .and_then(|n| n.checked_mul(heads))
        .ok_or_else(|| anyhow!("g shape overflow"))?;
    let state_len = batch
        .checked_mul(heads)
        .and_then(|n| n.checked_mul(key_dim))
        .and_then(|n| n.checked_mul(value_dim))
        .ok_or_else(|| anyhow!("state shape overflow"))?;

    ensure!(q.len() == q_len, "q length {} != {}", q.len(), q_len);
    ensure!(k.len() == k_len, "k length {} != {}", k.len(), k_len);
    ensure!(v.len() == v_len, "v length {} != {}", v.len(), v_len);
    ensure!(g.len() == g_len, "g length {} != {}", g.len(), g_len);
    ensure!(beta.len() == g_len, "beta length {} != {}", beta.len(), g_len);
    if let Some(st) = initial_state {
        ensure!(st.len() == state_len, "initial_state length {} != {}", st.len(), state_len);
    }

    let scale = 1.0 / (key_dim as f32).sqrt();
    let mut state = vec![0.0f32; state_len];
    if let Some(st) = initial_state {
        state.copy_from_slice(st);
    }

    let mut out = vec![0.0f32; batch * seq * heads * value_dim];
    let mut q_scaled = vec![0.0f32; key_dim];
    let mut k_norm = vec![0.0f32; key_dim];

    for b in 0..batch {
        for h in 0..heads {
            let state_base = (b * heads + h) * key_dim * value_dim;
            for s in 0..seq {
                let q_base = ((b * seq + s) * heads + h) * key_dim;
                let k_base = ((b * seq + s) * heads + h) * key_dim;
                let v_base = ((b * seq + s) * heads + h) * value_dim;
                let gb = (b * seq + s) * heads + h;

                if use_qk_l2norm {
                    let qn = l2norm_row(&q[q_base..q_base + key_dim]);
                    let kn = l2norm_row(&k[k_base..k_base + key_dim]);
                    for d in 0..key_dim {
                        q_scaled[d] = qn[d] * scale;
                        k_norm[d] = kn[d];
                    }
                } else {
                    for d in 0..key_dim {
                        q_scaled[d] = q[q_base + d] * scale;
                        k_norm[d] = k[k_base + d];
                    }
                }

                let g_t = g[gb].exp();
                let beta_t = beta[gb];

                for idx in state_base..state_base + key_dim * value_dim {
                    state[idx] *= g_t;
                }

                let mut kv_mem = vec![0.0f32; value_dim];
                for vd in 0..value_dim {
                    let mut acc = 0.0f32;
                    for kd in 0..key_dim {
                        acc += state[state_base + kd * value_dim + vd] * k_norm[kd];
                    }
                    kv_mem[vd] = acc;
                }

                let mut delta = vec![0.0f32; value_dim];
                for vd in 0..value_dim {
                    delta[vd] = (v[v_base + vd] - kv_mem[vd]) * beta_t;
                }

                for kd in 0..key_dim {
                    let ko = state_base + kd * value_dim;
                    let kv = k_norm[kd];
                    for vd in 0..value_dim {
                        state[ko + vd] += kv * delta[vd];
                    }
                }

                let out_base = ((b * seq + s) * heads + h) * value_dim;
                for vd in 0..value_dim {
                    let mut acc = 0.0f32;
                    for kd in 0..key_dim {
                        acc += state[state_base + kd * value_dim + vd] * q_scaled[kd];
                    }
                    out[out_base + vd] = acc;
                }
            }
        }
    }

    let final_state = if output_final_state { Some(state) } else { None };
    Ok((out, final_state))
}


/// Causal depthwise 1D convolution used by GatedDeltaNet short-conv.
///
/// Inputs:
/// - `x`: `[batch, channels, seq]`
/// - `state`: optional `[batch, channels, kernel]` (the last `kernel`
///   pre-conv inputs before `seq`)
/// - `weight`: `[channels, kernel]`
/// - `bias`: optional `[channels]`
/// - `new_state`: output state `[batch, channels, kernel]`
///
/// Returns the causal conv output (before activation, unless
/// `activation_silu` is true).
#[allow(clippy::too_many_arguments)]
pub(crate) fn causal_conv1d(
    batch: usize,
    channels: usize,
    seq: usize,
    kernel: usize,
    x: &[f32],
    state: Option<&[f32]>,
    weight: &[f32],
    bias: Option<&[f32]>,
    new_state: &mut [f32],
    activation_silu: bool,
) -> Result<Vec<f32>, anyhow::Error> {
    ensure!(x.len() == batch * channels * seq, "conv x shape mismatch");
    ensure!(weight.len() == channels * kernel, "conv weight shape mismatch");
    ensure!(new_state.len() == batch * channels * kernel, "conv new_state shape mismatch");
    if let Some(s) = state {
        ensure!(s.len() == batch * channels * kernel, "conv state shape mismatch");
    }
    if let Some(bs) = bias {
        ensure!(bs.len() == channels, "conv bias shape mismatch");
    }

    let mut out = vec![0.0f32; batch * channels * seq];

    for b in 0..batch {
        for c in 0..channels {
            let x_base = (b * channels + c) * seq;
            let w_base = c * kernel;
            let st_base = (b * channels + c) * kernel;
            let out_base = (b * channels + c) * seq;

            for t in 0..seq {
                let mut acc = 0.0f32;
                if let Some(st) = state {
                    for j in 0..kernel {
                        // effective input index for this window: t+1+j
                        let idx = t + 1 + j;
                        let val = if idx < kernel {
                            st[st_base + idx]
                        } else {
                            let xi = idx - kernel;
                            if xi < seq { x[x_base + xi] } else { 0.0 }
                        };
                        acc += weight[w_base + j] * val;
                    }
                } else {
                    // PyTorch's `nn.Conv1d(padding=k-1)` followed by keeping
                    // the first `seq` outputs is equivalent to this reversed
                    // causal-window indexing:
                    //   out[t] = sum_j w[j] * x[t + j - (k-1)]
                    // where x indices outside [0, seq) are zero.
                    for j in 0..kernel {
                        let offset = (j + 1) as isize - kernel as isize;
                        let idx = t as isize + offset;
                        if idx >= 0 && (idx as usize) < seq {
                            acc += weight[w_base + j] * x[x_base + idx as usize];
                        }
                    }
                }
                out[out_base + t] = acc;
            }

            if let Some(bs) = bias {
                let bc = bs[c];
                for t in 0..seq {
                    out[out_base + t] += bc;
                }
            }
            if activation_silu {
                for t in 0..seq {
                    let o = out[out_base + t];
                    out[out_base + t] = o / (1.0 + (-o).exp());
                }
            }

            // State update = last `kernel` effective inputs.
            if let Some(_st) = state {
                let in_len = kernel + seq;
                for j in 0..kernel {
                    let idx = in_len - kernel + j;
                    new_state[st_base + j] = if idx < kernel {
                        state.unwrap()[st_base + idx]
                    } else {
                        x[x_base + idx - kernel]
                    };
                }
            } else if seq >= kernel {
                for j in 0..kernel {
                    new_state[st_base + j] = x[x_base + seq - kernel + j];
                }
            } else {
                for j in 0..seq {
                    new_state[st_base + kernel - seq + j] = x[x_base + j];
                }
                for j in 0..kernel - seq {
                    new_state[st_base + j] = 0.0;
                }
            }
        }
    }

    Ok(out)
}

/// Qwen3.5 GatedDeltaNet attention core (without decoder residual/MLP).
///
/// This is the fused path that a future `sf.linear_attn` op-plan node will
/// call.  It performs:
///
/// 1. qkv/z/b/a projections
/// 2. causal depthwise short-conv + SiLU
/// 3. Gated DeltaNet recurrence (with q/k l2norm and final state update)
/// 4. gated RMSNorm
/// 5. output projection
///
/// All weights are expected as f32 row-major `[out, in]`; states are
/// updated in place.
#[allow(clippy::too_many_arguments)]
pub(crate) fn linear_attn_core(
    batch: usize,
    seq: usize,
    hidden: usize,
    num_k_heads: usize,
    num_v_heads: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel: usize,
    x: &[f32],
    qkv_w: &[f32],
    z_w: &[f32],
    b_w: &[f32],
    a_w: &[f32],
    conv_w: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    norm_w: &[f32],
    out_w: &[f32],
    recurrent_state: &mut [f32],
    conv_state: &mut [f32],
) -> Result<Vec<f32>, anyhow::Error> {
    let m = batch
        .checked_mul(seq)
        .ok_or_else(|| anyhow!("batch*seq overflow"))?;
    let qkv_channels = num_k_heads * key_dim * 2 + num_v_heads * value_dim;
    let key_total = num_k_heads * key_dim;
    let value_total = num_v_heads * value_dim;
    let state_len = num_v_heads * key_dim * value_dim;

    ensure!(x.len() == m * hidden, "linear_attn x shape mismatch");
    ensure!(qkv_w.len() == qkv_channels * hidden, "qkv weight shape mismatch");
    ensure!(z_w.len() == value_total * hidden, "z weight shape mismatch");
    ensure!(b_w.len() == num_v_heads * hidden, "b weight shape mismatch");
    ensure!(a_w.len() == num_v_heads * hidden, "a weight shape mismatch");
    ensure!(conv_w.len() == qkv_channels * conv_kernel, "conv weight shape mismatch");
    ensure!(a_log.len() == num_v_heads && dt_bias.len() == num_v_heads, "a_log/dt_bias shape mismatch");
    ensure!(norm_w.len() == value_dim, "norm weight shape mismatch");
    ensure!(out_w.len() == hidden * value_total, "out weight shape mismatch");
    ensure!(recurrent_state.len() == state_len, "recurrent state shape mismatch");
    ensure!(conv_state.len() == qkv_channels * conv_kernel, "conv state shape mismatch");

    let mut qkv = vec![0.0f32; m * qkv_channels];
    let mut z = vec![0.0f32; m * value_total];
    let mut b = vec![0.0f32; m * num_v_heads];
    let mut a = vec![0.0f32; m * num_v_heads];
    blas::sgemm_transb(x, m, qkv_channels, hidden, qkv_w, &mut qkv);
    blas::sgemm_transb(x, m, value_total, hidden, z_w, &mut z);
    blas::sgemm_transb(x, m, num_v_heads, hidden, b_w, &mut b);
    blas::sgemm_transb(x, m, num_v_heads, hidden, a_w, &mut a);

    // Convert [batch*seq, channels] -> [batch, channels, seq] for conv.
    let mut conv_x = vec![0.0f32; batch * qkv_channels * seq];
    for bb in 0..batch {
        for s in 0..seq {
            for c in 0..qkv_channels {
                conv_x[(bb * qkv_channels + c) * seq + s] = qkv[(bb * seq + s) * qkv_channels + c];
            }
        }
    }

    let old_conv_state = conv_state.to_vec();
    let mut new_conv_state = vec![0.0f32; qkv_channels * conv_kernel];
    let conv_out = causal_conv1d(
        batch,
        qkv_channels,
        seq,
        conv_kernel,
        &conv_x,
        Some(&old_conv_state),
        conv_w,
        None,
        &mut new_conv_state,
        true,
    )?;
    conv_state.copy_from_slice(&new_conv_state);

    // Convert conv output back to [batch*seq, channels].
    let mut conv_flat = vec![0.0f32; m * qkv_channels];
    for bb in 0..batch {
        for s in 0..seq {
            for c in 0..qkv_channels {
                conv_flat[(bb * seq + s) * qkv_channels + c] =
                    conv_out[(bb * qkv_channels + c) * seq + s];
            }
        }
    }

    let mut q = vec![0.0f32; m * key_total];
    let mut k = vec![0.0f32; m * key_total];
    let mut v = vec![0.0f32; m * value_total];
    for r in 0..m {
        let base = r * qkv_channels;
        q[r * key_total..(r + 1) * key_total]
            .copy_from_slice(&conv_flat[base..base + key_total]);
        k[r * key_total..(r + 1) * key_total]
            .copy_from_slice(&conv_flat[base + key_total..base + 2 * key_total]);
        v[r * value_total..(r + 1) * value_total]
            .copy_from_slice(&conv_flat[base + 2 * key_total..base + 2 * key_total + value_total]);
    }

    // Repeat Q/K to V heads when GQA has more value heads than key heads.
    let repeat = if num_v_heads % num_k_heads == 0 {
        num_v_heads / num_k_heads
    } else {
        1
    };
    let q_exp_len = m * num_v_heads * key_dim;
    let mut q_exp = vec![0.0f32; q_exp_len];
    let mut k_exp = vec![0.0f32; q_exp_len];
    if repeat > 1 {
        for r in 0..m {
            for kh in 0..num_k_heads {
                for vh in 0..repeat {
                    let dst_h = kh * repeat + vh;
                    let src = r * key_total + kh * key_dim;
                    let dst = r * num_v_heads * key_dim + dst_h * key_dim;
                    q_exp[dst..dst + key_dim].copy_from_slice(&q[src..src + key_dim]);
                    k_exp[dst..dst + key_dim].copy_from_slice(&k[src..src + key_dim]);
                }
            }
        }
    } else {
        q_exp.copy_from_slice(&q);
        k_exp.copy_from_slice(&k);
    }

    let mut g = vec![0.0f32; m * num_v_heads];
    let mut beta = vec![0.0f32; m * num_v_heads];
    for r in 0..m {
        for h in 0..num_v_heads {
            let av = a[r * num_v_heads + h] + dt_bias[h];
            // softplus: stable log1p(exp(x))
            let sp = if av > 20.0 { av } else { (1.0 + av.exp()).ln() };
            g[r * num_v_heads + h] = -(a_log[h].exp()) * sp;
            let bv = b[r * num_v_heads + h];
            beta[r * num_v_heads + h] = 1.0 / (1.0 + (-bv).exp());
        }
    }

    let (core, final_state) = gated_delta_rule_checked(
        batch,
        seq,
        num_v_heads,
        key_dim,
        value_dim,
        &q_exp,
        &k_exp,
        &v,
        &g,
        &beta,
        Some(recurrent_state),
        true,
        true,
    )?;
    recurrent_state.copy_from_slice(final_state.as_deref().unwrap_or(&[]));

    // Gated RMSNorm: rms(normalized) * weight * silu(z)
    let eps = 1.0e-6f32;
    let mut normalized = vec![0.0f32; m * value_total];
    for r in 0..m {
        for h in 0..num_v_heads {
            let base = r * value_total + h * value_dim;
            let mut sq = 0.0f32;
            for d in 0..value_dim {
                let cv = core[base + d];
                sq += cv * cv;
            }
            let rstd = 1.0 / (sq / value_dim as f32 + eps).sqrt();
            for d in 0..value_dim {
                let gz = z[base + d];
                let silu = gz / (1.0 + (-gz).exp());
                normalized[base + d] = core[base + d] * rstd * norm_w[d] * silu;
            }
        }
    }

    let mut output = vec![0.0f32; m * hidden];
    blas::sgemm_transb(&normalized, m, hidden, value_total, out_w, &mut output);
    Ok(output)
}



#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() <= tol,
                "index {}: {} != {} (delta {})",
                i,
                x,
                y,
                (x - y).abs()
            );
        }
    }

    #[test]
    fn gated_delta_rule_matches_python_golden() {
        // Generated from transformers.torch_recurrent_gated_delta_rule
        // (B=1, S=3, H=2, K=3, V=4, seed 0).
        let q = [
            -1.1258398, -1.1523602, 0.56665063, 0.79350835, 0.59883946, -1.5550951,
            -0.34136038, 1.8530061, 0.4680964, -0.15771244, -0.17339675, 0.18347794,
            1.3893661, 1.5863342, 0.94629836, -0.84367675, 0.9318266, 1.2590092,
        ];
        let k = [
            -0.49267697, 0.24841475, -0.23033547, -0.3917544, 0.54329473, -0.39515754,
            0.20552567, -0.45032975, -0.5730771, -0.5553584, -1.5311843, -1.234135,
            1.8197253, -0.5515287, -1.325326, 0.18855357, -0.06907269, -0.49492535,
        ];
        let v = [
            -1.478174, 2.5672328, -0.4731198, 0.33555075, 1.5091219, 2.0819554,
            1.7067117, 2.3803675, 1.941462, 0.79149806, -0.020251827, -0.43716955,
            1.645867, -1.3601689, 0.34456542, 0.5198677, -0.3656188, -1.3024404,
            0.09940346, 0.44182202, 0.2469264, 0.076887004, 0.3380058, 0.45440176,
        ];
        let g = [0.17528333, -0.9315211, -1.5054897, -0.66098255, 1.3232017, 0.037114304];
        let beta = [0.42925063, 0.46668902, 0.86908704, 0.9573461, 0.38736644, 0.41678432];
        let expected_out = [
            -0.049470507, 0.08591834, -0.01583404, 0.011229978, 0.1781866, 0.2458228,
            0.20151663, 0.2810572, -0.8035151, -0.27104136, -0.00033181952, 0.18098007,
            0.09758273, -0.28321803, -0.06419843, -0.08501552, -1.9838965, -1.1178111,
            0.06695024, 0.45246887, -0.44694993, 0.44546956, -0.09799357, -0.14390799,
        ];
        let expected_state = [
            0.4398737, -0.918928, 0.13968477, 0.009646177, -3.4257128, -0.9662882,
            -0.02301114, 0.73852575, -3.2960606, -1.7599183, 0.11672372, 0.6619915,
            -0.7545119, 0.19815496, -0.3108106, -0.4498261, -0.94208056, 1.308459,
            0.015043208, -0.006178005, -0.9022816, 0.25467047, -0.4592561, -0.64908355,
        ];

        let (out, state) = gated_delta_rule_checked(
            1, 3, 2, 3, 4, &q, &k, &v, &g, &beta, None, true, true,
        )
        .expect("gated delta rule should succeed");
        assert_close(&out, &expected_out, 1e-5);
        assert_close(state.as_deref().unwrap(), &expected_state, 1e-5);
    }

    #[test]
    fn gated_delta_rule_continues_from_state() {
        // Run two single-token steps and compare to one two-token run.
        // This is the decode-contract check for the recurrent cache.
        let (q, k, v, g, beta) = (
            vec![0.1f32, 0.2, 0.3, -0.4, 0.5, 0.6], // S=2 H=1 K=3
            vec![1.0f32, 0.0, 0.5, -1.0, 0.25, 0.75],
            vec![0.5f32, 0.25, -0.5, 1.0], // S=2 V=2
            vec![-0.1f32, -0.2],
            vec![0.9f32, 0.8],
        );
        let (out_once, state_once) = gated_delta_rule_checked(
            1, 2, 1, 3, 2, &q, &k, &v, &g, &beta, None, true, true,
        )
        .unwrap();

        let q0 = vec![0.1f32, 0.2, 0.3];
        let k0 = vec![1.0f32, 0.0, 0.5];
        let v0 = vec![0.5f32, 0.25];
        let g0 = vec![-0.1f32];
        let b0 = vec![0.9f32];
        let (out0, state0) = gated_delta_rule_checked(
            1, 1, 1, 3, 2, &q0, &k0, &v0, &g0, &b0, None, true, true,
        )
        .unwrap();
        let q1 = vec![-0.4f32, 0.5, 0.6];
        let k1 = vec![-1.0f32, 0.25, 0.75];
        let v1 = vec![-0.5f32, 1.0];
        let g1 = vec![-0.2f32];
        let b1 = vec![0.8f32];
        let (out1, state1) = gated_delta_rule_checked(
            1, 1, 1, 3, 2, &q1, &k1, &v1, &g1, &b1, state0.as_deref(), true, true,
        )
        .unwrap();

        assert_close(&out0, &out_once[..2], 1e-6);
        assert_close(&out1, &out_once[2..], 1e-6);
        assert_close(state1.as_deref().unwrap(), state_once.as_deref().unwrap(), 1e-6);
    }

    #[test]
    fn causal_conv1d_matches_torch_golden() {
        // B=1, C=2, K=3, S=4, seed 1.
        let x = [
            0.66135216, 0.2669241, 0.06167726, 0.6213173,
            -0.45190597, -0.16613023, -1.5227685, 0.38168392,
        ];
        let w = [
            -1.0276086, -0.5630528, -0.89229053,
            -0.058250178, -0.19550958, -0.96563596,
        ];
        let expected_no_state = [
            -0.5901183, -0.61055005, -0.8849376, -0.8634166,
            0.43637666, 0.24877328, 1.5292437, -0.06117476,
        ];
        let mut new_state = vec![0.0f32; 6];
        let out = causal_conv1d(
            1, 2, 4, 3, &x, None, &w, None, &mut new_state, false,
        )
        .unwrap();
        assert_close(&out, &expected_no_state, 1e-5);
        // new_state should be last three raw inputs for each channel.
        assert_close(
            &new_state,
            &[0.2669241, 0.06167726, 0.6213173, -0.16613023, -1.5227685, 0.38168392],
            1e-6,
        );

        // With a previous state, the first output uses state[1..] + x[0].
        let state = [
            0.42241532, 0.267317, -0.42119515,
            -0.51069999, -1.5726652, -0.12324776,
        ];
        let expected_state = [
            -0.6276604, -0.17772625, -0.8849376, -0.8634166,
            0.5520808, 0.25595248, 1.5292437, -0.06117476,
        ];
        let mut new_state2 = vec![0.0f32; 6];
        let out2 = causal_conv1d(
            1, 2, 4, 3, &x, Some(&state), &w, None, &mut new_state2, false,
        )
        .unwrap();
        assert_close(&out2, &expected_state, 1e-5);
        assert_close(
            &new_state2,
            &[0.2669241, 0.06167726, 0.6213173, -0.16613023, -1.5227685, 0.38168392],
            1e-6,
        );
    }

    #[test]
    fn linear_attn_core_smoke_shape() {
        // Small GQA-ish config: 1 key head, 2 value heads, seq=2.
        let (b, s, hidden, nk, nv, kd, vd, conv_k) = (1usize, 2usize, 4usize, 1usize, 2usize, 3usize, 2usize, 2usize);
        let x = vec![0.1f32; b * s * hidden];
        let qkv_w = vec![0.2f32; (nk * kd * 2 + nv * vd) * hidden];
        let z_w = vec![0.3f32; nv * vd * hidden];
        let b_w = vec![0.4f32; nv * hidden];
        let a_w = vec![0.5f32; nv * hidden];
        let conv_w = vec![0.6f32; (nk * kd * 2 + nv * vd) * conv_k];
        let a_log = vec![0.7f32; nv];
        let dt_bias = vec![0.8f32; nv];
        let norm_w = vec![0.9f32; vd];
        let out_w = vec![1.0f32; hidden * nv * vd];
        let mut rec = vec![0.0f32; nv * kd * vd];
        let mut conv_state = vec![0.0f32; (nk * kd * 2 + nv * vd) * conv_k];

        let out = linear_attn_core(
            b, s, hidden, nk, nv, kd, vd, conv_k,
            &x, &qkv_w, &z_w, &b_w, &a_w, &conv_w,
            &a_log, &dt_bias, &norm_w, &out_w,
            &mut rec, &mut conv_state,
        )
        .expect("linear_attn_core should run");
        assert_eq!(out.len(), b * s * hidden);
        assert!(rec.iter().any(|&v| v != 0.0));
        assert!(conv_state.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn linear_attn_core_matches_python_golden() {
        // Generated from a direct PyTorch replica of Qwen3.5 GatedDeltaNet
        // with B=1, S=2, H=4, nk=1, nv=2, kd=3, vd=2, conv_k=2, seed=2.
        let x = [
            0.39229682, -0.22356401, -0.31950027, -1.2050371,
            1.0444635, -0.6332277, 0.57310677, 0.54094744,
        ];
        let qkv_w = [
            -1.5071439, -0.4585608, -0.84800065, 0.52660435, 0.029916182,
            -0.049838036, 1.0650779, 0.8860367, 0.46401837, -0.49863246,
            0.12886369, 2.7630668, 0.14047647, 1.1191015, 0.31523156,
            1.7527765, -0.76496398, 1.8298852, -0.27840105, -0.2719453,
            -1.2944106, -0.024312533, -0.23535971, -0.7087095, 0.94661385,
            -1.7669007, -0.64220244, 2.5169554, -0.90217805, -0.015080526,
            1.3856398, 2.5785294, -0.3467131, 0.0873015, -1.0996896,
            0.49612108, 1.4098375, -0.38815606, 0.39802396, -1.1042764,
        ];
        let z_w = [
            0.5461433, -1.357513, 1.1453115, -0.96460056, -0.25854325,
            -0.042134635, -0.9555159, -0.13457002, -0.27794072, 0.55108863,
            0.015409281, -0.9184425, -1.2342043, 0.25392434, 0.29004243,
            -0.62324685,
        ];
        let b_w = [
            -1.3766499, -0.09680592, -0.96562183, 0.2583479,
            0.6798685, -0.38861853, -0.5133859, -0.12599526,
        ];
        let a_w = [
            -0.15056981, 1.240303, 1.4933516, 0.49872792,
            0.23187248, 1.1746274, -1.3967456, 0.8997811,
        ];
        let conv_w = [
            1.9977686, 1.4826752, 0.19454689, -1.1372466, 0.22088396,
            0.3501896, -1.4627492, -1.4155066, 1.0311496, -1.9556775,
            -0.14820482, 1.737566, -0.005099262, 0.9915627, -1.1680793,
            0.7854731, 0.54933023, 0.053876013, 0.26006395, 0.8570226,
        ];
        let a_log = [1.3494372, 0.8017718];
        let dt_bias = [-0.47166418, 0.7573155];
        let norm_w = [0.53413624, 1.3426954];
        let out_w = [
            0.69732004, -0.46476349, 0.4796967, -0.28820205,
            0.63958967, -0.7863882, -0.09891472, -0.770287,
            0.3024695, -0.08675606, -0.5062722, -1.2308071,
            1.7482779, 0.6534637, -0.30582025, -0.8808548,
        ];
        let expected_out = [
            0.4014938, 0.35059315, 0.32594204, 1.3999656,
            -0.6740677, -1.0196156, -0.79044306, -0.9628031,
        ];
        let expected_rec = [
            0.2150051, 0.41452282, 0.35691398, 0.66833436,
            -0.011956339, -0.020396587, -0.040389024, 0.6212938,
            -0.061939504, 0.70231265, 0.001560664, 0.009605613,
        ];
        let expected_conv = [
            -0.8523714, -1.484913, -1.3851218, 1.1525078, -3.0772607,
            2.3689246, -2.4079597, 0.56689775, -0.29253602, -2.2643726,
            0.42686096, -1.85483, -2.0614717, 3.101044, -3.900486,
            1.2562257, -0.4020251, -0.77927506, 1.8433778, 1.3490697,
        ];

        let mut rec = vec![0.0f32; 2 * 3 * 2];
        let mut conv_state = vec![0.0f32; 10 * 2];
        let out = linear_attn_core(
            1, 2, 4, 1, 2, 3, 2, 2,
            &x, &qkv_w, &z_w, &b_w, &a_w, &conv_w,
            &a_log, &dt_bias, &norm_w, &out_w,
            &mut rec, &mut conv_state,
        )
        .expect("linear_attn_core golden should run");
        assert_close(&conv_state, &expected_conv, 2e-5);
        assert_close(&rec, &expected_rec, 2e-5);
        assert_close(&out, &expected_out, 2e-5);
    }
}
