//! OPT-specific fused decoder-layer fast path (Path C prototype).
//!
//! This module exists to answer a capability question, not to become the
//! mainline executor: "can this machine + Accelerate + the current KV
//! contract reach the 125 tok/s upper bound when the per-layer functions
//! are fused in Rust?"  It is deliberately narrow:
//!
//! * contract-checked against the `main_Xa` / `main_Xb` ABI emitted by the
//!   KV split compiler for OPT-125M
//! * only replaces the decoder-layer function pair; `main_0`, `main_13`,
//!   `main_14`, and `main_15` keep their existing fast paths
//! * reuses [`BlockManager`] with the same per-layer table naming
//!   (`<request_id>_f<func_index>`) as the intercept path
//! * disabled by default (the graph runner only calls it when the
//!   `--opt-fused-fastpath` flag is enabled)
//!
//! Numerics follow the lowered MLIR sequence of the KV dylib:
//! layer_norm(mean/rstd/eps) -> QKV SGEMM with `CblasTrans` weights ->
//! `Q *= 0.125` -> causal softmax -> SGEMM attention output ->
//! out_proj -> residual -> final layer_norm -> fc1 -> ReLU -> fc2 ->
//! residual.  GEMMs use the same `cblas_sgemm` entry point as the dylib
//! BLAS bridge so the dominant arithmetic matches bit-for-bit.

use std::collections::HashMap;

use crate::cache::block::BlockManager;
use crate::engine::blas;
use crate::engine::kernels;
use crate::model::compute_graph::{ComputeGraph, FuncDef, InputBinding};
use crate::model::tensor::{Dtype, Tensor};

/// The KV dylib pre-scales Q before `scores = Q @ K^T` instead of
/// scaling inside the softmax.
const Q_SCALE: f32 = 0.125;

fn fused_phase_profile_enabled() -> bool {
    std::env::var_os("SERVEFORGE_FUSED_PROFILE").is_some()
}

fn fused_phase_ms(t0: std::time::Instant) -> f64 {
    t0.elapsed().as_secs_f64() * 1e3
}

/// Structural requirements of the OPT KV layer split, derived from the
/// ABI itself (head count and head dim are static in the layer outputs).
#[derive(Debug, Clone, Copy)]
pub(crate) struct OptFusedSpec {
    pub num_heads: usize,
    pub head_dim: usize,
    pub hidden_dim: usize,
}

/// Scratch buffers reused across the 24 fused layer functions of one
/// forward pass.  Reusing these avoids ~150 Vec allocations per decode
/// step.  Output tensors still own their final activation storage.
#[derive(Default)]
pub(crate) struct FusedLayerWorkspace {
    ln: Vec<f32>,
    q_flat: Vec<f32>,
    k_flat: Vec<f32>,
    v_flat: Vec<f32>,
    scores: Vec<f32>,
    probs: Vec<f32>,
    ctx_head: Vec<f32>,
    ctx_flat: Vec<f32>,
    attn_out: Vec<f32>,
    hidden1: Vec<f32>,
    ln2: Vec<f32>,
    fc1: Vec<f32>,
    fc2: Vec<f32>,
    out: Vec<f32>,
}

/// Discover the OPT fused-fastpath contract from the compute graph.
///
/// Returns `None` when no `main_Xa` function carries the expected static
/// `[B, heads, seq, head_dim]` layer output — in which case the caller
/// must stay on the dylib path.
pub(crate) fn opt_fused_spec(compute_graph: &ComputeGraph) -> Option<OptFusedSpec> {
    for func in &compute_graph.functions {
        if is_opt_layer_a(func) {
            let out = func.outputs.first()?;
            if out.rank == 4 && out.shape.len() == 4 {
                let heads = out.shape[1] as usize;
                let dim = out.shape[3] as usize;
                if heads > 0 && dim > 0 && heads.checked_mul(dim).is_some() {
                    return Some(OptFusedSpec {
                        num_heads: heads,
                        head_dim: dim,
                        hidden_dim: heads * dim,
                    });
                }
            }
        }
    }
    None
}

/// True for `_mlir_ciface_main_<n>a` layer functions.
pub(crate) fn is_opt_layer_a(func: &FuncDef) -> bool {
    func.symbol.ends_with('a')
        && func.outputs.len() == 3
        && func
            .outputs
            .get(1)
            .map(|o| o.consumed_internally)
            .unwrap_or(false)
        && func
            .outputs
            .get(2)
            .map(|o| o.consumed_internally)
            .unwrap_or(false)
}

/// True for `_mlir_ciface_main_<n>b` layer functions.
pub(crate) fn is_opt_layer_b(func: &FuncDef) -> bool {
    func.symbol.ends_with('b')
        && func.inputs.len() == 15
        && func.outputs.len() == 1
}

/// Resolve an `main_X*` input that must be an SSA edge from `main_0`.
///
/// `main_0` stores every output (computed tensors and weight
/// pass-throughs) in `func_outputs[0]` in ABI order, so this is an O(1)
/// Arc clone for weights.
fn f0_ssa_tensor(
    func_def: &FuncDef,
    bi: usize,
    func_outputs: &[Vec<Tensor>],
    what: &str,
) -> Option<Tensor> {
    let (binding, _io_def) = func_def.inputs.get(bi)?;
    let (producer, output_idx) = match binding {
        InputBinding::Ssa {
            producer_func,
            output_idx,
        } => (*producer_func, *output_idx),
        _ => {
            log::debug!("opt-fused fallback: {} is not an f0 SSA edge", what);
            return None;
        }
    };
    if producer != 0 {
        log::debug!(
            "opt-fused fallback: {} produced by func {} instead of main_0",
            what,
            producer
        );
        return None;
    }
    match func_outputs.get(0).and_then(|v| v.get(output_idx)) {
        Some(t) => Some(t.to_owned()),
        None => {
            log::debug!("opt-fused fallback: {} (f0:{} unavailable)", what, output_idx);
            None
        }
    }
}

fn vec_matches(tensor: &Tensor, expected: &[usize]) -> bool {
    tensor.shape
        .iter()
        .map(|&d| d as i64)
        .eq(expected.iter().map(|&d| d as i64))
}

/// Execute the fused `main_Xa` half (attention layer norm + QKV + cache
/// write).
///
/// Returns `Ok(true)` when the fused path handled this function and
/// `Ok(false)` when the ABI did not match the OPT contract (the caller
/// falls back to the dylib function).
pub(crate) fn run_fused_layer_a(
    func_def: &FuncDef,
    func_outputs: &mut [Vec<Tensor>],
    positions: &[u32],
    is_decode: bool,
    block_manager: Option<&mut BlockManager>,
    request_id: Option<&str>,
    kv_new: &mut HashMap<(usize, usize), Tensor>,
    spec: OptFusedSpec,
    ws: &mut FusedLayerWorkspace,
) -> Result<bool, anyhow::Error> {
    if !is_opt_layer_a(func_def) || func_def.inputs.len() != 11 {
        return Ok(false);
    }
    let heads = spec.num_heads;
    let dim = spec.head_dim;
    let hidden = spec.hidden_dim;
    let seq = positions.len();
    if seq == 0 {
        anyhow::bail!("opt-fused layer a: empty position list");
    }

    // ABI contract: out0=[B,heads,seq,dim] visible, out1=K, out2=V consumed.
    let expected_kv_shape = vec![0u64, heads as u64, 0u64, dim as u64];
    if func_def.outputs[0].shape != expected_kv_shape
        || func_def.outputs[1].shape != expected_kv_shape
        || func_def.outputs[2].shape != expected_kv_shape
        || func_def.outputs[0].consumed_internally
        || !func_def.outputs[1].consumed_internally
        || !func_def.outputs[2].consumed_internally
    {
        log::warn!("opt-fused fallback: {} output contract mismatch", func_def.symbol);
        return Ok(false);
    }

    // Compiler-split operand order:
    //   0=ones scalar  1=hidden  2=attention scale  3=k_bias  4=k_weight
    //   5=ln_bias  6=ln_weight  7=q_bias  8=q_weight  9=v_bias  10=v_weight
    // The hidden edge comes from main_0 for layer 0 and from the previous
    // `main_Xb` output for every later layer, so resolve it as a generic
    // SSA edge.
    let hidden_t = match ssa_tensor(func_def, 1, func_outputs, "layer a hidden") {
        Some(t) => t,
        None => return Ok(false),
    };
    let scale_t = match f0_ssa_tensor(func_def, 2, func_outputs, "layer a scale") {
        Some(t) => t,
        None => return Ok(false),
    };
    let ln_bias = match f0_ssa_tensor(func_def, 5, func_outputs, "attn ln bias") {
        Some(t) => t,
        None => return Ok(false),
    };
    let ln_weight = match f0_ssa_tensor(func_def, 6, func_outputs, "attn ln weight") {
        Some(t) => t,
        None => return Ok(false),
    };
    let q_bias = match f0_ssa_tensor(func_def, 7, func_outputs, "q bias") {
        Some(t) => t,
        None => return Ok(false),
    };
    let q_weight = match f0_ssa_tensor(func_def, 8, func_outputs, "q weight") {
        Some(t) => t,
        None => return Ok(false),
    };
    let k_bias = match f0_ssa_tensor(func_def, 3, func_outputs, "k bias") {
        Some(t) => t,
        None => return Ok(false),
    };
    let k_weight = match f0_ssa_tensor(func_def, 4, func_outputs, "k weight") {
        Some(t) => t,
        None => return Ok(false),
    };
    let v_bias = match f0_ssa_tensor(func_def, 9, func_outputs, "v bias") {
        Some(t) => t,
        None => return Ok(false),
    };
    let v_weight = match f0_ssa_tensor(func_def, 10, func_outputs, "v weight") {
        Some(t) => t,
        None => return Ok(false),
    };

    if !vec_matches(&hidden_t, &[1, seq, hidden])
        || scale_t.shape != vec![1]
        || scale_t.numel() != 1
        || !vec_matches(&ln_bias, &[hidden])
        || !vec_matches(&ln_weight, &[hidden])
        || !vec_matches(&q_bias, &[hidden])
        || !vec_matches(&q_weight, &[hidden, hidden])
        || !vec_matches(&k_bias, &[hidden])
        || !vec_matches(&k_weight, &[hidden, hidden])
        || !vec_matches(&v_bias, &[hidden])
        || !vec_matches(&v_weight, &[hidden, hidden])
    {
        log::warn!("opt-fused fallback: {} operand shape mismatch", func_def.symbol);
        return Ok(false);
    }

    let phase_profile = fused_phase_profile_enabled();
    let t_all = std::time::Instant::now();
    let mut t_ln = 0.0f64;
    let mut t_q = 0.0f64;
    let mut t_k = 0.0f64;
    let mut t_v = 0.0f64;
    let mut t_layout = 0.0f64;
    let mut t_write = 0.0f64;

    let mut t0 = std::time::Instant::now();
    kernels::layer_norm_into(
        hidden_t.as_slice(),
        seq,
        hidden,
        ln_weight.as_slice(),
        ln_bias.as_slice(),
        kernels::LN_EPS,
        &mut ws.ln,
    )?;
    t_ln = fused_phase_ms(t0);

    let q_scale = scale_t.as_slice()[0];
    if (q_scale - Q_SCALE).abs() > 1e-6 {
        log::warn!(
            "opt-fused fallback: {} attention scale is {} (expected {})",
            func_def.symbol,
            q_scale,
            Q_SCALE
        );
        return Ok(false);
    }

    t0 = std::time::Instant::now();
    blas::sgemm_transb_into(&ws.ln, seq, hidden, hidden, q_weight.as_slice(), &mut ws.q_flat);
    kernels::add_row_bias(&mut ws.q_flat, seq, hidden, q_bias.as_slice());
    for v in ws.q_flat.iter_mut() {
        *v *= q_scale;
    }
    t_q = fused_phase_ms(t0);
    t0 = std::time::Instant::now();
    blas::sgemm_transb_into(&ws.ln, seq, hidden, hidden, k_weight.as_slice(), &mut ws.k_flat);
    kernels::add_row_bias(&mut ws.k_flat, seq, hidden, k_bias.as_slice());
    t_k = fused_phase_ms(t0);
    t0 = std::time::Instant::now();
    blas::sgemm_transb_into(&ws.ln, seq, hidden, hidden, v_weight.as_slice(), &mut ws.v_flat);
    kernels::add_row_bias(&mut ws.v_flat, seq, hidden, v_bias.as_slice());
    t_v = fused_phase_ms(t0);

    t0 = std::time::Instant::now();

    // The dylib stores Q/K/V as BNSD (head-major).  Build that layout for
    // the SSA wires so a fallback `main_Xb` dylib call sees the exact same
    // tensors as the intercepted path would have produced.
    let q_bnsd = kernels::pmajor_to_bnsd(&ws.q_flat, seq, heads, dim);
    let k_bnsd = kernels::pmajor_to_bnsd(&ws.k_flat, seq, heads, dim);
    let v_bnsd = kernels::pmajor_to_bnsd(&ws.v_flat, seq, heads, dim);
    t_layout = fused_phase_ms(t0);

    // Reuse the BlockManager with the canonical per-layer table name.  The
    // cache layout is BNLD (position-major), which is exactly the layout of
    // our projection output — no extra transpose.
    t0 = std::time::Instant::now();
    if let Some(bm) = block_manager {
        let rid = request_id
            .ok_or_else(|| anyhow::anyhow!("opt-fused layer a: request_id required for KV write"))?;
        let layer_rid = format!("{}_f{}", rid, func_def.index);
        let start_pos = if is_decode { positions[0] as usize } else { 0 };
        bm.write_kv(&layer_rid, start_pos, &ws.k_flat, hidden, true)
            .map_err(|e| anyhow::anyhow!("opt-fused layer a K write: {}", e))?;
        bm.write_kv(&layer_rid, start_pos, &ws.v_flat, hidden, false)
            .map_err(|e| anyhow::anyhow!("opt-fused layer a V write: {}", e))?;
    }
    t_write = fused_phase_ms(t0);

    if phase_profile {
        eprintln!(
            "[fused-profile] a{} total={:.3}ms ln={:.3} q={:.3} k={:.3} v={:.3} layout={:.3} kv_write={:.3}",
            func_def.index,
            fused_phase_ms(t_all),
            t_ln,
            t_q,
            t_k,
            t_v,
            t_layout,
            t_write,
        );
    }

    kv_new.insert(
        (func_def.index, 1),
        Tensor::new_owned(vec![1, heads, seq, dim], k_bnsd, Dtype::F32),
    );
    kv_new.insert(
        (func_def.index, 2),
        Tensor::new_owned(vec![1, heads, seq, dim], v_bnsd, Dtype::F32),
    );
    func_outputs[func_def.index].push(Tensor::new_owned(
        vec![1, heads, seq, dim],
        q_bnsd,
        Dtype::F32,
    ));
    Ok(true)
}

/// Execute the fused `main_Xb` half (causal attention read + out_proj +
/// MLP + residuals).
///
/// Returns `Ok(true)` when handled and `Ok(false)` for a contract
/// mismatch (fallback to the dylib function).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_fused_layer_b(
    func_def: &FuncDef,
    func_outputs: &mut [Vec<Tensor>],
    positions: &[u32],
    is_decode: bool,
    block_manager: Option<&BlockManager>,
    request_id: Option<&str>,
    kv_new: &HashMap<(usize, usize), Tensor>,
    spec: OptFusedSpec,
    ws: &mut FusedLayerWorkspace,
) -> Result<bool, anyhow::Error> {
    if !is_opt_layer_b(func_def) {
        return Ok(false);
    }
    let heads = spec.num_heads;
    let dim = spec.head_dim;
    let hidden = spec.hidden_dim;
    let seq = positions.len();
    if seq == 0 {
        anyhow::bail!("opt-fused layer b: empty position list");
    }

    // ABI contract: out0 = [B, seq, hidden].
    if func_def.outputs[0].rank != 3 || func_def.outputs[0].shape != vec![0, 0, hidden as u64] {
        log::warn!("opt-fused fallback: {} output contract mismatch", func_def.symbol);
        return Ok(false);
    }

    // Split operand order is NOT uniform across layers:
    //   * layer 0 (`main_1b`): 1=Q 2=K 3=V 4=mask 5=hidden
    //   * layers 1..11:        1=hidden 2=Q 3=K 4=V 5=mask
    // Locate the hidden edge by its ABI rank/shape instead of a hardcoded
    // index.  The weight edges (6..=13) are uniform.
    let hidden_bi = if func_def
        .inputs
        .get(5)
        .map(|(_, d)| d.rank == 3 && d.shape == vec![0, 0, hidden as u64])
        .unwrap_or(false)
    {
        5
    } else {
        1
    };
    let hidden_t = match ssa_tensor(func_def, hidden_bi, func_outputs, "layer b hidden") {
        Some(t) => t,
        None => return Ok(false),
    };
    // The Q edge references producer output index 1 in the ABI, but the
    // runtime stores visible layer-a outputs contiguously (K/V consumed
    // outputs are never pushed), so the storage index is always 0.
    let q_t = match func_outputs
        .get(func_def.index.saturating_sub(1))
        .and_then(|v| v.first())
    {
        Some(t) => t.to_owned(),
        None => {
            log::debug!(
                "opt-fused fallback: layer b Q (f{}:0) unavailable",
                func_def.index.saturating_sub(1)
            );
            return Ok(false);
        }
    };
    let fc1_bias = match f0_ssa_tensor(func_def, 6, func_outputs, "fc1 bias") {
        Some(t) => t,
        None => return Ok(false),
    };
    let fc1_weight = match f0_ssa_tensor(func_def, 7, func_outputs, "fc1 weight") {
        Some(t) => t,
        None => return Ok(false),
    };
    let fc2_bias = match f0_ssa_tensor(func_def, 8, func_outputs, "fc2 bias") {
        Some(t) => t,
        None => return Ok(false),
    };
    let fc2_weight = match f0_ssa_tensor(func_def, 9, func_outputs, "fc2 weight") {
        Some(t) => t,
        None => return Ok(false),
    };
    let final_ln_bias = match f0_ssa_tensor(func_def, 10, func_outputs, "final ln bias") {
        Some(t) => t,
        None => return Ok(false),
    };
    let final_ln_weight = match f0_ssa_tensor(func_def, 11, func_outputs, "final ln weight") {
        Some(t) => t,
        None => return Ok(false),
    };
    let out_bias = match f0_ssa_tensor(func_def, 12, func_outputs, "out bias") {
        Some(t) => t,
        None => return Ok(false),
    };
    let out_weight = match f0_ssa_tensor(func_def, 13, func_outputs, "out weight") {
        Some(t) => t,
        None => return Ok(false),
    };

    if !vec_matches(&hidden_t, &[1, seq, hidden])
        || !vec_matches(&q_t, &[1, heads, seq, dim])
        || !vec_matches(&fc1_bias, &[hidden * 4])
        || !vec_matches(&fc1_weight, &[hidden * 4, hidden])
        || !vec_matches(&fc2_bias, &[hidden])
        || !vec_matches(&fc2_weight, &[hidden, hidden * 4])
        || !vec_matches(&final_ln_bias, &[hidden])
        || !vec_matches(&final_ln_weight, &[hidden])
        || !vec_matches(&out_bias, &[hidden])
        || !vec_matches(&out_weight, &[hidden, hidden])
    {
        log::warn!(
            "opt-fused fallback: {} operand shape mismatch: hidden={:?} q={:?} fc1_bias={:?} fc1_w={:?} fc2_bias={:?} fc2_w={:?} ln_bias={:?} ln_w={:?} out_bias={:?} out_w={:?}",
            func_def.symbol,
            hidden_t.shape,
            q_t.shape,
            fc1_bias.shape,
            fc1_weight.shape,
            fc2_bias.shape,
            fc2_weight.shape,
            final_ln_bias.shape,
            final_ln_weight.shape,
            out_bias.shape,
            out_weight.shape,
        );
        return Ok(false);
    }

    // Build the K/V view for attention.  With a BlockManager this reads
    // the canonical per-layer table (`<rid>_f<a_func_index>`) in BNLD
    // position-major layout; without one it uses the BNSD K/V tensors the
    // fused (or dylib) layer-a path stored in `kv_new`.
    let (k_slice, v_slice, kv_len, kv_row_stride, kv_bnsd_layout) =
        if let (Some(bm), Some(rid)) = (block_manager, request_id) {
            let layer_rid = format!("{}_f{}", rid, func_def.index - 1);
            let kv_len = if is_decode {
                positions[0] as usize + 1
            } else {
                seq
            };
            let (cached_k, cached_v) = bm
                .read_kv(&layer_rid, kv_len, hidden)
                .map_err(|e| anyhow::anyhow!("opt-fused layer b read_kv: {}", e))?;
            (
                std::borrow::Cow::Owned(cached_k),
                std::borrow::Cow::Owned(cached_v),
                kv_len,
                hidden,
                false,
            )
        } else {
            let k_new = kv_new
                .get(&(func_def.index - 1, 1))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "opt-fused layer b: no K tensor for func {} (run layers in order)",
                        func_def.index - 1
                    )
                })?;
            let v_new = kv_new
                .get(&(func_def.index - 1, 2))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "opt-fused layer b: no V tensor for func {} (run layers in order)",
                        func_def.index - 1
                    )
                })?;
            let kv_len = k_new.shape[2] as usize;
            if !vec_matches(k_new, &[1, heads, kv_len, dim])
                || !vec_matches(v_new, &[1, heads, kv_len, dim])
            {
                log::warn!("opt-fused fallback: {} KV operand shape mismatch", func_def.symbol);
                return Ok(false);
            }
            (
                std::borrow::Cow::Borrowed(k_new.as_slice()),
                std::borrow::Cow::Borrowed(v_new.as_slice()),
                kv_len,
                dim,
                true,
            )
        };

    let phase_profile = fused_phase_profile_enabled();
    let t_all = std::time::Instant::now();
    let mut t_attn = 0.0f64;
    let mut t_out = 0.0f64;
    let mut t_residual = 0.0f64;
    let mut t_ln2 = 0.0f64;
    let mut t_fc1 = 0.0f64;
    let mut t_fc2 = 0.0f64;

    let mut t0 = std::time::Instant::now();
    kernels::attention_forward(
        q_t.as_slice(),
        seq,
        &k_slice,
        &v_slice,
        kv_len,
        kv_row_stride,
        kv_bnsd_layout,
        heads,
        dim,
        is_decode,
        None,
        &mut ws.scores,
        &mut ws.probs,
        &mut ws.ctx_head,
        &mut ws.ctx_flat,
    )?;
    t_attn = fused_phase_ms(t0);

    t0 = std::time::Instant::now();
    blas::sgemm_transb_into(
        &ws.ctx_flat,
        seq,
        hidden,
        hidden,
        out_weight.as_slice(),
        &mut ws.attn_out,
    );
    kernels::add_row_bias(&mut ws.attn_out, seq, hidden, out_bias.as_slice());
    t_out = fused_phase_ms(t0);

    // attention residual
    t0 = std::time::Instant::now();
    ws.hidden1.resize(seq * hidden, 0.0);
    for (dst, (&a, &b)) in ws
        .hidden1
        .iter_mut()
        .zip(hidden_t.as_slice().iter().zip(ws.attn_out.iter()))
    {
        *dst = a + b;
    }
    t_residual = fused_phase_ms(t0);

    t0 = std::time::Instant::now();
    kernels::layer_norm_into(
        &ws.hidden1,
        seq,
        hidden,
        final_ln_weight.as_slice(),
        final_ln_bias.as_slice(),
        kernels::LN_EPS,
        &mut ws.ln2,
    )?;
    t_ln2 = fused_phase_ms(t0);

    t0 = std::time::Instant::now();
    blas::sgemm_transb_into(
        &ws.ln2,
        seq,
        hidden * 4,
        hidden,
        fc1_weight.as_slice(),
        &mut ws.fc1,
    );
    kernels::add_row_bias(&mut ws.fc1, seq, hidden * 4, fc1_bias.as_slice());
    for v in ws.fc1.iter_mut() {
        *v = v.max(0.0);
    }
    t_fc1 = fused_phase_ms(t0);

    t0 = std::time::Instant::now();
    blas::sgemm_transb_into(
        &ws.fc1,
        seq,
        hidden,
        hidden * 4,
        fc2_weight.as_slice(),
        &mut ws.fc2,
    );
    kernels::add_row_bias(&mut ws.fc2, seq, hidden, fc2_bias.as_slice());

    // MLP residual
    ws.out.resize(seq * hidden, 0.0);
    for (dst, (&a, &b)) in ws
        .out
        .iter_mut()
        .zip(ws.hidden1.iter().zip(ws.fc2.iter()))
    {
        *dst = a + b;
    }
    t_fc2 = fused_phase_ms(t0);

    if phase_profile {
        eprintln!(
            "[fused-profile] b{} total={:.3}ms attn={:.3} out={:.3} residual={:.3} ln2={:.3} fc1={:.3} fc2={:.3} kv_len={}",
            func_def.index,
            fused_phase_ms(t_all),
            t_attn,
            t_out,
            t_residual,
            t_ln2,
            t_fc1,
            t_fc2,
            kv_len,
        );
    }

    func_outputs[func_def.index].push(Tensor::new_owned(
        vec![1, seq, hidden],
        std::mem::take(&mut ws.out),
        Dtype::F32,
    ));
    Ok(true)
}

/// Resolve a generic SSA edge and return the producer's stored tensor.
fn ssa_tensor(
    func_def: &FuncDef,
    bi: usize,
    func_outputs: &[Vec<Tensor>],
    what: &str,
) -> Option<Tensor> {
    let (binding, _io_def) = func_def.inputs.get(bi)?;
    let (producer, output_idx) = match binding {
        InputBinding::Ssa {
            producer_func,
            output_idx,
        } => (*producer_func, *output_idx),
        _ => {
            log::debug!("opt-fused fallback: {} is not an SSA edge", what);
            return None;
        }
    };
    match func_outputs
        .get(producer)
        .and_then(|v| v.get(output_idx))
    {
        Some(t) => {
            log::debug!(
                "opt-fused SSA {} = f{}:{} shape={:?}",
                what,
                producer,
                output_idx,
                t.shape
            );
            Some(t.to_owned())
        }
        None => {
            log::debug!(
                "opt-fused fallback: {} (f{}:{}) unavailable",
                what,
                producer,
                output_idx
            );
            None
        }
    }
}
