//! Op-plan HAL kernels (Phase 5; dtype-aware projections since S2 Day 2).
//!
//! Every kernel is a pure function of its input tensors plus the node
//! attribute map.  The executor owns allocation/pooling and input
//! assembly; kernels own numerics and output-shape resolution only.
//! `linear_transb` accepts F32/F16/BF16 weights; other ops remain F32.

use std::collections::HashMap;

use crate::engine::blas;
use crate::engine::kernels;
use crate::model::tensor::{Dtype, Tensor};

pub(crate) const PLAN_OP_CATALOG: &[&str] = &[
    "layer_norm",
    "linear_transb",
    "attention_causal",
    "add",
    "relu",
    "mul",
    "view",
    "transpose",
    "identity",
];

// ── attribute helpers ───────────────────────────────────────────────

fn attr<'a>(attrs: &'a HashMap<String, String>, key: &str) -> Result<&'a str, anyhow::Error> {
    attrs
        .get(key)
        .map(|s| s.as_str())
        .ok_or_else(|| anyhow::anyhow!("missing required attribute {key:?}"))
}

fn parse_i64_list(s: &str) -> Result<Vec<i64>, anyhow::Error> {
    let inner = s.trim().trim_start_matches('[').trim_end_matches(']');
    if inner.is_empty() {
        return Ok(Vec::new());
    }
    inner
        .split(',')
        .map(|p| {
            p.trim()
                .parse::<i64>()
                .map_err(|e| anyhow::anyhow!("bad int list {s:?}: {}", e))
        })
        .collect()
}

fn parse_usize_list(s: &str) -> Result<Vec<usize>, anyhow::Error> {
    parse_i64_list(s).map(|v| v.into_iter().map(|x| x as usize).collect())
}

fn attr_usize(attrs: &HashMap<String, String>, key: &str) -> Result<usize, anyhow::Error> {
    Ok(attr(attrs, key)?.parse::<usize>()?)
}

fn attr_f32(attrs: &HashMap<String, String>, key: &str) -> Result<f32, anyhow::Error> {
    Ok(attr(attrs, key)?.parse::<f32>()?)
}

fn shape_numel(shape: &[usize]) -> Result<usize, anyhow::Error> {
    shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| anyhow::anyhow!("shape product overflow: {:?}", shape))
}

fn tensor_scalar_usize(t: &Tensor, what: &str) -> Result<usize, anyhow::Error> {
    anyhow::ensure!(t.numel() == 1, "{}: expected scalar tensor, got {:?}", what, t.shape);
    let v = t.as_slice()[0];
    anyhow::ensure!(v >= 0.0 && v.fract() == 0.0, "{}: non-integer value {}", what, v);
    Ok(v as usize)
}

// ── broadcasting (add/mul) ─────────────────────────────────────────

fn broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let rank = a.len().max(b.len());
    let mut out = vec![0usize; rank];
    for i in 0..rank {
        let da = if i >= rank - a.len() { a[i - (rank - a.len())] } else { 1 };
        let db = if i >= rank - b.len() { b[i - (rank - b.len())] } else { 1 };
        if da != db && da != 1 && db != 1 {
            return None;
        }
        out[i] = da.max(db);
    }
    Some(out)
}

fn binary_broadcast(
    a: &Tensor,
    b: &Tensor,
    out: &mut [f32],
    f: impl Fn(f32, f32) -> f32,
) -> Result<Vec<usize>, anyhow::Error> {
    let shape = broadcast_shape(&a.shape, &b.shape)
        .ok_or_else(|| anyhow::anyhow!("cannot broadcast {:?} with {:?}", a.shape, b.shape))?;
    let numel = shape_numel(&shape)?;
    anyhow::ensure!(out.len() == numel, "broadcast output buffer mismatch");
    let a_slice = a.as_slice();
    let b_slice = b.as_slice();

    // Fast paths for the M1 shapes: same-shape add and scalar broadcast.
    if a.shape == b.shape {
        for ((dst, &av), &bv) in out.iter_mut().zip(a_slice.iter()).zip(b_slice.iter()) {
            *dst = f(av, bv);
        }
        return Ok(shape);
    }
    if a.numel() == 1 {
        let av = a_slice[0];
        for (dst, &bv) in out.iter_mut().zip(b_slice.iter()) {
            *dst = f(av, bv);
        }
        return Ok(shape);
    }
    if b.numel() == 1 {
        let bv = b_slice[0];
        for (dst, &av) in out.iter_mut().zip(a_slice.iter()) {
            *dst = f(av, bv);
        }
        return Ok(shape);
    }

    // General numpy-style broadcast fallback: precompute source strides
    // once instead of re-deriving them per element.
    let a_strides = row_major_strides(&a.shape);
    let b_strides = row_major_strides(&b.shape);
    let out_strides = row_major_strides(&shape);
    let a_delta = shape.len() - a.shape.len();
    let b_delta = shape.len() - b.shape.len();
    for (idx, dst) in out.iter_mut().enumerate() {
        let mut rem = idx;
        let mut ai = 0usize;
        let mut bi = 0usize;
        for dim in 0..shape.len() {
            let coord = rem / out_strides[dim];
            rem %= out_strides[dim];
            if dim >= a_delta {
                let src_dim = a.shape[dim - a_delta];
                if src_dim != 1 {
                    ai += coord * a_strides[dim - a_delta];
                }
            }
            if dim >= b_delta {
                let src_dim = b.shape[dim - b_delta];
                if src_dim != 1 {
                    bi += coord * b_strides[dim - b_delta];
                }
            }
        }
        *dst = f(a_slice[ai], b_slice[bi]);
    }
    Ok(shape)
}

// ── output shape resolution ─────────────────────────────────────────

pub(crate) fn resolve_output_shape(
    op_name: &str,
    inputs: &[Tensor],
    attrs: &HashMap<String, String>,
) -> Result<Vec<usize>, anyhow::Error> {
    match op_name {
        "layer_norm" | "relu" | "identity" => Ok(inputs[0].shape.clone()),
        "linear_transb" => linear_out_shape(&inputs[0], &inputs[1]),
        "attention_causal" => Ok(inputs[0].shape.clone()),
        "add" | "mul" => broadcast_shape(&inputs[0].shape, &inputs[1].shape)
            .ok_or_else(|| anyhow::anyhow!("cannot broadcast {:?} with {:?}", inputs[0].shape, inputs[1].shape)),
        "transpose" => {
            let a = attr_usize(attrs, "dim0")?;
            let b = attr_usize(attrs, "dim1")?;
            let mut shape = inputs[0].shape.clone();
            anyhow::ensure!(a < shape.len() && b < shape.len(), "transpose dim out of range");
            shape.swap(a, b);
            Ok(shape)
        }
        "view" => view_out_shape(inputs, attrs),
        other => anyhow::bail!("no output-shape resolver for op {other:?}"),
    }
}

fn linear_out_shape(input: &Tensor, weight: &Tensor) -> Result<Vec<usize>, anyhow::Error> {
    anyhow::ensure!(weight.shape.len() == 2, "linear weight must be [n,k]");
    let n = weight.shape[0];
    let k = weight.shape[1];
    let (leading, reduction) = match input.shape.len() {
        0 => anyhow::bail!("linear input must have rank >= 1"),
        1 => (&[][..], input.shape[0]),
        _ => (&input.shape[..input.shape.len() - 1], input.shape[input.shape.len() - 1]),
    };
    anyhow::ensure!(reduction == k, "linear K mismatch: input {reduction}, weight {k}");
    let mut shape = leading.to_vec();
    shape.push(n);
    Ok(shape)
}

fn view_out_shape(inputs: &[Tensor], attrs: &HashMap<String, String>) -> Result<Vec<usize>, anyhow::Error> {
    let entries = parse_i64_list(attr(attrs, "shape")?)?;
    let input = &inputs[0];
    let input_numel = input.numel();
    let mut dyn_idx = 0usize;
    let mut shape = Vec::with_capacity(entries.len());
    let mut known_product = 1usize;
    let mut inferred = false;

    for &entry in &entries {
        if entry >= 0 {
            shape.push(entry as usize);
            known_product = known_product
                .checked_mul(entry as usize)
                .ok_or_else(|| anyhow::anyhow!("view shape overflow"))?;
        } else if entry < -1 {
            let idx = (-entry - 2) as usize;
            let tensor = inputs.get(idx + 1).ok_or_else(|| {
                anyhow::anyhow!("view references dyn operand {idx}, only {} provided", inputs.len().saturating_sub(1))
            })?;
            let v = tensor_scalar_usize(tensor, "view dyn operand")?;
            shape.push(v);
            known_product = known_product.checked_mul(v).ok_or_else(|| anyhow::anyhow!("view shape overflow"))?;
            dyn_idx = dyn_idx.max(idx + 1);
        } else if dyn_idx < inputs.len().saturating_sub(1) {
            let v = tensor_scalar_usize(&inputs[dyn_idx + 1], "view dyn operand")?;
            shape.push(v);
            known_product = known_product.checked_mul(v).ok_or_else(|| anyhow::anyhow!("view shape overflow"))?;
            dyn_idx += 1;
        } else {
            shape.push(0);
            inferred = true;
        }
    }

    if inferred {
        anyhow::ensure!(known_product > 0 && input_numel % known_product == 0,
            "view cannot infer dim: numel {} / known {}", input_numel, known_product);
        for dim in shape.iter_mut() {
            if *dim == 0 {
                *dim = input_numel / known_product;
            }
        }
    }
    anyhow::ensure!(shape_numel(&shape)? == input_numel,
        "view shape {:?} has numel {} but input has {}",
        shape, shape_numel(&shape)?, input_numel);
    Ok(shape)
}

// ── execution ───────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub(crate) fn execute(
    op_name: &str,
    inputs: &[Tensor],
    attrs: &HashMap<String, String>,
    is_decode: bool,
    out: &mut [f32],
) -> Result<Vec<usize>, anyhow::Error> {
    for (idx, t) in inputs.iter().enumerate() {
        let raw_linear_weight = op_name == "linear_transb"
            && idx == 1
            && matches!(t.dtype, Dtype::F16 | Dtype::BF16);
        anyhow::ensure!(
            t.dtype == Dtype::F32 || raw_linear_weight,
            "op {op_name} input {idx}: expected f32 input, got {}",
            t.dtype
        );
    }
    match op_name {
        "layer_norm" => layer_norm(inputs, attrs, out),
        "linear_transb" => linear_transb(inputs, out),
        "attention_causal" => attention_causal(inputs, attrs, is_decode, out),
        "add" => binary_broadcast(&inputs[0], &inputs[1], out, |a, b| a + b),
        "mul" => binary_broadcast(&inputs[0], &inputs[1], out, |a, b| a * b),
        "relu" => {
            let src = inputs[0].as_slice();
            anyhow::ensure!(out.len() == src.len(), "relu buffer mismatch");
            for (dst, &v) in out.iter_mut().zip(src.iter()) {
                *dst = v.max(0.0);
            }
            Ok(inputs[0].shape.clone())
        }
        "identity" => {
            let src = inputs[0].as_slice();
            anyhow::ensure!(out.len() == src.len(), "identity buffer mismatch");
            out.copy_from_slice(src);
            Ok(inputs[0].shape.clone())
        }
        "view" => {
            let shape = view_out_shape(inputs, attrs)?;
            let src = inputs[0].as_slice();
            anyhow::ensure!(out.len() == src.len(), "view buffer mismatch");
            out.copy_from_slice(src);
            Ok(shape)
        }
        "transpose" => transpose(inputs, attrs, out),
        other => anyhow::bail!("unsupported plan op {other:?}"),
    }
}

fn layer_norm(
    inputs: &[Tensor],
    attrs: &HashMap<String, String>,
    out: &mut [f32],
) -> Result<Vec<usize>, anyhow::Error> {
    anyhow::ensure!(inputs.len() == 3, "layer_norm expects [x, weight, bias]");
    let x = &inputs[0];
    let weight = &inputs[1];
    let bias = &inputs[2];
    let normalized = parse_usize_list(attr(attrs, "normalized_shape")?)?;
    let cols = shape_numel(&normalized)?;
    anyhow::ensure!(x.shape.last().copied().unwrap_or(1) == cols
        || (x.shape.len() >= normalized.len()
            && x.shape[x.shape.len() - normalized.len()..].iter().product::<usize>() == cols),
        "layer_norm normalized_shape {normalized:?} incompatible with input {:?}", x.shape);
    let rows = x.numel() / cols.max(1);
    anyhow::ensure!(weight.numel() == cols && bias.numel() == cols, "layer_norm affine shape mismatch");
    let mut scratch = Vec::new();
    kernels::layer_norm_into(
        x.as_slice(),
        rows,
        cols,
        weight.as_slice(),
        bias.as_slice(),
        kernels::LN_EPS,
        &mut scratch,
    )?;
    anyhow::ensure!(scratch.len() == out.len(), "layer_norm output buffer mismatch");
    out.copy_from_slice(&scratch);
    Ok(x.shape.clone())
}

fn linear_transb(inputs: &[Tensor], out: &mut [f32]) -> Result<Vec<usize>, anyhow::Error> {
    anyhow::ensure!(inputs.len() == 3, "linear_transb expects [x, weight, bias]");
    let x = &inputs[0];
    let weight = &inputs[1];
    let bias = &inputs[2];
    anyhow::ensure!(weight.shape.len() == 2, "linear weight must be [n,k]");
    let n = weight.shape[0];
    let k = weight.shape[1];
    let m = if x.shape.is_empty() {
        anyhow::bail!("linear input rank must be >= 1")
    } else if x.shape.len() == 1 {
        x.shape[0] / k.max(1)
    } else {
        x.numel() / x.shape[x.shape.len() - 1]
    };
    let reduction = if x.shape.len() == 1 { k } else { x.shape[x.shape.len() - 1] };
    anyhow::ensure!(reduction == k && m.checked_mul(k) == Some(x.numel()), "linear K/shape mismatch");
    let shape = linear_out_shape(x, weight)?;
    let expected = shape_numel(&shape)?;
    anyhow::ensure!(out.len() == expected && bias.numel() == n, "linear output buffer/bias mismatch");
    match weight.dtype {
        Dtype::F32 => {
            blas::sgemm_transb(x.as_slice(), m, n, k, weight.as_slice(), out);
        }
        Dtype::F16 | Dtype::BF16 => {
            // The spike showed row-threading pays only for large reduction
            // or output dims; qkv/out stay single-thread to avoid spawning
            // overhead on 768x768 matrices.
            let threads = if k >= 3072 || n >= 3072 { 4 } else { 1 };
            if m == 1 {
                crate::engine::gemv::gemv_threaded_into(
                    x.as_slice(),
                    n,
                    k,
                    weight.as_u16(),
                    weight.dtype,
                    threads,
                    out,
                );
            } else {
                // Prefill has m > 1.  Loop rows and use the single-thread
                // GEMV per row; prefill is not the gate hot path and this
                // keeps the kernel contract identical to decode.
                for row in 0..m {
                    crate::engine::gemv::gemv_threaded_into(
                        &x.as_slice()[row * k..(row + 1) * k],
                        n,
                        k,
                        weight.as_u16(),
                        weight.dtype,
                        1,
                        &mut out[row * n..(row + 1) * n],
                    );
                }
            }
        }
        other => anyhow::bail!("linear_transb: unsupported weight dtype {other}"),
    }
    kernels::add_row_bias(out, m, n, bias.as_slice());
    Ok(shape)
}

fn attention_causal(
    inputs: &[Tensor],
    attrs: &HashMap<String, String>,
    is_decode: bool,
    out: &mut [f32],
) -> Result<Vec<usize>, anyhow::Error> {
    anyhow::ensure!(inputs.len() == 4, "attention_causal expects [q, k, v, mask]");
    let q = &inputs[0];
    let k = &inputs[1];
    let v = &inputs[2];
    let mask = &inputs[3];
    anyhow::ensure!(q.shape.len() == 4 && k.shape.len() == 4 && v.shape.len() == 4,
        "attention BNSD rank-4 required: q={:?} k={:?} v={:?}", q.shape, k.shape, v.shape);
    anyhow::ensure!(q.shape[0] == 1 && k.shape[0] == 1 && v.shape[0] == 1, "batch must be 1");
    let heads = q.shape[1];
    let seq = q.shape[2];
    let dim = q.shape[3];
    anyhow::ensure!(k.shape[1] == heads && k.shape[3] == dim && v.shape[1] == heads && v.shape[3] == dim,
        "attention head/dim mismatch");
    let kv_len = k.shape[2];
    anyhow::ensure!(v.shape[2] == kv_len, "K/V length mismatch");
    let scale = attrs.get("scale").map(|s| s.parse::<f32>()).transpose()?.unwrap_or(1.0);
    anyhow::ensure!((scale - 1.0).abs() < 1e-6,
        "attention_causal scale {scale} != 1.0 is not supported by the M1 f32 kernel");

    let mut scores = Vec::new();
    let mut probs = Vec::new();
    let mut ctx_head = Vec::new();
    let mut ctx_flat = Vec::new();
    kernels::attention_forward(
        q.as_slice(),
        seq,
        k.as_slice(),
        v.as_slice(),
        kv_len,
        dim,
        true,
        heads,
        dim,
        is_decode,
        Some(mask),
        &mut scores,
        &mut probs,
        &mut ctx_head,
        &mut ctx_flat,
    )?;
    anyhow::ensure!(ctx_head.len() == out.len(), "attention output buffer mismatch");
    // The plan node outputs BNSD (head-major), matching the sf-dialect
    // SDPA result type [B, H, S, D].  `ctx_flat` is the position-major
    // interleave used only by the fused one-step path.
    out.copy_from_slice(&ctx_head);
    Ok(q.shape.clone())
}

fn transpose(
    inputs: &[Tensor],
    attrs: &HashMap<String, String>,
    out: &mut [f32],
) -> Result<Vec<usize>, anyhow::Error> {
    let a = attr_usize(attrs, "dim0")?;
    let b = attr_usize(attrs, "dim1")?;
    let src_shape = &inputs[0].shape;
    anyhow::ensure!(a < src_shape.len() && b < src_shape.len(), "transpose dim out of range");
    let mut dst_shape = src_shape.clone();
    dst_shape.swap(a, b);
    let src = inputs[0].as_slice();
    let dst_numel = shape_numel(&dst_shape)?;
    anyhow::ensure!(src.len() == dst_numel && out.len() == dst_numel, "transpose buffer mismatch");

    // Row-major strides for source and destination.
    let src_strides = row_major_strides(src_shape);
    let dst_strides = row_major_strides(&dst_shape);
    let rank = src_shape.len();
    let mut coords = vec![0usize; rank];
    for (dst_idx, slot) in out.iter_mut().enumerate() {
        let mut rem = dst_idx;
        for dim in 0..rank {
            coords[dim] = rem / dst_strides[dim];
            rem %= dst_strides[dim];
        }
        coords.swap(a, b);
        let mut src_idx = 0usize;
        for dim in 0..rank {
            src_idx += coords[dim] * src_strides[dim];
        }
        *slot = src[src_idx];
    }
    Ok(dst_shape)
}

fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len()).rev() {
        if i + 1 < shape.len() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(shape: Vec<usize>, data: Vec<f32>) -> Tensor {
        Tensor::new_owned(shape, data, Dtype::F32)
    }

    fn attrs(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    #[test]
    fn linear_transb_matches_reference() {
        let x = tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let w = tensor(vec![2, 3], vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let b = tensor(vec![2], vec![1.0, -1.0]);
        let mut out = vec![0.0; 4];
        let shape = execute("linear_transb", &[x, w, b], &HashMap::new(), true, &mut out).unwrap();
        assert_eq!(shape, vec![2, 2]);
        assert_eq!(out, vec![2.0, 1.0, 5.0, 4.0]);
    }

    #[test]
    fn linear_transb_f16_weight_matches_reference() {
        let x = tensor(vec![1, 3], vec![1.0, 2.0, 3.0]);
        let w = Tensor::new_owned_u16(
            vec![2, 3],
            vec![
                half::f16::from_f32(1.0).to_bits(),
                half::f16::from_f32(2.0).to_bits(),
                half::f16::from_f32(3.0).to_bits(),
                half::f16::from_f32(0.5).to_bits(),
                half::f16::from_f32(-0.5).to_bits(),
                half::f16::from_f32(0.25).to_bits(),
            ],
            Dtype::F16,
        );
        let b = tensor(vec![2], vec![0.25, -0.25]);
        let mut out = vec![0.0; 2];
        let shape = execute("linear_transb", &[x, w, b], &HashMap::new(), true, &mut out).unwrap();
        assert_eq!(shape, vec![1, 2]);
        assert!((out[0] - 14.25).abs() < 1e-3, "got {}", out[0]);
        assert!((out[1] - 0.0).abs() < 1e-3, "got {}", out[1]);
    }

    #[test]
    fn linear_transb_bf16_weight_matches_reference() {
        let bf = |v: f32| -> u16 { (v.to_bits() >> 16) as u16 };
        let x = tensor(vec![1, 3], vec![1.0, 2.0, 3.0]);
        let w = Tensor::new_owned_u16(
            vec![2, 3],
            vec![bf(1.0), bf(2.0), bf(3.0), bf(0.5), bf(-0.5), bf(0.25)],
            Dtype::BF16,
        );
        let b = tensor(vec![2], vec![0.25, -0.25]);
        let mut out = vec![0.0; 2];
        let shape = execute("linear_transb", &[x, w, b], &HashMap::new(), true, &mut out).unwrap();
        assert_eq!(shape, vec![1, 2]);
        assert!((out[0] - 14.25).abs() < 1e-3, "got {}", out[0]);
        assert!((out[1] - 0.0).abs() < 1e-3, "got {}", out[1]);
    }

    #[test]
    fn view_resolves_dynamic_dims() {
        let x = tensor(vec![8, 768], (0..8 * 768).map(|i| i as f32).collect());
        let batch = tensor(vec![1], vec![1.0]);
        let seq = tensor(vec![1], vec![8.0]);
        let out_shape = resolve_output_shape(
            "view",
            &[x, batch, seq],
            &attrs(&[("shape", "[-2, -3, 768]")]),
        )
        .unwrap();
        assert_eq!(out_shape, vec![1, 8, 768]);
    }

    #[test]
    fn transpose_matches_reference() {
        let x = tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut out = vec![0.0; 6];
        let shape = execute(
            "transpose",
            &[x],
            &attrs(&[("dim0", "0"), ("dim1", "1")]),
            true,
            &mut out,
        )
        .unwrap();
        assert_eq!(shape, vec![3, 2]);
        assert_eq!(out, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn broadcast_mul_scalar() {
        let x = tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let s = tensor(vec![1], vec![2.0]);
        let mut out = vec![0.0; 6];
        let shape = execute("mul", &[x, s], &HashMap::new(), true, &mut out).unwrap();
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(out, vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn layer_norm_2d_shape() {
        let x = tensor(vec![2, 2], vec![0.0, 2.0, 2.0, 4.0]);
        let w = tensor(vec![2], vec![1.0, 1.0]);
        let b = tensor(vec![2], vec![0.0, 0.0]);
        let mut out = vec![0.0; 4];
        let shape = execute(
            "layer_norm",
            &[x, w, b],
            &attrs(&[("normalized_shape", "[2]")]),
            true,
            &mut out,
        )
        .unwrap();
        assert_eq!(shape, vec![2, 2]);
        let a = out[0] + out[1];
        let c = out[2] + out[3];
        assert!((a + c).abs() < 1e-5);
    }
}
