//! HalRustExecutable — pure-Rust HAL backend dispatch.
//!
//! Implements ``traits::Executable`` by dispatching to the generated
//! ``*_cpu`` functions from ``hal_ops_cpu.rs`` (emitted by EmitRust).
//!
//! Dispatch logic lives in the sibling ``dispatch`` module — each
//! ``dispatch_*`` free function handles one HAL operation type.
//!
//! Each input/output Buffer is converted to a ``&[f32]``/``&mut [f32]``
//! slice, and an ``OpShapeMeta`` is constructed from the buffer shapes.
//! The generated ``*_cpu`` function is then called inline.

use crate::hal::traits;
use crate::hal::sfa::SfaMemRef;

// ── Free helpers ──────────────────────────────────────────────────────

/// Convert an SfaMemRef to a ``&[f32]`` slice.
///
/// # Safety
///
/// The memref must contain f32 data (element_size == 4).
pub(crate) unsafe fn sfa_as_f32_slice(sfa: &SfaMemRef) -> &[f32] {
    let ptr = sfa.data_ptr() as *const f32;
    let len = sfa.numel();
    std::slice::from_raw_parts(ptr, len)
}

/// Convert an SfaMemRef to a ``&mut [f32]`` slice.
///
/// # Safety
///
/// The memref must contain f32 data (element_size == 4) and be writable.
#[allow(clippy::mut_from_ref)]
pub(crate) unsafe fn sfa_as_f32_mut(sfa: &SfaMemRef) -> &mut [f32] {
    let ptr = sfa.data_ptr() as *mut f32;
    let len = sfa.numel();
    std::slice::from_raw_parts_mut(ptr, len)
}

/// Convert an SfaMemRef to raw bytes.
///
/// # Safety
///
/// The memref must point to valid data.
pub(crate) unsafe fn sfa_as_bytes(sfa: &SfaMemRef) -> &[u8] {
    let ptr = sfa.data_ptr();
    let len = sfa.byte_len();
    std::slice::from_raw_parts(ptr, len)
}

/// Convert an SfaMemRef to mutable raw bytes.
///
/// # Safety
///
/// The memref must point to valid, writable data.
#[allow(clippy::mut_from_ref)]
pub(crate) unsafe fn sfa_as_mut_bytes(sfa: &SfaMemRef) -> &mut [u8] {
    let ptr = sfa.data_ptr();
    let len = sfa.byte_len();
    std::slice::from_raw_parts_mut(ptr, len)
}

/// Build an ``OpShapeMeta`` from input and output memrefs.
pub(crate) fn build_shape_meta_from_sfa(
    inputs: &[SfaMemRef],
    outputs: &[SfaMemRef],
) -> crate::hal::hal_ops_cpu::OpShapeMeta {
    let input_shapes: Vec<Vec<i64>> = inputs
        .iter()
        .map(|m| m.sizes_i64())
        .collect();
    let output_shape: Vec<i64> = outputs
        .first()
        .map(|m| m.sizes_i64())
        .unwrap_or_default();
    crate::hal::hal_ops_cpu::OpShapeMeta::new(input_shapes, output_shape)
}

// ── HalRustExecutable ─────────────────────────────────────────────────

/// A pure-Rust HAL executable that dispatches to generated CPU kernels.
///
/// ``function_count`` is the number of functions (entry points) in the
/// model forward pass, set from the compute graph.
#[derive(Debug)]
pub struct HalRustExecutable {
    function_count: usize,
}

impl HalRustExecutable {
    /// Create a new ``HalRustExecutable``.
    ///
    /// ``function_count`` is the number of functions in the model's
    /// compute graph (typically 28 for a KV-cache model).
    pub fn new(function_count: usize) -> Self {
        Self { function_count }
    }

    // ── Op handlers ───────────────────────────────────────────────

    /// Handle gather op (both 2-input embedding lookup and 3-input sf.index).
    fn handle_gather(
        &self,
        inputs: &[SfaMemRef],
        outputs: &mut [SfaMemRef],
        meta: &crate::hal::hal_ops_cpu::OpShapeMeta,
    ) -> Result<(), anyhow::Error> {
        if inputs.len() < 2 {
            return Err(anyhow::anyhow!("gather: need at least 2 inputs"));
        }
        let out_sfa = outputs.first().ok_or_else(|| anyhow::anyhow!("gather: no output buffer"))?;

        if inputs.len() >= 3 {
            // sf.index: 3-input gather (data, batch_idx, position_idx)
            let data_sfa = &inputs[0];
            let idx0_sfa = &inputs[1];
            let idx1_sfa = &inputs[2];

            // SAFETY: Data buffer contains f32 tensor from SSA map.
            let data_slice = unsafe { sfa_as_f32_slice(data_sfa) };
            // SAFETY: Output buffer pre-allocated by runner.
            let out_slice = unsafe { sfa_as_f32_mut(out_sfa) };

            let data_shape = data_sfa.sizes();
            let rank = data_shape.len();

            let idx0: usize = if idx0_sfa.element_size() == 8 {
                // SAFETY: Index buffer is i64 (8 bytes/elem).
                let bytes = unsafe { sfa_as_bytes(idx0_sfa) };
                if bytes.len() >= 8 { i64::from_le_bytes(bytes[..8].try_into().unwrap_or([0;8])) as usize } else { 0 }
            } else {
                // SAFETY: Index buffer is f32 (4 bytes/elem).
                let slice = unsafe { sfa_as_f32_slice(idx0_sfa) };
                if !slice.is_empty() { slice[0] as usize } else { 0 }
            };

            let idx1: usize = if idx1_sfa.element_size() == 8 {
                // SAFETY: Index buffer is i64 (8 bytes/elem).
                let bytes = unsafe { sfa_as_bytes(idx1_sfa) };
                if bytes.len() >= 8 { i64::from_le_bytes(bytes[..8].try_into().unwrap_or([0;8])) as usize } else { 0 }
            } else {
                // SAFETY: Index buffer is f32 (4 bytes/elem).
                let slice = unsafe { sfa_as_f32_slice(idx1_sfa) };
                if !slice.is_empty() { slice[0] as usize } else { 0 }
            };

            if rank >= 2 {
                let inner: usize = data_shape[1..rank-1].iter().product();
                let last = data_shape[rank - 1];
                let src_base = idx0 * inner * last + idx1;
                for i in 0..inner {
                    let src_off = src_base + i * last;
                    let dst_off = i * last;
                    if src_off + last <= data_slice.len() && dst_off + last <= out_slice.len() {
                        out_slice[dst_off..dst_off + last]
                            .copy_from_slice(&data_slice[src_off..src_off + last]);
                    }
                }
            }
        } else {
            // Standard 2-input gather (embedding lookup)
            let weight_sfa = &inputs[0];
            let indices_sfa = &inputs[1];

            let embed_dim = meta.input_shapes.first()
                .map(|s| s.iter().skip(1).map(|&d| d as usize).product())
                .unwrap_or(768);

            // SAFETY: Weight buffer is f32 embedding table.
            let weight_slice = unsafe { sfa_as_f32_slice(weight_sfa) };
            // SAFETY: Indices buffer may be i64 or f32; accessed as raw bytes.
            let indices_bytes = unsafe { sfa_as_bytes(indices_sfa) };
            // SAFETY: Output buffer pre-allocated by runner.
            let out_slice = unsafe { sfa_as_f32_mut(out_sfa) };

            let index_dtype = if indices_sfa.element_size() == 8 {
                crate::model::tensor::Dtype::I64
            } else {
                crate::model::tensor::Dtype::F32
            };

            eprintln!(
                "[gather_debug] indices: element_size={}, numel={}, bytes={}, dtype={:?}, shape={:?}",
                indices_sfa.element_size(),
                indices_sfa.numel(),
                indices_sfa.byte_len(),
                index_dtype,
                indices_sfa.sizes(),
            );

            crate::hal::primitives::gather_from_bytes(
                weight_slice, indices_bytes, out_slice, embed_dim, index_dtype,
            ).map_err(|e| anyhow::anyhow!("{}", e))?;
        }

        Ok(())
    }

    /// Handle reshape op — byte-level copy preserving raw data.
    fn handle_reshape(
        &self,
        inputs: &[SfaMemRef],
        outputs: &mut [SfaMemRef],
    ) -> Result<(), anyhow::Error> {
        if inputs.is_empty() {
            return Err(anyhow::anyhow!("reshape: no input"));
        }
        let in_sfa = &inputs[0];
        let out_sfa = outputs.first().ok_or_else(|| anyhow::anyhow!("reshape: no output buffer"))?;

        // SAFETY: Input buffer accessed as raw bytes for byte-level copy.
        let in_bytes = unsafe { sfa_as_bytes(in_sfa) };
        // SAFETY: Output buffer accessed as mutable bytes for byte-level copy.
        let out_bytes = unsafe { sfa_as_mut_bytes(out_sfa) };

        if in_bytes.len() != out_bytes.len() {
            return Err(anyhow::anyhow!(
                "reshape: numel mismatch: input {} bytes != output {} bytes (in_shape={:?}, out_shape={:?})",
                in_bytes.len(), out_bytes.len(),
                in_sfa.sizes(), out_sfa.sizes(),
            ));
        }
        out_bytes.copy_from_slice(in_bytes);

        Ok(())
    }

    /// Handle layer_norm op (fused RMS norm).
    fn handle_layer_norm(
        &self,
        inputs: &[SfaMemRef],
        outputs: &mut [SfaMemRef],
        meta: &crate::hal::hal_ops_cpu::OpShapeMeta,
    ) -> Result<(), anyhow::Error> {
        if inputs.len() < 2 {
            return Err(anyhow::anyhow!("layer_norm: need at least 2 inputs"));
        }
        let in_sfa = &inputs[0];
        let weight_sfa = &inputs[1];
        let out_sfa = outputs.first().ok_or_else(|| anyhow::anyhow!("layer_norm: no output buffer"))?;

        // SAFETY: Input buffer is f32 tensor from SSA map.
        let in_slice = unsafe { sfa_as_f32_slice(in_sfa) };
        // SAFETY: Weight buffer is f32 layer norm parameters.
        let weight_slice = unsafe { sfa_as_f32_slice(weight_sfa) };
        // SAFETY: Output buffer pre-allocated by runner.
        let out_slice = unsafe { sfa_as_f32_mut(out_sfa) };

        let cols = meta.input_shapes.first()
            .map(|s| s.last().copied().unwrap_or(768) as usize)
            .unwrap_or(768);

        crate::hal::primitives::fused_rms_norm(in_slice, out_slice, weight_slice, cols, 1e-5);
        Ok(())
    }

    /// Handle linear/matmul op via BLAS.
    fn handle_linear_matmul(
        &self,
        inputs: &[SfaMemRef],
        outputs: &mut [SfaMemRef],
        meta: &crate::hal::hal_ops_cpu::OpShapeMeta,
    ) -> Result<(), anyhow::Error> {
        if inputs.len() < 2 {
            return Err(anyhow::anyhow!("linear: need at least 2 inputs"));
        }
        let in_sfa = &inputs[0];
        let weight_sfa = &inputs[1];
        let out_sfa = outputs.first().ok_or_else(|| anyhow::anyhow!("linear: no output buffer"))?;

        let a_shape = meta.input_shapes.first().cloned().unwrap_or_default();
        let b_shape = meta.input_shapes.get(1).cloned().unwrap_or_default();

        let in_slice = unsafe { sfa_as_f32_slice(in_sfa) };
        let weight_slice = unsafe { sfa_as_f32_slice(weight_sfa) };
        let out_slice = unsafe { sfa_as_f32_mut(out_sfa) };

        let total_in = a_shape.iter().product::<i64>() as usize;
        let total_w = b_shape.iter().product::<i64>() as usize;
        if in_slice.len() < total_in || weight_slice.len() < total_w {
            return Err(anyhow::anyhow!(
                "linear: shape mismatch — activation shape={:?} ({} elements, slice has {}), weight shape={:?} ({} elements, slice has {})",
                a_shape, total_in, in_slice.len(), b_shape, total_w, weight_slice.len(),
            ));
        }

        crate::hal::primitives::matmul_blas(in_slice, weight_slice, out_slice, &a_shape, &b_shape, true)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        Ok(())
    }

    /// Handle scaled_dot_product_attention op.
    fn handle_sdpa(
        &self,
        inputs: &[SfaMemRef],
        outputs: &mut [SfaMemRef],
        meta: &crate::hal::hal_ops_cpu::OpShapeMeta,
    ) -> Result<(), anyhow::Error> {
        if inputs.len() < 4 {
            return Err(anyhow::anyhow!("sdpa: need at least 4 inputs"));
        }
        let q_sfa = &inputs[0];
        let k_sfa = &inputs[1];
        let v_sfa = &inputs[2];
        let mask_sfa = &inputs[3];
        let out_sfa = outputs.first().ok_or_else(|| anyhow::anyhow!("sdpa: no output buffer"))?;

        // SAFETY: Input buffers are f32 tensors from SSA map.
        let q_slice = unsafe { sfa_as_f32_slice(q_sfa) };
        let k_slice = unsafe { sfa_as_f32_slice(k_sfa) };
        let v_slice = unsafe { sfa_as_f32_slice(v_sfa) };
        let mask_slice = unsafe { sfa_as_f32_slice(mask_sfa) };
        // SAFETY: Output buffer pre-allocated by runner.
        let out_slice = unsafe { sfa_as_f32_mut(out_sfa) };

        let q_shape = meta.input_shapes.first().cloned().unwrap_or_default();
        let k_shape = meta.input_shapes.get(1).cloned().unwrap_or_default();
        let v_shape = meta.input_shapes.get(2).cloned().unwrap_or_default();
        let mask_shape = meta.input_shapes.get(3).cloned().unwrap_or_default();

        crate::hal::primitives::fused_sdpa(
            q_slice, k_slice, v_slice, mask_slice, out_slice,
            &q_shape, &k_shape, &v_shape, &mask_shape,
        )
        .map_err(|e| anyhow::anyhow!("{}", e))?;
        Ok(())
    }

    /// Handle element_wise:* ops with f32 or i64 inputs, with broadcasting.
    fn handle_element_wise(
        &self,
        inputs: &[SfaMemRef],
        outputs: &mut [SfaMemRef],
        op_name: &str,
    ) -> Result<(), anyhow::Error> {
        if inputs.len() < 2 {
            return Err(anyhow::anyhow!("element_wise: need at least 2 inputs"));
        }
        let a_sfa = &inputs[0];
        let b_sfa = &inputs[1];
        let out_sfa = outputs.first().ok_or_else(|| anyhow::anyhow!("element_wise: no output buffer"))?;
        let kind = op_name.strip_prefix("element_wise:").unwrap_or("add");

        // Check if inputs are i64 or f32
        let is_i64 = a_sfa.element_size() == 8 && b_sfa.element_size() == 8;
        let is_f32 = a_sfa.element_size() == 4 && b_sfa.element_size() == 4;
        if is_i64 {
            // SAFETY: Buffers are valid for the lifetime of inputs/outputs refs.
            let a_bytes = unsafe { sfa_as_bytes(a_sfa) };
            let b_bytes = unsafe { sfa_as_bytes(b_sfa) };
            // SAFETY: Output buffer is pre-allocated by the runner.
            let out_bytes = unsafe { sfa_as_mut_bytes(out_sfa) };

            let num_elems = a_bytes.len().max(b_bytes.len()) / 8;
            let a_scalar = a_bytes.len() / 8 == 1;
            let b_scalar = b_bytes.len() / 8 == 1;

            for i in 0..num_elems {
                let a_idx = if a_scalar { 0 } else { i };
                let b_idx = if b_scalar { 0 } else { i };
                let a_val = i64::from_le_bytes(a_bytes[a_idx*8..(a_idx+1)*8].try_into().unwrap_or([0; 8]));
                let b_val = i64::from_le_bytes(b_bytes[b_idx*8..(b_idx+1)*8].try_into().unwrap_or([0; 8]));
                let result = match kind {
                    "add" => a_val.wrapping_add(b_val),
                    "sub" => a_val.wrapping_sub(b_val),
                    "mul" => a_val.wrapping_mul(b_val),
                    _ => a_val,
                };
                let result_bytes = result.to_le_bytes();
                out_bytes[i*8..(i+1)*8].copy_from_slice(&result_bytes);
            }
            return Ok(());
        }

        // Handle f32 element_wise ops with broadcasting
        if is_f32 {
            let a_slice = unsafe { sfa_as_f32_slice(a_sfa) };
            let b_slice = unsafe { sfa_as_f32_slice(b_sfa) };
            let out_slice = unsafe { sfa_as_f32_mut(out_sfa) };
            let a_scalar = a_slice.len() == 1;
            let b_scalar = b_slice.len() == 1;
            let num_elems = out_slice.len();
            let a_len = a_slice.len();
            let b_len = b_slice.len();
            let out_shape = out_sfa.sizes();
            let b_idx_of = {
                if b_scalar {
                    None
                } else if b_len == num_elems {
                    None
                } else if b_len >= num_elems {
                    Some(Box::new(move |i: usize| i % b_len) as Box<dyn Fn(usize) -> usize>)
                } else {
                    let b_shape = b_sfa.sizes();
                    let group = num_elems / b_len;
                    let is_suffix = {
                        let b_ndim = b_shape.len();
                        let out_ndim = out_shape.len();
                        b_ndim <= out_ndim && (0..b_ndim).all(|d| {
                            let od = out_ndim - b_ndim + d;
                            b_shape[d] == 1 || b_shape[d] == out_shape[od]
                        })
                    };
                    if is_suffix { Some(Box::new(move |i: usize| i % b_len) as Box<dyn Fn(usize) -> usize>) }
                    else { Some(Box::new(move |i: usize| i / group) as Box<dyn Fn(usize) -> usize>) }
                }
            };
            let a_idx_of = {
                if a_scalar || a_len == num_elems {
                    None
                } else if a_len >= num_elems {
                    Some(Box::new(move |i: usize| i % a_len) as Box<dyn Fn(usize) -> usize>)
                } else {
                    let a_shape = a_sfa.sizes();
                    let group = num_elems / a_len;
                    let is_suffix = {
                        let a_ndim = a_shape.len();
                        let out_ndim = out_shape.len();
                        a_ndim <= out_ndim && (0..a_ndim).all(|d| {
                            let od = out_ndim - a_ndim + d;
                            a_shape[d] == 1 || a_shape[d] == out_shape[od]
                        })
                    };
                    if is_suffix { Some(Box::new(move |i: usize| i % a_len) as Box<dyn Fn(usize) -> usize>) }
                    else { Some(Box::new(move |i: usize| i / group) as Box<dyn Fn(usize) -> usize>) }
                }
            };

            for i in 0..num_elems {
                let a_idx = if a_scalar { 0 }
                    else if let Some(ref f) = a_idx_of { f(i) }
                    else { i };
                let b_idx = if b_scalar { 0 }
                    else if let Some(ref f) = b_idx_of { f(i) }
                    else { i };
                let result = match kind {
                    "add" => a_slice[a_idx] + b_slice[b_idx],
                    "sub" => a_slice[a_idx] - b_slice[b_idx],
                    "mul" => a_slice[a_idx] * b_slice[b_idx],
                    "div" => a_slice[a_idx] / b_slice[b_idx],
                    _ => a_slice[a_idx],
                };
                out_slice[i] = result;
            }
            return Ok(());
        }

        // Neither i64 nor f32 — fall through as unsupported
        Err(anyhow::anyhow!("element_wise: unsupported element size {}", a_sfa.element_size()))
    }

    /// Handle transpose op (plain or with axis pairs like transpose:1,2).
    fn handle_transpose(
        &self,
        inputs: &[SfaMemRef],
        outputs: &mut [SfaMemRef],
        op_name: &str,
    ) -> Result<(), anyhow::Error> {
        let in_sfa = inputs.first().ok_or_else(|| anyhow::anyhow!("transpose: no input"))?;
        let out_sfa = outputs.first().ok_or_else(|| anyhow::anyhow!("transpose: no output"))?;

        // SAFETY: Input buffer contains f32 tensor from SSA map.
        let input_slice = unsafe { sfa_as_f32_slice(in_sfa) };
        // SAFETY: Output buffer is pre-allocated as f32 by the runner.
        let out_slice = unsafe { sfa_as_f32_mut(out_sfa) };

        let input_shape: Vec<i64> = in_sfa.sizes_i64();
        let output_shape: Vec<i64> = out_sfa.sizes_i64();

        let rank = input_shape.len();
        let perm: Vec<usize> = if let Some(kind_str) = op_name.strip_prefix("transpose:") {
            // Parse axis pairs from "transpose:1,2" and build full permutation.
            let dims: Vec<usize> = kind_str.split(',')
                .filter_map(|s| s.trim().parse::<usize>().ok())
                .collect();
            let mut p: Vec<usize> = (0..rank).collect();
            for pair in dims.chunks(2) {
                if pair.len() == 2 && pair[0] < rank && pair[1] < rank {
                    p.swap(pair[0], pair[1]);
                }
            }
            p
        } else {
            let mut p: Vec<usize> = (0..rank).collect();
            p.swap(rank - 2, rank - 1);
            p
        };

        crate::hal::primitives::transpose_nd(input_slice, out_slice, &input_shape, &output_shape, &perm)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        Ok(())
    }

    /// Handle scan:cumsum op (prefix sum).
    fn handle_scan_cumsum(
        &self,
        inputs: &[SfaMemRef],
        outputs: &mut [SfaMemRef],
    ) -> Result<(), anyhow::Error> {
        let in_sfa = inputs.first().ok_or_else(|| anyhow::anyhow!("scan: no input"))?;
        let out_sfa = outputs.first().ok_or_else(|| anyhow::anyhow!("scan: no output"))?;
        let input_slice = unsafe { sfa_as_f32_slice(in_sfa) };
        let out_slice = unsafe { sfa_as_f32_mut(out_sfa) };
        let mut running = 0.0f32;
        for i in 0..input_slice.len().min(out_slice.len()) {
            running += input_slice[i];
            out_slice[i] = running;
        }
        Ok(())
    }

    /// Handle softmax op.
    fn handle_softmax(
        &self,
        inputs: &[SfaMemRef],
        outputs: &mut [SfaMemRef],
    ) -> Result<(), anyhow::Error> {
        let in_sfa = inputs.first().ok_or_else(|| anyhow::anyhow!("softmax: no input"))?;
        let out_sfa = outputs.first().ok_or_else(|| anyhow::anyhow!("softmax: no output"))?;
        let input_slice = unsafe { sfa_as_f32_slice(in_sfa) };
        let out_slice = unsafe { sfa_as_f32_mut(out_sfa) };
        let shape = in_sfa.sizes();
        let last_dim = *shape.last().unwrap_or(&1);
        crate::hal::primitives::fused_softmax(input_slice, out_slice, last_dim);
        Ok(())
    }

    /// Handle element_wise:rsqrt op (1/sqrt(x)).
    fn handle_rsqrt(
        &self,
        inputs: &[SfaMemRef],
        outputs: &mut [SfaMemRef],
    ) -> Result<(), anyhow::Error> {
        let in_sfa = inputs.first().ok_or_else(|| anyhow::anyhow!("rsqrt: no input"))?;
        let out_sfa = outputs.first().ok_or_else(|| anyhow::anyhow!("rsqrt: no output"))?;
        let input_slice = unsafe { sfa_as_f32_slice(in_sfa) };
        let out_slice = unsafe { sfa_as_f32_mut(out_sfa) };
        for i in 0..input_slice.len().min(out_slice.len()) {
            out_slice[i] = 1.0 / input_slice[i].sqrt();
        }
        Ok(())
    }

    /// Handle reduce:mean op.
    fn handle_reduce_mean(
        &self,
        inputs: &[SfaMemRef],
        outputs: &mut [SfaMemRef],
        meta: &crate::hal::hal_ops_cpu::OpShapeMeta,
    ) -> Result<(), anyhow::Error> {
        let in_sfa = inputs.first().ok_or_else(|| anyhow::anyhow!("reduce: no input"))?;
        let out_sfa = outputs.first().ok_or_else(|| anyhow::anyhow!("reduce: no output"))?;
        let input_slice = unsafe { sfa_as_f32_slice(in_sfa) };
        let out_slice = unsafe { sfa_as_f32_mut(out_sfa) };
        let in_shape = meta.input_shapes.first().cloned().unwrap_or_else(|| vec![1]);
        let last_dim = *in_shape.last().unwrap_or(&1) as usize;
        if last_dim > 0 {
            let num_groups = input_slice.len() / last_dim;
            for g in 0..num_groups.min(out_slice.len()) {
                let start = g * last_dim;
                let end = (start + last_dim).min(input_slice.len());
                let group = &input_slice[start..end];
                if !group.is_empty() {
                    out_slice[g] = group.iter().sum::<f32>() / group.len() as f32;
                }
            }
        }
        Ok(())
    }

    /// Handle fill:arange op — writes i64 values when
    /// output dtype is i64 (declared in op output_dtypes), f32 otherwise.
    fn handle_fill_arange(
        &self,
        inputs: &[SfaMemRef],
        outputs: &mut [SfaMemRef],
    ) -> Result<(), anyhow::Error> {
        let _ = inputs;
        let out_sfa = outputs.first().ok_or_else(|| anyhow::anyhow!("fill:arange: no output"))?;
        if out_sfa.element_size() == 8 {
            // SAFETY: Output buffer is pre-allocated as i64 (8 bytes/elem) by runner.
            let out_bytes = unsafe { sfa_as_mut_bytes(out_sfa) };
            for (i, chunk) in out_bytes.chunks_mut(8).enumerate() {
                let val = i as i64;
                chunk.copy_from_slice(&val.to_le_bytes());
            }
        } else {
            // SAFETY: Output buffer is f32 (4 bytes/elem).
            let out_slice = unsafe { sfa_as_f32_mut(out_sfa) };
            for (i, o) in out_slice.iter_mut().enumerate() {
                *o = i as f32;
            }
        }
        Ok(())
    }

    /// Default dispatch path: KernelOp trait registry first, then generated dispatch fallback.
    fn handle_default(
        &self,
        op_name: &str,
        inputs: &[SfaMemRef],
        outputs: &mut [SfaMemRef],
        meta: &crate::hal::hal_ops_cpu::OpShapeMeta,
    ) -> Result<(), anyhow::Error> {
        let registry = crate::hal::primitives::traits::kernel_registry();
        if let Some(kernel) = registry.get(op_name) {
            let input_slices: Vec<&[f32]> = inputs
                .iter()
                // SAFETY: All inputs are f32 buffers (element_size == 4).
                .map(|sfa| unsafe { sfa_as_f32_slice(sfa) })
                .collect();

            if let Some(out_sfa) = outputs.first() {
                // SAFETY: Output buffer is pre-allocated as f32 by the runner.
                let out_slice = unsafe { sfa_as_f32_mut(out_sfa) };
                kernel.execute_typed(&input_slices, out_slice, meta)
                    .map_err(|e| anyhow::anyhow!("{}", e))?;
            } else {
                kernel.execute_typed(&input_slices, &mut [], meta)
                    .map_err(|e| anyhow::anyhow!("{}", e))?;
            }
        } else {
            // Legacy fallback for ops not yet in the KernelOp registry.
            let input_slices: Vec<&[f32]> = inputs
                .iter()
                // SAFETY: All inputs are f32 buffers (element_size == 4).
                .map(|sfa| unsafe { sfa_as_f32_slice(sfa) })
                .collect();

            if let Some(out_sfa) = outputs.first() {
                // SAFETY: Output buffer is pre-allocated as f32 by the runner.
                let out_slice = unsafe { sfa_as_f32_mut(out_sfa) };
                crate::hal::hal_ops_cpu::dispatch(op_name, &input_slices, out_slice, meta)
                    .map_err(|e| anyhow::anyhow!("{}", e))?;
            } else {
                crate::hal::hal_ops_cpu::dispatch(op_name, &input_slices, &mut [], meta)
                    .map_err(|e| anyhow::anyhow!("{}", e))?;
            }
        }
        Ok(())
    }
}

impl traits::Executable for HalRustExecutable {
    fn execute(
        &self,
        op_name: &str,
        _stream: &dyn traits::Stream,
        inputs: &[SfaMemRef],
        outputs: &mut [SfaMemRef],
    ) -> Result<Vec<Vec<i64>>, anyhow::Error> {
        let output_shapes: Vec<Vec<i64>> = outputs
            .iter()
            .map(|m| m.sizes_i64())
            .collect();
        let meta = build_shape_meta_from_sfa(inputs, outputs);

        match op_name {
            "gather" => self.handle_gather(inputs, outputs, &meta)?,
            "reshape" => self.handle_reshape(inputs, outputs)?,
            "layer_norm" => self.handle_layer_norm(inputs, outputs, &meta)?,
            "linear" | "matmul" => self.handle_linear_matmul(inputs, outputs, &meta)?,
            "scaled_dot_product_attention" => self.handle_sdpa(inputs, outputs, &meta)?,
            "scan:cumsum" => self.handle_scan_cumsum(inputs, outputs)?,
            "softmax" => self.handle_softmax(inputs, outputs)?,
            "element_wise:rsqrt" => self.handle_rsqrt(inputs, outputs)?,
            "reduce:mean" => self.handle_reduce_mean(inputs, outputs, &meta)?,
            "fill:arange" => self.handle_fill_arange(inputs, outputs)?,
            "cache_read" | "cache_write" => {
                // no-op: shapes are returned unchanged
            }
            op if op.starts_with("element_wise:") => self.handle_element_wise(inputs, outputs, op)?,
            op if op.starts_with("transpose") => self.handle_transpose(inputs, outputs, op)?,
            _ => self.handle_default(op_name, inputs, outputs, &meta)?,
        }
        Ok(output_shapes)
    }

    fn function_count(&self) -> usize {
        self.function_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hal::traits;
    use crate::hal::traits::Executable as _;

    /// A minimal buffer backed by a Vec<f32> for testing.
    #[derive(Debug)]
    struct TestBuf(Vec<u8>, usize, Vec<usize>);

    impl traits::Buffer for TestBuf {
        fn as_ptr(&self) -> *const u8 { self.0.as_ptr() }
        fn as_mut_ptr(&mut self) -> *mut u8 { self.0.as_mut_ptr() }
        fn len(&self) -> usize { self.0.len() }
        fn copy_from_host(&mut self, src: &[u8], _: &dyn traits::Stream) -> Result<(), anyhow::Error> {
            self.0.copy_from_slice(src);
            Ok(())
        }
        fn copy_to_host(&self, dst: &mut [u8], _: &dyn traits::Stream) -> Result<(), anyhow::Error> {
            dst.copy_from_slice(&self.0);
            Ok(())
        }
        fn element_size(&self) -> usize { self.1 }
        fn shape(&self) -> Vec<usize> { self.2.clone() }
        fn rank(&self) -> u8 { self.2.len() as u8 }
    }

    #[derive(Debug)]
    struct NoopStream;
    impl traits::Stream for NoopStream {
        fn synchronize(&self) -> Result<(), anyhow::Error> { Ok(()) }
        fn wait_event(&self, _: &dyn traits::Event) -> Result<(), anyhow::Error> { Ok(()) }
        fn record_event(&self, _: &dyn traits::Event) -> Result<(), anyhow::Error> { Ok(()) }
    }

    #[test]
    fn test_hal_rust_executable_new() {
        let exe = HalRustExecutable::new(28);
        assert_eq!(exe.function_count, 28);
    }

    #[test]
    fn test_hal_rust_executable_function_count() {
        let exe = HalRustExecutable::new(16);
        assert_eq!(exe.function_count(), 16);

        let exe2 = HalRustExecutable::new(28);
        assert_eq!(exe2.function_count(), 28);
    }

    #[test]
    fn test_hal_rust_executable_execute_unknown_op() {
        let exe = HalRustExecutable::new(1);
        let stream = NoopStream;
        let result = exe.execute("nonexistent_op", &stream, &[], &mut []);
        assert!(result.is_err(), "unknown op should return error");
        assert!(
            result.unwrap_err().to_string().contains("unknown op"),
            "error message should mention 'unknown op'"
        );
    }

    #[test]
    fn test_hal_rust_executable_cache_ops_noop() {
        // cache_read and cache_write are no-op stubs that return output shapes.
        let exe = HalRustExecutable::new(1);
        let stream = NoopStream;
        let input_vec = vec![0u8; 16];
        let in_ptr = input_vec.as_ptr() as *mut std::ffi::c_void;
        // Leak to keep the pointer valid; drop after the test.
        std::mem::forget(input_vec);
        let out_vec = vec![0u8; 16];
        let out_ptr = out_vec.as_ptr() as *mut std::ffi::c_void;
        std::mem::forget(out_vec);

        let inputs = [SfaMemRef::r1(in_ptr, [4], [1], 4)];
        let mut outputs = [SfaMemRef::r1(out_ptr, [4], [1], 4)];

        let result = exe.execute("cache_read", &stream, &inputs, &mut outputs);
        assert!(result.is_ok(), "cache_read should be a no-op");
        let shapes = result.unwrap();
        assert_eq!(shapes, vec![vec![4i64]]);

        let result2 = exe.execute("cache_write", &stream, &inputs, &mut outputs);
        assert!(result2.is_ok(), "cache_write should be a no-op");

        // Reclaim leaked memory.
        unsafe {
            let _ = Vec::from_raw_parts(in_ptr as *mut u8, 16, 16);
            let _ = Vec::from_raw_parts(out_ptr as *mut u8, 16, 16);
        }
    }
}
