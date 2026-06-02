//! Utility functions for the HAL IR graph runner.
//!
//! These helpers are used by `run_hal_function_graph` for shape estimation,
//! output buffer allocation, error recovery, and cross-function wiring.

use std::collections::HashMap;

use crate::hal::traits;
use crate::sfa_tensor::SFATensor;
use crate::weight_loader::WeightProvider;
use crate::hal_runner::types::{HalFunction, HalIR, HalTensorDef};

// ── Helpers ────────────────────────────────────────────────────────────

/// Zero-fill output buffers after a failed op execution.
#[allow(dead_code)]
pub(super) fn zero_fill_outputs(
    output_vecs: &mut [Vec<u8>],
    output_bufs: &[Box<dyn traits::Buffer>],
    fi: usize,
    oi: usize,
    _op_name: &str,
) -> Vec<Vec<i64>> {
    for (idx, out_vec) in output_vecs.iter_mut().enumerate() {
        out_vec.fill(0);
        log::trace!(
            "hal_runner: zero-filled func[{}] op[{}] output[{}] ({} bytes)",
            fi, oi, idx, out_vec.len(),
        );
    }
    output_bufs
        .iter()
        .map(|b| b.shape().iter().map(|&d| d as i64).collect())
        .collect()
}

/// Estimate the number of f32 elements for a tensor with the given
/// HAL IR shape representation (strings with "?" for dynamic dims).
///
/// First "?" = batch (always 1), subsequent "?" = sequence length.
pub(super) fn estimate_numel_from_shape(shape: &[String], seq_len: usize) -> usize {
    let mut numel: usize = 1;
    let mut first_dyn = true;
    for d in shape {
        let dim = if d == "?" || d == "-1" {
            if first_dyn {
                first_dyn = false;
                1 // batch
            } else {
                seq_len
            }
        } else {
            d.parse::<usize>().unwrap_or(1)
        };
        numel = numel.saturating_mul(dim);
    }
    // Generous minimum to accommodate intermediate tensors whose
    // declared shapes in the function output list are unreliable
    // (e.g., shape_of output declared as [1] but actually [rank],
    //  gather output declared as [1] but actually [N × embed_dim]).
    numel.max(65536) // 64K elements = 256 KB
}

/// Find the main (wire) output of a HAL function for cross-function wiring.
///
/// The wire is the hidden state output that gets passed to the next
/// function as `%arg0`.  It is identified by:
///   a) NOT being `consumed_internally` (excludes KV cache intermediates)
///   b) having at least one dynamic dimension ("?")
///   c) having rank >= 2 (excludes scalars and offsets)
///   d) preferring rank-3 shapes `[?, ?, X]` typical of hidden states
///
/// Returns `Some((ssa_name, data_bytes))` when found, `None` on failure.
pub(super) fn find_main_output(
    function: &HalFunction,
    ssa_map: &HashMap<String, SFATensor>,
) -> Option<(String, SFATensor)> {
    let mut best_score: i64 = -1;
    let mut best: Option<(&HalTensorDef, &SFATensor)> = None;

    for output in &function.outputs {
        // Skip internally-consumed tensors (KV cache intermediates).
        if output.consumed_internally {
            continue;
        }
        let dyn_count = output.shape.iter().filter(|d| *d == "?" || *d == "-1").count();
        if dyn_count == 0 {
            continue; // skip fully static outputs (weights, constants)
        }
        let rank = output.shape.len();
        if rank < 2 {
            continue; // skip scalars and 1D offsets
        }

        // Score: prefer higher rank, more "?" dims, and rank 3 being ideal.
        let score = (rank as i64) * 10 + (dyn_count as i64) + if rank == 3 { 100 } else { 0 };

        if let Some(tensor) = ssa_map.get(&output.name) {
            if score > best_score {
                best_score = score;
                best = Some((output, tensor));
            }
        }
    }

    best.map(|(output, tensor)| (output.name.clone(), tensor.clone_data()))
}

/// Fallback shape builder for op outputs: converts the declared shape
/// strings to a `Vec<usize>`, substituting `"?` and `"-1"` with 1 (batch).
/// Uses `numel` as the flat fallback when the shape is all-dynamic.
#[allow(dead_code)]
pub(super) fn fallback_shape(shape: &[String], numel: usize) -> Vec<usize> {
    let dims: Vec<usize> = shape
        .iter()
        .map(|d| {
            if d == "?" || d == "-1" {
                1
            } else {
                d.parse::<usize>().unwrap_or(1)
            }
        })
        .collect();
    if dims.is_empty() || dims.iter().all(|&d| d == 0) {
        vec![numel]
    } else {
        dims
    }
}

/// Find the shape definition for a tensor name in a function's
/// input, output, or weight list.
#[allow(dead_code)]
pub(super) fn find_output_shape(function: &HalFunction, name: &str) -> (Vec<String>, bool) {
    for output in &function.outputs {
        if output.name == name {
            return (output.shape.clone(), false);
        }
    }
    for input in &function.inputs {
        if input.name == name {
            return (input.shape.clone(), false);
        }
    }
    // Check weight list for invisible constant SSAs (e.g. %1 for position emb).
    for weight_entry in &function.weights {
        if weight_entry.ssa == name {
            return (weight_entry.shape.clone(), false);
        }
    }
    (vec!["?".to_string()], false)
}

/// Find the shape for any SSA name in a function (inputs + outputs).
#[allow(dead_code)]
pub(super) fn find_any_shape(function: &HalFunction, name: &str) -> Vec<String> {
    find_output_shape(function, name).0
}

/// Try to find a matching tensor from prior function outputs for the
/// given function input definition.  Uses shape matching when names
/// don't directly correspond (the common case across functions).
#[allow(dead_code)]
pub(super) fn find_matching_output(
    hal_ir: &HalIR,
    _current_func: &HalFunction,
    input_def: &HalTensorDef,
    ssa_map: &HashMap<String, SFATensor>,
    seq_len: usize,
) -> Option<SFATensor> {
    let input_numel = estimate_numel_from_shape(&input_def.shape, seq_len);

    // Search all prior functions' outputs for a tensor with matching
    // numel and compatible shape.
    for func in &hal_ir.functions {
        for output in &func.outputs {
            if output.name == input_def.name {
                // Same name — direct match (should already be in map).
                return ssa_map.get(&output.name).map(|t| t.clone_data());
            }

            let output_numel = estimate_numel_from_shape(&output.shape, seq_len);

            // Check if the output is in the SSA map and has matching size.
            if output_numel == input_numel && ssa_map.contains_key(&output.name) {
                // Also check shape compatibility (same rank, or both 1D).
                if output.shape.len() == input_def.shape.len() {
                    return ssa_map.get(&output.name).map(|t| t.clone_data());
                }
            }
        }
    }

    None
}

// ── compute_output_shape ──────────────────────────────────────────────

/// Compute the output buffer numel and shape for an op from its input
/// shapes and semantics, rather than from the function's declared output
/// shape (which is often wrong — e.g. `[1]` instead of `[1,4,768]`).
///
/// This is the core fix for the two-phase shape problem:
///   1. Phase 1 (op execution) → `execute()` returns output shapes → stored in `ssa_shapes`.
///   2. Phase 2 (next op's input resolution) → looks up `ssa_shapes` for correct shapes.
///
/// The cascade starts at the FIRST op with wrong shapes.  By computing
/// numel/shape from ACTUAL INPUT SHAPES, every op gets the right buffer.
// ── Weight injection ───────────────────────────────────────────────────

/// Inject weight data for a SINGLE function from WeightProvider into the SSA map.
///
/// Called per-function inside the execution loop, AFTER cross-function wiring
/// so that wire data is not overwritten by weight injection.
pub(super) fn inject_function_weights(
    weight_provider: Option<&WeightProvider>,
    function: &HalFunction,
    ssa_map: &mut HashMap<String, SFATensor>,
    ssa_shapes: &mut HashMap<String, Vec<usize>>,
    ssa_dtypes: &mut HashMap<String, crate::tensor::Dtype>,
) {
    if let Some(wp) = weight_provider {
        // Case (a): weights list with inline SSA names.
        for weight_entry in &function.weights {
            if weight_entry.ssa.is_empty() {
                continue;
            }
            if let Some((desc, _dtype)) = wp.get_weight_memref(&weight_entry.name) {
                let n = desc.numel();
                let weight_dtype = crate::tensor::Dtype::from_hal_str(&weight_entry.dtype);
                let weight_dims: Vec<usize> = weight_entry.shape.iter().map(|d| {
                    if d == "?" || d == "-1" { 1 } else { d.parse::<usize>().unwrap_or(1) }
                }).collect();
                let t = if weight_dtype == crate::tensor::Dtype::I64 {
                    // SAFETY: desc.aligned points to valid i64 weight data.
                    let data: Vec<i64> = unsafe {
                        let raw = desc.aligned as *const i64;
                        let slice = std::slice::from_raw_parts(raw, n);
                        slice.to_vec()
                    };
                    SFATensor::from_vec_i64(data, weight_dims.clone())
                } else if weight_dtype == crate::tensor::Dtype::F32
                    && weight_entry.name.starts_with("_const_") {
                    let data: Vec<f32> = unsafe {
                        let raw = desc.aligned as *const f32;
                        let slice = std::slice::from_raw_parts(raw, n);
                        slice.to_vec()
                    };
                    SFATensor::from_vec_f32(data, weight_dims.clone())
                } else {
                    // SAFETY: desc.aligned points to valid f16 weight data.
                    let data: Vec<f32> = unsafe {
                        let raw = desc.aligned as *const u16;
                        let slice = std::slice::from_raw_parts(raw, n);
                        slice
                            .iter()
                            .map(|&h| half::f16::from_bits(h).to_f32())
                            .collect()
                    };
                    SFATensor::from_vec_f32(data, weight_dims.clone())
                };
                ssa_map.insert(weight_entry.ssa.clone(), t);
                ssa_dtypes.insert(weight_entry.ssa.clone(), weight_dtype);
                ssa_shapes.insert(weight_entry.ssa.clone(), weight_dims);
                log::debug!(
                    "hal_runner: loaded weight '{}' -> SSA '{}' ({} elements)",
                    weight_entry.name,
                    weight_entry.ssa,
                    n,
                );
            }
        }

        // Case (b): weight_inputs mapping (function args → compiled names).
        for (ssa_name, compiled_name) in &function.weight_inputs {
            if let Some((desc, _dtype)) = wp.get_weight_memref(compiled_name) {
                let n = desc.numel();
                let weight_dtype = function.inputs.iter()
                    .find(|i| i.name == *ssa_name)
                    .map(|i| crate::tensor::Dtype::from_hal_str(&i.dtype))
                    .unwrap_or(crate::tensor::Dtype::F32);
                let input_dims: Vec<usize> = function.inputs.iter()
                    .find(|i| i.name == *ssa_name)
                    .map(|input_def| {
                        input_def.shape.iter().map(|d| {
                            if d == "?" || d == "-1" { 1 } else { d.parse::<usize>().unwrap_or(1) }
                        }).collect()
                    })
                    .unwrap_or_else(|| vec![n]);
                let t = if weight_dtype == crate::tensor::Dtype::I64 {
                    // SAFETY: desc.aligned points to valid i64 weight data.
                    let data: Vec<i64> = unsafe {
                        let raw = desc.aligned as *const i64;
                        let slice = std::slice::from_raw_parts(raw, n);
                        slice.to_vec()
                    };
                    SFATensor::from_vec_i64(data, input_dims.clone())
                } else if weight_dtype == crate::tensor::Dtype::F32
                    && compiled_name.starts_with("_const_") {
                    let data: Vec<f32> = unsafe {
                        let raw = desc.aligned as *const f32;
                        let slice = std::slice::from_raw_parts(raw, n);
                        slice.to_vec()
                    };
                    SFATensor::from_vec_f32(data, input_dims.clone())
                } else {
                    // SAFETY: desc.aligned points to valid f16 weight data.
                    let data: Vec<f32> = unsafe {
                        let raw = desc.aligned as *const u16;
                        let slice = std::slice::from_raw_parts(raw, n);
                        slice
                            .iter()
                            .map(|&h| half::f16::from_bits(h).to_f32())
                            .collect()
                    };
                    SFATensor::from_vec_f32(data, input_dims.clone())
                };
                ssa_map.insert(ssa_name.clone(), t);
                ssa_dtypes.insert(ssa_name.clone(), weight_dtype);
                ssa_shapes.insert(ssa_name.clone(), input_dims);
                log::debug!(
                    "hal_runner: loaded weight '{}' -> SSA '{}' ({} elements)",
                    compiled_name,
                    ssa_name,
                    n,
                );
            } else {
                log::warn!(
                    "hal_runner: weight '{}' for SSA '{}' not found in WeightProvider",
                    compiled_name,
                    ssa_name,
                );
            }
        }
    }
}
