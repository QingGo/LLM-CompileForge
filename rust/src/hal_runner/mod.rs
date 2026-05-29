//! HAL IR graph runner — iterates over HalFunction ops, assembles inputs
//! from global inputs / weights / SSA wires, dispatches through
//! `executable.execute(op_name, stream, &input_bufs, &output_bufs)`,
//! and extracts the final output Tensor.
//!
//! Path B (pure-Rust) counterpart to `compute_graph_runner.rs`.
//! All kernel dispatch goes through the HAL Executable trait — no direct
//! ciface / lookup_typed calls.

pub mod helpers;
pub mod prep;
pub mod types;

#[cfg(test)]
mod tests;

pub use types::{HalIR, HalFunction, HalOp, HalRustRunner, HalTensorDef, HalWeightEntry};

use crate::hal::cpu::buffer::CpuBuffer as InnerCpuBuffer;
use crate::hal::cpu::CpuBuffer;
use crate::hal::traits;
use crate::tensor::Tensor;
use crate::weight_loader::WeightProvider;

use crate::hal_runner::helpers::{
    compute_output_shape, estimate_numel_from_shape, fallback_shape, find_any_shape,
    find_main_output, find_output_shape, inject_weights_into_ssa, zero_fill_outputs,
};
use crate::hal_runner::prep::{
    extract_global_output, prepare_ssa_maps, wire_cross_function_inputs,
};

// ── run_hal_function_graph ────────────────────────────────────────────

/// Execute a complete HAL IR function graph through a HAL Executable.
///
/// Iterates over every function and its ops in order, maintaining an SSA
/// value map (`HashMap<String, Vec<f32>>`) across all functions.
///
/// # Arguments
///
/// * `executable` — HAL executable that dispatches op_name → CPU kernel.
/// * `hal_ir` — parsed HAL IR (28 functions, 634 ops for opt-125m).
/// * `weight_provider` — optional weight loader for weight tensors.
///   When `Some(...)`, known weights are loaded before execution.
///   When `None`, all weights are zero-filled.
/// * `stream` — HAL stream (no-op for CPU).
/// * `input_ids` — token IDs (length = sequence length).
/// * `positions` — position IDs (length = sequence length).
///
/// # Returns
///
/// The global output tensor (typically logits from the last function).
pub fn run_hal_function_graph(
    executable: &dyn traits::Executable,
    hal_ir: &HalIR,
    weight_provider: Option<&WeightProvider>,
    stream: &dyn traits::Stream,
    input_ids: &[u32],
    positions: &[u32],
) -> Result<Tensor, anyhow::Error> {
    let seq_len = input_ids.len();

    // ── Build initial SSA maps (global inputs + zero-fill) ────────────
    let (mut ssa_map, mut ssa_shapes) =
        prepare_ssa_maps(hal_ir, seq_len, input_ids, positions);

    // ── Inject weights from WeightProvider ────────────────────────────
    inject_weights_into_ssa(weight_provider, hal_ir, &mut ssa_map, &mut ssa_shapes);

    // ── Execute each function's ops ──────────────────────────────────
    for (fi, function) in hal_ir.functions.iter().enumerate() {
        log::debug!(
            "hal_runner: executing function[{}] '{}' (layer={}, {} ops)",
            fi,
            function.name,
            function.layer,
            function.ops.len(),
        );

        // ── Cross-function wiring for functions >= 1 ────────────────
        if fi >= 1 {
            let prev_func = &hal_ir.functions[fi - 1];
            wire_cross_function_inputs(
                fi,
                function,
                prev_func,
                &mut ssa_map,
                &mut ssa_shapes,
                seq_len,
            );
        }

        // DEBUG: track time for starting function execution
        for (oi, op) in function.ops.iter().enumerate() {

            // Skip runtime-level cache ops — handled by block_manager/kv_cache.
            if op.op == "cache_read" || op.op == "cache_write" {
                log::trace!(
                    "hal_runner: func[{}] op[{}] skipping '{}' (runtime-level)",
                    fi,
                    oi,
                    op.op,
                );
                continue;
            }

            // ── Resolve input buffers from SSA map ──────────────────
            let mut input_bufs: Vec<Box<dyn traits::Buffer>> =
                Vec::with_capacity(op.inputs.len());

            for input_name in &op.inputs {
                let data = ssa_map.get(input_name).ok_or_else(|| {
                    anyhow::anyhow!(
                        "hal_runner: SSA value '{}' not found in map (func[{}] op[{}]: {:?})",
                        input_name,
                        fi,
                        oi,
                        op,
                    )
                })?;

                let elem_size = 4;

                // Look up shape from ssa_shapes FIRST.  This is the critical
                // fix for cross-function shape propagation: ssa_shapes tracks
                // tensor shapes across function boundaries (populated from
                // actual execute() output shapes and function I/O metadata).
                // Without this, tensors flowing between functions default to
                // flat 1D (e.g. [65536]) causing matmul "expected rank >= 2".
                let dims: Vec<usize> = if let Some(shape) = ssa_shapes.get(input_name) {
                    shape.clone()
                } else {
                    // Fall back to function I/O list lookup (existing behavior).
                    let mut declared_shape: Vec<String> = vec![];
                    for output in &function.outputs {
                        if output.name == *input_name {
                            declared_shape = output.shape.clone();
                            break;
                        }
                    }
                    if declared_shape.is_empty() {
                        // Look up from function input list (for %arg names).
                        for input_def in &function.inputs {
                            if input_def.name == *input_name {
                                declared_shape = input_def.shape.clone();
                                break;
                            }
                        }
                    }
                    if declared_shape.is_empty() {
                        // Look up from function weight list (invisible constants
                        // populated by weight injection, e.g. %1 for position emb).
                        for weight_entry in &function.weights {
                            if weight_entry.ssa == *input_name {
                                declared_shape = weight_entry.shape.clone();
                                break;
                            }
                        }
                    }
                    if declared_shape.is_empty() {
                        // Fall back to flat 1D from data length.
                        declared_shape = vec![(data.len() / elem_size).to_string()];
                    }
                    let dims: Vec<usize> = declared_shape
                        .iter()
                        .map(|d| {
                            if d == "?" || d == "-1" {
                                1 // batch dim for dynamic shapes
                            } else {
                                d.parse::<usize>().unwrap_or(data.len() / elem_size)
                            }
                        })
                        .collect();
                    // For buffers with no shape metadata, fall back to flat 1D.
                    if dims.is_empty() || dims.iter().all(|&d| d == 0) {
                        vec![data.len() / elem_size]
                    } else {
                        dims
                    }
                };

                let raw_buf = InnerCpuBuffer::from_raw_parts(
                    data.as_ptr() as *mut u8,
                    data.len(),
                    true, // borrowed
                )
                .map_err(|e| anyhow::anyhow!("InnerCpuBuffer: {}", e))?;

                let cpu_buf = CpuBuffer::with_meta(raw_buf, elem_size, dims);
                input_bufs.push(Box::new(cpu_buf));
            }

            // ── Pre-allocate output buffers ─────────────────────────
            let mut output_vecs: Vec<Vec<f32>> = Vec::with_capacity(op.outputs.len());
            let mut output_bufs: Vec<Box<dyn traits::Buffer>> =
                Vec::with_capacity(op.outputs.len());

            for (out_idx, output_name) in op.outputs.iter().enumerate() {
                let (numel, output_dims) = if op.op == "reshape" {
                    // Reshape: output numel = input numel (flat copy)
                    let inp_numel = op.inputs.first()
                        .and_then(|n| ssa_map.get(n))
                        .map(|d| d.len() / 4)
                        .unwrap_or(1);
                    let target_shape = op.shape.as_ref()
                        .map(|s| s.iter().map(|d| d.parse::<usize>().unwrap_or(1)).collect())
                        .unwrap_or_else(|| vec![inp_numel]);
                    (inp_numel.max(1), target_shape)
                } else {
                    compute_output_shape(
                        op, out_idx, &ssa_shapes, &ssa_map, function, seq_len,
                    )
                };
                let numel = numel.max(1);

                let mut vec = vec![0.0f32; numel];

                let raw_buf = InnerCpuBuffer::from_raw_parts(
                    vec.as_mut_ptr() as *mut u8,
                    numel * 4,
                    true, // borrowed
                )
                .map_err(|e| anyhow::anyhow!("InnerCpuBuffer: {}", e))?;

                let cpu_buf = CpuBuffer::with_meta(raw_buf, 4 /* f32 */, output_dims);
                output_vecs.push(vec);
                output_bufs.push(Box::new(cpu_buf));
            }

            // ── Execute ─────────────────────────────────────────────
            let input_refs: Vec<&dyn traits::Buffer> =
                input_bufs.iter().map(|b| b.as_ref()).collect();
            let output_refs: Vec<&dyn traits::Buffer> =
                output_bufs.iter().map(|b| b.as_ref()).collect();

            let op_name = match &op.kind {
                Some(kind) => format!("{}:{}", op.op, kind),
                None => op.op.clone(),
            };

            let exe_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                executable.execute(&op_name, stream, &input_refs, &output_refs)
            }));
            let output_shapes = match exe_result {
                Ok(Ok(shapes)) => shapes,
                Ok(Err(e)) => {
                    log::warn!(
                        "hal_runner: func[{}] op[{}] '{}' error: {}. \
                         Zero-filling and continuing.",
                        fi, oi, op_name, e,
                    );
                    zero_fill_outputs(&mut output_vecs, &output_bufs, fi, oi, &op_name)
                }
                Err(panic_payload) => {
                    let msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                        s.to_string()
                    } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "unknown panic".to_string()
                    };
                    log::warn!(
                        "hal_runner: func[{}] op[{}] '{}' PANIC: {}. \
                         Zero-filling and continuing.",
                        fi, oi, op_name, msg,
                    );
                    zero_fill_outputs(&mut output_vecs, &output_bufs, fi, oi, &op_name)
                }
            };
            log::info!(
                "hal_runner: func[{}] op[{}] '{}' DONE ({} outputs, {:?})",
                fi, oi, op_name, output_shapes.len(), output_shapes,
            );
            for (idx, output_name) in op.outputs.iter().enumerate() {
                let out_vec = std::mem::take(&mut output_vecs[idx]);
                let raw_bytes: Vec<u8> =
                    out_vec.iter().flat_map(|&v| v.to_le_bytes()).collect();

                ssa_map.insert(output_name.clone(), raw_bytes);

                if let Some(shapes) = output_shapes.get(idx) {
                    let shape_usize: Vec<usize> = shapes.iter()
                        .map(|&s| std::cmp::max(1, s as usize))
                        .collect();
                    ssa_shapes.insert(output_name.clone(), shape_usize);
                }
            }
        }

        // ── Capture wire output for cross-function wiring ─────────────
        if fi < hal_ir.functions.len().saturating_sub(1) {
            let wire = find_main_output(function, &ssa_map);
            if let Some((name, data)) = wire {
                log::debug!(
                    "hal_runner: function[{}] main wire '{}' ({} bytes) ready for next function",
                    fi,
                    name,
                    data.len(),
                );
            }
        }
    }

    // ── Extract global output ───────────────────────────────────────
    extract_global_output(hal_ir, &ssa_map, seq_len)
}
