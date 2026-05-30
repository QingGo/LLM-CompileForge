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
pub mod shape_inference;
pub mod types;

#[cfg(test)]
mod tests;

pub use types::{HalIR, HalFunction, HalOp, HalRustRunner, HalTensorDef, HalWeightEntry};

use crate::hal::cpu::buffer::CpuBuffer as InnerCpuBuffer;
use crate::hal::cpu::CpuBuffer;
use crate::hal::traits;
use crate::tensor::{Dtype, Tensor};
use crate::weight_loader::WeightProvider;

use crate::hal_runner::helpers::{
    find_main_output, inject_function_weights,
};
use crate::hal_runner::prep::{
    extract_global_output, prepare_ssa_maps, wire_cross_function_inputs,
};
use crate::hal_runner::shape_inference::compute_output_shape;

// ── dtype inference ──────────────────────────────────────────────────

fn infer_output_dtype(
    op_name: &str,
    input_names: &[String],
    ssa_dtypes: &std::collections::HashMap<String, Dtype>,
) -> Dtype {
    match op_name {
        "reshape" => input_names.first()
            .and_then(|n| ssa_dtypes.get(n))
            .copied()
            .unwrap_or(Dtype::F32),
        "gather" => Dtype::F32,
        "shape_of" => Dtype::F32,
        "fill" => Dtype::F32,
        "compare" => Dtype::F32,
        _ => input_names.first()
            .and_then(|n| ssa_dtypes.get(n))
            .copied()
            .unwrap_or(Dtype::F32),
    }
}

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
    let (mut ssa_map, mut ssa_shapes, mut ssa_dtypes) =
        prepare_ssa_maps(hal_ir, seq_len, input_ids, positions);

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
                &mut ssa_dtypes,
                seq_len,
            );
        }

        // ── Inject weights for this function (AFTER wiring) ─────────
        inject_function_weights(weight_provider, function, &mut ssa_map, &mut ssa_shapes, &mut ssa_dtypes);

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
                        input_name, fi, oi, op,
                    )
                })?;

                let elem_size = ssa_dtypes
                    .get(input_name)
                    .map(|d| d.element_size())
                    .unwrap_or(4);

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
            let mut output_vecs: Vec<Vec<u8>> = Vec::with_capacity(op.outputs.len());
            let mut output_bufs: Vec<Box<dyn traits::Buffer>> =
                Vec::with_capacity(op.outputs.len());

            // Infer output dtype from op semantics
            let output_dtype = infer_output_dtype(&op.op, &op.inputs, &ssa_dtypes);

            for (out_idx, _output_name) in op.outputs.iter().enumerate() {
                let (numel, output_dims) = compute_output_shape(
                    op, out_idx, &ssa_shapes, &ssa_map, &ssa_dtypes, function, seq_len,
                );
                let numel = numel.max(1);
                let out_elem_size = output_dtype.element_size();

                eprintln!("[alloc] func[{}] op[{}] {} out={} numel={} esize={} size={}", fi, oi, op.op, op.outputs.first().unwrap_or(&String::new()), numel, out_elem_size, numel * out_elem_size);
                let mut vec = vec![0u8; numel * out_elem_size];

                let raw_buf = InnerCpuBuffer::from_raw_parts(
                    vec.as_mut_ptr(),
                    numel * out_elem_size,
                    true,
                )
                .map_err(|e| anyhow::anyhow!("InnerCpuBuffer: {}", e))?;

                let cpu_buf = CpuBuffer::with_meta(raw_buf, out_elem_size, output_dims);
                output_vecs.push(vec);
                output_bufs.push(Box::new(cpu_buf));
            }

            // ── Execute ─────────────────────────────────────────────
            let input_refs: Vec<&dyn traits::Buffer> =
                input_bufs.iter().map(|b| b.as_ref()).collect();
            let output_refs: Vec<&dyn traits::Buffer> =
                output_bufs.iter().map(|b| b.as_ref()).collect();

            // Handle shape_of with dim: compute directly from input shape
            // instead of calling kernel (output buffer is only 1 element).
            if op.op == "shape_of" {
                if let Some(dim) = op.dim {
                    let input_shape = op.inputs.first()
                        .and_then(|n| ssa_shapes.get(n))
                        .cloned()
                        .unwrap_or_default();
                    let dim_val = if dim < input_shape.len() {
                        input_shape[dim]
                    } else {
                        1
                    };
                    let out_name = op.outputs.first().unwrap().clone();
                    let out_bytes = (dim_val as f32).to_le_bytes().to_vec();
                    ssa_map.insert(out_name.clone(), out_bytes);
                    ssa_dtypes.insert(out_name.clone(), output_dtype);
                    ssa_shapes.insert(out_name.clone(), vec![1]);
                    log::debug!(
                        "hal_runner: func[{}] op[{}] shape_of dim={} -> {} (ssa_shapes={:?})",
                        fi, oi, dim, dim_val, vec![1usize],
                    );
                    continue;
                }
            }

            let op_name = if let Some(kind) = &op.kind {
                format!("{}:{}", op.op, kind)
            } else if op.op == "transpose" {
                if let Some(ref dims) = op.dims {
                    let dims_str = dims.iter()
                        .map(|d| d.to_string())
                        .collect::<Vec<_>>()
                        .join(",");
                    format!("transpose:{}", dims_str)
                } else {
                    "transpose".to_string()
                }
            } else {
                op.op.clone()
            };

            let exe_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                executable.execute(&op_name, stream, &input_refs, &output_refs)
            }));
            let output_shapes = match exe_result {
                Ok(Ok(shapes)) => shapes,
                Ok(Err(e)) => {
                    return Err(anyhow::anyhow!(
                        "{}",
                        crate::error::HalExecutionError::OpFailed {
                            func_idx: fi,
                            op_idx: oi,
                            op_name,
                            message: e.to_string(),
                        }
                    ));
                }
                Err(panic_payload) => {
                    let msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                        s.to_string()
                    } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "unknown panic".to_string()
                    };
                    return Err(anyhow::anyhow!(
                        "{}",
                        crate::error::HalExecutionError::OpPanic {
                            func_idx: fi,
                            op_idx: oi,
                            op_name,
                            panic_msg: msg,
                        }
                    ));
                }
            };
            log::info!(
                "hal_runner: func[{}] op[{}] '{}' DONE ({} outputs, {:?})",
                fi, oi, op_name, output_shapes.len(), output_shapes,
            );
            drop(output_bufs);
            for (idx, output_name) in op.outputs.iter().enumerate() {
                let out_vec = std::mem::take(&mut output_vecs[idx]);

                ssa_map.insert(output_name.clone(), out_vec);
                ssa_dtypes.insert(output_name.clone(), output_dtype);

                if let Some(shapes) = output_shapes.get(idx) {
                    let shape_usize: Vec<usize> = shapes.iter()
                        .map(|&s| std::cmp::max(1, s as usize))
                        .collect();
                    ssa_shapes.insert(output_name.clone(), shape_usize);
                }
            }

            if op.op == "shape_of" {
                if let Some(out_name) = op.outputs.first() {
                    if let Some(data) = ssa_map.get(out_name) {
                        let dims: Vec<usize> = data.chunks(4)
                            .map(|b| {
                                let bytes: [u8; 4] = b.try_into().unwrap_or([0; 4]);
                                f32::from_le_bytes(bytes) as usize
                            })
                            .collect();
                        if !dims.is_empty() {
                            log::debug!(
                                "hal_runner: shape_of '{}' updated ssa_shapes {:?} -> {:?}",
                                out_name,
                                ssa_shapes.get(out_name),
                                dims,
                            );
                            ssa_shapes.insert(out_name.clone(), dims);
                        }
                    }
                }
            }

            // ── Debug: dump func[0] gather outputs ──────────────────
            if fi == 0 && op.op == "gather" {
                if let Some(out_name) = op.outputs.first() {
                    if let Some(data) = ssa_map.get(out_name) {
                        use std::io::Write;
                        let path = format!("/tmp/func0_gather_{}.bin", out_name.trim_start_matches('%'));
                        let shape = ssa_shapes.get(out_name).cloned().unwrap_or_default();
                        let mut f = std::fs::File::create(&path).unwrap();
                        let rank = shape.len() as i32;
                        f.write_all(&rank.to_le_bytes()).unwrap();
                        for &d in &shape { f.write_all(&(d as i32).to_le_bytes()).unwrap(); }
                        f.write_all(data).unwrap();
                        let dtype = ssa_dtypes.get(out_name).copied().unwrap_or(Dtype::F32);
                        let indices_name = op.inputs.get(1).cloned().unwrap_or_default();
                        let indices_dtype = ssa_dtypes.get(&indices_name).copied().unwrap_or(Dtype::F32);
                        eprintln!("[debug] func[0] gather '{}': shape={:?}, dtype={:?}, indices={}, indices_dtype={:?}, {} bytes -> {}",
                            out_name, shape, dtype, indices_name, indices_dtype, data.len(), path);
                    }
                }
            }
            // Debug: dump func[0] position indices %251
            if fi == 0 && op.op == "element_wise" && op.outputs.first().map(|n| n.as_str()) == Some("%251") {
                if let Some(data) = ssa_map.get("%251") {
                    let vals: Vec<i64> = data.chunks(8).map(|b| {
                        let arr: [u8; 8] = b.try_into().unwrap_or([0; 8]);
                        i64::from_le_bytes(arr)
                    }).collect();
                    eprintln!("[debug] position indices %251: I64 values = {:?}", vals);
                }
            }
            // Also dump %200 value  
            if fi == 0 && op.op == "element_wise" && op.outputs.first().map(|n| n.as_str()) == Some("%251") {
                if let Some(data) = ssa_map.get("%200") {
                    let vals: Vec<i64> = data.chunks(8).map(|b| {
                        let arr: [u8; 8] = b.try_into().unwrap_or([0; 8]);
                        i64::from_le_bytes(arr)
                    }).collect();
                    eprintln!("[debug] constant %200: I64 values = {:?}", vals);
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

            // ── Debug: dump func[0] embedding output ──────────────────
            if fi == 0 {
                if let Some(output_name) = function.outputs.iter()
                    .find(|o| o.shape.len() >= 3)
                    .map(|o| &o.name)
                {
                    if let Some(data) = ssa_map.get(output_name) {
                        let path = "/tmp/func0_embedding.bin";
                        let shape = ssa_shapes.get(output_name).cloned().unwrap_or_default();
                        let mut f = std::fs::File::create(path).unwrap();
                        use std::io::Write;
                        let rank = shape.len() as i32;
                        f.write_all(&rank.to_le_bytes()).unwrap();
                        for &d in &shape { f.write_all(&(d as i32).to_le_bytes()).unwrap(); }
                        f.write_all(data).unwrap();
                        eprintln!("[debug] func[0] output '{}': shape={:?}, {} bytes -> {}",
                            output_name, shape, data.len(), path);
                    }
                }
            }
    }

    // ── Extract global output ───────────────────────────────────────
    extract_global_output(hal_ir, &ssa_map, &ssa_shapes, &ssa_dtypes, seq_len)
}
