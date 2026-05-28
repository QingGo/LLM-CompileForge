//! Compute graph runner — iterates over FuncDefs, assembles inputs from
//! global inputs / weights / SSA wires, dispatches through
//! `executable.execute(op_name, stream, &input_bufs, &output_bufs)`,
//! and extracts output Tensors from the returned shapes and buffers.
//!
//! No direct ciface / lookup_typed / parse_sret_descriptor calls — all
//! kernel dispatch goes through the HAL Executable trait.

use std::cell::RefCell;
use std::collections::HashMap;

use half::f16;

use crate::block_manager::BlockManager;
use crate::compute_graph::{ComputeGraph, InputBinding};
use crate::hal::cpu::buffer::CpuBuffer as InnerCpuBuffer;
use crate::hal::cpu::CpuBuffer;
use crate::hal::cpu::CpuStream;
use crate::hal::traits;
use crate::kv_cache_intercept::{intercept_consumed_input, intercept_consumed_output};
use crate::tensor::{Dtype, Tensor};
use crate::weight_loader::WeightProvider;

/// Walk every function in `compute_graph` in order.
///
/// For each function:
///   1. Build input buffers from global inputs, weights (with f16→f32
///      conversion), or SSA wires (from prior functions).
///   2. Build pre-allocated output buffers sized from the compute graph
///      metadata and actual sequence length.
///   3. Call `executable.execute(op_name, stream, &input_bufs, &output_bufs)`.
///   4. Use the returned output shapes to construct Tensor values.
///
/// Returns the global output tensor specified by
/// `compute_graph.global_output`.
pub fn run_function_graph(
    compute_graph: &ComputeGraph,
    executable: &dyn traits::Executable,
    weight_provider: &WeightProvider,
    weight_cache: &RefCell<HashMap<String, Tensor>>,
    func_outputs: &mut Vec<Vec<Tensor>>,
    input_ids: &[u32],
    positions: &[u32],
) -> Result<Tensor, anyhow::Error> {
    let stream = CpuStream;

    for func_def in &compute_graph.functions {
        let fi = func_def.index;

        let mut input_bufs: Vec<Box<dyn traits::Buffer>> =
            Vec::with_capacity(func_def.num_inputs);
        let mut _raw_global: Vec<Vec<u8>> = Vec::new();
        let mut _tensors: Vec<Tensor> = Vec::with_capacity(func_def.num_inputs);

        for (bi, (binding, io_def)) in func_def.inputs.iter().enumerate() {
            match binding {
                InputBinding::GlobalInput => {
                    let (_, raw) =
                        crate::global_input::fill_global_input(input_ids, positions, io_def, bi)?;
                    _raw_global.push(raw);
                    let raw_bytes = _raw_global.last().expect("raw_global last");
                    let raw_buf = InnerCpuBuffer::from_raw_parts(
                        raw_bytes.as_ptr() as *mut u8,
                        raw_bytes.len(),
                    )
                    .map_err(|e| anyhow::anyhow!("{}", e))?;
                    let rank = io_def.rank as usize;
                    let dims: Vec<usize> = (0..rank)
                        .map(|i| {
                            if io_def.shape[i] <= 0 {
                                if i == 0 { 1 } else { input_ids.len() }
                            } else {
                                io_def.shape[i] as usize
                            }
                        })
                        .collect();
                    let cpu_buf = CpuBuffer::with_meta(raw_buf, 8 /* i64 */, dims);
                    input_bufs.push(Box::new(cpu_buf));
                }
                InputBinding::Weight(key) => {
                    let mut cache = weight_cache.borrow_mut();
                    let tensor: Tensor = if let Some(cached) = cache.get(key) {
                        cached.to_owned()
                    } else {
                        let desc = weight_provider
                            .get_weight_memref(key)
                            .ok_or_else(|| {
                                anyhow::anyhow!("weight not found: {}", key)
                            })?;
                        let n = desc.numel();
                        let data: Vec<f32> = unsafe {
                            let raw = desc.aligned as *const u16;
                            let slice = std::slice::from_raw_parts(raw, n);
                            slice
                                .iter()
                                .map(|&h| f16::from_bits(h).to_f32())
                                .collect()
                        };
                        let shape: Vec<usize> =
                            io_def.shape.iter().map(|&d| d as usize).collect();
                        let tensor = Tensor::new_owned(shape, data, Dtype::F32);
                        cache.insert(key.clone(), tensor.to_owned());
                        tensor
                    };
                    let raw_buf = InnerCpuBuffer::from_raw_parts(
                        tensor.as_slice().as_ptr() as *mut u8,
                        tensor.as_slice().len() * 4,
                    )
                    .map_err(|e| anyhow::anyhow!("{}", e))?;
                    let cpu_buf = CpuBuffer::with_meta(raw_buf, 4 /* f32 */, tensor.shape.clone());
                    _tensors.push(tensor);
                    input_bufs.push(Box::new(cpu_buf));
                }
                InputBinding::Ssa {
                    producer_func,
                    output_idx,
                } => {
                    let ref_tensor = &func_outputs[*producer_func][*output_idx];
                    let raw_buf = InnerCpuBuffer::from_raw_parts(
                        ref_tensor.as_slice().as_ptr() as *mut u8,
                        ref_tensor.as_slice().len() * 4,
                    )
                    .map_err(|e| anyhow::anyhow!("{}", e))?;
                    let dims = ref_tensor.shape.clone();
                    let cpu_buf = CpuBuffer::with_meta(raw_buf, 4 /* f32 */, dims);
                    input_bufs.push(Box::new(cpu_buf));
                }
            }
        }

        // Pre-allocate output buffers sized from the compute graph metadata.
        // The dims use io_def.shape with 0→1 fallback so that rank() returns
        // the correct rank for sret descriptor parsing.
        let seq_len = input_ids.len();
        let mut output_vecs: Vec<Vec<f32>> = Vec::with_capacity(func_def.outputs.len());
        let mut output_bufs: Vec<Box<dyn traits::Buffer>> =
            Vec::with_capacity(func_def.outputs.len());
        for io_def in &func_def.outputs {
            let numel = estimate_output_numel(&io_def.shape, seq_len);
            let mut vec = Vec::with_capacity(numel);
            // SAFETY: set_len to capacity — the buffer is filled by
            // execute() before any f32 reads.  execute() checks that
            // the output does not exceed numel.
            unsafe { vec.set_len(numel); }
            let raw_buf = InnerCpuBuffer::from_raw_parts(
                vec.as_mut_ptr() as *mut u8,
                numel * 4,
            )
            .map_err(|e| anyhow::anyhow!("{}", e))?;
            let shape_fallback: Vec<usize> = io_def.shape.iter()
                .map(|&d| if d == 0 { 1 } else { d as usize })
                .collect();
            let cpu_buf = CpuBuffer::with_meta(raw_buf, 4 /* f32 */, shape_fallback);
            output_vecs.push(vec);
            output_bufs.push(Box::new(cpu_buf));
        }

        // Collect trait-object references for the execute call.
        let input_refs: Vec<&dyn traits::Buffer> =
            input_bufs.iter().map(|b| b.as_ref()).collect();
        let output_refs: Vec<&dyn traits::Buffer> =
            output_bufs.iter().map(|b| b.as_ref()).collect();

        // SAFETY: executable.execute handles the ciface kernel call
        // internally, including sret allocation, lookup_typed, and
        // parsing output descriptors.
        let output_shapes = executable.execute(&func_def.symbol, &stream, &input_refs, &output_refs)?;

        // Extract Tensors from output buffers using the returned shapes.
        for (oi, shapes) in output_shapes.iter().enumerate() {
            let actual_n: usize = shapes.iter().map(|&s| std::cmp::max(0, s) as usize).product();
            let mut out_vec = std::mem::take(&mut output_vecs[oi]);
            if actual_n < out_vec.len() {
                // SAFETY: actual_n ≤ out_vec.len() (checked by execute).
                // Data is written by execute() via copy_nonoverlapping.
                unsafe { out_vec.set_len(actual_n); }
            }
            let shape_usize: Vec<usize> = shapes.iter().map(|&s| std::cmp::max(1, s) as usize).collect();
            func_outputs[fi].push(Tensor::new_owned(shape_usize, out_vec, Dtype::F32));
        }
    }

    let (g_func, g_idx) = compute_graph.global_output;
    let result = &func_outputs[g_func][g_idx];
    Ok(result.to_owned())
}

/// Estimate the number of f32 elements for an output tensor.
///
/// The compute graph encodes dynamic dimensions as 0.  We assume the first
/// dynamic dimension is batch (= 1) and subsequent dynamic dimensions are
/// sequence length (= `seq_len`).
fn estimate_output_numel(shape: &[u64], seq_len: usize) -> usize {
    let mut numel: usize = 1;
    let mut first_zero = true;
    for &d in shape {
        let dim = if d == 0 {
            if first_zero {
                first_zero = false;
                1 // batch is always 1
            } else {
                seq_len
            }
        } else {
            d as usize
        };
        numel = numel.saturating_mul(dim);
    }
    // Minimum sensible size: 16 elements (avoids empty-output issues).
    numel.max(16)
}

/// Same as [`run_function_graph`] but with KV cache intercept callbacks.
///
/// Iterates all `FuncDef`s in order, with the same dispatch logic as
/// `run_function_graph`, except:
///
/// 1. SSA inputs that reference a `consumed_internally=true` output are
///    **overridden** via [`intercept_consumed_input`] (which reads from
///    the KV cache during decode).
/// 2. `consumed_internally=true` outputs are stored in a local `kv_new`
///    map and written to the [`BlockManager`] via
///    [`intercept_consumed_output`] instead of being pushed to
///    `func_outputs`.
/// 3. Normal (non-consumed) outputs are pushed to `func_outputs` as usual.
///
/// Returns the global output tensor (same as `run_function_graph`).
pub fn run_function_graph_with_kv_intercept(
    compute_graph: &ComputeGraph,
    executable: &dyn traits::Executable,
    weight_provider: &WeightProvider,
    weight_cache: &RefCell<HashMap<String, Tensor>>,
    func_outputs: &mut Vec<Vec<Tensor>>,
    input_ids: &[u32],
    positions: &[u32],
    mut block_manager: Option<&mut BlockManager>,
    request_id: Option<&str>,
) -> Result<Tensor, anyhow::Error> {
    let is_decode = input_ids.len() == 1;
    let mut kv_new: HashMap<(usize, usize), Tensor> = HashMap::new();
    let stream = CpuStream;

    for func_def in &compute_graph.functions {
        let fi = func_def.index;

        let mut input_bufs: Vec<Box<dyn traits::Buffer>> =
            Vec::with_capacity(func_def.num_inputs);
        let mut _raw_global: Vec<Vec<u8>> = Vec::new();
        let mut _tensors: Vec<Tensor> = Vec::with_capacity(func_def.num_inputs);

        for (bi, (binding, io_def)) in func_def.inputs.iter().enumerate() {
            // GlobalInput: handle with early continue (consumes `bi`)
            if let InputBinding::GlobalInput = binding {
                let (_, raw) =
                    crate::global_input::fill_global_input(input_ids, positions, io_def, bi)?;
                _raw_global.push(raw);
                let raw_bytes = _raw_global.last().expect("raw_global last");
                let raw_buf = InnerCpuBuffer::from_raw_parts(
                    raw_bytes.as_ptr() as *mut u8,
                    raw_bytes.len(),
                )
                .map_err(|e| anyhow::anyhow!("{}", e))?;
                let rank = io_def.rank as usize;
                let dims: Vec<usize> = (0..rank)
                    .map(|i| {
                        if io_def.shape[i] <= 0 {
                            if i == 0 { 1 } else { input_ids.len() }
                        } else {
                            io_def.shape[i] as usize
                        }
                    })
                    .collect();
                let cpu_buf = CpuBuffer::with_meta(raw_buf, 8 /* i64 */, dims);
                input_bufs.push(Box::new(cpu_buf));
                continue;
            }

            let tensor: Tensor = match binding {
                InputBinding::GlobalInput => unreachable!(), // handled above
                InputBinding::Weight(key) => {
                    let mut cache = weight_cache.borrow_mut();
                    if let Some(cached) = cache.get(key) {
                        cached.to_owned()
                    } else {
                        let desc = weight_provider
                            .get_weight_memref(key)
                            .ok_or_else(|| {
                                anyhow::anyhow!("weight not found: {}", key)
                            })?;
                        let n = desc.numel();
                        let data: Vec<f32> = unsafe {
                            let raw = desc.aligned as *const u16;
                            let slice = std::slice::from_raw_parts(raw, n);
                            slice
                                .iter()
                                .map(|&h| f16::from_bits(h).to_f32())
                                .collect()
                        };
                        let shape: Vec<usize> =
                            io_def.shape.iter().map(|&d| d as usize).collect();
                        let t = Tensor::new_owned(shape, data, Dtype::F32);
                        cache.insert(key.clone(), t.to_owned());
                        t
                    }
                }
                InputBinding::Ssa {
                    producer_func,
                    output_idx,
                } => {
                    let prod_output_def =
                        &compute_graph.functions[*producer_func].outputs[*output_idx];
                    if prod_output_def.consumed_internally {
                        intercept_consumed_input(
                            *producer_func,
                            *output_idx,
                            compute_graph,
                            &kv_new,
                            block_manager.as_deref(),
                            request_id,
                            positions,
                            is_decode,
                        )?
                    } else {
                        let producer_outputs =
                            &compute_graph.functions[*producer_func].outputs;
                        let ci_before = producer_outputs[..*output_idx]
                            .iter()
                            .filter(|o| o.consumed_internally)
                            .count();
                        let adjusted_idx = *output_idx - ci_before;
                        let ref_tensor = &func_outputs[*producer_func][adjusted_idx];
                        ref_tensor.to_owned()
                    }
                }
            };

            let raw_buf = InnerCpuBuffer::from_raw_parts(
                tensor.as_slice().as_ptr() as *mut u8,
                tensor.as_slice().len() * 4,
            )
            .map_err(|e| anyhow::anyhow!("{}", e))?;
            let cpu_buf = CpuBuffer::with_meta(raw_buf, 4 /* f32 */, tensor.shape.clone());
            _tensors.push(tensor);
            input_bufs.push(Box::new(cpu_buf));
        }

        // Pre-allocate output buffers (same pattern as run_function_graph)
        let seq_len = input_ids.len();
        let mut output_vecs: Vec<Vec<f32>> = Vec::with_capacity(func_def.outputs.len());
        let mut output_bufs: Vec<Box<dyn traits::Buffer>> =
            Vec::with_capacity(func_def.outputs.len());
        for io_def in &func_def.outputs {
            let numel = estimate_output_numel(&io_def.shape, seq_len);
            let mut vec = Vec::with_capacity(numel);
            unsafe { vec.set_len(numel); }
            let raw_buf = InnerCpuBuffer::from_raw_parts(
                vec.as_mut_ptr() as *mut u8,
                numel * 4,
            )
            .map_err(|e| anyhow::anyhow!("{}", e))?;
            let shape_fallback: Vec<usize> = io_def.shape.iter()
                .map(|&d| if d == 0 { 1 } else { d as usize })
                .collect();
            let cpu_buf = CpuBuffer::with_meta(raw_buf, 4 /* f32 */, shape_fallback);
            output_vecs.push(vec);
            output_bufs.push(Box::new(cpu_buf));
        }

        let input_refs: Vec<&dyn traits::Buffer> =
            input_bufs.iter().map(|b| b.as_ref()).collect();
        let output_refs: Vec<&dyn traits::Buffer> =
            output_bufs.iter().map(|b| b.as_ref()).collect();

        let output_shapes =
            executable.execute(&func_def.symbol, &stream, &input_refs, &output_refs)?;

        for (oi, shapes) in output_shapes.iter().enumerate() {
            let actual_n: usize = shapes.iter().map(|&s| std::cmp::max(0, s) as usize).product();
            let mut out_vec = std::mem::take(&mut output_vecs[oi]);
            if actual_n < out_vec.len() {
                unsafe { out_vec.set_len(actual_n); }
            }
            let shape_usize: Vec<usize> =
                shapes.iter().map(|&s| std::cmp::max(1, s) as usize).collect();
            let tensor = Tensor::new_owned(shape_usize, out_vec, Dtype::F32);

            let io_def = &func_def.outputs[oi];
            if io_def.consumed_internally {
                intercept_consumed_output(
                    fi,
                    oi,
                    &tensor,
                    &mut kv_new,
                    block_manager.as_deref_mut(),
                    request_id,
                    positions,
                    is_decode,
                    &func_def.outputs,
                )?;
            } else {
                func_outputs[fi].push(tensor);
            }
        }
    }

    let (g_func, g_idx) = compute_graph.global_output;
    let result = &func_outputs[g_func][g_idx];
    Ok(result.to_owned())
}

/// Parse a single sret output descriptor (ranked MemRef) from raw bytes.
///
/// # Safety
///
/// `slice` must contain valid MemRef descriptor binary data written by a
/// ciface kernel.  The layout is:
///   struct { allocated: ptr, aligned: ptr, offset: i64,
///            sizes: [i64; RANK], strides: [i64; RANK] }
/// with `RANK` = `rank` parameter.
///
/// Returns `(CpuBuffer, actual_sizes)` where the CpuBuffer wraps the
/// dylib-allocated output memory.
///
/// NOTE: This function is kept for backward compatibility of existing
/// `parse_sret_descriptor` tests in `executor_tests.rs`.  The
/// compute-graph runner no longer uses it — sret parsing is handled
/// inside `CpuExecutable::execute()`.
#[allow(dead_code)]
pub(crate) unsafe fn parse_sret_descriptor(
    slice: &[u8],
    rank: usize,
) -> Result<(InnerCpuBuffer, Vec<i64>), String> {
    let min_len = 24 + rank * 8;
    if slice.len() < min_len {
        return Err(format!("slice too short: {} < {}", slice.len(), min_len));
    }
    let aligned = std::ptr::read_unaligned(slice.as_ptr().add(8) as *const *mut u8);
    if aligned.is_null() {
        return Err("aligned pointer is null".to_string());
    }
    let sizes: Vec<i64> = (0..rank)
        .map(|i| std::ptr::read_unaligned(slice.as_ptr().add(24 + i * 8) as *const i64))
        .collect();
    let n: usize = sizes.iter().map(|&s| std::cmp::max(0, s) as usize).product();
    let n_bytes = n * 4; // f32 element size
    let cpu_buf = InnerCpuBuffer::from_raw_parts(aligned, n_bytes)?;
    Ok((cpu_buf, sizes))
}
