//! Compute graph runner — iterates over FuncDefs, assembles inputs from
//! global inputs / weights / SSA wires, calls ciface kernels via sret,
//! and parses outputs back into Tensors.
//!
//! Extracted from `executor.rs` to keep the ModelExecutor focused on
//! high-level orchestration (KV cache, DUMP_LAYERS, etc.).

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;

use half::f16;

use crate::block_manager::BlockManager;
use crate::compute_graph::{ComputeGraph, InputBinding};
use crate::hal::cpu::{Executable, MemRefDescAny};
use crate::kv_cache_intercept::{intercept_consumed_input, intercept_consumed_output};
use crate::tensor::{Dtype, Tensor};
use crate::weight_loader::WeightProvider;

/// Walk every function in `compute_graph` in order.
///
/// For each function:
///   1. Look up the kernel symbol in `executable`.
///   2. Assemble input descriptors from global inputs, weights (with
///      f16→f32 conversion), or SSA wires (from prior functions).
///   3. Call the ciface kernel via a unified sret buffer.
///   4. Parse the sret buffer into `Tensor` values and store them in
///      `func_outputs`.
///
/// Returns the global output tensor specified by
/// `compute_graph.global_output`.
pub fn run_function_graph(
    compute_graph: &ComputeGraph,
    executable: &Executable,
    weight_provider: &WeightProvider,
    weight_cache: &RefCell<HashMap<String, Tensor<'static>>>,
    func_outputs: &mut Vec<Vec<Tensor<'static>>>,
    input_ids: &[u32],
    positions: &[u32],
) -> Result<Tensor<'static>, anyhow::Error> {
    for func_def in &compute_graph.functions {
        let fi = func_def.index;
        let kernel = executable
            .lookup_typed(&func_def.symbol, func_def.total_args())?;

        let mut input_descs: Vec<MemRefDescAny> =
            Vec::with_capacity(func_def.num_inputs);
        let mut input_ptrs: Vec<*const c_void> =
            Vec::with_capacity(func_def.num_inputs);
        let mut _tensors: Vec<Tensor<'static>> = Vec::with_capacity(func_def.num_inputs);
        let mut _raw_buffers: Vec<Vec<u8>> = Vec::new();

        for (bi, (binding, io_def)) in func_def.inputs.iter().enumerate() {
            let shape: Vec<usize> =
                io_def.shape.iter().map(|&d| d as usize).collect();
            let tensor: Tensor = match binding {
                InputBinding::GlobalInput => {
                    let (desc, raw) =
                        crate::global_input::fill_global_input(input_ids, positions, io_def, bi)?;
                    _raw_buffers.push(raw);
                    input_descs.push(desc);
                    input_ptrs.push(input_descs.last()
                        .expect("input_descs has entry for GlobalInput")
                        .as_input_ptr());
                    continue;
                }
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
                            slice.iter().map(|&h| f16::from_bits(h).to_f32()).collect()
                        };
                        let tensor = Tensor::new_owned(shape, data, Dtype::F32);
                        cache.insert(key.clone(), tensor.to_owned());
                        tensor
                    }
                }
                InputBinding::Ssa {
                    producer_func,
                    output_idx,
                } => {
                    let ref_tensor = &func_outputs[*producer_func][*output_idx];
                    ref_tensor.to_owned()
                }
            };

            let desc = MemRefDescAny::from_f32(&tensor.shape, tensor.as_slice())
                .map_err(|e| anyhow::anyhow!("weight desc: {}", e))?;
            input_descs.push(desc);
            input_ptrs.push(input_descs.last()
                .expect("input_descs has entry for Weight/Ssa input").as_input_ptr());
            _tensors.push(tensor);
        }
        debug_assert!(_tensors.len() <= input_ptrs.len());

        const SRET_BUF_SIZE: usize = 131072;
        let mut sret: Vec<u8> = vec![0u8; SRET_BUF_SIZE];
        let sret_ptr = sret.as_mut_ptr() as *mut c_void;

        let mut all_args: Vec<*const c_void> = Vec::with_capacity(1 + input_ptrs.len());
        all_args.push(sret_ptr);
        all_args.extend(input_ptrs.iter().copied());
        // SAFETY: kernel was loaded from the compiled .dylib and validated
        // by Executable::lookup_typed().  sret_ptr and input_ptrs point to
        // writable/readable buffers of appropriate size.  The kernel is
        // _mlir_ciface_* — a C ABI function that reads MemRef descriptors
        // from input_ptrs and writes output descriptors to sret_ptr.
        unsafe {
            let raw_ptr = kernel.as_raw_ptr();
            crate::ciface_high::call_high_arity(raw_ptr, &all_args);
        }

        let mut sret_offset: usize = 0;
        for (oi, io_def) in func_def.outputs.iter().enumerate() {
            let r = io_def.rank as usize;
            let desc_size = 24 + 16 * r;
            let end = sret_offset + desc_size;
            if end > SRET_BUF_SIZE {
                anyhow::bail!(
                    "sret overflow: func {} output {} desc_size={} offset={} exceeds {}",
                    fi, oi, desc_size, sret_offset, SRET_BUF_SIZE,
                );
            }
            let ptr_slice = &sret[sret_offset..end];
            // SAFETY: parse_sret_descriptor reads structured binary data
            // from the sret buffer written by the MLIR ciface kernel.
            // desc_size was computed from the known rank r.  The slice
            // bounds are validated above (end <= SRET_BUF_SIZE).
            let (aligned, runtime_sizes) = match unsafe { parse_sret_descriptor(ptr_slice, r) } {
                Ok(result) => result,
                Err(e) => {
                    eprintln!("[executor] func_{} output_{}: {} — skipping", fi, oi, e);
                    sret_offset += desc_size;
                    continue;
                }
            };
            let fallback: Vec<i64> = io_def.shape.iter().map(|&d|
                if d == 0 { 1 } else { d as i64 }
            ).collect();
            let sizes: Vec<i64> = runtime_sizes.iter().zip(fallback.iter()).map(|(&r, &f)|
                if r <= 0 || r > 1_000_000_000 { f } else { r }
            ).collect();
            let n: usize = sizes.iter().map(|&s| s as usize).product();
            let data: Vec<f32> = if aligned.is_null() {
                Vec::new()
            } else {
                unsafe {
                    let slice = std::slice::from_raw_parts(aligned as *const f32, n);
                    slice.to_vec()
                }
            };
            let shape: Vec<usize> = sizes.iter().map(|&s| s as usize).collect();
            func_outputs[fi].push(Tensor::new_owned(shape, data, Dtype::F32));
            sret_offset += desc_size;
        }
    }

    let (g_func, g_idx) = compute_graph.global_output;
    let result = &func_outputs[g_func][g_idx];
    Ok(result.to_owned())
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
    executable: &Executable,
    weight_provider: &WeightProvider,
    weight_cache: &RefCell<HashMap<String, Tensor<'static>>>,
    func_outputs: &mut Vec<Vec<Tensor<'static>>>,
    input_ids: &[u32],
    positions: &[u32],
    mut block_manager: Option<&mut BlockManager>,
    request_id: Option<&str>,
) -> Result<Tensor<'static>, anyhow::Error> {
    let is_decode = input_ids.len() == 1;
    let mut kv_new: HashMap<(usize, usize), Tensor<'static>> = HashMap::new();

    for func_def in &compute_graph.functions {
        let fi = func_def.index;
        let kernel = executable
            .lookup_typed(&func_def.symbol, func_def.total_args())?;

        let mut input_descs: Vec<MemRefDescAny> =
            Vec::with_capacity(func_def.num_inputs);
        let mut input_ptrs: Vec<*const c_void> =
            Vec::with_capacity(func_def.num_inputs);
        let mut _tensors: Vec<Tensor<'static>> = Vec::with_capacity(func_def.num_inputs);
        let mut _raw_buffers: Vec<Vec<u8>> = Vec::new();

        for (bi, (binding, io_def)) in func_def.inputs.iter().enumerate() {
            let shape: Vec<usize> =
                io_def.shape.iter().map(|&d| d as usize).collect();

            // GlobalInput: handled with early continue
            if let InputBinding::GlobalInput = binding {
                let (desc, raw) =
                    crate::global_input::fill_global_input(input_ids, positions, io_def, bi)?;
                _raw_buffers.push(raw);
                input_descs.push(desc);
                input_ptrs.push(input_descs.last()
                    .expect("input_descs has entry for GlobalInput")
                    .as_input_ptr());
                continue;
            }

            let tensor: Tensor = match binding {
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
                            slice.iter().map(|&h| f16::from_bits(h).to_f32()).collect()
                        };
                        let tensor = Tensor::new_owned(shape, data, Dtype::F32);
                        cache.insert(key.clone(), tensor.to_owned());
                        tensor
                    }
                }
                InputBinding::Ssa {
                    producer_func,
                    output_idx,
                } => {
                    let prod_output_def = &compute_graph.functions[*producer_func].outputs[*output_idx];
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
                        // Normal SSA input — read from func_outputs with
                        // adjusted index to account for consumed_internally
                        // outputs skipped in func_outputs.
                        let producer_outputs = &compute_graph.functions[*producer_func].outputs;
                        let ci_before = producer_outputs[..*output_idx]
                            .iter()
                            .filter(|o| o.consumed_internally)
                            .count();
                        let adjusted_idx = *output_idx - ci_before;
                        let ref_tensor = &func_outputs[*producer_func][adjusted_idx];
                        ref_tensor.to_owned()
                    }
                }
                _ => unreachable!(), // GlobalInput handled above
            };

            let desc = MemRefDescAny::from_f32(&tensor.shape, tensor.as_slice())
                .map_err(|e| anyhow::anyhow!("desc: {}", e))?;
            input_descs.push(desc);
            input_ptrs.push(input_descs.last()
                .expect("input_descs has entry").as_input_ptr());
            _tensors.push(tensor);
        }
        debug_assert!(_tensors.len() <= input_ptrs.len());

        const SRET_BUF_SIZE: usize = 131072;
        let mut sret: Vec<u8> = vec![0u8; SRET_BUF_SIZE];
        let sret_ptr = sret.as_mut_ptr() as *mut c_void;

        let mut all_args: Vec<*const c_void> = Vec::with_capacity(1 + input_ptrs.len());
        all_args.push(sret_ptr);
        all_args.extend(input_ptrs.iter().copied());
        // SAFETY: kernel was loaded from the compiled .dylib and validated
        // by Executable::lookup_typed().  See run_function_graph for details.
        unsafe {
            let raw_ptr = kernel.as_raw_ptr();
            crate::ciface_high::call_high_arity(raw_ptr, &all_args);
        }

        let mut sret_offset: usize = 0;
        for (oi, io_def) in func_def.outputs.iter().enumerate() {
            let r = io_def.rank as usize;
            let desc_size = 24 + 16 * r;
            let end = sret_offset + desc_size;
            if end > SRET_BUF_SIZE {
                anyhow::bail!(
                    "sret overflow: func {} output {} desc_size={} offset={} exceeds {}",
                    fi, oi, desc_size, sret_offset, SRET_BUF_SIZE,
                );
            }
            let ptr_slice = &sret[sret_offset..end];
            // SAFETY: parse_sret_descriptor reads structured binary data
            // from the sret buffer written by the MLIR ciface kernel.
            let (aligned, runtime_sizes) = match unsafe { parse_sret_descriptor(ptr_slice, r) } {
                Ok(result) => result,
                Err(e) => {
                    eprintln!("[executor] func_{} output_{}: {} — skipping", fi, oi, e);
                    sret_offset += desc_size;
                    continue;
                }
            };
            let fallback: Vec<i64> = io_def.shape.iter().map(|&d|
                if d == 0 { 1 } else { d as i64 }
            ).collect();
            let sizes: Vec<i64> = runtime_sizes.iter().zip(fallback.iter()).map(|(&r, &f)|
                if r <= 0 || r > 1_000_000_000 { f } else { r }
            ).collect();
            let n: usize = sizes.iter().map(|&s| s as usize).product();
            let data: Vec<f32> = if aligned.is_null() {
                Vec::new()
            } else {
                unsafe {
                    let slice = std::slice::from_raw_parts(aligned as *const f32, n);
                    slice.to_vec()
                }
            };
            let shape: Vec<usize> = sizes.iter().map(|&s| s as usize).collect();
            let tensor = Tensor::new_owned(shape, data, Dtype::F32);

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
            sret_offset += desc_size;
        }
    }

    let (g_func, g_idx) = compute_graph.global_output;
    let result = &func_outputs[g_func][g_idx];
    Ok(result.to_owned())
}

pub(crate) unsafe fn parse_sret_descriptor(
    slice: &[u8],
    rank: usize,
) -> Result<(*mut u8, Vec<i64>), String> {
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
    Ok((aligned, sizes))
}
