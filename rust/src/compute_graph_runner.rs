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

use crate::compute_graph::{ComputeGraph, InputBinding};
use crate::hal::cpu::memref::MemRefDesc1;
use crate::hal::cpu::{Executable, MemRefDescAny, MemRefDesc2};
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
                    // shape[i] == 0 is the SFCF dynamic sentinel
                    let is_dynamic = shape.iter().any(|&d| d == 0);
                    if is_dynamic {
                        let rank = io_def.rank as usize;
                        let data_source: &[u32] = if bi == 1 { positions } else { input_ids };
                        match rank {
                            1 => {
                                let n_tokens = data_source.len();
                                let raw: Vec<u8> = data_source.iter()
                                    .flat_map(|&v| (v as i64).to_ne_bytes())
                                    .collect();
                                let p = raw.as_ptr();
                                let memref = MemRefDesc1 {
                                    allocated: p as *mut c_void,
                                    aligned: p as *mut c_void,
                                    offset: 0,
                                    sizes: [n_tokens as i64],
                                    strides: [1],
                                };
                                _raw_buffers.push(raw);
                                let desc = MemRefDescAny::R1(memref);
                                input_descs.push(desc);
                                input_ptrs.push(input_descs.last()
                                    .expect("input_descs has entry for GlobalInput")
                                    .as_input_ptr());
                                continue;
                            }
                            2 => {
                                let n_tokens = data_source.len() as i64;
                                let raw: Vec<u8> = data_source.iter()
                                    .flat_map(|&v| (v as i64).to_ne_bytes())
                                    .collect();
                                let p = raw.as_ptr();
                                let memref = MemRefDesc2 {
                                    allocated: p as *mut c_void,
                                    aligned: p as *mut c_void,
                                    offset: 0,
                                    sizes: [1, n_tokens],
                                    strides: [n_tokens, 1],
                                };
                                _raw_buffers.push(raw);
                                let desc = MemRefDescAny::R2(memref);
                                input_descs.push(desc);
                                input_ptrs.push(input_descs.last()
                                    .expect("input_descs has entry for GlobalInput")
                                    .as_input_ptr());
                                continue;
                            }
                            r => anyhow::bail!(
                                "run_function_graph: unsupported rank {} for \
                                 dynamic GlobalInput (shape={:?})",
                                r, shape,
                            ),
                        }
                    }
                    let data_source: &[u32] = if bi == 1 { positions } else { input_ids };
                    let expected_numel: usize = shape.iter().product();
                    let n_tokens = data_source.len().min(expected_numel);
                    let padded: Vec<i64> = (0..expected_numel).map(|i| {
                        if i < n_tokens {
                            data_source[i] as i64
                        } else {
                            0i64
                        }
                    }).collect();
                    let raw: Vec<u8> = padded.iter().flat_map(|&v| v.to_ne_bytes()).collect();
                    let p = raw.as_ptr();
                    let memref = MemRefDesc2 {
                        allocated: p as *mut c_void,
                        aligned: p as *mut c_void,
                        offset: 0,
                        sizes: [shape[0] as i64, shape.get(1).copied().unwrap_or(1) as i64],
                        strides: [shape.get(1).copied().unwrap_or(1) as i64, 1],
                    };
                    _raw_buffers.push(raw);
                    let desc = MemRefDescAny::R2(memref);
                    input_descs.push(desc);
                    input_ptrs.push(input_descs.last()
                        .expect("input_descs has entry for GlobalInput").as_input_ptr());
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
            let (aligned, runtime_sizes) = match unsafe { crate::executor::parse_sret_descriptor(ptr_slice, r) } {
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
