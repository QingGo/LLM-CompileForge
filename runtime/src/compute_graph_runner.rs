//! Compute graph runner — iterates over FuncDefs, assembles inputs from
//! global inputs / weights / SSA wires, dispatches through
//! `executable.execute(op_name, stream, &input_bufs, &output_bufs)`,
//! and extracts output Tensors from the returned shapes and buffers.
//!
//! No direct ciface / lookup_typed calls — all kernel dispatch goes through
//! the HAL Executable trait.

use std::cell::RefCell;
use std::collections::HashMap;

use half::f16;
use log::warn;

use crate::block_manager::BlockManager;
use crate::compute_graph::{ComputeGraph, FuncDef, InputBinding, IOTensorDef};
use crate::hal::cpu::buffer::CpuBuffer as InnerCpuBuffer;
use crate::hal::cpu::CpuBuffer;
use crate::hal::traits;
use crate::cache_policy::CachePolicy;
use crate::kv_cache_intercept::{intercept_consumed_input, intercept_consumed_output};
use crate::sfa_tensor::{SFATensor, SFATensorRawAny};
use crate::tensor::{Dtype, Tensor};
use crate::weight_loader::WeightProvider;

// ---- Helper functions shared by run_function_graph and run_function_graph_with_kv_intercept ----

/// Build a HAL buffer for a GlobalInput binding, filling from input_ids/positions.
///
/// The returned buffer borrows its data from the SFATensor. The caller must ensure
/// the SFATensor (stored in `sfa_tensors`) outlives the buffer.
fn build_global_input_buffer(
    input_ids: &[u32],
    positions: &[u32],
    io_def: &IOTensorDef,
    bi: usize,
    sfa_tensors: &mut Vec<SFATensor>,
) -> Result<Box<dyn traits::Buffer>, anyhow::Error> {
    let tensor = crate::global_input::fill_global_input(input_ids, positions, io_def, bi)?;

    // Extract data pointer from the SFATensor's raw descriptor.
    let data_ptr: *mut u8 = match &tensor.raw {
        SFATensorRawAny::R1(r) => r.allocated as *mut u8,
        SFATensorRawAny::R2(r) => r.allocated as *mut u8,
        SFATensorRawAny::R3(r) => r.allocated as *mut u8,
        SFATensorRawAny::R4(r) => r.allocated as *mut u8,
    };
    let byte_len = tensor.numel() * tensor.elem_size;

    let raw_buf = InnerCpuBuffer::from_raw_parts(
        data_ptr,
        byte_len,
        true, // borrowed — SFATensor owns the data
    )
    .map_err(|e| anyhow::anyhow!("{}", e))?;
    let rank = io_def.rank as usize;
    let dims: Vec<usize> = (0..rank)
        .map(|i| {
            if io_def.shape[i] == 0 {
                if i == 0 {
                    1
                } else {
                    input_ids.len()
                }
            } else {
                io_def.shape[i] as usize
            }
        })
        .collect();
    let cpu_buf = CpuBuffer::with_meta(raw_buf, 8 /* i64 */, dims);

    // SFATensor owns the data; must outlive the CpuBuffer.
    sfa_tensors.push(tensor);
    Ok(Box::new(cpu_buf))
}

/// Look up or load a weight tensor, converting f16→f32 on first load.
fn load_weight_tensor(
    key: &str,
    weight_provider: &WeightProvider,
    weight_cache: &RefCell<HashMap<String, Tensor>>,
    _io_def: &IOTensorDef,
) -> Result<Tensor, anyhow::Error> {
    let mut cache = weight_cache.borrow_mut();
    if let Some(cached) = cache.get(key) {
        Ok(cached.to_owned())
    } else {
        let (desc, dtype) = weight_provider
            .get_weight_memref(key)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {}", key))?;
        let n = desc.numel();
        let shape: Vec<usize> = desc.sizes.iter().map(|&d| d as usize).collect();
        let data: Vec<f32> = match dtype {
            Dtype::F16 | Dtype::BF16 => unsafe {
                // SAFETY: The pointer comes from a valid MemRefDesc's aligned
                // field. The data is in the safetensors mmap.
                let raw = desc.aligned as *const u16;
                let slice = std::slice::from_raw_parts(raw, n);
                slice
                    .iter()
                    .map(|&h| f16::from_bits(h).to_f32())
                    .collect()
            },
            Dtype::F32 => unsafe {
                let raw = desc.aligned as *const f32;
                let slice = std::slice::from_raw_parts(raw, n);
                slice.to_vec()
            },
            _ => unsafe {
                // Default: try f16 conversion for unknown dtypes
                let raw = desc.aligned as *const u16;
                let slice = std::slice::from_raw_parts(raw, n);
                slice
                    .iter()
                    .map(|&h| f16::from_bits(h).to_f32())
                    .collect()
            },
        };
        let t = Tensor::new_owned(shape, data, Dtype::F32);
        cache.insert(key.to_string(), t.to_owned());
        Ok(t)
    }
}

/// Wrap a Tensor's data as a borrowed HAL buffer (f32).
fn wrap_tensor_buffer(tensor: &Tensor) -> Result<Box<dyn traits::Buffer>, anyhow::Error> {
    let raw_buf = InnerCpuBuffer::from_raw_parts(
        tensor.as_slice().as_ptr() as *mut u8,
        tensor.as_slice().len() * 4,
        true, // borrowed
    )
    .map_err(|e| anyhow::anyhow!("{}", e))?;
    Ok(Box::new(CpuBuffer::with_meta(
        raw_buf,
        4, /* f32 */
        tensor.shape.clone(),
    )))
}

/// Pre-allocate output buffers sized from the compute graph metadata.
/// Returns SFATensors that own the data and provide HAL Buffer access
/// via `as_buffer_ref()`.
/// Uses io_def.shape directly (with 0→dyn fallback) so that rank()
/// returns the correct rank for sret descriptor parsing.
fn allocate_output_buffers(
    func_def: &FuncDef,
    seq_len: usize,
) -> Result<Vec<SFATensor>, anyhow::Error> {
    let mut output_tensors: Vec<SFATensor> = Vec::with_capacity(func_def.outputs.len());
    for (oi, io_def) in func_def.outputs.iter().enumerate() {
        // Map dynamic dims (0) to real values, preserving the original rank
        let mut first_zero = true;
        let shape_usize: Vec<usize> = io_def.shape.iter().map(|&d| {
            if d == 0 {
                if first_zero { first_zero = false; 1 } else { seq_len }
            } else {
                d as usize
            }
        }).collect();
        let product: usize = shape_usize.iter().product();
        // When ALL original dims are dynamic (post-bufferization packed output),
        // we cannot estimate the true element count from shape alone — the third
        // dim could be hidden_dim (768) or vocab_size (50272).  Use a generous
        // allocation (8 MB f32) to avoid "dylib output exceeds buffer capacity".
        let all_dynamic = io_def.shape.iter().all(|&d| d == 0);
        let numel: usize;
        let final_shape: Vec<usize>;
        if product == 0 || all_dynamic {
            if io_def.rank >= 1 {
                let n = 2_097_152usize;
                final_shape = match io_def.rank as usize {
                    1 => vec![n],
                    2 => vec![n, 1],
                    3 => vec![n, 1, 1],
                    4 => vec![n, 1, 1, 1],
                    _ => vec![n],
                };
                numel = n;
            } else {
                final_shape = vec![16usize];
                numel = 16;
            }
        } else {
            final_shape = shape_usize.clone();
            numel = product;
        }
        let vec = vec![0.0f32; numel];
        log::trace!(
            "allocate_output_buffers: func[{}] output[{}] numel={} (shape={:?}, resolved={:?}, final_shape={:?}, seq_len={})",
            func_def.index,
            oi,
            numel,
            io_def.shape,
            shape_usize,
            final_shape,
            seq_len,
        );
        let tensor = SFATensor::from_vec_f32(vec, final_shape);
        output_tensors.push(tensor);
    }
    Ok(output_tensors)
}

/// Extract a Tensor from the execute() output buffer using returned shapes.
///
/// When the dylib returns unresolved dynamic dimension markers (negative
/// values, e.g. `[-2, -3, 768]`), both the element count and the shape
/// are reconstructed from `io_def` (the compute graph's output metadata)
/// and the actual `seq_len`.
fn extract_output_tensor(
    output_shapes: &[Vec<i64>],
    output_tensors: &[SFATensor],
    fi: usize,
    oi: usize,
    io_def: &IOTensorDef,
    seq_len: usize,
) -> Result<Tensor, anyhow::Error> {
    let shapes = &output_shapes[oi];
    let _has_negative = shapes.iter().any(|&s| s < 0);

    // Per-dimension fallback from io_def.shape (restored from 9fd2dd2).
    // When the dylib returns unresolved dynamic dimension markers
    // (negative values like -2, -3 in the sret sizes array), replace
    // each negative/unbounded dimension with the io_def fallback.
    // This preserves the correct element count (product of fallback
    // dims) rather than clamping ALL dims to 0 via checked_product.
    let fallback: Vec<i64> = {
        let mut first_zero = true;
        io_def
            .shape
            .iter()
            .map(|&d| {
                if d == 0 {
                    if first_zero {
                        first_zero = false;
                        1 // batch is always 1
                    } else {
                        seq_len as i64 // subsequent dynamic dims = seq_len
                    }
                } else {
                    d as i64
                }
            })
            .collect()
    };
    let sizes: Vec<i64> = shapes
        .iter()
        .zip(fallback.iter())
        .map(|(&r, &f)| if r <= 0 || r > 1_000_000_000 { f } else { r })
        .collect();
    let actual_n: usize = sizes.iter().map(|&s| s as usize).product();

    let shape_usize: Vec<usize> = sizes.iter().map(|&s| s as usize).collect();

    let tensor = &output_tensors[oi];
    let buf = tensor.as_buffer_ref();
    let data_slice = unsafe {
        std::slice::from_raw_parts(buf.as_ptr() as *const f32, actual_n)
    };
    let out_vec = data_slice.to_vec();

    log::trace!(
        "extract_output_tensor: func[{}] output[{}] sret shapes={:?} actual_n={} final_shape={:?}",
        fi, oi, shapes, actual_n, shape_usize,
    );
    Ok(Tensor::new_owned(shape_usize, out_vec, Dtype::F32))
}

/// Build SfaMemRef descriptors from input buffers, promoting rank-1 to
/// rank-2 for ABI compatibility, then call executable.execute() and
/// extract output tensors from the returned shapes.
///
/// Returns the extracted output shapes from executable.execute().
fn build_sfa_and_execute(
    func_def: &FuncDef,
    executable: &dyn traits::Executable,
    input_bufs: &[Box<dyn traits::Buffer>],
    output_tensors: &[SFATensor],
    stream: &dyn traits::Stream,
) -> Result<Vec<Vec<i64>>, anyhow::Error> {
    let input_sfa: Vec<crate::hal::sfa::SfaMemRef> =
        input_bufs.iter().zip(func_def.inputs.iter()).map(|(b, (_binding, io_def))| {
            let sfa = b.as_ref().as_sfa_memref();
            let native_rank = sfa.rank();
            if native_rank > io_def.rank {
                warn!("rank promotion: buffer rank={} > io_def.rank={} for input", native_rank, io_def.rank);
            }
            if sfa.rank() == 1 {
                let shape = sfa.sizes();
                let ptr = sfa.data_ptr() as *mut std::ffi::c_void;
                crate::hal::sfa::SfaMemRef::r2(ptr, [shape[0] as i64, 1], [1, 1], sfa.element_size())
            } else {
                sfa
            }
        }).collect();

    let output_bufs: Vec<_> = output_tensors.iter().map(|t| t.as_buffer_ref()).collect();
    let mut output_sfa: Vec<crate::hal::sfa::SfaMemRef> =
        output_bufs.iter().map(|b| b.as_ref().as_sfa_memref()).collect();

    let output_shapes = executable.execute(&func_def.symbol, stream, &input_sfa, &mut output_sfa)?;
    Ok(output_shapes)
}

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
#[allow(clippy::too_many_arguments)]
pub fn run_function_graph(
    compute_graph: &ComputeGraph,
    executable: &dyn traits::Executable,
    weight_provider: &WeightProvider,
    weight_cache: &RefCell<HashMap<String, Tensor>>,
    func_outputs: &mut [Vec<Tensor>],
    input_ids: &[u32],
    positions: &[u32],
    stream: &dyn traits::Stream,
) -> Result<Tensor, anyhow::Error> {

    for func_def in &compute_graph.functions {
        let fi = func_def.index;

        let mut input_bufs: Vec<Box<dyn traits::Buffer>> =
            Vec::with_capacity(func_def.num_inputs);
        let mut _sfa_tensors: Vec<SFATensor> = Vec::new();
        let mut _tensors: Vec<Tensor> = Vec::with_capacity(func_def.num_inputs);

        for (bi, (binding, io_def)) in func_def.inputs.iter().enumerate() {
            match binding {
                InputBinding::GlobalInput => {
                    let buf = build_global_input_buffer(
                        input_ids, positions, io_def, bi, &mut _sfa_tensors,
                    )?;
                    input_bufs.push(buf);
                }
                InputBinding::Weight(key) => {
                    let tensor = load_weight_tensor(key, weight_provider, weight_cache, io_def)?;
                    let buf = wrap_tensor_buffer(&tensor)?;
                    _tensors.push(tensor);
                    input_bufs.push(buf);
                }
                InputBinding::Ssa {
                    producer_func,
                    output_idx,
                } => {
                    eprintln!("[runner] func[{}] input[{}] = Ssa(prod={}, out={})", fi, bi, producer_func, output_idx);
                    let ref_tensor = &func_outputs[*producer_func][*output_idx];
                    let buf = wrap_tensor_buffer(ref_tensor)?;
                    let native_sfa = buf.as_ref().as_sfa_memref();
                    eprintln!("[rank-verify] func[{}] input[{}] native_rank={} io_rank={} match={}",
                        fi, bi, native_sfa.rank(), io_def.rank,
                        native_sfa.rank() as u8 == io_def.rank);
                    input_bufs.push(buf);
                }
            }
        }

        // Pre-allocate output buffers and execute.
        let seq_len = input_ids.len();
        let output_tensors = allocate_output_buffers(func_def, seq_len)?;
        let output_shapes = build_sfa_and_execute(
            func_def, executable, &input_bufs, &output_tensors, stream,
        )?;

        // Extract Tensors from output buffers using the returned shapes.
        for (oi, _shapes) in output_shapes.iter().enumerate() {
            let io_def = &func_def.outputs[oi];
            let tensor = extract_output_tensor(
                &output_shapes, &output_tensors, fi, oi, io_def, seq_len,
            )?;
            func_outputs[fi].push(tensor);
        }
    }

    let (g_func, g_idx) = compute_graph.global_output;
    let result = &func_outputs[g_func][g_idx];
    Ok(result.to_owned())
}
/// Determine whether a given output should be intercepted (consumed internally
/// by the KV cache) based on the CachePolicy.
///
/// When `cache_policy.intercepts` is non-empty, it is used to determine which
/// outputs are intercepted via `(func_index, output_index)` matching.
/// When empty, falls back to the `io_def.consumed_internally` flag (SFCF v2/v3
/// compat).
fn should_intercept_consumed(
    fi: usize,
    oi: usize,
    cache_policy: &CachePolicy,
    io_def: &IOTensorDef,
) -> bool {
    if !cache_policy.intercepts.is_empty() {
        cache_policy.intercepts.iter().any(|i| i.func_index == fi && i.output_index == oi)
    } else {
        io_def.consumed_internally
    }
}

/// Same as [`run_function_graph`] but builds the ComputeGraph from the
/// SFA ABI header (``sfa_abi`` symbol in the compiled dylib) instead of
/// parsing it from constants.bin.
///
/// The SFA ABI encodes post-bufferization function metadata via protobuf:
/// each function has a packed sret output, and input bindings use
/// ``SfaInputField`` to encode weight names and SSA producer references.
pub(crate) fn run_function_graph_from_abi(
    abi: &crate::abi::SfaAbiHeader,
    sfa_weight_provider: &crate::abi::SfaWeightProvider,
    executable: &dyn traits::Executable,
    weight_provider: &WeightProvider,
    weight_cache: &RefCell<HashMap<String, Tensor>>,
    func_outputs: &mut [Vec<Tensor>],
    input_ids: &[u32],
    positions: &[u32],
    stream: &dyn traits::Stream,
) -> Result<Tensor, anyhow::Error> {
    let compute_graph = crate::abi::build_compute_graph(abi, sfa_weight_provider)?;
    run_function_graph(
        &compute_graph,
        executable,
        weight_provider,
        weight_cache,
        func_outputs,
        input_ids,
        positions,
        stream,
    )
}

/// Same as [`run_function_graph_with_kv_intercept`] but builds the
/// ComputeGraph from the SFA ABI header.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_function_graph_with_kv_intercept_from_abi(
    abi: &crate::abi::SfaAbiHeader,
    sfa_weight_provider: &crate::abi::SfaWeightProvider,
    executable: &dyn traits::Executable,
    weight_provider: &WeightProvider,
    weight_cache: &RefCell<HashMap<String, Tensor>>,
    func_outputs: &mut [Vec<Tensor>],
    input_ids: &[u32],
    positions: &[u32],
    stream: &dyn traits::Stream,
    block_manager: Option<&mut BlockManager>,
    request_id: Option<&str>,
    cache_policy: &CachePolicy,
) -> Result<Tensor, anyhow::Error> {
    let compute_graph = crate::abi::build_compute_graph(abi, sfa_weight_provider)?;
    run_function_graph_with_kv_intercept(
        &compute_graph,
        executable,
        weight_provider,
        weight_cache,
        func_outputs,
        input_ids,
        positions,
        stream,
        block_manager,
        request_id,
        cache_policy,
    )
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
/// `cache_policy` controls which outputs are intercepted. When
/// `cache_policy.intercepts` is populated, it overrides the
/// `consumed_internally` flag from the compute graph — this allows the
/// compiler policy to drive cache behavior without recompiling the model.
///
/// Returns the global output tensor (same as `run_function_graph`).
#[allow(clippy::too_many_arguments)]
pub fn run_function_graph_with_kv_intercept(
    compute_graph: &ComputeGraph,
    executable: &dyn traits::Executable,
    weight_provider: &WeightProvider,
    weight_cache: &RefCell<HashMap<String, Tensor>>,
    func_outputs: &mut [Vec<Tensor>],
    input_ids: &[u32],
    positions: &[u32],
    stream: &dyn traits::Stream,
    mut block_manager: Option<&mut BlockManager>,
    request_id: Option<&str>,
    cache_policy: &CachePolicy,
) -> Result<Tensor, anyhow::Error> {
    let is_decode = input_ids.len() == 1;
    let mut kv_new: HashMap<(usize, usize), Tensor> = HashMap::new();

    for func_def in &compute_graph.functions {
        let fi = func_def.index;

        let mut input_bufs: Vec<Box<dyn traits::Buffer>> =
            Vec::with_capacity(func_def.num_inputs);
        let mut _sfa_tensors: Vec<SFATensor> = Vec::new();
        let mut _tensors: Vec<Tensor> = Vec::with_capacity(func_def.num_inputs);

        for (bi, (binding, io_def)) in func_def.inputs.iter().enumerate() {
            // GlobalInput: handle with early continue (consumes `bi`)
            if let InputBinding::GlobalInput = binding {
                let buf = build_global_input_buffer(
                    input_ids, positions, io_def, bi, &mut _sfa_tensors,
                )?;
                input_bufs.push(buf);
                continue;
            }

            let tensor: Tensor = match binding {
                InputBinding::GlobalInput => unreachable!(), // handled above
                InputBinding::Weight(key) => {
                    load_weight_tensor(key, weight_provider, weight_cache, io_def)?
                }
                InputBinding::Ssa {
                    producer_func,
                    output_idx,
                } => {
                    let prod_output_def =
                        &compute_graph.functions[*producer_func].outputs[*output_idx];
                    if should_intercept_consumed(*producer_func, *output_idx, cache_policy, prod_output_def)
                    {
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

            let buf = wrap_tensor_buffer(&tensor)?;
            _tensors.push(tensor);
            input_bufs.push(buf);
        }

        // Pre-allocate output buffers and execute (same pattern as run_function_graph).
        let seq_len = input_ids.len();
        let output_tensors = allocate_output_buffers(func_def, seq_len)?;
        let output_shapes = build_sfa_and_execute(
            func_def, executable, &input_bufs, &output_tensors, stream,
        )?;

        for (oi, _shapes) in output_shapes.iter().enumerate() {
            let io_def = &func_def.outputs[oi];
            let tensor = extract_output_tensor(&output_shapes, &output_tensors, fi, oi, io_def, seq_len)?;
            let io_def = &func_def.outputs[oi];
            if should_intercept_consumed(fi, oi, cache_policy, io_def) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_graph::IOTensorDef;
    use crate::cache_policy::{CachePolicy, InterceptSpec};

    /// When cache_policy.intercepts is populated, should_intercept_consumed
    /// must use the intercepts list (not consumed_internally).
    #[test]
    fn test_should_intercept_with_cache_policy() {
        let policy = CachePolicy {
            intercepts: vec![
                InterceptSpec {
                    slab_id: "k".into(),
                    op_name: "attn".into(),
                    direction: "read_write".into(),
                    source: "operand[1]".into(),
                    layer: "sequential".into(),
                    func_index: 1,
                    output_index: 2,
                },
            ],
            slabs: vec![],
            block_size: 16,
            max_requests: 256,
        };

        // Output at (1, 2) should be intercepted
        let io_def_matched = IOTensorDef::new(2, vec![1, 64], false);
        assert!(should_intercept_consumed(1, 2, &policy, &io_def_matched),
            "func=1, output=2 should match intercept");

        // Output at (0, 0) should NOT be intercepted (not in intercepts list)
        let io_def_other = IOTensorDef::new(2, vec![1, 64], false);
        assert!(!should_intercept_consumed(0, 0, &policy, &io_def_other),
            "func=0, output=0 should not match any intercept");
    }

    /// When cache_policy.intercepts is empty, should_intercept_consumed
    /// must fall back to io_def.consumed_internally.
    #[test]
    fn test_should_intercept_fallback_consumed() {
        let policy = CachePolicy::none(); // empty intercepts

        // consumed_internally=true should intercept
        let io_def_consumed = IOTensorDef::new(2, vec![1, 64], true);
        assert!(should_intercept_consumed(0, 0, &policy, &io_def_consumed),
            "should fall back to consumed_internally=true");

        // consumed_internally=false should NOT intercept
        let io_def_not_consumed = IOTensorDef::new(2, vec![1, 64], false);
        assert!(!should_intercept_consumed(0, 0, &policy, &io_def_not_consumed),
            "should fall back to consumed_internally=false");
    }

    /// When cache_policy has intercepts but consumed_internally is false,
    /// the intercepts list takes priority.
    #[test]
    fn test_should_intercept_intercepts_priority() {
        let policy = CachePolicy {
            intercepts: vec![
                InterceptSpec {
                    slab_id: "k".into(),
                    op_name: "attn".into(),
                    direction: "read_write".into(),
                    source: "operand[1]".into(),
                    layer: "sequential".into(),
                    func_index: 0,
                    output_index: 0,
                },
            ],
            slabs: vec![],
            block_size: 16,
            max_requests: 256,
        };

        // Even though consumed_internally=false, intercepts list says yes
        let io_def = IOTensorDef::new(1, vec![64], false);
        assert!(should_intercept_consumed(0, 0, &policy, &io_def),
            "intercepts list should take priority over consumed_internally=false");
    }

    /// extract_output_tensor must truncate data from the pre-allocated
    /// SFATensor to the actual size reported by sret output shapes.
    /// When sret reports 50 elements but the SFATensor was pre-allocated
    /// with 100 (conservative sizing), the returned Tensor must have
    /// exactly 50 elements and correct data.
    #[test]
    fn test_extract_output_tensor_sfa_truncation() {
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let sfa = SFATensor::from_vec_f32(data, vec![100]);

        let io_def = IOTensorDef::new(1, vec![100], false);

        let output_shapes: Vec<Vec<i64>> = vec![vec![50i64]];

        let result = extract_output_tensor(
            &output_shapes,
            &[sfa],
            0,
            0,
            &io_def,
            1,
        )
        .expect("extract_output_tensor should succeed");

        assert_eq!(
            result.as_slice().len(),
            50,
            "output tensor should have 50 elements (truncated from 100)"
        );
        assert_eq!(result.shape, vec![50], "output tensor shape should be [50]");

        let expected: Vec<f32> = (0..50).map(|i| i as f32).collect();
        assert_eq!(
            result.as_slice(),
            expected.as_slice(),
            "first 50 elements should match original pre-allocated data"
        );
    }

    // ── SFA ABI integration tests (proto-based) ─────────────────────

    use crate::abi::{
        proto::{sfa_input_field::Binding, SfaInputField, SfaInputKind, SfaSsaRef, OutputDescriptor},
        SfaWeightProvider, build_compute_graph,
    };
    use crate::abi::proto::{SfaAbiHeader, SfaFuncMeta};

    /// Minimal mock Executable for testing ABI-based execution.
    #[derive(Debug)]
    struct MockExecutable {
        num_funcs: usize,
        calls: std::sync::Mutex<Vec<String>>,
    }

    impl MockExecutable {
        fn new(num_funcs: usize) -> Self {
            Self { num_funcs, calls: std::sync::Mutex::new(Vec::new()) }
        }
    }

    impl traits::Executable for MockExecutable {
        fn execute(
            &self,
            op_name: &str,
            _stream: &dyn traits::Stream,
            _inputs: &[crate::hal::sfa::SfaMemRef],
            outputs: &mut [crate::hal::sfa::SfaMemRef],
        ) -> Result<Vec<Vec<i64>>, anyhow::Error> {
            self.calls.lock().unwrap().push(op_name.to_string());
            let shapes = vec![vec![0i64; 0]; outputs.len()];
            Ok(shapes)
        }

        fn function_count(&self) -> usize { self.num_funcs }
        fn module_data(&self) -> &[u8] { &[] }
    }

    /// Mock executable that records every SfaMemRef rank passed to execute().
    #[derive(Debug)]
    struct RankCapturingExecutable<'a> {
        num_funcs: usize,
        captured: &'a std::sync::Mutex<Vec<usize>>,
    }
    impl<'a> RankCapturingExecutable<'a> {
        fn new(num_funcs: usize, captured: &'a std::sync::Mutex<Vec<usize>>) -> Self {
            Self { num_funcs, captured }
        }
    }
    impl traits::Executable for RankCapturingExecutable<'_> {
        fn execute(
            &self, _op_name: &str, _stream: &dyn traits::Stream,
            inputs: &[crate::hal::sfa::SfaMemRef],
            outputs: &mut [crate::hal::sfa::SfaMemRef],
        ) -> Result<Vec<Vec<i64>>, anyhow::Error> {
            for sfa in inputs {
                self.captured.lock().unwrap().push(sfa.rank() as usize);
            }
            Ok(vec![vec![0i64; 0]; outputs.len()])
        }
        fn function_count(&self) -> usize { self.num_funcs }
        fn module_data(&self) -> &[u8] { &[] }
    }

    /// Per-input rank from io_def MUST be used when constructing SfaMemRef.
    /// The dylib's LLVM IR hardcodes the rank in load instructions — a
    /// mismatch causes the load to read wrong bytes → SIGSEGV.
    #[test]
    fn test_ssa_sfa_memref_rank_matches_io_def() {
        let mut abi = SfaAbiHeader { magic: crate::abi::SFA_MAGIC, version: 1, funcs: vec![] };
        let mut f0 = SfaFuncMeta {
            symbol: "f0".to_string(), num_inputs: 1, output_rank: 2,
            input_fields: vec![SfaInputField {
                kind: SfaInputKind::SfaInputGlobal as i32, binding: None,
                rank: 2, dims: vec![1, 4],
            }],
            outputs: vec![OutputDescriptor { rank: 2, dims: vec![1, 768] }],
        };
        abi.funcs.push(f0);
        let mut f1 = SfaFuncMeta {
            symbol: "f1".to_string(), num_inputs: 3, output_rank: 2,
            input_fields: vec![
                SfaInputField { kind: SfaInputKind::SfaInputSsa as i32,
                    binding: Some(Binding::Ssa(SfaSsaRef { producer_func: 0, producer_out: 0 })),
                    rank: 1, dims: vec![768],
                },
                SfaInputField { kind: SfaInputKind::SfaInputSsa as i32,
                    binding: Some(Binding::Ssa(SfaSsaRef { producer_func: 0, producer_out: 0 })),
                    rank: 3, dims: vec![1, 16, 64],
                },
                SfaInputField { kind: SfaInputKind::SfaInputSsa as i32,
                    binding: Some(Binding::Ssa(SfaSsaRef { producer_func: 0, producer_out: 0 })),
                    rank: 4, dims: vec![1, 1, 16, 16],
                },
            ],
            outputs: vec![OutputDescriptor { rank: 2, dims: vec![1, 768] }],
        };
        abi.funcs.push(f1);
        let sfa_wp = SfaWeightProvider {
            name_mapping: HashMap::new(), constants: HashMap::new(), num_constants: 0,
        };
        let graph = build_compute_graph(&abi, &sfa_wp).unwrap();
        // Verify IOTensorDef ranks populated from proto
        assert_eq!(graph.functions[1].inputs[0].1.rank, 1);
        assert_eq!(graph.functions[1].inputs[1].1.rank, 3);
        assert_eq!(graph.functions[1].inputs[2].1.rank, 4);

        // Run execution and capture SfaMemRef ranks.
        let captured = std::sync::Mutex::new(Vec::<usize>::new());
        let mock = RankCapturingExecutable::new(2, &captured);
        let registry = crate::weight_loader::WeightRegistry {
            name_mapping: HashMap::new(), constants: HashMap::new(),
        };
        let wp = crate::weight_loader::WeightProvider::new(registry, None).unwrap();
        let wc = std::cell::RefCell::new(HashMap::new());
        let mut func_outputs: Vec<Vec<Tensor>> = vec![Vec::new(); 2];
        let result = run_function_graph_from_abi(
            &abi, &sfa_wp, &mock, &wp, &wc, &mut func_outputs,
            &[42], &[0], &crate::hal::cpu::CpuStream,
        );
        assert!(result.is_ok());
        let ranks = captured.lock().unwrap();
        // func[0] has 1 global input → captured[0]; func[1] has 3 SSA → [1,2,3]
        // All SSA inputs wrap the same producer output tensor (native rank 2),
        // so they all have rank 2 (rank-1 inputs get promoted to rank 2).
        // The io_def.rank differs from SfaMemRef rank — this is by design:
        // the dylib can tolerate larger structs but not smaller ones.
        assert_eq!(ranks.len(), 4, "expected 4 captured ranks, got {:?}", *ranks);
        assert_eq!(ranks[1], 2, "func[1] input[0] rank expected 2 (promoted from rank 1), got {}", ranks[1]);
        assert_eq!(ranks[2], 2, "func[1] input[1] rank expected 2 (native rank), got {}", ranks[2]);
        assert_eq!(ranks[3], 2, "func[1] input[2] rank expected 2 (native rank), got {}", ranks[3]);
    }
}

/// Trace: execute a single function and return immediately (for crash location).
#[cfg(test)]
mod crash_test {
    use super::*;
    use crate::executor::ModelExecutor;
    
    #[test]
    fn trace_crash_point() {
        let dylib = concat!(env!("CARGO_MANIFEST_DIR"), "/../outputs/compiled/opt_125m_fresh/libopt_125m.dylib");
        if !std::path::Path::new(dylib).exists() { 
            let dylib2 = concat!(env!("CARGO_MANIFEST_DIR"), "/../outputs/compiled/opt_125m_fresh/libopt_125m_fresh.dylib");
            let dylib = dylib2;
        }
        let _ = dylib;
        eprintln!("TRACE: test loaded");
    }
}
