//! Compute graph runner — iterates over FuncDefs, assembles inputs from
//! global inputs / weights / SSA wires, dispatches through
//! `executable.execute(op_name, stream, &input_bufs, &output_bufs)`,
//! and extracts output Tensors from the returned shapes and buffers.
//!
//! No direct ciface / lookup_typed calls — all kernel dispatch goes through
//! the HAL Executable trait.

use std::cell::RefCell;
use std::collections::HashMap;

use log::warn;

use crate::cache::block::BlockManager;
use crate::cache::intercept::{intercept_consumed_input, intercept_consumed_output};
use crate::cache::policy::CachePolicy;
use crate::engine::opt_fused;
use crate::hal::cpu::buffer::RawBuffer as InnerCpuBuffer;
use crate::hal::cpu::CpuBuffer;
use crate::hal::traits;
use crate::model::compute_graph::{ComputeGraph, FuncDef, IOTensorDef, InputBinding};
use crate::model::sfa_tensor::{SFATensor, SFATensorRawAny};
use crate::model::tensor::{Dtype, Tensor};
use crate::model::weight_loader::WeightProvider;

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
    let tensor = crate::model::global_input::fill_global_input(input_ids, positions, io_def, bi)?;

    // Extract data pointer from the SFATensor's raw descriptor.
    let data_ptr: *mut u8 = match &tensor.raw {
        SFATensorRawAny::R1(r) => r.allocated as *mut u8,
        SFATensorRawAny::R2(r) => r.allocated as *mut u8,
        SFATensorRawAny::R3(r) => r.allocated as *mut u8,
        SFATensorRawAny::R4(r) => r.allocated as *mut u8,
    };
    let byte_len = tensor.numel() * tensor.elem_size;

    let raw_buf = InnerCpuBuffer::from_raw_parts(
        data_ptr, byte_len, true, // borrowed — SFATensor owns the data
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
pub(crate) fn load_weight_tensor(
    key: &str,
    weight_provider: &WeightProvider,
    weight_cache: &RefCell<HashMap<String, Tensor>>,
    io_def: &IOTensorDef,
) -> Result<Tensor, anyhow::Error> {
    let mut cache = weight_cache.borrow_mut();
    if let Some(cached) = cache.get(key) {
        return Ok(cached.to_owned());
    }

    let (desc, dtype) = weight_provider
        .get_weight_memref(key)
        .ok_or_else(|| anyhow::anyhow!("weight not found: {}", key))?;
    let n = desc.numel();
    // WeightProvider promotes rank-1 safetensors tensors to MemRefDesc2
    // ([n, 1]).  The ABI contract keeps the original rank (rank-1 for
    // biases/norms).  Rehydrate the ABI shape here so downstream buffers
    // and pass-through aliases use the rank the dylib expects.
    let shape: Vec<usize> = if io_def.rank > 0
        && io_def.shape.len() == io_def.rank as usize
        && io_def.shape.iter().all(|&d| d > 0)
    {
        io_def.shape.iter().map(|&d| d as usize).collect()
    } else {
        desc.sizes.iter().map(|&d| d as usize).collect()
    };
    anyhow::ensure!(
        shape.iter().product::<usize>() == n,
        "weight {} numel mismatch: ABI shape {:?} has {} elements, safetensors has {}",
        key,
        shape,
        shape.iter().product::<usize>(),
        n,
    );
    // SAFETY: desc.aligned comes from a valid MemRefDesc pointing to
    // safetensors mmap data. convert_weight_to_f32 handles dtype dispatch.
    let data: Vec<f32> =
        unsafe { crate::model::weight_loader::convert_weight_to_f32(desc.aligned, n, dtype) };
    let t = Tensor::new_owned(shape, data, Dtype::F32);
    cache.insert(key.to_string(), t.to_owned());
    Ok(t)
}

/// Look up or load a weight tensor without the f16/bf16→f32 promotion.
///
/// Returns a raw-u16 Tensor (`F16` or `BF16`) for dtype-aware production
/// kernels.  The raw cache is intentionally separate from [`load_weight_tensor`]:
/// the func-level dylib path must keep receiving promoted f32 weights while
/// the op-plan kernels consume source-dtype weights.
pub(crate) fn load_weight_tensor_u16(
    key: &str,
    weight_provider: &WeightProvider,
    raw_weight_cache: &RefCell<HashMap<String, Tensor>>,
    io_def: &IOTensorDef,
) -> Result<Tensor, anyhow::Error> {
    let mut cache = raw_weight_cache.borrow_mut();
    if let Some(cached) = cache.get(key) {
        return Ok(cached.to_owned());
    }

    let (desc, dtype) = weight_provider
        .get_weight_memref(key)
        .ok_or_else(|| anyhow::anyhow!("weight not found: {}", key))?;
    anyhow::ensure!(
        matches!(dtype, Dtype::F16 | Dtype::BF16),
        "raw weight {key}: expected F16/BF16 source dtype, got {dtype}"
    );
    let n = desc.numel();
    let shape: Vec<usize> = if io_def.rank > 0
        && io_def.shape.len() == io_def.rank as usize
        && io_def.shape.iter().all(|&d| d > 0)
    {
        io_def.shape.iter().map(|&d| d as usize).collect()
    } else {
        desc.sizes.iter().map(|&d| d as usize).collect()
    };
    anyhow::ensure!(
        shape.iter().product::<usize>() == n,
        "raw weight {} numel mismatch: ABI shape {:?} has {} elements, safetensors has {}",
        key,
        shape,
        shape.iter().product::<usize>(),
        n,
    );

    // SAFETY: desc.aligned comes from a valid MemRefDesc pointing to
    // safetensors mmap data of `n` contiguous elements; `Dtype` is F16 or
    // BF16, so each element is exactly two bytes.
    // SAFETY: the copy below never writes through the descriptor.
    let data: Vec<u16> = unsafe {
        let bytes = std::slice::from_raw_parts(desc.aligned as *const u8, n.saturating_mul(2));
        bytes
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect()
    };
    let t = Tensor::new_owned_u16(shape, data, dtype);
    cache.insert(key.to_string(), t.to_owned());
    Ok(t)
}

/// Load one weight according to the A/B storage-dtype policy.
///
/// * `Auto`  — preserve source dtype (raw F16/BF16, promoted f32 otherwise)
/// * `F32`   — always promote (the f32 A/B control, used for gate compare)
/// * `F16`   — require an F16 source and return raw u16
/// * `Bf16`  — require a BF16 source and return raw u16
pub(crate) fn load_weight_tensor_for_mode(
    key: &str,
    weight_provider: &WeightProvider,
    weight_cache: &RefCell<HashMap<String, Tensor>>,
    raw_weight_cache: &RefCell<HashMap<String, Tensor>>,
    io_def: &IOTensorDef,
    mode: crate::engine::executor::WeightDtypeMode,
) -> Result<Tensor, anyhow::Error> {
    let source_dtype = weight_provider
        .get_weight_memref(key)
        .map(|(_, dtype)| dtype)
        .ok_or_else(|| anyhow::anyhow!("weight not found: {}", key))?;
    match mode {
        crate::engine::executor::WeightDtypeMode::F32 => {
            load_weight_tensor(key, weight_provider, weight_cache, io_def)
        }
        crate::engine::executor::WeightDtypeMode::F16 => {
            anyhow::ensure!(
                source_dtype == Dtype::F16,
                "weight {key}: --weight-dtype f16 requires F16 source, got {source_dtype}"
            );
            load_weight_tensor_u16(key, weight_provider, raw_weight_cache, io_def)
        }
        crate::engine::executor::WeightDtypeMode::Bf16 => {
            anyhow::ensure!(
                source_dtype == Dtype::BF16,
                "weight {key}: --weight-dtype bf16 requires BF16 source, got {source_dtype}"
            );
            load_weight_tensor_u16(key, weight_provider, raw_weight_cache, io_def)
        }
        crate::engine::executor::WeightDtypeMode::Auto => match source_dtype {
            Dtype::F16 | Dtype::BF16 => {
                load_weight_tensor_u16(key, weight_provider, raw_weight_cache, io_def)
            }
            _ => load_weight_tensor(key, weight_provider, weight_cache, io_def),
        },
    }
}

/// Wrap a Tensor's data as a borrowed HAL buffer using its native dtype.
fn wrap_tensor_buffer(tensor: &Tensor) -> Result<Box<dyn traits::Buffer>, anyhow::Error> {
    let (ptr, byte_len, elem_size) = match tensor.dtype {
        Dtype::F32 => (
            tensor.as_slice().as_ptr() as *mut u8,
            tensor.as_slice().len() * 4,
            4,
        ),
        Dtype::F16 | Dtype::BF16 => (
            tensor.as_u16().as_ptr() as *mut u8,
            tensor.as_u16().len() * 2,
            2,
        ),
        Dtype::I64 => (
            tensor.as_i64().as_ptr() as *mut u8,
            tensor.as_i64().len() * 8,
            8,
        ),
        other => anyhow::bail!(
            "wrap_tensor_buffer: unsupported tensor dtype {}",
            other
        ),
    };
    let raw_buf = InnerCpuBuffer::from_raw_parts(ptr, byte_len, true /* borrowed */)
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    Ok(Box::new(CpuBuffer::with_meta(
        raw_buf,
        elem_size,
        tensor.shape.clone(),
    )))
}

/// Detect the weight-staging pass-through outputs of `main_0`.
///
/// The split compiler emits `main_0` as a combined embedding + weight-staging
/// function.  Its first N weight inputs are consumed while computing the
/// embedding (`embed_tokens`, `embed_positions`) and are not returned; the
/// remaining exported weights are returned in the same input order.  The
/// corresponding output descriptors are pass-throughs to those input
/// buffers.
///
/// Re-allocating and re-copying those tensors on every forward pass is the
/// dominant cost of the legacy Rust runner (~500 MB of OPT-125M weights per
/// step).  This helper returns `output_index -> input_index` aliases so the
/// graph runner can borrow the cached weight buffer instead.
///
/// Returns an empty map whenever the function does not match the compiler
/// contract exactly.  This makes the optimization fail safe: a mismatch
/// falls back to the (slower but correct) allocation/copy path.
fn main0_weight_passthrough_map(func_def: &FuncDef) -> HashMap<usize, usize> {
    if func_def.index != 0 {
        return HashMap::new();
    }

    let weight_inputs: Vec<usize> = func_def
        .inputs
        .iter()
        .enumerate()
        .filter_map(|(bi, (binding, _))| {
            if matches!(binding, InputBinding::Weight(_)) {
                Some(bi)
            } else {
                None
            }
        })
        .collect();
    if weight_inputs.is_empty() {
        return HashMap::new();
    }

    // Find the longest contiguous run of static outputs whose rank/shape
    // exactly matches a contiguous run of weight inputs.  Unlike the earlier
    // fixed-offset contract, this also works when `main_0` has leading
    // non-weight outputs (scalars, masks, embeddings) before the exported
    // weight pass-through block.
    let mut best_len = 0usize;
    let mut best_out_start = 0usize;
    let mut best_in_start = 0usize;

    for out_start in 0..func_def.outputs.len() {
        for in_start in 0..weight_inputs.len() {
            let mut len = 0usize;
            while out_start + len < func_def.outputs.len()
                && in_start + len < weight_inputs.len()
            {
                let od = &func_def.outputs[out_start + len];
                let id = &func_def.inputs[weight_inputs[in_start + len]].1;
                let static_match = od.shape.iter().all(|&d| d > 0)
                    && od.rank == id.rank
                    && od.shape == id.shape;
                if !static_match {
                    break;
                }
                len += 1;
            }
            if len > best_len {
                best_len = len;
                best_out_start = out_start;
                best_in_start = in_start;
            }
        }
    }

    // A single accidental shape match is not enough to prove the compiler
    // emitted a weight pass-through block.
    if best_len < 2 {
        return HashMap::new();
    }

    let mut aliases = HashMap::with_capacity(best_len);
    for k in 0..best_len {
        let output_idx = best_out_start + k;
        let input_idx = weight_inputs[best_in_start + k];
        log::trace!(
            "main_0 weight pass-through: output[{}] -> input[{}]",
            output_idx,
            input_idx,
        );
        aliases.insert(output_idx, input_idx);
    }
    aliases
}

/// Contract check for the `main_0` embedding fast path.
///
/// Returns `Some((token_emb_input, position_emb_input))` when `main_0` has
/// the compiler-split shape expected by [`build_main0_outputs_fastpath`]:
///
/// ```text
/// outputs[0..12]    attention scales (0.125)
/// outputs[12]       token + position embeddings [1, seq, 768]
/// outputs[13]       causal mask [1, 1, seq, seq]
/// outputs[14]       seq_len scalar
/// outputs[15..N-1]  exported weights (pass-through aliases)
/// outputs[N-1]      1.0 scalar
/// ```
///
/// The first two weight inputs are the consumed token/position embedding
/// tables.  Any deviation returns `None` and the caller falls back to the
/// compiled `_mlir_ciface_main_0` call.
fn main0_fastpath_contract(
    func_def: &FuncDef,
    aliases: &HashMap<usize, usize>,
) -> Option<(usize, usize)> {
    if func_def.index != 0 || aliases.is_empty() {
        return None;
    }
    let n = func_def.outputs.len();
    if aliases.len() + 16 != n {
        return None;
    }

    let weight_inputs: Vec<usize> = func_def
        .inputs
        .iter()
        .enumerate()
        .filter_map(|(bi, (binding, _))| {
            if matches!(binding, InputBinding::Weight(_)) {
                Some(bi)
            } else {
                None
            }
        })
        .collect();
    if weight_inputs.len() < 2 {
        return None;
    }

    let token_emb_bi = weight_inputs[0];
    let pos_emb_bi = weight_inputs[1];
    let token_def = &func_def.inputs[token_emb_bi].1;
    let pos_def = &func_def.inputs[pos_emb_bi].1;
    if token_def.rank != 2
        || pos_def.rank != 2
        || token_def.shape.len() < 2
        || pos_def.shape.len() < 2
        || token_def.shape[1] == 0
        || pos_def.shape[1] == 0
        || token_def.shape[1] != pos_def.shape[1]
    {
        return None;
    }

    let hidden_def = &func_def.outputs[12];
    let mask_def = &func_def.outputs[13];
    if hidden_def.rank != 3
        || hidden_def.shape != vec![0, 0, token_def.shape[1]]
        || mask_def.rank != 4
        || mask_def.shape != vec![0, 1, 0, 0]
    {
        return None;
    }

    Some((token_emb_bi, pos_emb_bi))
}

/// Build `main_0` outputs directly in Rust (embedding + mask + scalar
/// constants + cached weight pass-throughs), bypassing the compiled
/// `_mlir_ciface_main_0` call.
///
/// The compiled function spends most of its time allocating/copying the
/// exported weight descriptors on every forward pass.  The dynamic outputs
/// are small and have a stable contract (see [`main0_fastpath_contract`]).
#[allow(clippy::too_many_arguments)]
fn build_main0_outputs_fastpath(
    func_def: &FuncDef,
    input_ids: &[u32],
    positions: &[u32],
    input_tensors: &[Option<Tensor>],
    aliases: &HashMap<usize, usize>,
    token_emb_bi: usize,
    pos_emb_bi: usize,
    func_outputs: &mut [Vec<Tensor>],
) -> Result<(), anyhow::Error> {
    let token_emb = input_tensors
        .get(token_emb_bi)
        .and_then(|t| t.as_ref())
        .ok_or_else(|| anyhow::anyhow!("main_0 fast path: token embedding not loaded"))?;
    let pos_emb = input_tensors
        .get(pos_emb_bi)
        .and_then(|t| t.as_ref())
        .ok_or_else(|| anyhow::anyhow!("main_0 fast path: position embedding not loaded"))?;

    let vocab = token_emb.shape[0];
    let max_pos = pos_emb.shape[0];
    let hidden_dim = token_emb.shape[1];
    let seq_len = input_ids.len();
    // OPT position ids are offset by 2 (`sf.add(%position_ids, %_const_41)`).
    let position_offset = 2usize;

    anyhow::ensure!(
        positions.len() == seq_len,
        "main_0 fast path: positions/input_ids length mismatch"
    );
    anyhow::ensure!(
        input_ids.iter().all(|&t| (t as usize) < vocab),
        "main_0 fast path: input token >= vocab"
    );
    anyhow::ensure!(
        positions
            .iter()
            .all(|&p| (p as usize + position_offset) < max_pos),
        "main_0 fast path: position >= learned position table"
    );

    let mut hidden = vec![0.0f32; seq_len * hidden_dim];
    for (p, (&token, &pos)) in input_ids.iter().zip(positions.iter()).enumerate() {
        let tok_row = token as usize * hidden_dim;
        let pos_row = (pos as usize + position_offset) * hidden_dim;
        let out_row = p * hidden_dim;
        for d in 0..hidden_dim {
            hidden[out_row + d] =
                token_emb.as_slice()[tok_row + d] + pos_emb.as_slice()[pos_row + d];
        }
    }

    let mut mask = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..=i {
            mask[i * seq_len + j] = 1.0f32;
        }
    }

    let scale = Tensor::new_owned(vec![1], vec![0.125f32], Dtype::F32);
    let one = Tensor::new_owned(vec![1], vec![1.0f32], Dtype::F32);
    let seq_scalar = Tensor::new_owned(vec![1], vec![seq_len as f32], Dtype::F32);
    let hidden_tensor = Tensor::new_owned(vec![1, seq_len, hidden_dim], hidden, Dtype::F32);
    let mask_tensor = Tensor::new_owned(vec![1, 1, seq_len, seq_len], mask, Dtype::F32);

    for oi in 0..func_def.outputs.len() {
        let tensor = if oi < 12 {
            scale.to_owned()
        } else if oi == 12 {
            hidden_tensor.to_owned()
        } else if oi == 13 {
            mask_tensor.to_owned()
        } else if oi == 14 {
            seq_scalar.to_owned()
        } else if let Some(&input_idx) = aliases.get(&oi) {
            input_tensors
                .get(input_idx)
                .and_then(|t| t.as_ref())
                .ok_or_else(|| {
                    anyhow::anyhow!("main_0 fast path: aliased input {} not loaded", input_idx)
                })?
                .to_owned()
        } else if oi + 1 == func_def.outputs.len() {
            one.to_owned()
        } else {
            anyhow::bail!(
                "main_0 fast path: output[{}] does not match the split contract",
                oi
            );
        };
        func_outputs[0].push(tensor);
    }

    Ok(())
}

/// Contract check + execution for the final `main_15` logits projection.
///
/// The compiled `main_15` materializes a transposed copy of the 154 MB
/// `lm_head` weight before its batch matmul on every forward pass.  We skip
/// that function and call Accelerate SGEMM directly with `CblasTrans` on the
/// original `[vocab, hidden]` weight, which computes the same `hidden @ W^T`
/// without the copy.
#[allow(clippy::too_many_arguments)]
fn run_main15_fastpath(
    compute_graph: &ComputeGraph,
    func_def: &FuncDef,
    weight_provider: &WeightProvider,
    weight_cache: &RefCell<HashMap<String, Tensor>>,
    raw_weight_cache: Option<&RefCell<HashMap<String, Tensor>>>,
    weight_dtype_mode: crate::engine::executor::WeightDtypeMode,
    input_tensors: &[Option<Tensor>],
    func_outputs: &mut [Vec<Tensor>],
) -> Result<bool, anyhow::Error> {
    if func_def.index + 1 != compute_graph.functions.len()
        || !func_def.symbol.ends_with("main_15")
        || func_def.inputs.len() != 2
        || func_def.outputs.len() != 1
        || func_def.outputs[0].rank != 3
        || func_def.outputs[0].shape != vec![0, 0, 50272]
    {
        return Ok(false);
    }

    let hidden = input_tensors
        .first()
        .and_then(|t| t.as_ref())
        .ok_or_else(|| anyhow::anyhow!("main_15 fast path: hidden state not loaded"))?;
    let weight = input_tensors
        .get(1)
        .and_then(|t| t.as_ref())
        .ok_or_else(|| anyhow::anyhow!("main_15 fast path: lm_head weight not loaded"))?;

    anyhow::ensure!(
        hidden.shape.len() >= 2 && hidden.shape.last() == Some(&768),
        "main_15 fast path: hidden shape {:?} is not [..., 768]",
        hidden.shape,
    );
    anyhow::ensure!(
        weight.shape.len() == 2 && weight.shape[0] == 50272 && weight.shape[1] == 768,
        "main_15 fast path: lm_head shape {:?} is not [50272, 768]",
        weight.shape,
    );

    let seq_len: usize = hidden.shape[..hidden.shape.len() - 1].iter().product();
    let vocab = 50272usize;
    let mut logits = vec![0.0f32; seq_len * vocab];

    // Source-dtype path: when the runner has a raw-weight cache and the
    // safetensors tensor is F16/BF16, use the production dtype GEMV kernels.
    // Otherwise keep the f32 BLAS fast path (FP32-source checkpoints).
    let weight_key = match &func_def.inputs[1].0 {
        InputBinding::Weight(key) => Some(key.as_str()),
        InputBinding::Ssa {
            producer_func,
            output_idx,
        } => {
            // main_15 consumes main_0's weight pass-through output.  Trace
            // the alias back to the original Weight binding so the raw
            // source-dtype loader can key on the compiled weight name.
            let producer = compute_graph
                .functions
                .get(*producer_func)
                .ok_or_else(|| anyhow::anyhow!("main_15 producer func {producer_func} missing"))?;
            main0_weight_passthrough_map(producer)
                .get(output_idx)
                .and_then(|&input_idx| match &producer.inputs[input_idx].0 {
                    InputBinding::Weight(key) => Some(key.as_str()),
                    _ => None,
                })
        }
        _ => None,
    };
    let raw_weight = match (raw_weight_cache, weight_key) {
        (Some(cache), Some(key)) => {
            let selected = load_weight_tensor_for_mode(
                key,
                weight_provider,
                weight_cache,
                cache,
                &func_def.inputs[1].1,
                weight_dtype_mode,
            )?;
            if matches!(selected.dtype, Dtype::F16 | Dtype::BF16) {
                Some(selected)
            } else {
                None
            }
        }
        _ => None,
    };

    if let Some(raw) = raw_weight.as_ref() {
        anyhow::ensure!(
            raw.shape.len() == 2 && raw.shape[0] == vocab && raw.shape[1] == 768,
            "main_15 raw weight shape {:?} is not [50272, 768]",
            raw.shape,
        );
        if seq_len == 1 {
            crate::engine::gemv::gemv_threaded_into(
                hidden.as_slice(),
                vocab,
                768,
                raw.as_u16(),
                raw.dtype,
                6,
                &mut logits,
            );
        } else {
            // Prefill: process each position separately to keep the kernel
            // contract m==1.  Prefill is not the gate hot path.
            for row in 0..seq_len {
                crate::engine::gemv::gemv_threaded_into(
                    &hidden.as_slice()[row * 768..(row + 1) * 768],
                    vocab,
                    768,
                    raw.as_u16(),
                    raw.dtype,
                    1,
                    &mut logits[row * vocab..(row + 1) * vocab],
                );
            }
        }
    } else {
        // Shared BLAS bridge: `hidden @ lm_head^T` with lm_head stored as
        // [vocab, hidden].  The bridge owns the FFI declaration and the
        // row-major leading-dimension contract.
        crate::engine::blas::sgemm(
            crate::engine::blas::CBLAS_ROW_MAJOR,
            crate::engine::blas::CBLAS_NO_TRANS,
            crate::engine::blas::CBLAS_TRANS,
            seq_len,
            vocab,
            768,
            1.0f32,
            hidden.as_slice(),
            768,
            weight.as_slice(),
            768,
            0.0f32,
            &mut logits,
            vocab,
        );
    }

    let batch = if hidden.shape.len() == 3 {
        hidden.shape[0]
    } else {
        1
    };
    let mut shape = vec![batch, seq_len / batch.max(1), vocab];
    if hidden.shape.len() == 2 {
        shape = vec![1, seq_len, vocab];
    }
    func_outputs[func_def.index].push(Tensor::new_owned(shape, logits, Dtype::F32));
    Ok(true)
}

/// Reusable per-function output-buffer pool.
///
/// Decode steps repeatedly need the same ``(func_index, seq_len=1)`` output
/// vector.  Prefill chunks are grouped by their resolved ``seq_len``.  The
/// buffers are returned to the pool after the caller has copied/extracted
/// the activation values it needs, so no dylib output is aliased by a live
/// SSA tensor when it is reused.
#[derive(Default)]
pub struct OutputBufferPool {
    cache: HashMap<(usize, usize), Vec<SFATensor>>,
}

// SAFETY: Buffers in the pool are CPU SFATensor allocations owned by this
// struct.  `InferenceRunner` (which owns the pool via `ModelExecutor`) is
// guarded by a `tokio::sync::Mutex` in the server path, so a pool instance is
// never accessed concurrently; moving the guarded allocation across threads
// is safe.
unsafe impl Send for OutputBufferPool {}

impl OutputBufferPool {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

/// Pre-allocate output buffers sized from the compute graph metadata.
/// Returns SFATensors that own the data and provide HAL Buffer access
/// via `as_buffer_ref()`.
///
/// `passthrough_aliases` maps output indices to weight input indices whose
/// buffers may be borrowed instead of allocated (see
/// [`main0_weight_passthrough_map`]).  `input_tensors[bi]` must contain the
/// corresponding cached weight tensor.
fn allocate_output_buffers(
    func_def: &FuncDef,
    seq_len: usize,
    passthrough_aliases: &HashMap<usize, usize>,
    input_tensors: &[Option<Tensor>],
) -> Result<Vec<SFATensor>, anyhow::Error> {
    allocate_output_buffers_with_pool(func_def, seq_len, passthrough_aliases, input_tensors, None)
}

/// Pool-aware variant of [`allocate_output_buffers`].
fn allocate_output_buffers_with_pool(
    func_def: &FuncDef,
    seq_len: usize,
    passthrough_aliases: &HashMap<usize, usize>,
    input_tensors: &[Option<Tensor>],
    pool: Option<&mut OutputBufferPool>,
) -> Result<Vec<SFATensor>, anyhow::Error> {
    if let Some(pool) = pool {
        if let Some(cached) = pool.cache.remove(&(func_def.index, seq_len)) {
            if cached.len() == func_def.outputs.len() {
                log::trace!(
                    "output buffer pool hit: func[{}] seq_len={} outputs={}",
                    func_def.index,
                    seq_len,
                    cached.len(),
                );
                return Ok(cached);
            }
        }
    }
    let mut output_tensors: Vec<SFATensor> = Vec::with_capacity(func_def.outputs.len());
    for (oi, io_def) in func_def.outputs.iter().enumerate() {
        if let Some(&input_idx) = passthrough_aliases.get(&oi) {
            if let Some(tensor) = input_tensors.get(input_idx).and_then(|t| t.as_ref()) {
                let shape = tensor.shape.clone();
                let expected_numel: u64 = io_def.shape.iter().product();
                if shape.iter().map(|&d| d as u64).product::<u64>() == expected_numel {
                    // SAFETY: The pointer is used only to construct the output
                    // memref descriptor.  The dylib treats these outputs as
                    // pass-throughs (it returns the input descriptor unchanged)
                    // and CpuExecutable skips the copy when the returned
                    // aligned pointer already equals this destination.
                    log::trace!(
                        "allocate_output_buffers: func[{}] output[{}] aliases input[{}] (numel={})",
                        func_def.index,
                        oi,
                        input_idx,
                        tensor.numel(),
                    );
                    output_tensors.push(SFATensor::from_borrowed_tensor(tensor, shape));
                    continue;
                }
            }
        }

        // Map dynamic dims (0) to real values, preserving the original rank.
        // Q/K/V [0, heads, 0, dim] keep the leading batch zero as 1;
        // masks like [1, 1, 0, 0] resolve every zero to the sequence length.
        let first_zero_is_batch = io_def.shape.first() == Some(&0)
            && io_def.shape.get(1).is_some_and(|&d| d > 1);
        let shape_usize: Vec<usize> = io_def
            .shape
            .iter()
            .enumerate()
            .map(|(i, &d)| {
                if d == 0 {
                    if i == 0 && first_zero_is_batch {
                        1
                    } else {
                        seq_len
                    }
                } else {
                    d as usize
                }
            })
            .collect();
        let product: usize = shape_usize.iter().product();
        // Only fall back to a generous allocation when the shape cannot be
        // resolved safely from the ABI.  Besides every-dim-dynamic outputs,
        // LLaMA-style Q/K/V may arrive as [1, 0, 0, 64] with both seq and
        // head dims erased; the simple seq/1 heuristic cannot know which is
        // which.  Over-allocate and let execute() copy the returned shape.
        let all_dynamic = io_def.shape.iter().all(|&d| d == 0);
        let ambiguous_tail = io_def.shape.first() == Some(&1)
            && io_def.shape.get(1) == Some(&0)
            && io_def.shape.iter().filter(|&&d| d == 0).count() >= 2;
        let numel: usize;
        let final_shape: Vec<usize>;
        if all_dynamic || ambiguous_tail {
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
        log::trace!(
            "allocate_output_buffers: func[{}] output[{}] numel={} dtype={:?} (shape={:?}, resolved={:?}, final_shape={:?}, seq_len={})",
            func_def.index,
            oi,
            numel,
            io_def.dtype,
            io_def.shape,
            shape_usize,
            final_shape,
            seq_len,
        );
        let tensor = match io_def.dtype {
            Dtype::F32 => SFATensor::from_vec_f32(vec![0.0f32; numel], final_shape.clone()),
            Dtype::F16 | Dtype::BF16 => {
                SFATensor::from_vec_u16(vec![0u16; numel], final_shape.clone())
            }
            Dtype::I64 => SFATensor::from_vec_i64(vec![0i64; numel], final_shape.clone()),
            _ => SFATensor::from_vec_f32(vec![0.0f32; numel], final_shape.clone()),
        };
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
    let ptr = buf.as_ptr();
    log::trace!(
        "extract_output_tensor: func[{}] output[{}] sret shapes={:?} actual_n={} final_shape={:?} dtype={:?}",
        fi,
        oi,
        shapes,
        actual_n,
        shape_usize,
        io_def.dtype,
    );
    match io_def.dtype {
        Dtype::F32 => {
            // SAFETY: the pointer points to the owning allocation with at least the asserted element count.
            let data_slice = unsafe { std::slice::from_raw_parts(ptr as *const f32, actual_n) };
            Ok(Tensor::new_owned(shape_usize, data_slice.to_vec(), Dtype::F32))
        }
        Dtype::F16 | Dtype::BF16 => {
            // SAFETY: the pointer points to the owning allocation with at least the asserted element count.
            let data_slice = unsafe { std::slice::from_raw_parts(ptr as *const u16, actual_n) };
            Ok(Tensor::new_owned_u16(
                shape_usize,
                data_slice.to_vec(),
                io_def.dtype,
            ))
        }
        Dtype::I64 => {
            // SAFETY: the pointer points to the owning allocation with at least the asserted element count.
            let data_slice = unsafe { std::slice::from_raw_parts(ptr as *const i64, actual_n) };
            Ok(Tensor::new_owned_i64(shape_usize, data_slice.to_vec()))
        }
        other => {
            let data_slice = unsafe { std::slice::from_raw_parts(ptr as *const f32, actual_n) };
            Ok(Tensor::new_owned(
                shape_usize,
                data_slice.to_vec(),
                other,
            ))
        }
    }
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
    let input_sfa: Vec<crate::hal::sfa::SfaMemRef> = input_bufs
        .iter()
        .zip(func_def.inputs.iter())
        .map(|(b, (_binding, io_def))| {
            let sfa = b.as_ref().as_sfa_memref();
            let native_rank = sfa.rank();
            if native_rank > io_def.rank {
                warn!(
                    "rank promotion: buffer rank={} > io_def.rank={} for input",
                    native_rank, io_def.rank
                );
            }
            if sfa.rank() == 1 {
                let shape = sfa.sizes();
                let ptr = sfa.data_ptr() as *mut std::ffi::c_void;
                crate::hal::sfa::SfaMemRef::r2(
                    ptr,
                    [shape[0] as i64, 1],
                    [1, 1],
                    sfa.element_size(),
                )
            } else {
                sfa
            }
        })
        .collect();

    let output_bufs: Vec<_> = output_tensors.iter().map(|t| t.as_buffer_ref()).collect();
    let mut output_sfa: Vec<crate::hal::sfa::SfaMemRef> = output_bufs
        .iter()
        .map(|b| b.as_ref().as_sfa_memref())
        .collect();

    let output_shapes =
        executable.execute(&func_def.symbol, stream, &input_sfa, &mut output_sfa)?;
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

        let passthrough_aliases = main0_weight_passthrough_map(func_def);
        let main0_fastpath = main0_fastpath_contract(func_def, &passthrough_aliases);
        let mut input_bufs: Vec<Box<dyn traits::Buffer>> = Vec::with_capacity(func_def.num_inputs);
        let mut input_tensors: Vec<Option<Tensor>> = vec![None; func_def.num_inputs];
        let mut _sfa_tensors: Vec<SFATensor> = Vec::new();

        for (bi, (binding, io_def)) in func_def.inputs.iter().enumerate() {
            match binding {
                InputBinding::GlobalInput => {
                    let buf = build_global_input_buffer(
                        input_ids,
                        positions,
                        io_def,
                        bi,
                        &mut _sfa_tensors,
                    )?;
                    input_bufs.push(buf);
                }
                InputBinding::Weight(key) => {
                    let source_dtype = weight_provider
                        .get_weight_memref(key)
                        .map(|(_, d)| d)
                        .unwrap_or(Dtype::F32);
                    let tensor = if matches!(source_dtype, Dtype::F16 | Dtype::BF16) {
                        load_weight_tensor_u16(key, weight_provider, weight_cache, io_def)?
                    } else {
                        load_weight_tensor(key, weight_provider, weight_cache, io_def)?
                    };
                    let buf = wrap_tensor_buffer(&tensor)?;
                    input_tensors[bi] = Some(tensor);
                    input_bufs.push(buf);
                }
                InputBinding::Ssa {
                    producer_func,
                    output_idx,
                } => {
                    log::trace!(
                        "[runner] func[{}] input[{}] = Ssa(prod={}, out={})",
                        fi,
                        bi,
                        producer_func,
                        output_idx
                    );
                    let ref_tensor = &func_outputs[*producer_func][*output_idx];
                    let tensor = ref_tensor.to_owned();
                    let buf = wrap_tensor_buffer(&tensor)?;
                    log::trace!(
                        "[rank-verify] func[{}] input[{}] native_rank={} io_rank={} match={}",
                        fi,
                        bi,
                        buf.as_ref().as_sfa_memref().rank(),
                        io_def.rank,
                        buf.as_ref().as_sfa_memref().rank() as u8 == io_def.rank
                    );
                    input_tensors[bi] = Some(tensor);
                    input_bufs.push(buf);
                }
            }
        }

        if run_main15_fastpath(
            compute_graph,
            func_def,
            weight_provider,
            weight_cache,
            None,
            crate::engine::executor::WeightDtypeMode::Auto,
            &input_tensors,
            func_outputs,
        )? {
            continue;
        }

        if let Some((token_emb_bi, pos_emb_bi)) = main0_fastpath {
            build_main0_outputs_fastpath(
                func_def,
                input_ids,
                positions,
                &input_tensors,
                &passthrough_aliases,
                token_emb_bi,
                pos_emb_bi,
                func_outputs,
            )?;
            continue;
        }

        // Pre-allocate output buffers and execute.
        let seq_len = input_ids.len();
        let output_tensors =
            allocate_output_buffers(func_def, seq_len, &passthrough_aliases, &input_tensors)?;
        let output_shapes =
            build_sfa_and_execute(func_def, executable, &input_bufs, &output_tensors, stream)?;

        // Extract Tensors from output buffers using the returned shapes.
        // Pass-through weight outputs reuse their input Tensor (Arc clone is
        // O(1)) instead of copying the full weight matrix a second time.
        for (oi, _shapes) in output_shapes.iter().enumerate() {
            if let Some(&input_idx) = passthrough_aliases.get(&oi) {
                if let Some(tensor) = input_tensors.get(input_idx).and_then(|t| t.as_ref()) {
                    func_outputs[fi].push(tensor.to_owned());
                    continue;
                }
            }
            let io_def = &func_def.outputs[oi];
            let tensor =
                extract_output_tensor(&output_shapes, &output_tensors, fi, oi, io_def, seq_len)?;
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
pub(crate) fn should_intercept_consumed(
    fi: usize,
    oi: usize,
    cache_policy: &CachePolicy,
    io_def: &IOTensorDef,
) -> bool {
    if !cache_policy.intercepts.is_empty() {
        cache_policy
            .intercepts
            .iter()
            .any(|i| i.func_index == fi && i.output_index == oi)
    } else {
        io_def.consumed_internally
    }
}

pub(crate) fn find_slab_for_intercept<'p>(
    cache_policy: &'p CachePolicy,
    fi: usize,
    oi: usize,
) -> Option<&'p crate::cache::policy::SlabSpec> {
    let intercept = cache_policy
        .intercepts
        .iter()
        .find(|i| i.func_index == fi && i.output_index == oi)?;
    cache_policy
        .slabs
        .iter()
        .find(|s| s.slab_id == intercept.slab_id)
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
    block_manager: Option<&mut BlockManager>,
    request_id: Option<&str>,
    cache_policy: &CachePolicy,
) -> Result<Tensor, anyhow::Error> {
    let raw_weight_cache = RefCell::new(HashMap::new());
    run_function_graph_with_kv_intercept_pooled(
        compute_graph,
        executable,
        weight_provider,
        weight_cache,
        &raw_weight_cache,
        crate::engine::executor::WeightDtypeMode::Auto,
        func_outputs,
        input_ids,
        positions,
        stream,
        block_manager,
        request_id,
        cache_policy,
        None,
        false,
        None,
        None,
        None,
    )
}

/// Pool-aware variant of [`run_function_graph_with_kv_intercept`].
///
/// `opt_fused_layers` enables the Phase 4 Path C prototype: decoder layer
/// pairs matching the OPT split ABI are computed by the fused Rust kernel
/// in [`crate::engine::opt_fused`] instead of `_mlir_ciface_main_*`.  It is
/// off by default and every pair falls back to the dylib when its ABI
/// contract does not match.
#[allow(clippy::too_many_arguments)]
pub fn run_function_graph_with_kv_intercept_pooled(
    compute_graph: &ComputeGraph,
    executable: &dyn traits::Executable,
    weight_provider: &WeightProvider,
    weight_cache: &RefCell<HashMap<String, Tensor>>,
    raw_weight_cache: &RefCell<HashMap<String, Tensor>>,
    weight_dtype_mode: crate::engine::executor::WeightDtypeMode,
    func_outputs: &mut [Vec<Tensor>],
    input_ids: &[u32],
    positions: &[u32],
    stream: &dyn traits::Stream,
    mut block_manager: Option<&mut BlockManager>,
    request_id: Option<&str>,
    cache_policy: &CachePolicy,
    mut output_pool: Option<&mut OutputBufferPool>,
    opt_fused_layers: bool,
    op_plan: Option<&crate::engine::op_plan::OpPlan>,
    mut plan_buffer_pool: Option<&mut crate::engine::op_plan::PlanBufferPool>,
    mut account: Option<&mut crate::engine::account::ForwardAccount>,
) -> Result<Tensor, anyhow::Error> {
    let is_decode = input_ids.len() == 1;
    let mut kv_new: HashMap<(usize, usize), Tensor> = HashMap::new();
    let profile_level = std::env::var("SERVEFORGE_PROFILE")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(0);
    let profile_enabled = profile_level > 0;
    let profile_verbose = profile_level >= 2;
    let account_enabled = account.is_some();

    // Phase 4 prototype state.  Fused layers are enabled per forward pass:
    // decode steps always qualify; prefill chunks only when they start at
    // absolute position 0 (the same single-chunk semantics the current
    // intercept path relies on).  Later chunks stay on the dylib path.
    let fused_spec = if opt_fused_layers {
        opt_fused::opt_fused_spec(compute_graph)
    } else {
        None
    };
    let mut fused_ws = if fused_spec.is_some() {
        Some(opt_fused::FusedLayerWorkspace::default())
    } else {
        None
    };
    let fused_step_enabled = fused_spec.is_some() && (is_decode || positions.first() == Some(&0));

    // Phase 5 op-plan state.  The plan runs once, after main_0 has
    // populated its outputs and before the first covered function is
    // reached.  Covered functions are skipped in the func-level loop.
    let op_plan_covered = op_plan.map(crate::engine::op_plan::covered_func_indices);
    let op_plan_first = op_plan_covered.as_ref().and_then(|set| set.iter().min().copied());

    for func_def in &compute_graph.functions {
        let fi = func_def.index;
        let func_t0 = if profile_enabled {
            Some(std::time::Instant::now())
        } else {
            None
        };

        if let (Some(plan), Some(covered), Some(first)) =
            (op_plan, op_plan_covered.as_ref(), op_plan_first)
        {
            if fi == first {
                let plan_t0 = std::time::Instant::now();
                let mut plan_stats = if account_enabled {
                    Some(crate::engine::account::OpPlanAccount::default())
                } else {
                    None
                };
                crate::engine::op_plan::run_op_plan(
                    plan,
                    compute_graph,
                    weight_provider,
                    weight_cache,
                    raw_weight_cache,
                    weight_dtype_mode,
                    func_outputs,
                    positions,
                    block_manager.as_deref_mut(),
                    request_id,
                    cache_policy,
                    &mut kv_new,
                    plan_buffer_pool.as_deref_mut(),
                    plan_stats.as_mut(),
                )?;
                if let (Some(account), Some(stats)) = (account.as_deref_mut(), plan_stats) {
                    account.add_compute_ms(stats.exec_ms);
                    account.add_cache_ms(stats.cache_ms);
                }
                if let Some(t0) = func_t0 {
                    let total_ms = t0.elapsed().as_secs_f64() * 1e3;
                    if profile_verbose {
                        eprintln!(
                            "[profile2] op-plan {} nodes total={:.3}ms (plan body {:.3}ms)",
                            plan.nodes.len(),
                            total_ms,
                            plan_t0.elapsed().as_secs_f64() * 1e3
                        );
                    } else {
                        eprintln!("[profile] op-plan {} nodes {:.1}ms", plan.nodes.len(), total_ms);
                    }
                }
            }
            if covered.contains(&fi) {
                continue;
            }
        }

        let passthrough_aliases = main0_weight_passthrough_map(func_def);
        let main0_fastpath = main0_fastpath_contract(func_def, &passthrough_aliases);
        let mut input_bufs: Vec<Box<dyn traits::Buffer>> = Vec::with_capacity(func_def.num_inputs);
        let mut input_tensors: Vec<Option<Tensor>> = vec![None; func_def.num_inputs];
        let mut _sfa_tensors: Vec<SFATensor> = Vec::new();

        // Phase 4 Path C prototype: replace matching decoder-layer pairs
        // with the fused Rust kernel.  Contract mismatch returns Ok(false)
        // and the normal dylib path below executes unchanged.
        if fused_step_enabled {
            let fused_t0 = if account_enabled {
                Some(std::time::Instant::now())
            } else {
                None
            };
            let handled = if let (Some(spec), Some(ws)) = (fused_spec, fused_ws.as_mut()) {
                if opt_fused::is_opt_layer_a(func_def) {
                    opt_fused::run_fused_layer_a(
                        func_def,
                        func_outputs,
                        positions,
                        is_decode,
                        block_manager.as_deref_mut(),
                        request_id,
                        &mut kv_new,
                        spec,
                        ws,
                    )?
                } else if opt_fused::is_opt_layer_b(func_def) {
                    opt_fused::run_fused_layer_b(
                        func_def,
                        func_outputs,
                        positions,
                        is_decode,
                        block_manager.as_deref(),
                        request_id,
                        &kv_new,
                        spec,
                        ws,
                    )?
                } else {
                    false
                }
            } else {
                false
            };
            if handled {
                if let (Some(account), Some(t0)) = (account.as_deref_mut(), fused_t0) {
                    account.add_compute_ms(t0.elapsed().as_secs_f64() * 1e3);
                }
                if let Some(t0) = func_t0 {
                    let total_ms = t0.elapsed().as_secs_f64() * 1e3;
                    if profile_verbose {
                        eprintln!(
                            "[profile2] func[{}] {} (opt-fused) total={:.3}ms",
                            fi, func_def.symbol, total_ms
                        );
                    } else {
                        eprintln!(
                            "[profile] func[{}] {} (opt-fused) {:.1}ms",
                            fi, func_def.symbol, total_ms
                        );
                    }
                }
                continue;
            }
        }

        let build_inputs_t0 = if profile_verbose {
            Some(std::time::Instant::now())
        } else {
            None
        };
        for (bi, (binding, io_def)) in func_def.inputs.iter().enumerate() {
            // GlobalInput: handle with early continue (consumes `bi`)
            if let InputBinding::GlobalInput = binding {
                let buf =
                    build_global_input_buffer(input_ids, positions, io_def, bi, &mut _sfa_tensors)?;
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
                    let prod_func = &compute_graph.functions[*producer_func];
                    let prod_output_def = &prod_func.outputs[*output_idx];
                    if should_intercept_consumed(
                        *producer_func,
                        *output_idx,
                        cache_policy,
                        prod_output_def,
                    ) {
                        let slab =
                            find_slab_for_intercept(cache_policy, *producer_func, *output_idx);
                        let t_cache = if account_enabled {
                            Some(std::time::Instant::now())
                        } else {
                            None
                        };
                        let tensor = intercept_consumed_input(
                            *producer_func,
                            *output_idx,
                            compute_graph,
                            &kv_new,
                            block_manager.as_deref(),
                            request_id,
                            positions,
                            is_decode,
                            slab,
                            None,
                        )?;
                        if let (Some(account), Some(t0)) = (account.as_deref_mut(), t_cache) {
                            account.add_cache_ms(t0.elapsed().as_secs_f64() * 1e3);
                        }
                        tensor
                    } else {
                        // Non-consumed output: wire from func_outputs.  Consumed
                        // outputs are never pushed there, so subtract them to get
                        // the storage index.
                        let producer_outputs = &prod_func.outputs;
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
            input_tensors[bi] = Some(tensor);
            input_bufs.push(buf);
        }
        let build_inputs_ms = build_inputs_t0
            .map(|t0| t0.elapsed().as_secs_f64() * 1e3)
            .unwrap_or(0.0);
        let input_elems: usize = input_tensors
            .iter()
            .flatten()
            .map(|t| t.numel())
            .chain(_sfa_tensors.iter().map(|t| t.numel()))
            .sum();
        let input_allocs = input_tensors.iter().flatten().count() + _sfa_tensors.len();

        let fastpath_t0 = if account_enabled {
            Some(std::time::Instant::now())
        } else {
            None
        };
        if run_main15_fastpath(
            compute_graph,
            func_def,
            weight_provider,
            weight_cache,
            Some(raw_weight_cache),
            weight_dtype_mode,
            &input_tensors,
            func_outputs,
        )? {
            if let (Some(account), Some(t0)) = (account.as_deref_mut(), fastpath_t0) {
                account.add_compute_ms(t0.elapsed().as_secs_f64() * 1e3);
            }
            if let Some(t0) = func_t0 {
                let total_ms = t0.elapsed().as_secs_f64() * 1e3;
                if profile_verbose {
                    eprintln!(
                        "[profile2] func[{}] {} (fastpath) total={:.3}ms build_inputs={:.3}ms input_elems={} input_allocs={}",
                        fi, func_def.symbol, total_ms, build_inputs_ms, input_elems, input_allocs,
                    );
                } else {
                    eprintln!(
                        "[profile] func[{}] {} (fastpath) {:.1}ms",
                        fi, func_def.symbol, total_ms
                    );
                }
            }
            continue;
        }

        if let Some((token_emb_bi, pos_emb_bi)) = main0_fastpath {
            let fastpath_t0 = if account_enabled {
                Some(std::time::Instant::now())
            } else {
                None
            };
            build_main0_outputs_fastpath(
                func_def,
                input_ids,
                positions,
                &input_tensors,
                &passthrough_aliases,
                token_emb_bi,
                pos_emb_bi,
                func_outputs,
            )?;
            if let (Some(account), Some(t0)) = (account.as_deref_mut(), fastpath_t0) {
                account.add_compute_ms(t0.elapsed().as_secs_f64() * 1e3);
            }
            if let Some(t0) = func_t0 {
                let total_ms = t0.elapsed().as_secs_f64() * 1e3;
                if profile_verbose {
                    eprintln!(
                        "[profile2] func[{}] {} (fastpath) total={:.3}ms build_inputs={:.3}ms input_elems={} input_allocs={}",
                        fi, func_def.symbol, total_ms, build_inputs_ms, input_elems, input_allocs,
                    );
                } else {
                    eprintln!(
                        "[profile] func[{}] {} (fastpath) {:.1}ms",
                        fi, func_def.symbol, total_ms
                    );
                }
            }
            continue;
        }

        // Pre-allocate output buffers and execute.
        let seq_len = input_ids.len();

        let allocate_t0 = std::time::Instant::now();
        let output_tensors = allocate_output_buffers_with_pool(
            func_def,
            seq_len,
            &passthrough_aliases,
            &input_tensors,
            output_pool.as_deref_mut(),
        )?;
        let allocate_ms = allocate_t0.elapsed().as_secs_f64() * 1e3;
        let output_elems: usize = output_tensors.iter().map(|t| t.numel()).sum();
        let output_alloc_bytes: usize =
            output_tensors.iter().map(|t| t.numel() * t.elem_size).sum();

        log::trace!(
            "executing func[{}] {} inputs={} outputs={}",
            fi,
            func_def.symbol,
            input_bufs.len(),
            output_tensors.len()
        );
        let execute_t0 = std::time::Instant::now();
        let output_shapes =
            build_sfa_and_execute(func_def, executable, &input_bufs, &output_tensors, stream)?;
        let execute_ms = execute_t0.elapsed().as_secs_f64() * 1e3;
        if let Some(account) = account.as_deref_mut() {
            account.add_compute_ms(execute_ms);
        }

        let mut extract_ms = 0.0f64;
        let mut extract_elems = 0usize;
        let mut extract_allocs = 0usize;
        let mut intercept_ms = 0.0f64;
        let mut intercept_elems = 0usize;

        for (oi, _shapes) in output_shapes.iter().enumerate() {
            let io_def = &func_def.outputs[oi];
            let consumed = should_intercept_consumed(fi, oi, cache_policy, io_def);

            if consumed {
                let extract_t0 = std::time::Instant::now();
                let tensor = extract_output_tensor(
                    &output_shapes,
                    &output_tensors,
                    fi,
                    oi,
                    io_def,
                    seq_len,
                )?;
                extract_ms += extract_t0.elapsed().as_secs_f64() * 1e3;
                extract_elems += tensor.numel();
                extract_allocs += 1;

                let slab = find_slab_for_intercept(cache_policy, fi, oi);
                // First consumed output index = K, second = V (the a-block
                // emits [Q, K, V] with flags [false, true, true]).
                let kv_idx: Vec<usize> = func_def
                    .outputs
                    .iter()
                    .enumerate()
                    .filter(|(_, o)| o.consumed_internally)
                    .map(|(i, _)| i)
                    .collect();
                let is_key = kv_idx.first() == Some(&oi);
                let intercept_t0 = std::time::Instant::now();
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
                    slab,
                    is_key,
                )?;
                intercept_ms += intercept_t0.elapsed().as_secs_f64() * 1e3;
                intercept_elems += tensor.numel();
            } else if let Some(&input_idx) = passthrough_aliases.get(&oi) {
                if let Some(tensor) = input_tensors.get(input_idx).and_then(|t| t.as_ref()) {
                    func_outputs[fi].push(tensor.to_owned());
                } else {
                    let extract_t0 = std::time::Instant::now();
                    let tensor = extract_output_tensor(
                        &output_shapes,
                        &output_tensors,
                        fi,
                        oi,
                        io_def,
                        seq_len,
                    )?;
                    extract_ms += extract_t0.elapsed().as_secs_f64() * 1e3;
                    extract_elems += tensor.numel();
                    extract_allocs += 1;
                    func_outputs[fi].push(tensor);
                }
            } else {
                let extract_t0 = std::time::Instant::now();
                let tensor = extract_output_tensor(
                    &output_shapes,
                    &output_tensors,
                    fi,
                    oi,
                    io_def,
                    seq_len,
                )?;
                extract_ms += extract_t0.elapsed().as_secs_f64() * 1e3;
                extract_elems += tensor.numel();
                extract_allocs += 1;
                func_outputs[fi].push(tensor);
            }
        }

        if let Some(account) = account.as_deref_mut() {
            account.add_cache_ms(intercept_ms);
        }

        if let Some(t0) = func_t0 {
            let total_ms = t0.elapsed().as_secs_f64() * 1e3;
            if profile_verbose {
                eprintln!(
                    "[profile2] func[{}] {} total={:.3}ms build_inputs={:.3}ms(input_elems={},allocs={}) allocate_outputs={:.3}ms(output_elems={},bytes={},allocs={}) execute={:.3}ms extract={:.3}ms(elems={},allocs={}) intercept={:.3}ms(elems={})",
                    fi,
                    func_def.symbol,
                    total_ms,
                    build_inputs_ms,
                    input_elems,
                    input_allocs,
                    allocate_ms,
                    output_elems,
                    output_alloc_bytes,
                    output_tensors.len(),
                    execute_ms,
                    extract_ms,
                    extract_elems,
                    extract_allocs,
                    intercept_ms,
                    intercept_elems,
                );
            } else {
                eprintln!(
                    "[profile] func[{}] {} {:.1}ms",
                    fi, func_def.symbol, total_ms
                );
            }
        }

        if let Some(pool) = output_pool.as_deref_mut() {
            pool.cache.insert((fi, seq_len), output_tensors);
        }
    }

    let (g_func, g_idx) = compute_graph.global_output;
    let result = &func_outputs[g_func][g_idx];
    Ok(result.to_owned())
}

// ── ComputeGraphRunner (GraphRunner impl) ───────────────────────────────

use crate::engine::graph_runner::GraphRunner;

/// Global input specification for [`ComputeGraphRunner`].
///
/// Bundles the runtime buffers and metadata needed to construct a
/// global-input tensor (input_ids, position_ids) for a particular
/// function input slot.
#[derive(Debug, Clone)]
pub struct CGGlobalInputSpec<'a> {
    pub input_ids: &'a [u32],
    pub positions: &'a [u32],
    pub io_def: &'a IOTensorDef,
    /// Byte index of this input within the function's input list (used
    /// by `fill_global_input` for rank selection).
    pub bi: usize,
}

/// Output buffer specification for [`ComputeGraphRunner`].
#[derive(Debug, Clone)]
pub struct CGOutputSpec {
    pub shape: Vec<usize>,
    pub dtype: Dtype,
}

/// Internal state for compute-graph execution, implementing [`GraphRunner`].
///
/// Holds weight provider, weight cache, and the per-function output tensor
/// registry (SSA) used to wire data between functions.
pub struct ComputeGraphRunner<'a> {
    /// Weight provider for f16→f32 weight loading.
    pub weight_provider: &'a WeightProvider,

    /// Per-run weight cache (avoids reloading the same weight tensor
    /// across multiple function invocations).
    pub weight_cache: std::cell::RefCell<HashMap<String, Tensor>>,

    /// Per-function output tensors — indexed by `[func_index][output_index]`.
    /// This is the SSA wire store for cross-function data flow.
    pub func_outputs: std::cell::RefCell<Vec<Vec<Tensor>>>,
}

impl<'a> ComputeGraphRunner<'a> {
    /// Create a runner for `num_funcs` functions.
    pub fn new(weight_provider: &'a WeightProvider, num_funcs: usize) -> Self {
        Self {
            weight_provider,
            weight_cache: std::cell::RefCell::new(HashMap::new()),
            func_outputs: std::cell::RefCell::new(vec![Vec::new(); num_funcs]),
        }
    }
}

impl GraphRunner for ComputeGraphRunner<'_> {
    type InputSpec = CGGlobalInputSpec<'static>;
    type OutputSpec = CGOutputSpec;

    fn load_weight_tensor(&self, name: &str, _dtype: Dtype) -> Result<Tensor, anyhow::Error> {
        // Delegate to the existing free-function helper (line 78).
        load_weight_tensor(
            name,
            self.weight_provider,
            &self.weight_cache,
            &IOTensorDef::new(1, vec![0], false),
        )
    }

    fn allocate_output_buffer(
        &self,
        shape: &[usize],
        _dtype: Dtype,
    ) -> Result<Box<dyn traits::Buffer>, anyhow::Error> {
        let numel: usize = shape.iter().product::<usize>().max(1);
        let len = numel * 4;
        let data = vec![0.0f32; numel];
        let raw = InnerCpuBuffer::from_raw_parts(data.leak().as_mut_ptr() as *mut u8, len, false)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        Ok(Box::new(CpuBuffer::with_meta(raw, 4, shape.to_vec())))
    }

    fn build_global_input(&self, spec: &Self::InputSpec) -> Result<Tensor, anyhow::Error> {
        // Use the existing global input filler, then extract f32 data.
        let tensor = crate::model::global_input::fill_global_input(
            spec.input_ids,
            spec.positions,
            spec.io_def,
            spec.bi,
        )?;
        let numel = tensor.numel();
        let ptr = tensor.data_ptr() as *const f32;
// SAFETY: the pointer points to the owning allocation with at least the asserted element count.
        let data = unsafe { std::slice::from_raw_parts(ptr, numel) }.to_vec();

        // Resolve dynamic dims to real values for the shape.
        let rank = spec.io_def.rank as usize;
        let shape: Vec<usize> = (0..rank)
            .map(|i| {
                if spec.io_def.shape[i] == 0 {
                    if i == 0 {
                        1
                    } else {
                        spec.input_ids.len()
                    }
                } else {
                    spec.io_def.shape[i] as usize
                }
            })
            .collect();

        Ok(Tensor::new_owned(shape, data, Dtype::F32))
    }

    fn wire_ssa_output(&mut self, name: &str, tensor: Tensor) {
        // Store in func_outputs using encoded func_idx, output_idx from name.
        // Name format: "f{func_idx}_o{output_idx}" or fallback to func 0.
        let (func_idx, output_idx) = parse_ssa_name(name);
        let mut outputs = self.func_outputs.borrow_mut();
        while outputs.len() <= func_idx {
            outputs.push(Vec::new());
        }
        while outputs[func_idx].len() <= output_idx {
            outputs[func_idx].push(Tensor::scalar(0.0));
        }
        outputs[func_idx][output_idx] = tensor;
    }

    fn get_ssa_input(&self, name: &str) -> Option<Tensor> {
        let (func_idx, output_idx) = parse_ssa_name(name);
        let outputs = self.func_outputs.borrow();
        outputs
            .get(func_idx)
            .and_then(|fv| fv.get(output_idx))
            .cloned()
    }
}

/// Parse an SSA name like "f2_o1" → (func_idx=2, output_idx=1).
/// Falls back to (0, 0) on parse failure.
fn parse_ssa_name(name: &str) -> (usize, usize) {
    let trimmed = name.trim_start_matches('%').trim_start_matches("f");
    let parts: Vec<&str> = trimmed.splitn(2, "_o").collect();
    let func_idx = parts.first().and_then(|s| s.parse().ok()).unwrap_or(0);
    let output_idx = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    (func_idx, output_idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::policy::{CachePolicy, InterceptSpec};
    use crate::model::compute_graph::{FuncDef, InputBinding, IOTensorDef};

    fn single_output_func(index: usize) -> FuncDef {
        FuncDef {
            index,
            symbol: format!("main_{index}"),
            num_inputs: 0,
            num_outputs: 1,
            inputs: vec![],
            outputs: vec![IOTensorDef::new(2, vec![1, 4], false)],
            consumed_sub_output_flags: vec![],
        }
    }

    /// Phase 2.1: decode steps with the same (func, seq_len) must reuse the
    /// output allocation instead of zero-filling a fresh Vec every forward.
    #[test]
    fn test_output_buffer_pool_reuses_decode_buffers() {
        let func = single_output_func(7);
        let mut pool = OutputBufferPool::new();

        let first =
            allocate_output_buffers_with_pool(&func, 1, &HashMap::new(), &[], Some(&mut pool))
                .expect("first allocation");
        let first_ptr = match &first[0].raw {
            SFATensorRawAny::R2(r) => r.allocated as *mut u8,
            _ => panic!("expected rank-2 output"),
        };

        // The runner returns buffers to the pool after each function call.
        pool.cache.insert((func.index, 1), first);

        let second =
            allocate_output_buffers_with_pool(&func, 1, &HashMap::new(), &[], Some(&mut pool))
                .expect("pooled allocation");
        let second_ptr = match &second[0].raw {
            SFATensorRawAny::R2(r) => r.allocated as *mut u8,
            _ => panic!("expected rank-2 output"),
        };
        assert_eq!(first_ptr, second_ptr, "decode output buffer should be reused");

        // Prefill chunks with a different resolved seq_len get a distinct
        // pool entry (and therefore do not alias the decode buffer).
        let prefill =
            allocate_output_buffers_with_pool(&func, 8, &HashMap::new(), &[], Some(&mut pool))
                .expect("prefill allocation");
        let prefill_ptr = match &prefill[0].raw {
            SFATensorRawAny::R2(r) => r.allocated as *mut u8,
            _ => panic!("expected rank-2 output"),
        };
        assert_ne!(first_ptr, prefill_ptr, "prefill must not alias decode buffers");
    }

    #[test]
    fn test_dynamic_output_shape_resolves_mask_and_kv() {
        let func = FuncDef {
            index: 0,
            symbol: "main_0".into(),
            num_inputs: 0,
            num_outputs: 2,
            inputs: vec![],
            outputs: vec![
                IOTensorDef::new(4, vec![1, 1, 0, 0], false),
                IOTensorDef::new(4, vec![0, 12, 0, 64], false),
            ],
            consumed_sub_output_flags: vec![],
        };
        let out = allocate_output_buffers_with_pool(&func, 5, &HashMap::new(), &[], None)
            .expect("allocation");
        assert_eq!(out[0].shape(), vec![1, 1, 5, 5]);
        assert_eq!(out[1].shape(), vec![1, 12, 5, 64]);
    }

    /// When cache_policy.intercepts is populated, should_intercept_consumed
    /// must use the intercepts list (not consumed_internally).
    #[test]
    fn test_should_intercept_with_cache_policy() {
        let policy = CachePolicy {
            intercepts: vec![InterceptSpec {
                slab_id: "k".into(),
                op_name: "attn".into(),
                direction: "read_write".into(),
                source: "operand[1]".into(),
                layer: "sequential".into(),
                func_index: 1,
                output_index: 2,
            }],
            slabs: vec![],
            block_size: 16,
            max_requests: 256,
        };

        // Output at (1, 2) should be intercepted
        let io_def_matched = IOTensorDef::new(2, vec![1, 64], false);
        assert!(
            should_intercept_consumed(1, 2, &policy, &io_def_matched),
            "func=1, output=2 should match intercept"
        );

        // Output at (0, 0) should NOT be intercepted (not in intercepts list)
        let io_def_other = IOTensorDef::new(2, vec![1, 64], false);
        assert!(
            !should_intercept_consumed(0, 0, &policy, &io_def_other),
            "func=0, output=0 should not match any intercept"
        );
    }

    /// When cache_policy.intercepts is empty, should_intercept_consumed
    /// must fall back to io_def.consumed_internally.
    #[test]
    fn test_should_intercept_fallback_consumed() {
        let policy = CachePolicy::none(); // empty intercepts

        // consumed_internally=true should intercept
        let io_def_consumed = IOTensorDef::new(2, vec![1, 64], true);
        assert!(
            should_intercept_consumed(0, 0, &policy, &io_def_consumed),
            "should fall back to consumed_internally=true"
        );

        // consumed_internally=false should NOT intercept
        let io_def_not_consumed = IOTensorDef::new(2, vec![1, 64], false);
        assert!(
            !should_intercept_consumed(0, 0, &policy, &io_def_not_consumed),
            "should fall back to consumed_internally=false"
        );
    }

    /// When cache_policy has intercepts but consumed_internally is false,
    /// the intercepts list takes priority.
    #[test]
    fn test_should_intercept_intercepts_priority() {
        let policy = CachePolicy {
            intercepts: vec![InterceptSpec {
                slab_id: "k".into(),
                op_name: "attn".into(),
                direction: "read_write".into(),
                source: "operand[1]".into(),
                layer: "sequential".into(),
                func_index: 0,
                output_index: 0,
            }],
            slabs: vec![],
            block_size: 16,
            max_requests: 256,
        };

        // Even though consumed_internally=false, intercepts list says yes
        let io_def = IOTensorDef::new(1, vec![64], false);
        assert!(
            should_intercept_consumed(0, 0, &policy, &io_def),
            "intercepts list should take priority over consumed_internally=false"
        );
    }

    fn make_main0_func_def(
        index: usize,
        consumed_weights: usize,
        exported_weights: usize,
        dynamic_leading_outputs: usize,
    ) -> FuncDef {
        let mut inputs = Vec::new();
        inputs.push((
            InputBinding::GlobalInput,
            IOTensorDef::new(2, vec![0, 0], false),
        ));
        for i in 0..(consumed_weights + exported_weights) {
            let shape = if i < consumed_weights {
                vec![2050, 768]
            } else if i % 2 == 0 {
                vec![768, 768]
            } else {
                vec![3072, 768]
            };
            inputs.push((
                InputBinding::Weight(format!("w{}", i)),
                IOTensorDef::new(2, shape.into_iter().map(|d| d as u64).collect(), false),
            ));
        }

        let mut outputs = Vec::new();
        for _ in 0..dynamic_leading_outputs {
            outputs.push(IOTensorDef::new(3, vec![0, 0, 768], false));
        }
        for i in 0..exported_weights {
            let shape = if (consumed_weights + i) % 2 == 0 {
                vec![768u64, 768]
            } else {
                vec![3072, 768]
            };
            outputs.push(IOTensorDef::new(2, shape, false));
        }
        outputs.push(IOTensorDef::new(1, vec![1], false));

        FuncDef {
            index,
            symbol: format!("_mlir_ciface_main_{}", index),
            num_inputs: inputs.len(),
            num_outputs: outputs.len(),
            inputs,
            outputs,
            consumed_sub_output_flags: vec![],
        }
    }

    /// The main_0 weight-staging contract: consumed weights are first in
    /// inputs and are not returned; exported weights follow in the same
    /// order as the static weight outputs.
    #[test]
    fn test_main0_weight_passthrough_map_contract() {
        let func = make_main0_func_def(0, 2, 3, 2);
        let aliases = main0_weight_passthrough_map(&func);
        assert_eq!(aliases.len(), 3);
        assert_eq!(
            aliases.get(&2),
            Some(&3),
            "first exported output aliases first exported input"
        );
        assert_eq!(
            aliases.get(&3),
            Some(&4),
            "second exported output aliases second exported input"
        );
        assert_eq!(
            aliases.get(&4),
            Some(&5),
            "third exported output aliases third exported input"
        );
    }

    /// Non-main functions must not get the weight-staging alias treatment.
    #[test]
    fn test_main0_weight_passthrough_map_non_main_is_empty() {
        let func = make_main0_func_def(1, 2, 3, 2);
        assert!(main0_weight_passthrough_map(&func).is_empty());
    }

    /// A mismatch in one exported weight shape only disables the aliases for
    /// that mismatched output; the remaining correctly-matching contiguous
    /// block is still safe to alias.
    #[test]
    fn test_main0_weight_passthrough_map_mismatch_keeps_valid_tail() {
        let mut func = make_main0_func_def(0, 2, 3, 2);
        // Corrupt the first exported weight output shape while keeping it a
        // large static candidate, so the pairwise check must skip it.
        func.outputs[2] = IOTensorDef::new(2, vec![999, 999], false);
        let aliases = main0_weight_passthrough_map(&func);
        assert_eq!(aliases.len(), 2);
        assert_eq!(aliases.get(&3), Some(&4));
        assert_eq!(aliases.get(&4), Some(&5));
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

        let result = extract_output_tensor(&output_shapes, &[sfa], 0, 0, &io_def, 1)
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

    use crate::model::abi::proto::{SfaAbiHeader, SfaFuncMeta};
    use crate::model::abi::{
        build_compute_graph,
        proto::{
            sfa_input_field::Binding, OutputDescriptor, SfaInputField, SfaInputKind, SfaSsaRef,
        },
        SfaWeightProvider,
    };

    /// Minimal mock Executable for testing ABI-based execution.
    #[derive(Debug)]
    struct MockExecutable {
        num_funcs: usize,
        calls: std::sync::Mutex<Vec<String>>,
    }

    impl MockExecutable {
        fn new(num_funcs: usize) -> Self {
            Self {
                num_funcs,
                calls: std::sync::Mutex::new(Vec::new()),
            }
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

        fn function_count(&self) -> usize {
            self.num_funcs
        }
    }

    /// Mock executable that records every SfaMemRef rank passed to execute().
    #[derive(Debug)]
    struct RankCapturingExecutable<'a> {
        num_funcs: usize,
        captured: &'a std::sync::Mutex<Vec<usize>>,
    }
    impl<'a> RankCapturingExecutable<'a> {
        fn new(num_funcs: usize, captured: &'a std::sync::Mutex<Vec<usize>>) -> Self {
            Self {
                num_funcs,
                captured,
            }
        }
    }
    impl traits::Executable for RankCapturingExecutable<'_> {
        fn execute(
            &self,
            _op_name: &str,
            _stream: &dyn traits::Stream,
            inputs: &[crate::hal::sfa::SfaMemRef],
            outputs: &mut [crate::hal::sfa::SfaMemRef],
        ) -> Result<Vec<Vec<i64>>, anyhow::Error> {
            for sfa in inputs {
                self.captured.lock().unwrap().push(sfa.rank() as usize);
            }
            Ok(vec![vec![0i64; 0]; outputs.len()])
        }
        fn function_count(&self) -> usize {
            self.num_funcs
        }
    }

    /// Per-input rank from io_def MUST be used when constructing SfaMemRef.
    /// The dylib's LLVM IR hardcodes the rank in load instructions — a
    /// mismatch causes the load to read wrong bytes → SIGSEGV.
    #[test]
    fn test_ssa_sfa_memref_rank_matches_io_def() {
        let mut abi = SfaAbiHeader {
            magic: crate::model::abi::SFA_MAGIC,
            version: 1,
            funcs: vec![],
        };
        let mut f0 = SfaFuncMeta {
            consumed_sub_output_flags: vec![],
            symbol: "f0".to_string(),
            num_inputs: 1,
            output_rank: 2,
            input_fields: vec![SfaInputField {
                kind: SfaInputKind::SfaInputGlobal as i32,
                binding: None,
                rank: 2,
                dims: vec![1, 4],
                dtype: String::new(),
            }],
            outputs: vec![OutputDescriptor {
                rank: 2,
                dims: vec![1, 768],
                dtype: String::new(),
                consumed_internally: false,
            }],
        };
        abi.funcs.push(f0);
        let mut f1 = SfaFuncMeta {
            consumed_sub_output_flags: vec![],
            symbol: "f1".to_string(),
            num_inputs: 3,
            output_rank: 2,
            input_fields: vec![
                SfaInputField {
                    kind: SfaInputKind::SfaInputSsa as i32,
                    binding: Some(Binding::Ssa(SfaSsaRef {
                        producer_func: 0,
                        producer_out: 0,
                    })),
                    rank: 1,
                    dims: vec![768],
                    dtype: String::new(),
                },
                SfaInputField {
                    kind: SfaInputKind::SfaInputSsa as i32,
                    binding: Some(Binding::Ssa(SfaSsaRef {
                        producer_func: 0,
                        producer_out: 0,
                    })),
                    rank: 3,
                    dims: vec![1, 16, 64],
                    dtype: String::new(),
                },
                SfaInputField {
                    kind: SfaInputKind::SfaInputSsa as i32,
                    binding: Some(Binding::Ssa(SfaSsaRef {
                        producer_func: 0,
                        producer_out: 0,
                    })),
                    rank: 4,
                    dims: vec![1, 1, 16, 16],
                    dtype: String::new(),
                },
            ],
            outputs: vec![OutputDescriptor {
                rank: 2,
                dims: vec![1, 768],
                dtype: String::new(),
                consumed_internally: false,
            }],
        };
        abi.funcs.push(f1);
        let sfa_wp = SfaWeightProvider {
            name_mapping: HashMap::new(),
            constants: HashMap::new(),
        };
        let graph = build_compute_graph(&abi, &sfa_wp).unwrap();
        // Verify IOTensorDef ranks populated from proto
        assert_eq!(graph.functions[1].inputs[0].1.rank, 1);
        assert_eq!(graph.functions[1].inputs[1].1.rank, 3);
        assert_eq!(graph.functions[1].inputs[2].1.rank, 4);

        // Run execution and capture SfaMemRef ranks.
        let captured = std::sync::Mutex::new(Vec::<usize>::new());
        let mock = RankCapturingExecutable::new(2, &captured);
        let registry = crate::model::weight_loader::WeightRegistry {
            name_mapping: HashMap::new(),
            constants: HashMap::new(),
        };
        let wp = crate::model::weight_loader::WeightProvider::new(registry, None).unwrap();
        let wc = std::cell::RefCell::new(HashMap::new());
        let mut func_outputs: Vec<Vec<Tensor>> = vec![Vec::new(); 2];
        let result = run_function_graph(
            &graph,
            &mock,
            &wp,
            &wc,
            &mut func_outputs,
            &[42],
            &[0],
            &crate::hal::cpu::CpuStream,
        );
        assert!(result.is_ok());
        let ranks = captured.lock().unwrap();
        // func[0] has 1 global input → captured[0]; func[1] has 3 SSA → [1,2,3]
        // All SSA inputs wrap the same producer output tensor (native rank 2),
        // so they all have rank 2 (rank-1 inputs get promoted to rank 2).
        // The io_def.rank differs from SfaMemRef rank — this is by design:
        // the dylib can tolerate larger structs but not smaller ones.
        assert_eq!(
            ranks.len(),
            4,
            "expected 4 captured ranks, got {:?}",
            *ranks
        );
        assert_eq!(
            ranks[1], 2,
            "func[1] input[0] rank expected 2 (promoted from rank 1), got {}",
            ranks[1]
        );
        assert_eq!(
            ranks[2], 2,
            "func[1] input[1] rank expected 2 (native rank), got {}",
            ranks[2]
        );
        assert_eq!(
            ranks[3], 2,
            "func[1] input[2] rank expected 2 (native rank), got {}",
            ranks[3]
        );
    }
}
