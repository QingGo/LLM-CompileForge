//! KV cache intercept logic for the graph executor.
//!
//! Handles the SSA override and BlockManager integration for
//! `consumed_internally` outputs (K/V projections from attention-split
//! functions). Extracted from `executor.rs` `forward_with_kv` for
//! modularity.

use std::collections::HashMap;

use crate::cache::block::BlockManager;
use crate::model::compute_graph::{ComputeGraph, IOTensorDef};
use crate::model::tensor::{Dtype, Tensor};

/// Called after parsing a `consumed_internally` output from the sret
/// buffer during output-parsing in `forward_with_kv`.
///
/// Stores the tensor in `kv_new` for downstream SSA overrides, and
/// optionally writes it to the [`BlockManager`] KV cache (with BNSD→BNLD
/// transpose for prefill).
#[allow(clippy::too_many_arguments)]
pub fn intercept_consumed_output(
    fi: usize,
    oi: usize,
    tensor: &Tensor,
    kv_new: &mut HashMap<(usize, usize), Tensor>,
    block_manager: Option<&mut BlockManager>,
    request_id: Option<&str>,
    positions: &[u32],
    is_decode: bool,
    func_def_outputs: &[IOTensorDef],
) -> Result<(), anyhow::Error> {
    kv_new.insert((fi, oi), tensor.clone());

    if let Some(bm) = block_manager {
        let rid = request_id.ok_or_else(|| {
            anyhow::anyhow!(
                "intercept_consumed_output: request_id required when block_manager is set"
            )
        })?;
        let layer_rid = format!("{}_f{}", rid, fi);
        let num_tokens = positions.len();
        #[allow(clippy::manual_checked_ops)]
        let hidden_dim = if tensor.shape.len() >= 4 {
            tensor.shape[1] * tensor.shape[3]
        } else if num_tokens > 0 {
            tensor.numel() / num_tokens
        } else {
            *func_def_outputs[oi]
                .shape
                .last()
                .unwrap_or(&768) as usize
        };
        let start_pos = if is_decode {
            positions[0] as usize
        } else {
            0 // prefill starts at position 0
        };

        // Determine if this output is K or V by checking ordering of
        // consumed_internally outputs.
        let kv_indices: Vec<usize> = func_def_outputs
            .iter()
            .enumerate()
            .filter(|(_, o)| o.consumed_internally)
            .map(|(i, _)| i)
            .collect();
        let is_key = kv_indices.first() == Some(&oi);

        // BNSD→BNLD: prefill K/V is head-major [b, h, s, d];
        // write_kv expects position-major where each hidden_dim chunk
        // = one position.
        // When the tensor's first dim already encodes num_tokens
        // (e.g., [seq, heads, seq, dim] from SDPA-split), the
        // numel may be num_tokens * heads * seq * dim rather than
        // batch * heads * seq * dim.  Use a loop-based transpose only
        // when the tensor actually contains num_tokens * nh * hd
        // elements; otherwise pass through as-is.
        let nh = tensor.shape[1];
        let hd = tensor.shape[3];
        let expected_elems = num_tokens * nh * hd;
        let write_data: Vec<f32> = if tensor.shape.len() >= 4 && num_tokens > 1
            && tensor.numel() >= expected_elems
        {
            let sl = tensor.shape[2];
            let src = tensor.as_slice();
            let mut dst = vec![0.0f32; num_tokens * hidden_dim];
            for p in 0..num_tokens {
                for h in 0..nh {
                    let src_off = h * (sl * hd) + p * hd;
                    let dst_off = p * hidden_dim + h * hd;
                    dst[dst_off..dst_off + hd]
                        .copy_from_slice(&src[src_off..src_off + hd]);
                }
            }
            dst
        } else {
            tensor.as_slice().to_vec()
        };

        if let Err(e) = bm.write_kv(&layer_rid, start_pos, &write_data, hidden_dim, is_key) {
            log::warn!(
                "intercept_consumed_output: write_kv failed for func_{} output_{}: {}",
                fi,
                oi,
                e,
            );
        }
    }

    Ok(())
}

/// Called when an SSA input binding references a `consumed_internally`
/// output (i.e. a K or V tensor produced by an attention-split function).
///
/// Returns the override tensor:
///   - **Decode** (with `block_manager`): reads cached K/V for positions
///     `[0..pos)`, concatenates with the new single-position K/V, and
///     transposes BNLD→BNSD.
///   - **Prefill** (or no cache): returns the new K/V tensor directly.
#[allow(clippy::too_many_arguments)]
pub fn intercept_consumed_input(
    producer_func: usize,
    output_idx: usize,
    compute_graph: &ComputeGraph,
    kv_new: &HashMap<(usize, usize), Tensor>,
    block_manager: Option<&BlockManager>,
    request_id: Option<&str>,
    positions: &[u32],
    _is_decode: bool,
) -> Result<Tensor, anyhow::Error> {
    let new_tensor = kv_new
        .get(&(producer_func, output_idx))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "intercept_consumed_input: func_{} output_{} is consumed_internally \
                 but no KV data available (funcs must execute in topological order)",
                producer_func,
                output_idx,
            )
        })?;

    if let Some(bm) = block_manager.as_ref() {
        // Decode: concat cached K/V with new K/V
        let pos = positions[0] as usize;
        let hidden_dim = new_tensor.numel();
        let rid = request_id.ok_or_else(|| {
            anyhow::anyhow!(
                "intercept_consumed_input: request_id required when block_manager is set"
            )
        })?;
        let layer_rid = format!("{}_f{}", rid, producer_func);

        let (cached_key, cached_val) = bm
            .read_kv(&layer_rid, pos, hidden_dim)
            .map_err(|e| anyhow::anyhow!("intercept_consumed_input: read_kv: {}", e))?;

        // Determine if this SSA binding refers to K or V.
        // The producer func has two consumed_internally outputs:
        // first (lower output_idx) is K, second is V.
        let kv_indices: Vec<usize> = compute_graph.functions[producer_func]
            .outputs
            .iter()
            .enumerate()
            .filter(|(_, o)| o.consumed_internally)
            .map(|(i, _)| i)
            .collect();
        let is_k = kv_indices.first() == Some(&output_idx);

        let cached_data = if is_k { &cached_key } else { &cached_val };
        let n_cached_tokens = cached_data.len() / hidden_dim.max(1);
        let n_new_tokens = new_tensor.numel() / hidden_dim.max(1);
        let total_tokens = n_cached_tokens + n_new_tokens;

        // Concat: cached (BNLD from read_kv) ++ new (1 token, BNSD but
        // flat-equivalent to BNLD for a single position).
        let mut bnld = Vec::with_capacity(total_tokens * hidden_dim);
        bnld.extend_from_slice(cached_data);
        bnld.extend_from_slice(new_tensor.as_slice());

        // Transpose BNLD→BNSD so main_Xb receives correct layout.
        // new_tensor.shape is BNSD [b, h, s, d] for seq=1.
        if new_tensor.shape.len() >= 4 {
            let nh = new_tensor.shape[1];
            let hd = new_tensor.shape[3];
            let mut bnsd = Vec::with_capacity(nh * total_tokens * hd);
            for h in 0..nh {
                for p in 0..total_tokens {
                    let off = p * hidden_dim + h * hd;
                    bnsd.extend_from_slice(&bnld[off..off + hd]);
                }
            }
            Ok(Tensor::new_owned(
                vec![1, nh, total_tokens, hd],
                bnsd,
                Dtype::F32,
            ))
        } else {
            Ok(Tensor::new_owned(
                vec![1, total_tokens, hidden_dim],
                bnld,
                Dtype::F32,
            ))
        }
    } else {
        // Prefill or no cache: pass K/V directly
        Ok(new_tensor.clone())
    }
}
