//! KV cache intercept logic for the graph executor.
//!
//! Handles the SSA override and BlockManager integration for
//! `consumed_internally` outputs (K/V projections from attention-split
//! functions). Extracted from `executor.rs` `forward_with_kv` for
//! modularity.

use std::collections::HashMap;

use crate::cache::block::BlockManager;
use crate::cache::policy::SlabSpec;
use crate::model::compute_graph::{ComputeGraph, IOTensorDef};
use crate::model::tensor::{Dtype, Tensor};

/// Called after parsing a `consumed_internally` output from the sret
/// buffer during output-parsing in `forward_with_kv`.
///
/// Stores the tensor in `kv_new` for downstream SSA overrides, and
/// optionally writes it to the [`BlockManager`] KV cache (with BNSD→BNLD
/// transpose for prefill).
///
/// `slab` is the [`SlabSpec`] for the cache slab this output feeds into
/// (looked up from the [`crate::cache::policy::CachePolicy`] by
/// `slab_id`). When provided, `nh`/`hd` are read from the contract
/// (`slab.dims["heads"]` / `slab.dims["dim"]`) instead of being guessed
/// from `tensor.shape`, so tensors whose axis order differs from BNSD
/// (e.g. BSND emitted by some SDPA-split variants) are transposed
/// correctly. When `None`, the legacy shape heuristic is used as a
/// fallback for callers that have not yet been migrated to the
/// contract-driven path.
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
    slab: Option<&SlabSpec>,
    override_is_key: bool,
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

        let (nh, hd) = match slab {
            Some(s) => (
                *s.dims.get("heads").unwrap_or(&(tensor.shape[1] as usize)),
                *s.dims.get("dim").unwrap_or(&(tensor.shape[3] as usize)),
            ),
            None => (tensor.shape[1] as usize, tensor.shape[3] as usize),
        };
        let hidden_dim = nh * hd;
        let start_pos = if is_decode {
            positions[0] as usize
        } else {
            0
        };

        let kv_indices: Vec<usize> = func_def_outputs
            .iter()
            .enumerate()
            .filter(|(_, o)| o.consumed_internally)
            .map(|(i, _)| i)
            .collect();
        let is_key = if kv_indices.len() == 1 && func_def_outputs.len() == 1
            && func_def_outputs[0].consumed_internally
        {
            // When the ABI has a single packed output with consumed_internally,
            // use the caller-provided override (computed from
            // consumed_sub_output_flags).  Otherwise fall back to the
            // kv_indices heuristic which works for multi-output ABI.
            override_is_key
        } else {
            kv_indices.first() == Some(&oi)
        };

        let write_data: Vec<f32> = if slab.is_some() && tensor.shape.len() >= 4 && num_tokens > 0 {
            transpose_to_bnld_from_slab(tensor, num_tokens, nh, hd, slab.unwrap())?
        } else if tensor.shape.len() >= 4 && num_tokens > 1 {
            let sl = tensor.shape[2] as usize;
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

/// Transpose a producer tensor into the flat BNLD representation the
/// [`BlockManager`] stores, using the [`SlabSpec`] contract to decide
/// the source axis order.
///
/// `slab.layout` is the *storage* layout of the slab (always BNLD inside
/// the BlockManager). The producer tensor's layout is inferred from the
/// contract's `nh`/`hd` paired with the tensor's shape: the axis whose
/// size equals `nh` is the head axis, the axis whose size equals `hd`
/// is the dim axis, and the remaining axis of size `num_tokens` is the
/// position axis (batch is collapsed for single-batch prefill).
///
/// This avoids the legacy hardcoded `shape[1]`/`shape[3]` indexing
/// which assumed BNSD and broke for BSND (`[batch, seq, heads, dim]`)
/// tensors emitted by some SDPA-split variants.
fn transpose_to_bnld_from_slab(
    tensor: &Tensor,
    num_tokens: usize,
    nh: usize,
    hd: usize,
    _slab: &SlabSpec,
) -> Result<Vec<f32>, anyhow::Error> {
    let shape = &tensor.shape;
    if shape.len() < 4 {
        return Ok(tensor.as_slice().to_vec());
    }
    let hidden_dim = nh * hd;
    let src = tensor.as_slice();

    let batch = shape[0] as usize;
    if batch != 1 {
        return Ok(src.to_vec());
    }

    let axis_seq = shape
        .iter()
        .enumerate()
        .skip(1)
        .find(|(_, &s)| s as usize == num_tokens && s as usize != nh && s as usize != hd)
        .map(|(i, _)| i);
    let axis_heads = shape
        .iter()
        .enumerate()
        .skip(1)
        .find(|(_, &s)| s as usize == nh)
        .map(|(i, _)| i);
    let axis_dim = shape
        .iter()
        .enumerate()
        .skip(1)
        .find(|(_, &s)| s as usize == hd)
        .map(|(i, _)| i);

    let (ax_s, ax_h, _ax_d) = match (axis_seq, axis_heads, axis_dim) {
        (Some(s), Some(h), Some(d)) if s != h && s != d && h != d => (s, h, d),
        _ => {
            let sl = shape[2] as usize;
            let mut dst = vec![0.0f32; num_tokens * hidden_dim];
            for p in 0..num_tokens {
                for h in 0..nh {
                    let src_off = h * (sl * hd) + p * hd;
                    let dst_off = p * hidden_dim + h * hd;
                    dst[dst_off..dst_off + hd]
                        .copy_from_slice(&src[src_off..src_off + hd]);
                }
            }
            return Ok(dst);
        }
    };

    let strides = {
        let mut acc = 1usize;
        let mut s = vec![0usize; shape.len()];
        for i in (0..shape.len()).rev() {
            s[i] = acc;
            acc *= shape[i] as usize;
        }
        s
    };

    let mut dst = vec![0.0f32; num_tokens * hidden_dim];
    for p in 0..num_tokens {
        for h in 0..nh {
            // dim axis (ax_d) is always read at index 0, so it contributes 0.
            let src_off = p * strides[ax_s] + h * strides[ax_h];
            let dst_off = p * hidden_dim + h * hd;
            dst[dst_off..dst_off + hd]
                .copy_from_slice(&src[src_off..src_off + hd]);
        }
    }
    Ok(dst)
}

/// Called when an SSA input binding references a `consumed_internally`
/// output (i.e. a K or V tensor produced by an attention-split function).
///
/// Returns the override tensor:
///   - **Decode** (single token, with `block_manager`): reads cached K/V
///     for positions `[0..pos)`, concatenates with the new single-position
///     K/V, and transposes BNLD→BNSD.
///   - **Prefill** (multi-token) or no cache: returns the new K/V tensor
///     directly. There is no prior cache to concatenate during prefill,
///     and the producer already emits BNSD layout, so the consumer (SDPA)
///     receives exactly what it would on the no-cache path.
#[allow(clippy::too_many_arguments)]
pub fn intercept_consumed_input(
    producer_func: usize,
    output_idx: usize,
    compute_graph: &ComputeGraph,
    kv_new: &HashMap<(usize, usize), Tensor>,
    block_manager: Option<&BlockManager>,
    request_id: Option<&str>,
    positions: &[u32],
    is_decode: bool,
    slab: Option<&SlabSpec>,
    consumed_sub_flags: Option<&[bool]>,
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

    // Prefill (multi-token): there is no prior cache to concatenate, and the
    // producer emits BNSD K/V directly. Pass the new tensor through unchanged
    // so the SDPA consumer sees the same layout as on the no-cache path.
    // The cache itself is still populated by `intercept_consumed_output`.
    if !is_decode {
        return Ok(new_tensor.clone());
    }

    if let Some(bm) = block_manager.as_ref() {
        let pos = positions[0] as usize;
        let (nh, hd) = match slab {
            Some(s) => (
                *s.dims.get("heads").unwrap_or(&new_tensor.shape[1]),
                *s.dims.get("dim").unwrap_or(&new_tensor.shape[3]),
            ),
            None => (new_tensor.shape[1], new_tensor.shape[3]),
        };
        let hidden_dim = nh * hd;
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
        // When consumed_sub_flags is provided (from the packed-output
        // ABI), use it to find the first consumed sub-output index
        // (K). Otherwise fall back to the ABI output descriptors.
        let is_k = if let Some(flags) = consumed_sub_flags {
            flags.iter().position(|&f| f) == Some(output_idx)
        } else {
            let kv_indices: Vec<usize> = compute_graph.functions[producer_func]
                .outputs.iter().enumerate()
                .filter(|(_, o)| o.consumed_internally)
                .map(|(i, _)| i)
                .collect();
            kv_indices.first() == Some(&output_idx)
        };

        let cached_data = if is_k { &cached_key } else { &cached_val };
        let n_cached_tokens = cached_data.len() / hidden_dim.max(1);
        let n_new_tokens = new_tensor.numel() / hidden_dim.max(1);
        let total_tokens = n_cached_tokens + n_new_tokens;

        // Concat: cached (BNLD from read_kv) ++ new (1 token, BNSD but
        // flat-equivalent to BNLD for a single position).
        let mut bnld = Vec::with_capacity(total_tokens * hidden_dim);
        bnld.extend_from_slice(cached_data);
        bnld.extend_from_slice(new_tensor.as_slice());

        if new_tensor.shape.len() >= 4 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::compute_graph::{
        ComputeGraph, FuncDef, IOTensorDef, InputBinding,
    };

    /// Build a minimal ComputeGraph with one function that has two
    /// `consumed_internally` outputs (K at output 1, V at output 2),
    /// mirroring the KV-split attention functions produced by the
    /// CachePolicy compilation path.
    fn make_kv_graph() -> ComputeGraph {
        let f = FuncDef {
            index: 0,
            symbol: "f0".into(),
            num_inputs: 0,
            num_outputs: 3,
            inputs: vec![],
            outputs: vec![
                IOTensorDef::new(2, vec![1, 0], false), // logits
                IOTensorDef::new(4, vec![1, 12, 0, 64], true), // K (consumed)
                IOTensorDef::new(4, vec![1, 12, 0, 64], true), // V (consumed)
            ],
            consumed_sub_output_flags: vec![],
        };
        ComputeGraph {
            functions: vec![f],
            global_input: (0, 0),
            global_output: (0, 0),
        }
    }

    /// Make a prefill K tensor shaped [1, 12, seq, 64] (BNSD) with
    /// distinct, position-identifiable values so corruption is easy to
    /// spot. Each (head, pos) block is filled with a unique base value.
    fn make_prefill_k(seq: usize) -> Tensor {
        let nh = 12usize;
        let hd = 64usize;
        let mut data = vec![0.0f32; nh * seq * hd];
        for h in 0..nh {
            for p in 0..seq {
                let base = (h * 1000 + p) as f32;
                let off = h * (seq * hd) + p * hd;
                for d in 0..hd {
                    data[off + d] = base + d as f32;
                }
            }
        }
        Tensor::new_owned(vec![1, nh, seq, hd], data, Dtype::F32)
    }

    // ── Reproduces the E2E failure in test_kv_cache_vs_full_recompute ──

    #[test]
    fn prefill_with_block_manager_passes_through_new_tensor() {
        // Prefill (multi-token) with a BlockManager must return the
        // newly-produced K/V tensor unchanged (BNSD layout), since there
        // is no prior cache to concatenate. This is what the SDPA consumer
        // expects — identical to the no-cache path.
        let seq = 6usize;
        let nh = 12usize;
        let hd = 64usize;
        let graph = make_kv_graph();

        let mut bm = BlockManager::new_with_cache(64, 16, nh, hd).unwrap();
        // Allocate blocks for the request and per-layer cache (mirrors
        // the E2E test's setup in e2e_tests.rs).
        bm.allocate("req", seq).unwrap();
        bm.allocate("req_f0", seq + 10).unwrap();

        let new_k = make_prefill_k(seq);
        let positions: Vec<u32> = (0..seq as u32).collect();
        let mut kv_new = HashMap::new();
        kv_new.insert((0, 1), new_k.clone());

        let out = intercept_consumed_input(
            0,            // producer_func
            1,            // output_idx (K)
            &graph,
            &kv_new,
            Some(&bm),
            Some("req"),
            &positions,
            false,        // is_decode = false (prefill)
            None,
            None,
        )
        .expect("prefill intercept should succeed");

        // The returned tensor must equal the new K tensor (BNSD passthrough).
        assert_eq!(out.shape, vec![1, nh, seq, hd], "prefill output shape must be BNSD");
        assert_eq!(
            out.as_slice(),
            new_k.as_slice(),
            "prefill with BM must pass new_tensor through unchanged"
        );
    }

    #[test]
    fn prefill_without_block_manager_passes_through_new_tensor() {
        // Sanity: the no-cache path already does passthrough (this is the
        // "correct" reference behavior the BM path must also satisfy).
        let seq = 6usize;
        let graph = make_kv_graph();
        let new_k = make_prefill_k(seq);
        let positions: Vec<u32> = (0..seq as u32).collect();
        let mut kv_new = HashMap::new();
        kv_new.insert((0, 1), new_k.clone());

        let out = intercept_consumed_input(
            0, 1, &graph, &kv_new, None, None, &positions, false, None, None,
        )
        .expect("prefill no-cache intercept should succeed");

        assert_eq!(out.shape, new_k.shape);
        assert_eq!(out.as_slice(), new_k.as_slice());
    }

    // Ensure the unused import warning stays suppressed for the enum variant.
    #[allow(dead_code)]
    fn _silence_input_binding() -> InputBinding {
        InputBinding::GlobalInput
    }

    // ── Contract-driven intercept: SlabSpec must override shape heuristics ──
    //
    // Reproduces the E2E failure where intercept_consumed_output guessed
    // nh/hd from tensor.shape[1]/[3] instead of reading them from the
    // SfaSlabSpec contract. With SDPA-split models the tensor may carry
    // a different axis order (e.g. [batch, seq, heads, dim]), so
    // shape[1] is `seq`, not `heads`. The slab contract is the only
    // authoritative source of layout semantics.

    use crate::cache::policy::SlabSpec;

    fn make_slab(slab_id: &str, nh: usize, hd: usize, layout: &str) -> SlabSpec {
        let mut dims = HashMap::with_capacity(4);
        dims.insert("layers".to_string(), 12);
        dims.insert("heads".to_string(), nh);
        dims.insert("dim".to_string(), hd);
        dims.insert("blocks".to_string(), 64);
        SlabSpec {
            slab_id: slab_id.to_string(),
            storage: "paged".to_string(),
            dims,
            layout: layout.to_string(),
            dtype: "float32".to_string(),
        }
    }

    /// Build a prefill K tensor whose shape axis order is
    /// `[batch, seq, heads, dim]` (BSND, the layout the SDPA-split
    /// producer actually emits for some split variants).
    /// `shape[1]` is `seq`, NOT `heads` — so the legacy heuristic
    /// `nh = tensor.shape[1]` would pick the wrong axis.
    fn make_bsnd_prefill_k(batch: usize, seq: usize, nh: usize, hd: usize) -> Tensor {
        let mut data = vec![0.0f32; batch * seq * nh * hd];
        for b in 0..batch {
            for s in 0..seq {
                for h in 0..nh {
                    let base = ((b * seq + s) * nh + h) as f32 * 1000.0;
                    let off = ((b * seq + s) * nh + h) * hd;
                    for d in 0..hd {
                        data[off + d] = base + d as f32;
                    }
                }
            }
        }
        Tensor::new_owned(vec![batch, seq, nh, hd], data, Dtype::F32)
    }

    #[test]
    fn intercept_consumed_output_uses_slab_spec_dims_not_shape_heuristic() {
        // Contract says: BNSD layout, nh=12, hd=64.
        // But the producer emits a BSND tensor `[1, 6, 12, 64]`
        // (batch=1, seq=6, heads=12, dim=64). The slab's num_heads=12
        // and head_dim=64 MUST drive the transpose — not shape[1]=6.
        let seq = 6usize;
        let nh = 12usize;
        let hd = 64usize;
        let hidden_dim = nh * hd;

        let slab = make_slab("k", nh, hd, "BNSD");

        // Tensor laid out as BSND: shape[1]=seq=6, shape[3]=hd=64.
        // The legacy code reads nh=shape[1]=6 (WRONG) and hd=shape[3]=64.
        let tensor = make_bsnd_prefill_k(1, seq, nh, hd);
        assert_eq!(
            tensor.shape[1], seq,
            "test setup: tensor.shape[1] must be seq, not nh, to expose the bug",
        );

        let mut bm = BlockManager::new_with_cache(64, 16, nh, hd).unwrap();
        bm.allocate("req_f0", seq + 10).unwrap();

        let positions: Vec<u32> = (0..seq as u32).collect();
        let mut kv_new = HashMap::new();

        let func_def_outputs: Vec<IOTensorDef> = vec![
            IOTensorDef::new(2, vec![1, 0], false),
            IOTensorDef::new(4, vec![1, seq as u64, nh as u64, hd as u64], true), // K (consumed)
            IOTensorDef::new(4, vec![1, seq as u64, nh as u64, hd as u64], true), // V (consumed)
        ];

        intercept_consumed_output(
            0,
            1,
            &tensor,
            &mut kv_new,
            Some(&mut bm),
            Some("req"),
            &positions,
            false, // prefill
            &func_def_outputs,
            Some(&slab),
            true,
        )
        .expect("intercept with slab should succeed");

        // Read back from cache and verify each token's hidden_dim slice
        // contains the correct (head, dim) values from the source tensor.
        let (cached_k, _cached_v) = bm
            .read_kv("req_f0", seq, hidden_dim)
            .expect("read_kv should succeed");

        // The BNLD cache should hold, for each position p in [0..seq),
        // a hidden_dim chunk equal to concat over heads of tensor[b=0, s=p, h=:, d=:].
        for p in 0..seq {
            for h in 0..nh {
                let expected_base = ((p * nh + h) as f32) * 1000.0;
                for d in 0..hd {
                    let expected = expected_base + d as f32;
                    let got = cached_k[p * hidden_dim + h * hd + d];
                    assert_eq!(
                        got, expected,
                        "position {} head {} dim {}: cache corruption (got {} expected {})",
                        p, h, d, got, expected,
                    );
                }
            }
        }
    }

    #[test]
    fn intercept_consumed_input_decode_uses_slab_spec_for_bnld_to_bnsd() {
        let nh = 12usize;
        let hd = 64usize;
        let hidden_dim = nh * hd;

        let graph = make_kv_graph();
        let mut bm = BlockManager::new_with_cache(64, 16, nh, hd).unwrap();
        bm.allocate("req_f0", 32).unwrap();

        let cached_positions = 5usize;
        let mut cached_k = vec![0.0f32; cached_positions * hidden_dim];
        for p in 0..cached_positions {
            for h in 0..nh {
                let base = (p * 100 + h) as f32;
                for d in 0..hd {
                    cached_k[p * hidden_dim + h * hd + d] = base + d as f32;
                }
            }
        }
        bm.write_kv("req_f0", 0, &cached_k, hidden_dim, true).unwrap();

        let mut new_k_data = vec![0.0f32; hidden_dim];
        for h in 0..nh {
            let base = (cached_positions * 100 + h) as f32;
            for d in 0..hd {
                new_k_data[h * hd + d] = base + d as f32;
            }
        }
        let new_k = Tensor::new_owned(vec![1, nh, 1, hd], new_k_data.clone(), Dtype::F32);

        let mut kv_new = HashMap::new();
        kv_new.insert((0, 1), new_k);

        let slab = make_slab("k", nh, hd, "BNSD");

        let positions: Vec<u32> = [cached_positions as u32].to_vec();

        let out = intercept_consumed_input(
            0,
            1,
            &graph,
            &kv_new,
            Some(&bm),
            Some("req"),
            &positions,
            true,
            Some(&slab),
            None,
        )
        .expect("decode intercept with slab should succeed");

        let total_tokens = cached_positions + 1;
        assert_eq!(out.shape, vec![1, nh, total_tokens, hd]);

        for p in 0..total_tokens {
            for h in 0..nh {
                let base = (p * 100 + h) as f32;
                for d in 0..hd {
                    let expected = base + d as f32;
                    let got = out.as_slice()[h * (total_tokens * hd) + p * hd + d];
                    assert_eq!(
                        got, expected,
                        "BNSD[{}, {}, {}, {}]: got {} expected {}",
                        0, h, p, d, got, expected,
                    );
                }
            }
        }
    }
}
