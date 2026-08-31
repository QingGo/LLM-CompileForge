//! Cross-contract assertions between the compute graph (`sfa_abi`) and the
//! cache policy (`sfa_cache_policy`) — the dual-contract guarantee.
//!
//! Both contracts are emitted independently by the compiler (the
//! `consumed_internally` flags on ABI output descriptors, and the bound
//! intercept table on the cache policy).  They must agree; a mismatch is a
//! compiler bug, not a runtime condition, so these checks are hard errors
//! (not warnings) whenever the policy came from the proto symbol.
//!
//! Legacy dylibs without `sfa_cache_policy` keep the JSON/heuristic
//! fallback and never reach this module (see `ModelExecutor::load_with_device`).

use anyhow::bail;

use crate::cache::policy::CachePolicy;
use crate::model::compute_graph::ComputeGraph;

/// Assert the two cache contracts agree, for dylibs that export
/// `sfa_cache_policy`.
///
/// Checks, per function:
///
/// 1. `consumed_internally == true`  ⟺  a bound intercept exists at
///    `(func_index, output_index)` — the flags and the policy must
///    describe the same set of cache-consumed outputs.
/// 2. `consumed_sub_output_flags` (when present) agree per-index with the
///    output descriptors, and their length matches the descriptor count
///    (the packed-sret ABI is no longer produced by the compiler).
/// 3. Static ABI head dims (`dims[1]`) equal the slab's `heads` — the
///    GQA/kv-head pre-check.
pub fn cross_assert_cache_contract(
    graph: &ComputeGraph,
    policy: &CachePolicy,
) -> Result<(), anyhow::Error> {
    if policy.slabs.is_empty() {
        // No slabs → nothing is cached → no contract to enforce.
        return Ok(());
    }

    for func in &graph.functions {
        if !func.consumed_sub_output_flags.is_empty()
            && func.consumed_sub_output_flags.len() != func.outputs.len()
        {
            bail!(
                "cache contract: func[{}] has {} consumed_sub_output_flags but {} output \
                 descriptors — packed-sret ABI no longer supported",
                func.index,
                func.consumed_sub_output_flags.len(),
                func.outputs.len()
            );
        }

        for (oi, out) in func.outputs.iter().enumerate() {
            let intercept = policy
                .intercepts
                .iter()
                .find(|i| i.func_index == func.index && i.output_index == oi);

            match (out.consumed_internally, intercept) {
                (true, None) => bail!(
                    "cache contract: func[{}] output[{}] is consumed_internally but \
                     sfa_cache_policy has no matching intercept",
                    func.index,
                    oi
                ),
                (false, Some(i)) => bail!(
                    "cache contract: intercept (func={}, output={}, slab={}) exists but \
                     the ABI output is not consumed_internally",
                    func.index,
                    oi,
                    i.slab_id
                ),
                _ => {}
            }

            if let Some(i) = intercept {
                if let Some(&flag) = func.consumed_sub_output_flags.get(oi) {
                    if flag != out.consumed_internally {
                        bail!(
                            "cache contract: func[{}] output[{}] consumed_sub_output_flags={} \
                             disagrees with consumed_internally={}",
                            func.index,
                            oi,
                            flag,
                            out.consumed_internally
                        );
                    }
                }
                if let Some(slab) = policy.slabs.iter().find(|s| s.slab_id == i.slab_id) {
                    // GQA pre-check: when the ABI carries a static head
                    // dimension it must match the slab contract.
                    if let Some(&heads) = slab.dims.get("heads") {
                        if out.rank >= 2 && out.shape.len() >= 2 && out.shape[1] > 0 && heads > 0 {
                            if out.shape[1] != heads as u64 {
                                bail!(
                                    "cache contract: func[{}] output[{}] ABI dims[1]={} != \
                                     slab '{}' num_heads={} (head-count mismatch)",
                                    func.index,
                                    oi,
                                    out.shape[1],
                                    i.slab_id,
                                    heads
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::policy::{CachePolicy, InterceptSpec, SlabSpec};
    use crate::model::compute_graph::{ComputeGraph, FuncDef, IOTensorDef};
    use std::collections::HashMap;

    fn make_slab(slab_id: &str, nh: usize, hd: usize) -> SlabSpec {
        let mut dims = HashMap::new();
        dims.insert("layers".to_string(), 12usize);
        dims.insert("heads".to_string(), nh);
        dims.insert("dim".to_string(), hd);
        SlabSpec {
            slab_id: slab_id.to_string(),
            storage: "paged".to_string(),
            dims,
            layout: "BNLD".to_string(),
            dtype: "float32".to_string(),
        }
    }

    fn make_intercept(fi: usize, oi: usize, slab_id: &str) -> InterceptSpec {
        InterceptSpec {
            slab_id: slab_id.to_string(),
            op_name: "scaled_dot_product_attention".to_string(),
            direction: "read_write".to_string(),
            source: "operand[1]".to_string(),
            layer: "sequential".to_string(),
            func_index: fi,
            output_index: oi,
        }
    }

    fn make_policy(intercepts: Vec<InterceptSpec>) -> CachePolicy {
        CachePolicy {
            slabs: vec![make_slab("k", 12, 64), make_slab("v", 12, 64)],
            intercepts,
            block_size: 16,
            max_requests: 256,
        }
    }

    /// a-block function: outputs [Q, K, V] with flags [false, true, true].
    fn make_a_block_func(fi: usize, flags: Option<Vec<bool>>) -> FuncDef {
        FuncDef {
            index: fi,
            symbol: format!("_mlir_ciface_main_{}a", fi),
            num_inputs: 0,
            num_outputs: 3,
            inputs: vec![],
            outputs: vec![
                IOTensorDef::new(4, vec![1, 12, 0, 64], false), // Q
                IOTensorDef::new(4, vec![1, 12, 0, 64], true),  // K
                IOTensorDef::new(4, vec![1, 12, 0, 64], true),  // V
            ],
            consumed_sub_output_flags: flags.unwrap_or_else(|| vec![false, true, true]),
        }
    }

    fn make_graph(funcs: Vec<FuncDef>) -> ComputeGraph {
        ComputeGraph {
            functions: funcs,
            global_input: (0, 0),
            global_output: (0, 0),
        }
    }

    fn bound_policy() -> CachePolicy {
        make_policy(vec![make_intercept(0, 1, "k"), make_intercept(0, 2, "v")])
    }

    #[test]
    fn consistent_contract_passes() {
        let graph = make_graph(vec![make_a_block_func(0, None)]);
        cross_assert_cache_contract(&graph, &bound_policy()).expect("consistent contract");
    }

    #[test]
    fn consumed_without_intercept_fails() {
        // K/V consumed but only the K intercept bound → V must fail.
        let graph = make_graph(vec![make_a_block_func(0, None)]);
        let policy = make_policy(vec![make_intercept(0, 1, "k")]);
        let err = cross_assert_cache_contract(&graph, &policy).unwrap_err();
        assert!(
            err.to_string().contains("output[2] is consumed_internally"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn intercept_without_consumed_fails() {
        let graph = make_graph(vec![make_a_block_func(0, None)]);
        let policy = make_policy(vec![
            make_intercept(0, 1, "k"),
            make_intercept(0, 2, "v"),
            make_intercept(0, 0, "k"), // Q is NOT consumed
        ]);
        let err = cross_assert_cache_contract(&graph, &policy).unwrap_err();
        assert!(
            err.to_string().contains("not consumed_internally"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn head_dim_mismatch_fails() {
        let graph = make_graph(vec![make_a_block_func(0, None)]);
        // Slab claims 8 heads, ABI says dims[1]=12.
        let mut policy = bound_policy();
        policy.slabs[0] = make_slab("k", 8, 64);
        let err = cross_assert_cache_contract(&graph, &policy).unwrap_err();
        assert!(
            err.to_string().contains("num_heads"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn flags_disagreeing_with_descriptors_fails() {
        let graph = make_graph(vec![make_a_block_func(0, Some(vec![false, false, true]))]);
        let err = cross_assert_cache_contract(&graph, &bound_policy()).unwrap_err();
        assert!(
            err.to_string()
                .contains("disagrees with consumed_internally"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn packed_sret_abi_rejected() {
        // flags [F,T,T] but a single packed output descriptor → reject.
        let mut func = make_a_block_func(0, None);
        func.outputs = vec![IOTensorDef::new(3, vec![0, 0, 0], true)];
        let graph = make_graph(vec![func]);
        let err = cross_assert_cache_contract(&graph, &bound_policy()).unwrap_err();
        assert!(
            err.to_string().contains("packed-sret"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn empty_policy_skips_assertions() {
        // No slabs → no contract (legacy/heuristic path unaffected).
        let graph = make_graph(vec![make_a_block_func(0, None)]);
        cross_assert_cache_contract(&graph, &CachePolicy::none()).expect("empty policy");
    }

    #[test]
    fn dynamic_head_dim_skips_gqa_check() {
        // dims[1] = 0 (dynamic) must not trip the head-count check.
        let mut func = make_a_block_func(0, None);
        func.outputs[1].shape = vec![1, 0, 0, 64];
        let graph = make_graph(vec![func]);
        cross_assert_cache_contract(&graph, &bound_policy()).expect("dynamic heads");
    }
}
