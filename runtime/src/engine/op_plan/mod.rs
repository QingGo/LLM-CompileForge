//! Op-plan executor — executes the Phase 5 HAL kernel graph.
//!
//! The plan replaces only the decoder-layer function pairs.  The caller
//! runs the embedding prefix and output tail through the func-level path
//! exactly as before; boundary values are projected with
//! `FUNC_OUTPUT` inputs and plan outputs are pushed back into
//! `func_outputs` via [`OpPlanFuncOutput`] so later functions see the same
//! SSA contract.

mod kernels;

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use crate::cache::block::BlockManager;
use crate::cache::intercept::{intercept_consumed_input, intercept_consumed_output};
use crate::cache::policy::CachePolicy;
use crate::engine::compute_graph_runner::{
    find_slab_for_intercept, load_weight_tensor, load_weight_tensor_for_mode,
    should_intercept_consumed,
};
use crate::engine::executor::WeightDtypeMode;
use crate::model::abi::proto;
use crate::model::compute_graph::{ComputeGraph, IOTensorDef};
use crate::model::tensor::{Dtype, Tensor};
use crate::model::weight_loader::WeightProvider;

pub(crate) use kernels::PLAN_OP_CATALOG;
pub use proto::OpPlan;
pub use proto::OpPlanInput;
pub use proto::OpPlanNode;

/// Persistent per-node output buffer pool.
///
/// Key is `(node_index, output_index, resolved_shape)`.  Buffers are only
/// returned after every SSA consumer in the step has finished with the
/// tensor, so reuse never aliases live activation data.
#[derive(Default)]
pub struct PlanBufferPool {
    cache: HashMap<(u32, u32, Vec<usize>), Vec<f32>>,
}

// SAFETY: CPU-owned Vec<f32> buffers; the owning ModelExecutor is guarded by
// the same server mutex as OutputBufferPool, so it is never accessed
// concurrently.
unsafe impl Send for PlanBufferPool {}

impl PlanBufferPool {
    pub fn new() -> Self {
        Self::default()
    }

    fn acquire(&mut self, key: (u32, u32, Vec<usize>), numel: usize) -> Vec<f32> {
        match self.cache.remove(&key) {
            Some(buf) if buf.len() == numel => buf,
            _ => vec![0.0f32; numel],
        }
    }

    fn release(&mut self, key: (u32, u32, Vec<usize>), buf: Vec<f32>) {
        self.cache.entry(key).or_insert(buf);
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

/// Hard load-time contract assertions (plan invariants from
/// `.omo/plans/p1-phase5-op-split-design.md` §2).
pub fn validate_op_plan(
    plan: &OpPlan,
    compute_graph: &ComputeGraph,
    weight_provider: &WeightProvider,
    cache_policy: &CachePolicy,
) -> Result<(), anyhow::Error> {
    anyhow::ensure!(!plan.nodes.is_empty(), "OpPlan has no nodes");
    anyhow::ensure!(
        !plan.func_outputs.is_empty(),
        "OpPlan has no func-output projections"
    );

    let supported: HashSet<&str> = PLAN_OP_CATALOG.iter().copied().collect();
    for (idx, node) in plan.nodes.iter().enumerate() {
        anyhow::ensure!(node.index as usize == idx, "node index monotonicity broken at {idx}");
        anyhow::ensure!(
            supported.contains(node.op_name.as_str()),
            "op-plan node {idx} uses unregistered op {op:?}",
            op = node.op_name
        );
        anyhow::ensure!(
            !node.source_func_indices.is_empty(),
            "op-plan node {idx} has no source func projection"
        );
        for &sfi in &node.source_func_indices {
            anyhow::ensure!(
                (sfi as usize) < compute_graph.functions.len(),
                "op-plan node {idx} source func {sfi} out of range"
            );
        }
        for input in &node.inputs {
            validate_input(plan, compute_graph, weight_provider, idx, input)?;
        }
    }

    validate_cache_projection(plan, compute_graph, cache_policy)?;
    validate_func_output_projection(plan, compute_graph)?;
    Ok(())
}

fn validate_input(
    plan: &OpPlan,
    compute_graph: &ComputeGraph,
    weight_provider: &WeightProvider,
    node_idx: usize,
    input: &OpPlanInput,
) -> Result<(), anyhow::Error> {
    use proto::op_plan_input::{Binding, Source};
    let source = Source::try_from(input.source)
        .map_err(|_| anyhow::anyhow!("op-plan node {node_idx}: invalid input source {}", input.source))?;
    match source {
        Source::Ssa => {
            let Some(Binding::Producer(ref p)) = input.binding else {
                anyhow::bail!("op-plan node {node_idx}: SSA input missing producer");
            };
            anyhow::ensure!(
                (p.node_index as usize) < node_idx,
                "op-plan node {node_idx}: SSA producer {} is not strictly earlier",
                p.node_index
            );
            let producer = plan.nodes.get(p.node_index as usize).ok_or_else(|| {
                anyhow::anyhow!("op-plan node {node_idx}: SSA producer {} out of range", p.node_index)
            })?;
            anyhow::ensure!(
                (p.output_index as usize) < producer.outputs.len(),
                "op-plan node {node_idx}: SSA output {} out of range",
                p.output_index
            );
        }
        Source::FuncOutput => {
            let Some(Binding::FuncProducer(ref f)) = input.binding else {
                anyhow::bail!("op-plan node {node_idx}: FUNC_OUTPUT input missing func producer");
            };
            let func = compute_graph.functions.get(f.func_index as usize).ok_or_else(|| {
                anyhow::anyhow!("op-plan node {node_idx}: func producer {} out of range", f.func_index)
            })?;
            anyhow::ensure!(
                (f.output_index as usize) < func.outputs.len(),
                "op-plan node {node_idx}: func output {} out of range for func {}",
                f.output_index,
                f.func_index
            );
        }
        Source::Weight => {
            let Some(Binding::WeightName(name)) = &input.binding else {
                anyhow::bail!("op-plan node {node_idx}: WEIGHT input missing weight_name");
            };
            anyhow::ensure!(
                weight_provider.name_mapping().contains_key(name.as_str())
                    || weight_provider.constants().contains_key(name.as_str()),
                "op-plan node {node_idx}: weight {name:?} not in sfa_weights"
            );
        }
        Source::Constant => {
            let Some(Binding::ConstantName(name)) = &input.binding else {
                anyhow::bail!("op-plan node {node_idx}: CONSTANT input missing constant_name");
            };
            anyhow::ensure!(
                weight_provider.constants().contains_key(name.as_str()),
                "op-plan node {node_idx}: constant {name:?} not in sfa_weights"
            );
        }
        Source::Global => {
            let Some(Binding::GlobalIndex(idx)) = input.binding else {
                anyhow::bail!("op-plan node {node_idx}: GLOBAL input missing global_index");
            };
            anyhow::ensure!(
                (idx as usize) < plan.global_inputs.len(),
                "op-plan node {node_idx}: global index {idx} out of range"
            );
        }
    }
    anyhow::ensure!(
        input.spec.as_ref().is_some() && matches!(input.spec.as_ref().unwrap().dtype.as_str(), "float32"),
        "op-plan node {node_idx}: only float32 plan inputs are supported in M1"
    );
    Ok(())
}

fn validate_cache_projection(
    plan: &OpPlan,
    compute_graph: &ComputeGraph,
    cache_policy: &CachePolicy,
) -> Result<(), anyhow::Error> {
    let mut covered_funcs = HashSet::new();
    for node in &plan.nodes {
        covered_funcs.extend(node.source_func_indices.iter().map(|&f| f as usize));
    }

    let mut projected = HashSet::new();
    for node in &plan.nodes {
        for output in &node.outputs {
            if output.cache.is_some() {
                let cache = output.cache.as_ref().unwrap();
                let fi = cache.source_func_index as usize;
                let oi = cache.source_output_index as usize;
                anyhow::ensure!(
                    cache_policy.intercepts.iter().any(|i| {
                        i.func_index == fi && i.output_index == oi && i.slab_id == cache.slab_id
                    }),
                    "op-plan cache projection ({fi},{oi},{}) missing from SfaCachePolicy",
                    cache.slab_id
                );
                projected.insert((fi, oi));
            }
        }
    }

    for intercept in &cache_policy.intercepts {
        if covered_funcs.contains(&intercept.func_index) {
            let key = (intercept.func_index, intercept.output_index);
            anyhow::ensure!(
                projected.contains(&key),
                "SfaCachePolicy intercept {key:?} is not projected onto the op plan"
            );
        }
    }

    // Every projected output must match the ABI consumed_internally flag.
    for node in &plan.nodes {
        for output in &node.outputs {
            if let Some(cache) = &output.cache {
                let func = compute_graph
                    .functions
                    .get(cache.source_func_index as usize)
                    .ok_or_else(|| anyhow::anyhow!("cache source func {} out of range", cache.source_func_index))?;
                let out_def = func
                    .outputs
                    .get(cache.source_output_index as usize)
                    .ok_or_else(|| anyhow::anyhow!("cache source output {} out of range", cache.source_output_index))?;
                anyhow::ensure!(
                    out_def.consumed_internally,
                    "cache projection ({},{}) references a visible output",
                    cache.source_func_index,
                    cache.source_output_index
                );
            }
        }
    }
    Ok(())
}

fn validate_func_output_projection(plan: &OpPlan, compute_graph: &ComputeGraph) -> Result<(), anyhow::Error> {
    let mut covered = HashSet::new();
    for node in &plan.nodes {
        covered.extend(node.source_func_indices.iter().map(|&f| f as usize));
    }
    let mut seen = HashMap::new();
    for fo in &plan.func_outputs {
        let fi = fo.func_index as usize;
        let oi = fo.output_index as usize;
        anyhow::ensure!(covered.contains(&fi), "func-output projection for uncovered func {fi}");
        let func = compute_graph.functions.get(fi).ok_or_else(|| anyhow::anyhow!("func {fi} out of range"))?;
        let out_def = func.outputs.get(oi).ok_or_else(|| anyhow::anyhow!("func {fi} output {oi} out of range"))?;
        anyhow::ensure!(
            fo.consumed_internally == out_def.consumed_internally,
            "func-output projection ({fi},{oi}) consumed flag differs from ABI"
        );
        anyhow::ensure!(
            (fo.value.as_ref().map(|r| r.node_index as usize).unwrap_or(usize::MAX)) < plan.nodes.len(),
            "func-output projection ({fi},{oi}) node out of range"
        );
        anyhow::ensure!(
            seen.insert((fi, oi), ()).is_none(),
            "duplicate func-output projection ({fi},{oi})"
        );
    }
    Ok(())
}

// ── input assembly ──────────────────────────────────────────────────

fn resolve_input(
    plan: &OpPlan,
    node: &OpPlanNode,
    input: &OpPlanInput,
    input_idx: usize,
    compute_graph: &ComputeGraph,
    weight_provider: &WeightProvider,
    weight_cache: &RefCell<HashMap<String, Tensor>>,
    raw_weight_cache: &RefCell<HashMap<String, Tensor>>,
    weight_dtype_mode: WeightDtypeMode,
    func_outputs: &[Vec<Tensor>],
    values: &HashMap<(u32, u32), Tensor>,
    kv_new: &HashMap<(usize, usize), Tensor>,
    positions: &[u32],
    is_decode: bool,
    block_manager: Option<&BlockManager>,
    request_id: Option<&str>,
    cache_policy: &CachePolicy,
    mut cache_ms: Option<&mut f64>,
) -> Result<Tensor, anyhow::Error> {
    use proto::op_plan_input::{Binding, Source};
    let source = Source::try_from(input.source)
        .map_err(|_| anyhow::anyhow!("invalid op-plan input source {}", input.source))?;
    let io_def = spec_to_io_def(&input.spec);
    // Only the linear weight matrix is consumed in source dtype.  Biases and
    // layer-norm affine vectors stay f32-promoted (small, precision-sensitive).
    let use_raw_weight =
        node.op_name == "linear_transb" && input_idx == 1 && source == Source::Weight;

    match source {
        Source::Ssa => {
            let Binding::Producer(ref p) = input.binding.as_ref().expect("validated") else {
                unreachable!()
            };
            let producer = &plan.nodes[p.node_index as usize];
            let producer_out = &producer.outputs[p.output_index as usize];
            if let Some(cache) = &producer_out.cache {
                let slab = find_slab_for_intercept(cache_policy, cache.source_func_index as usize, cache.source_output_index as usize);
                let t0 = cache_ms.as_ref().map(|_| std::time::Instant::now());
                let tensor = intercept_consumed_input(
                    cache.source_func_index as usize,
                    cache.source_output_index as usize,
                    compute_graph,
                    kv_new,
                    block_manager,
                    request_id,
                    positions,
                    is_decode,
                    slab,
                    None,
                )?;
                if let (Some(t0), Some(slot)) = (t0, cache_ms.as_deref_mut()) {
                    *slot += t0.elapsed().as_secs_f64() * 1e3;
                }
                Ok(tensor)
            } else {
                values
                    .get(&(p.node_index, p.output_index))
                    .cloned()
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "op-plan node {}: SSA producer ({},{}) not available",
                            node.index,
                            p.node_index,
                            p.output_index
                        )
                    })
            }
        }
        Source::FuncOutput => {
            let Binding::FuncProducer(ref f) = input.binding.as_ref().expect("validated") else {
                unreachable!()
            };
            let fi = f.func_index as usize;
            let oi = f.output_index as usize;
            let producer = &compute_graph.functions[fi];
            let out_def = &producer.outputs[oi];
            if should_intercept_consumed(fi, oi, cache_policy, out_def) {
                let slab = find_slab_for_intercept(cache_policy, fi, oi);
                let t0 = cache_ms.as_ref().map(|_| std::time::Instant::now());
                let tensor = intercept_consumed_input(
                    fi,
                    oi,
                    compute_graph,
                    kv_new,
                    block_manager,
                    request_id,
                    positions,
                    is_decode,
                    slab,
                    None,
                )?;
                if let (Some(t0), Some(slot)) = (t0, cache_ms.as_deref_mut()) {
                    *slot += t0.elapsed().as_secs_f64() * 1e3;
                }
                Ok(tensor)
            } else {
                let consumed_before = producer.outputs[..oi]
                    .iter()
                    .filter(|o| o.consumed_internally)
                    .count();
                let storage_idx = oi - consumed_before;
                func_outputs[fi]
                    .get(storage_idx)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("op-plan FUNC_OUTPUT ({fi},{oi}) not produced yet"))
            }
        }
        Source::Weight => {
            let Binding::WeightName(name) = input.binding.as_ref().expect("validated") else {
                unreachable!()
            };
            if use_raw_weight {
                load_weight_tensor_for_mode(
                    name,
                    weight_provider,
                    weight_cache,
                    raw_weight_cache,
                    &io_def,
                    weight_dtype_mode,
                )
            } else {
                load_weight_tensor(name, weight_provider, weight_cache, &io_def)
            }
        }
        Source::Constant => {
            let Binding::ConstantName(name) = input.binding.as_ref().expect("validated") else {
                unreachable!()
            };
            load_weight_tensor(name, weight_provider, weight_cache, &io_def)
        }
        Source::Global => {
            let Binding::GlobalIndex(idx) = input.binding.as_ref().expect("validated") else {
                unreachable!()
            };
            let name = &plan.global_inputs[*idx as usize];
            if weight_provider.name_mapping().contains_key(name.as_str())
                || weight_provider.constants().contains_key(name.as_str())
            {
                load_weight_tensor(name, weight_provider, weight_cache, &io_def)
            } else {
                // M1 plans never emit GLOBAL activation inputs (all
                // non-weight boundaries are FUNC_OUTPUT projections).  This
                // branch exists so the contract is explicit and fails
                // loudly instead of fabricating a tensor.
                anyhow::bail!(
                    "op-plan node {}: GLOBAL input {name:?} is an activation; \
                     M1 requires FUNC_OUTPUT boundary projection",
                    node.index
                )
            }
        }
    }
}

fn spec_to_io_def(spec: &Option<proto::OpTensorSpec>) -> IOTensorDef {
    match spec {
        Some(s) => {
            let dtype = match s.dtype.as_str() {
                "f16" | "float16" => Dtype::F16,
                "bf16" | "bfloat16" | "bfloat" => Dtype::BF16,
                "i64" | "int64" => Dtype::I64,
                "i32" | "int32" => Dtype::I32,
                "i8" | "int8" => Dtype::I8,
                "u8" | "uint8" => Dtype::U8,
                _ => Dtype::F32,
            };
            IOTensorDef {
                rank: s.rank as u8,
                shape: s.dims.clone(),
                consumed_internally: false,
                dtype,
            }
        }
        None => IOTensorDef {
            rank: 0,
            shape: Vec::new(),
            consumed_internally: false,
            dtype: Dtype::F32,
        },
    }
}

// ── execution ───────────────────────────────────────────────────────

/// Run one op plan over the covered decoder-layer functions.
///
/// `func_outputs` must already contain `main_0` outputs.  Visible covered
/// outputs are pushed into `func_outputs` by the plan's func-output
/// projection so the later func-level tail (final norm / vocab) works
/// unchanged.
#[allow(clippy::too_many_arguments)]
pub fn run_op_plan(
    plan: &OpPlan,
    compute_graph: &ComputeGraph,
    weight_provider: &WeightProvider,
    weight_cache: &RefCell<HashMap<String, Tensor>>,
    raw_weight_cache: &RefCell<HashMap<String, Tensor>>,
    weight_dtype_mode: WeightDtypeMode,
    func_outputs: &mut [Vec<Tensor>],
    positions: &[u32],
    mut block_manager: Option<&mut BlockManager>,
    request_id: Option<&str>,
    cache_policy: &CachePolicy,
    kv_new: &mut HashMap<(usize, usize), Tensor>,
    mut pool: Option<&mut PlanBufferPool>,
    mut stats: Option<&mut crate::engine::account::OpPlanAccount>,
) -> Result<(), anyhow::Error> {
    let is_decode = positions.len() == 1;
    let profile_verbose = std::env::var("SERVEFORGE_PROFILE")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(0)
        >= 2;
    let timing = profile_verbose || stats.is_some();

    let mut values: HashMap<(u32, u32), Tensor> = HashMap::new();

    for node in &plan.nodes {
        let t_build = timing.then(std::time::Instant::now);
        let mut cache_read_ms = 0.0f64;
        let mut inputs: Vec<Tensor> = Vec::with_capacity(node.inputs.len());
        for (input_idx, input) in node.inputs.iter().enumerate() {
            inputs.push(resolve_input(
                plan,
                node,
                input,
                input_idx,
                compute_graph,
                weight_provider,
                weight_cache,
                raw_weight_cache,
                weight_dtype_mode,
                func_outputs,
                &values,
                kv_new,
                positions,
                is_decode,
                block_manager.as_deref(),
                request_id,
                cache_policy,
                if timing {
                    Some(&mut cache_read_ms)
                } else {
                    None
                },
            )?);
        }

        let shape = kernels::resolve_output_shape(&node.op_name, &inputs, &node.attributes)?;
        let build_ms = t_build
            .map(|t0| t0.elapsed().as_secs_f64() * 1e3)
            .unwrap_or(0.0);
        let numel: usize = shape.iter().product();
        let key = (node.index, 0u32, shape.clone());
        let t_alloc = timing.then(std::time::Instant::now);
        let mut out = match pool.as_deref_mut() {
            Some(p) => p.acquire(key.clone(), numel),
            None => vec![0.0f32; numel],
        };
        let alloc_ms = t_alloc
            .map(|t0| t0.elapsed().as_secs_f64() * 1e3)
            .unwrap_or(0.0);

        let t_exec = timing.then(std::time::Instant::now);
        kernels::execute(&node.op_name, &inputs, &node.attributes, is_decode, &mut out)?;
        let exec_ms = t_exec
            .map(|t0| t0.elapsed().as_secs_f64() * 1e3)
            .unwrap_or(0.0);

        let output = node
            .outputs
            .first()
            .ok_or_else(|| anyhow::anyhow!("op-plan node {} has no output", node.index))?;
        let tensor = Tensor::new_owned(shape, out, Dtype::F32);

        if let Some(cache) = &output.cache {
            // K/V: preserve the exact BlockManager/BNLD semantics of the
            // func path, including the kv_new SSA override.
            let fi = cache.source_func_index as usize;
            let oi = cache.source_output_index as usize;
            let slab = find_slab_for_intercept(cache_policy, fi, oi);
            let func_def = &compute_graph.functions[fi];
            let kv_indices: Vec<usize> = func_def
                .outputs
                .iter()
                .enumerate()
                .filter(|(_, o)| o.consumed_internally)
                .map(|(i, _)| i)
                .collect();
            let is_key = kv_indices.first() == Some(&oi);
            let t_intercept = timing.then(std::time::Instant::now);
            intercept_consumed_output(
                fi,
                oi,
                &tensor,
                kv_new,
                block_manager.as_deref_mut(),
                request_id,
                positions,
                is_decode,
                &func_def.outputs,
                slab,
                is_key,
            )?;
            let intercept_ms = t_intercept
                .map(|t0| t0.elapsed().as_secs_f64() * 1e3)
                .unwrap_or(0.0);
            if let Some(stats) = stats.as_deref_mut() {
                stats.exec_ms += exec_ms;
                stats.cache_ms += cache_read_ms + intercept_ms;
            }
            if profile_verbose {
                eprintln!(
                    "[plan] n{} {} total={:.3}ms build={:.3} alloc={:.3} exec={:.3} kv_write={:.3}",
                    node.index,
                    node.op_name,
                    build_ms + alloc_ms + exec_ms + intercept_ms,
                    build_ms,
                    alloc_ms,
                    exec_ms,
                    intercept_ms
                );
            }
            if let Some(p) = pool.as_deref_mut() {
                // Cache-owned data is cloned into kv_new; the plan buffer
                // can be recycled immediately.
                if let crate::model::tensor::TensorData::Owned(arc) = tensor.data {
                    if let Ok(buf) = std::sync::Arc::try_unwrap(arc) {
                        p.release(key, buf);
                    }
                }
            }
        } else {
            values.insert((node.index, 0), tensor);
            if let Some(stats) = stats.as_deref_mut() {
                stats.exec_ms += exec_ms;
                stats.cache_ms += cache_read_ms;
            }
            if profile_verbose {
                eprintln!(
                    "[plan] n{} {} total={:.3}ms build={:.3} alloc={:.3} exec={:.3}",
                    node.index,
                    node.op_name,
                    build_ms + alloc_ms + exec_ms,
                    build_ms,
                    alloc_ms,
                    exec_ms
                );
            }
        }
    }

    // Push visible function outputs back into the func-level SSA store.
    for fo in &plan.func_outputs {
        let fi = fo.func_index as usize;
        let oi = fo.output_index as usize;
        if fo.consumed_internally {
            anyhow::ensure!(
                kv_new.contains_key(&(fi, oi)),
                "consumed op-plan output ({fi},{oi}) missing from kv_new"
            );
            continue;
        }
        let ref_node = fo.value.as_ref().ok_or_else(|| anyhow::anyhow!("func-output ({fi},{oi}) missing value ref"))?;
        let tensor = values
            .remove(&(ref_node.node_index, ref_node.output_index))
            .ok_or_else(|| anyhow::anyhow!("func-output ({fi},{oi}) value not available"))?;
        anyhow::ensure!(
            func_outputs[fi].is_empty(),
            "func_outputs[{fi}] already populated before op-plan projection"
        );
        func_outputs[fi].push(tensor);
    }

    // Recycle remaining plan-owned buffers into the persistent pool.
    if let Some(p) = pool.as_deref_mut() {
        for ((ni, oi), tensor) in values {
            let shape = tensor.shape.clone();
            if let crate::model::tensor::TensorData::Owned(arc) = tensor.data {
                if let Ok(buf) = std::sync::Arc::try_unwrap(arc) {
                    p.release((ni, oi, shape), buf);
                }
            }
        }
    }
    Ok(())
}

/// Covered func indices for logging/skip decisions.
pub fn covered_func_indices(plan: &OpPlan) -> HashSet<usize> {
    plan.nodes
        .iter()
        .flat_map(|n| n.source_func_indices.iter().map(|&f| f as usize))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_buffer_pool_reuses_same_shape() {
        let mut pool = PlanBufferPool::new();
        let key = (0u32, 0u32, vec![2usize, 3]);
        let mut first = pool.acquire(key.clone(), 6);
        let ptr = first.as_mut_ptr();
        first[0] = 1.0;
        pool.release(key.clone(), first);
        let second = pool.acquire(key, 6);
        assert_eq!(second.as_ptr(), ptr);
        assert_eq!(second[0], 1.0);
    }

    #[test]
    fn spec_to_io_def_maps_rank_and_dims() {
        let def = spec_to_io_def(&Some(proto::OpTensorSpec {
            rank: 2,
            dims: vec![1, 0],
            dtype: "float32".to_string(),
        }));
        assert_eq!(def.rank, 2);
        assert_eq!(def.shape, vec![1, 0]);
    }

    #[test]
    fn covered_func_indices_are_unique_set() {
        let plan = OpPlan {
            global_inputs: vec![],
            nodes: vec![
                OpPlanNode {
                    index: 0,
                    op_name: "add".to_string(),
                    inputs: vec![],
                    outputs: vec![],
                    source_func_indices: vec![1],
                    attributes: HashMap::new(),
                },
                OpPlanNode {
                    index: 1,
                    op_name: "relu".to_string(),
                    inputs: vec![],
                    outputs: vec![],
                    source_func_indices: vec![1, 2],
                    attributes: HashMap::new(),
                },
            ],
            global_output: None,
            func_outputs: vec![],
        };
        let covered = covered_func_indices(&plan);
        assert_eq!(covered, HashSet::from([1, 2]));
    }
}
