use std::cell::RefCell;

use crate::cache::block::BlockManager;
use crate::cache::policy::CachePolicy;
use crate::engine::account::ForwardAccount;
use crate::engine::compute_graph_runner::OutputBufferPool;
use crate::engine::op_plan::{OpPlan, PlanBufferPool};
use crate::hal::cpu::CpuDevice;
use crate::hal::traits;
#[cfg(test)]
pub(crate) use crate::hal::traits::Buffer;
use crate::model::abi;
use crate::model::compute_graph::ComputeGraph;
use crate::model::tensor::{Dtype, Tensor};
use crate::model::weight_loader::WeightProvider;

/// Phase 5 execution-path selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecPlanMode {
    /// Use the op plan when the dylib exports `sfa_op_plan` and the load
    /// contract assertions pass; otherwise use the func path.
    Auto,
    /// Always use the func-level `_mlir_ciface_*` path.
    Func,
    /// Require the op plan and fail when it is missing.
    Op,
}

/// Weight storage dtype policy for dtype-aware op-plan kernels.
///
/// `Auto` preserves the safetensors source dtype.  Explicit values are A/B
/// switches for gate comparison; per G10 they never cross-convert a source
/// dtype (F16 source + `--weight-dtype bf16` is a hard error, not a silent
/// cast).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightDtypeMode {
    Auto,
    F32,
    F16,
    Bf16,
}

impl WeightDtypeMode {
    pub(crate) fn permits(self, source: Dtype) -> bool {
        match self {
            WeightDtypeMode::Auto => matches!(source, Dtype::F32 | Dtype::F16 | Dtype::BF16),
            WeightDtypeMode::F32 => source == Dtype::F32,
            WeightDtypeMode::F16 => source == Dtype::F16,
            WeightDtypeMode::Bf16 => source == Dtype::BF16,
        }
    }
}

pub struct ModelExecutor {
    pub executable: Box<dyn traits::Executable>,
    pub weight_provider: WeightProvider,
    pub compute_graph: ComputeGraph,
    pub weight_cache: RefCell<std::collections::HashMap<String, Tensor>>,
    /// Source-dtype (F16/BF16) raw weights for op-plan kernels.  Kept
    /// separate from `weight_cache`, which always holds f32-promoted tensors
    /// for the func-level dylib path.
    pub raw_weight_cache: RefCell<std::collections::HashMap<String, Tensor>>,
    pub cache_policy: CachePolicy,
    pub output_buffer_pool: RefCell<OutputBufferPool>,
    /// Phase 4 Path C prototype flag.  Off by default; enables the OPT
    /// fused decoder-layer path in [`forward_with_kv`].
    pub opt_fused_fastpath: bool,
    /// Phase 5 additive op plan (present only on rebuilt KV dylibs).
    pub op_plan: Option<OpPlan>,
    /// `--exec-plan` selection.
    pub exec_plan_mode: ExecPlanMode,
    /// `--weight-dtype` A/B selection for dtype-aware kernels.
    pub weight_dtype_mode: WeightDtypeMode,
    /// Persistent per-node output buffers for the op-plan path.
    pub plan_buffer_pool: RefCell<PlanBufferPool>,
}

impl ModelExecutor {
    /// Load a model with the default CPU device.
    #[allow(dead_code)]
    pub fn load(dylib_path: &str, safetensors_path: Option<&str>) -> Result<Self, anyhow::Error> {
        let device = CpuDevice::new();
        Self::load_with_device(&device, dylib_path, safetensors_path)
    }

    /// Load a model using a specific HAL device.
    #[allow(dead_code)]
    pub fn load_with_device(
        device: &dyn traits::Device,
        dylib_path: &str,
        safetensors_path: Option<&str>,
    ) -> Result<Self, anyhow::Error> {
        let executable: Box<dyn traits::Executable> = {
            let dylib_bytes = dylib_path.as_bytes();
            device
                .compile(dylib_bytes)
                .map_err(|e| anyhow::anyhow!("Device rejected dylib '{}': {}", dylib_path, e))?
        };

        // Load compute graph, weight provider, and the additive op plan
        // from the proto ABI symbols.
        let (weight_provider, compute_graph, proto_cache_policy, op_plan) =
            abi::load_from_dylib_full(dylib_path, safetensors_path)?;
        log::info!(
            "Loaded compute graph from sfa_abi: {} functions, op plan: {}",
            compute_graph.functions.len(),
            op_plan.as_ref().map(|p| p.nodes.len()).unwrap_or(0)
        );

        // Feature-detect: the sfa_cache_policy proto symbol exists only on
        // newly compiled dylibs.  Its presence switches on the hard
        // cross-contract assertion below.
        let policy_from_proto = proto_cache_policy.is_some();

        // Resolve cache policy: proto first, JSON metadata.json fallback.
        let cache_policy = if let Some(pol) = proto_cache_policy {
            pol
        } else {
            let dylib_p = std::path::Path::new(dylib_path);
            let meta_path = if let Some(parent) = dylib_p.parent() {
                parent.join("metadata.json")
            } else {
                std::path::PathBuf::from("metadata.json")
            };
            if meta_path.exists() {
                match std::fs::read_to_string(&meta_path) {
                    Ok(contents) => match serde_json::from_str::<serde_json::Value>(&contents) {
                        Ok(meta) => {
                            if let Some(cp_json) = meta.get("cache_policy") {
                                log::warn!(
                                    "Using JSON CachePolicy fallback — migrate to proto format"
                                );
                                CachePolicy::from_dict(cp_json).unwrap_or_else(|e| {
                                    log::error!("Failed to parse cache_policy from JSON: {}", e);
                                    CachePolicy::none()
                                })
                            } else {
                                CachePolicy::none()
                            }
                        }
                        Err(e) => {
                            log::warn!("Failed to parse metadata.json: {}", e);
                            CachePolicy::none()
                        }
                    },
                    Err(e) => {
                        log::warn!("Failed to read metadata.json: {}", e);
                        CachePolicy::none()
                    }
                }
            } else {
                CachePolicy::none()
            }
        };

        // Dual-contract hard assertion (proto-policy dylibs only):
        // consumed_internally flags ⟺ bound intercepts, plus the GQA
        // head-count pre-check.  A mismatch here is a compiler bug and
        // must fail loudly at load, not misbehave at decode time.
        if policy_from_proto {
            crate::cache::contract::cross_assert_cache_contract(&compute_graph, &cache_policy)?;
        }

        // Phase 5 load-time hard assertions: any malformed op plan fails
        // here, before a single node executes.
        if let Some(plan) = &op_plan {
            crate::engine::op_plan::validate_op_plan(
                plan,
                &compute_graph,
                &weight_provider,
                &cache_policy,
            )?;
            log::info!("Op plan contract validated: {} nodes", plan.nodes.len());
        }

        Ok(Self {
            executable,
            weight_provider,
            compute_graph,
            weight_cache: RefCell::new(std::collections::HashMap::new()),
            raw_weight_cache: RefCell::new(std::collections::HashMap::new()),
            cache_policy,
            output_buffer_pool: RefCell::new(OutputBufferPool::new()),
            opt_fused_fastpath: false,
            op_plan,
            exec_plan_mode: ExecPlanMode::Auto,
            weight_dtype_mode: WeightDtypeMode::Auto,
            plan_buffer_pool: RefCell::new(PlanBufferPool::new()),
        })
    }

    /// Enable/disable the Phase 4 OPT fused decoder-layer prototype.
    pub fn set_opt_fused_fastpath(&mut self, enabled: bool) {
        self.opt_fused_fastpath = enabled;
    }

    /// Select the Phase 5 execution path (`--exec-plan`).
    pub fn set_exec_plan_mode(&mut self, mode: ExecPlanMode) {
        self.exec_plan_mode = mode;
    }

    /// Select weight storage dtype policy (`--weight-dtype`).
    pub fn set_weight_dtype_mode(&mut self, mode: WeightDtypeMode) {
        self.weight_dtype_mode = mode;
    }

    /// Resolve the effective op plan for one forward pass.
    fn effective_op_plan(&self) -> Result<Option<&OpPlan>, anyhow::Error> {
        match self.exec_plan_mode {
            ExecPlanMode::Auto => Ok(self.op_plan.as_ref()),
            ExecPlanMode::Func => Ok(None),
            ExecPlanMode::Op => {
                anyhow::ensure!(
                    self.op_plan.is_some(),
                    "--exec-plan op requested but this dylib has no sfa_op_plan symbol \
                     (rebuild with compiler/compile_dylib.py)"
                );
                Ok(self.op_plan.as_ref())
            }
        }
    }

    #[allow(dead_code)]
    pub fn forward(&self, input_ids: &[u32]) -> Result<Tensor, anyhow::Error> {
        // Default: use sequential positions [0, 1, ..., N-1] (full prefill)
        let positions: Vec<u32> = (0..input_ids.len() as u32).collect();
        self.forward_with_positions(input_ids, &positions)
    }

    /// Like forward() but accepts explicit positions for each token.
    /// positions[i] gives the position of input_ids[i] in the sequence.
    pub fn forward_with_positions(
        &self,
        input_ids: &[u32],
        positions: &[u32],
    ) -> Result<Tensor, anyhow::Error> {
        let num_funcs = self.compute_graph.functions.len();
        let mut func_outputs: Vec<Vec<Tensor>> = vec![Vec::new(); num_funcs];
        let stream: &dyn traits::Stream = &crate::hal::cpu::CpuStream;

        let result = crate::engine::compute_graph_runner::run_function_graph(
            &self.compute_graph,
            &*self.executable,
            &self.weight_provider,
            &self.weight_cache,
            &mut func_outputs,
            input_ids,
            positions,
            stream,
        )?;

        dump_layers(&func_outputs);

        Ok(result)
    }

    /// Accounted variant of [`forward_with_positions`] for the rare
    /// non-KV runner configuration.  The non-KV graph runner has no
    /// per-function accounting seam yet, so the whole forward call is
    /// conservatively attributed to `compute` (no cache exists by
    /// definition).
    pub(crate) fn forward_with_positions_accounted(
        &self,
        input_ids: &[u32],
        positions: &[u32],
    ) -> Result<(Tensor, ForwardAccount), anyhow::Error> {
        let t0 = std::time::Instant::now();
        let tensor = self.forward_with_positions(input_ids, positions)?;
        let mut account = ForwardAccount {
            total_ms: t0.elapsed().as_secs_f64() * 1e3,
            compute_ms: 0.0,
            cache_ms: 0.0,
            executor_ms: 0.0,
        };
        account.compute_ms = account.total_ms;
        Ok((tensor, account))
    }

    /// Thin wrapper around [`run_function_graph_with_kv_intercept`].
    pub fn forward_with_kv(
        &self,
        input_ids: &[u32],
        positions: &[u32],
        block_manager: Option<&mut BlockManager>,
        request_id: Option<&str>,
    ) -> Result<Tensor, anyhow::Error> {
        self.forward_with_kv_inner(input_ids, positions, block_manager, request_id, None)
    }

    /// Accounted variant used by `SERVEFORGE_ACCOUNT=1`.  The caller opted
    /// into timing, so this method measures the full forward call and
    /// derives the executor residual.
    pub(crate) fn forward_with_kv_accounted(
        &self,
        input_ids: &[u32],
        positions: &[u32],
        block_manager: Option<&mut BlockManager>,
        request_id: Option<&str>,
    ) -> Result<(Tensor, ForwardAccount), anyhow::Error> {
        let t0 = std::time::Instant::now();
        let mut account = ForwardAccount::default();
        let tensor =
            self.forward_with_kv_inner(input_ids, positions, block_manager, request_id, Some(&mut account))?;
        account.total_ms = t0.elapsed().as_secs_f64() * 1e3;
        account.finalize();
        Ok((tensor, account))
    }

    fn forward_with_kv_inner(
        &self,
        input_ids: &[u32],
        positions: &[u32],
        block_manager: Option<&mut BlockManager>,
        request_id: Option<&str>,
        account: Option<&mut ForwardAccount>,
    ) -> Result<Tensor, anyhow::Error> {
        let num_funcs = self.compute_graph.functions.len();
        let mut func_outputs: Vec<Vec<Tensor>> = vec![Vec::new(); num_funcs];
        let stream: &dyn traits::Stream = &crate::hal::cpu::CpuStream;
        let mut output_pool = self.output_buffer_pool.borrow_mut();
        let op_plan_ref = self.effective_op_plan()?;
        let mut plan_buffer_pool = self.plan_buffer_pool.borrow_mut();

        let result =
            crate::engine::compute_graph_runner::run_function_graph_with_kv_intercept_pooled(
                &self.compute_graph,
                &*self.executable,
                &self.weight_provider,
                &self.weight_cache,
                &self.raw_weight_cache,
                self.weight_dtype_mode,
                &mut func_outputs,
                input_ids,
                positions,
                stream,
                block_manager,
                request_id,
                &self.cache_policy,
                Some(&mut output_pool),
                self.opt_fused_fastpath,
                op_plan_ref,
                Some(&mut plan_buffer_pool),
                account,
            )?;

        dump_layers(&func_outputs);

        Ok(result)
    }
}

use crate::debug::dump::dump_layers;

#[cfg(test)]
#[path = "../tests/executor_tests.rs"]
mod tests;
