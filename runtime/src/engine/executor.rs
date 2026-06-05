use std::cell::RefCell;

use crate::model::abi;
use crate::cache::block::BlockManager;
use crate::model::compute_graph::ComputeGraph;
use crate::hal::cpu::CpuDevice;
use crate::hal::traits;
#[cfg(test)]
pub(crate) use crate::hal::traits::Buffer;
use crate::cache::policy::CachePolicy;
use crate::model::tensor::Tensor;
use crate::model::weight_loader::WeightProvider;

pub struct ModelExecutor {
    pub executable: Box<dyn traits::Executable>,
    pub weight_provider: WeightProvider,
    pub compute_graph: ComputeGraph,
    pub weight_cache: RefCell<std::collections::HashMap<String, Tensor>>,
    pub cache_policy: CachePolicy,
}

impl ModelExecutor {
    /// Load a model with the default CPU device.
    #[allow(dead_code)]
    pub fn load(
        dylib_path: &str,
        safetensors_path: Option<&str>,
    ) -> Result<Self, anyhow::Error> {
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
            device.compile(dylib_bytes)
                .map_err(|e| anyhow::anyhow!("Device rejected dylib '{}': {}", dylib_path, e))?
        };

        // Load compute graph and weight provider from proto ABI symbols.
        let (weight_provider, compute_graph, proto_cache_policy) =
            abi::load_from_dylib(dylib_path, safetensors_path)?;
        log::info!(
            "Loaded compute graph from sfa_abi: {} functions",
            compute_graph.functions.len()
        );

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
                    Ok(contents) => {
                        match serde_json::from_str::<serde_json::Value>(&contents) {
                            Ok(meta) => {
                                if let Some(cp_json) = meta.get("cache_policy") {
                                    log::warn!(
                                        "Using JSON CachePolicy fallback — migrate to proto format"
                                    );
                                    CachePolicy::from_dict(cp_json)
                                        .unwrap_or_else(|e| {
                                            log::error!(
                                                "Failed to parse cache_policy from JSON: {}",
                                                e
                                            );
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
                        }
                    }
                    Err(e) => {
                        log::warn!("Failed to read metadata.json: {}", e);
                        CachePolicy::none()
                    }
                }
            } else {
                CachePolicy::none()
            }
        };

        Ok(Self {
            executable,
            weight_provider,
            compute_graph,
            weight_cache: RefCell::new(std::collections::HashMap::new()),
            cache_policy,
        })
    }

    #[allow(dead_code)]
    pub fn forward(&self, input_ids: &[u32]) -> Result<Tensor, anyhow::Error> {
        // Default: use sequential positions [0, 1, ..., N-1] (full prefill)
        let positions: Vec<u32> = (0..input_ids.len() as u32).collect();
        self.forward_with_positions(input_ids, &positions)
    }

    /// Like forward() but accepts explicit positions for each token.
    /// positions[i] gives the position of input_ids[i] in the sequence.
    pub fn forward_with_positions(&self, input_ids: &[u32], positions: &[u32]) -> Result<Tensor, anyhow::Error> {
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

    /// Thin wrapper around [`run_function_graph_with_kv_intercept`].
    pub fn forward_with_kv(
        &self,
        input_ids: &[u32],
        positions: &[u32],
        block_manager: Option<&mut BlockManager>,
        request_id: Option<&str>,
    ) -> Result<Tensor, anyhow::Error> {
        let num_funcs = self.compute_graph.functions.len();
        let mut func_outputs: Vec<Vec<Tensor>> = vec![Vec::new(); num_funcs];
        let stream: &dyn traits::Stream = &crate::hal::cpu::CpuStream;

        let result = crate::engine::compute_graph_runner::run_function_graph_with_kv_intercept(
            &self.compute_graph,
            &*self.executable,
            &self.weight_provider,
            &self.weight_cache,
            &mut func_outputs,
            input_ids,
            positions,
            stream,
            block_manager,
            request_id,
            &self.cache_policy,
        )?;

        dump_layers(&func_outputs);

        Ok(result)
    }
}

use crate::debug::dump::dump_layers;

#[cfg(test)]
#[path = "../tests/executor_tests.rs"]
mod tests;
