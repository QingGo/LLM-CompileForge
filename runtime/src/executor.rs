use std::cell::RefCell;

use crate::block_manager::BlockManager;
use crate::compute_graph::ComputeGraph;
use crate::error::ExecutorError;
use crate::hal::cpu::CpuDevice;
use crate::hal::cpu::CpuStream;
use crate::kernel_catalog::KernelCatalog;
use crate::hal::traits;
#[cfg(test)]
pub(crate) use crate::hal::traits::Buffer;
use crate::kv_cache::CachePolicy;
use crate::tensor::Tensor;
use crate::weight_loader::{WeightProvider, WeightRegistry};

/// Load a pure-Rust HAL executable (no dylib dependency).
///
/// The generated ``hal_ops_cpu.rs`` functions are compiled directly into
/// the binary via ``#[cfg(feature = "hal-rust")]``.
///
/// ``function_count`` is the number of functions in the compute graph
/// (typically 16 for no-cache or 28 for KV-cache models).
#[cfg(feature = "hal-rust")]
#[allow(dead_code)]
pub fn load_hal_rust_executable(function_count: usize) -> Box<dyn traits::Executable> {
    Box::new(crate::hal::rust::executable::HalRustExecutable::new(function_count))
}

#[cfg(feature = "hal-rust")]
#[allow(dead_code)]
fn load_dylib_executable(device: &dyn traits::Device, dylib_path: &str) -> Result<Box<dyn traits::Executable>, anyhow::Error> {
    let dylib_bytes = dylib_path.as_bytes();
    device.compile(dylib_bytes)
        .map_err(|e| anyhow::anyhow!("Device rejected dylib '{}': {}", dylib_path, e))
}

#[cfg(feature = "hal-rust")]
#[allow(dead_code)]
fn load_hal_rust_executable_from_dir(dylib_dir: &std::path::Path) -> Result<Box<dyn traits::Executable>, anyhow::Error> {
    let constants_path = dylib_dir.join("constants.bin");
    let hal_ir_path = dylib_dir.join("generated").join("hal_ir.json");

    let constants_data = std::fs::read(&constants_path)
        .map_err(|e| anyhow::anyhow!("Failed to read constants.bin from {:?}: {}", constants_path, e))?;
    let hal_ir_content = std::fs::read_to_string(&hal_ir_path)
        .map_err(|e| anyhow::anyhow!("Failed to read hal_ir.json from {:?}: {}", hal_ir_path, e))?;
    let hal_ir: serde_json::Value = serde_json::from_str(&hal_ir_content)
        .map_err(|e| anyhow::anyhow!("Failed to parse hal_ir.json: {}", e))?;
    let function_count = hal_ir["num_functions"]
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("hal_ir.json: missing or invalid 'num_functions'"))?
        as usize;
    Ok(Box::new(
        crate::hal::rust::executable::HalRustExecutable::with_blob(
            function_count, constants_data,
        )
    ))
}

/// Try to load compute graph and weight provider from ``sfa_abi`` and
/// ``sfa_weights`` symbols embedded in the compiled dylib.
///
/// Also attempts to load cache policy from the ``sfa_cache_policy``
/// symbol (proto format).  Returns ``None`` if the symbol is absent
/// (older models).
///
/// Returns ``Ok((WeightProvider, ComputeGraph, Option<CachePolicy>))``
/// on success, or an error if either symbol is missing or the ABI data
/// is malformed.
fn try_load_sfa_abi(
    dylib_path: &str,
    safetensors_path: Option<&str>,
) -> Result<(WeightProvider, ComputeGraph, Option<CachePolicy>), anyhow::Error> {
    // SAFETY: Loading a compiled dylib produced by the SFA toolchain.
    // The sfa_abi / sfa_weights symbols point to protobuf-encoded
    // data embedded at compile time, decoded into owned Rust types.
    let lib = unsafe { libloading::Library::new(dylib_path)? };
    let abi = unsafe { crate::abi::load_sfa_abi(&lib)? };
    let sfa_wp = unsafe { crate::abi::load_sfa_weights(&lib)? };
    let cache_policy_proto = unsafe { crate::abi::load_sfa_cache_policy(&lib)? };
    // build_compute_graph takes references to the decoded proto types
    // and clones all data into the owned ComputeGraph.
    let compute_graph = crate::abi::build_compute_graph(&abi, &sfa_wp)?;
    let registry = WeightRegistry {
        name_mapping: sfa_wp.name_mapping,
        constants: sfa_wp.constants,
    };
    let st_path = safetensors_path.map(std::path::Path::new);
    let weight_provider = WeightProvider::new(registry, st_path)?;
    let cache_policy = match cache_policy_proto {
        Some(p) => Some(CachePolicy::from_proto(&p).map_err(|e| anyhow::anyhow!("{}", e))?),
        None => None,
    };
    Ok((weight_provider, compute_graph, cache_policy))
}

pub struct ModelExecutor {
    pub executable: Box<dyn traits::Executable>,
    pub weight_provider: WeightProvider,
    pub compute_graph: ComputeGraph,
    pub weight_cache: RefCell<std::collections::HashMap<String, Tensor>>,
    #[allow(dead_code)]
    pub catalog: Option<Box<dyn KernelCatalog>>,
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

    /// Load a model using a specific HAL device.  The device is used for
    /// compiling/loading the executable.
    #[allow(unused_variables)]
    #[allow(dead_code)]
    pub fn load_with_device(
        device: &dyn traits::Device,
        dylib_path: &str,
        safetensors_path: Option<&str>,
    ) -> Result<Self, anyhow::Error> {
        #[cfg(not(feature = "hal-rust"))]
        let executable: Box<dyn traits::Executable> = {
            let dylib_bytes = dylib_path.as_bytes();
            device.compile(dylib_bytes)
                .map_err(|e| anyhow::anyhow!("Device rejected dylib '{}': {}", dylib_path, e))?
        };

        #[cfg(feature = "hal-rust")]
        let executable: Box<dyn traits::Executable> = {
            let dylib_p = std::path::Path::new(dylib_path);
            // Use HalRustExecutable only when the caller passes a DIRECTORY
            // (forward_check_hal pattern).  When a .dylib file is passed
            // (forward_check / ModelExecutor legacy path), always use the
            // dylib executable to avoid ciface symbol resolution failures.
            if dylib_p.is_dir() {
                load_hal_rust_executable_from_dir(dylib_p)?
            } else {
                load_dylib_executable(device, dylib_path)?
            }
        };

        // Try to load compute graph and weight provider from sfa_abi
        // symbols when the path points to a compiled dylib file.
        let dylib_p = std::path::Path::new(dylib_path);
        let is_dylib = dylib_p.is_file()
            && dylib_p
                .extension()
                .map_or(false, |ext| ext == "dylib" || ext == "so");

        let (weight_provider, compute_graph, proto_cache_policy, sfcf_version, mut data_pos, data_for_contract): (
            WeightProvider,
            ComputeGraph,
            Option<CachePolicy>,
            u32,
            usize,
            Option<&[u8]>,
        ) = if is_dylib {
            match try_load_sfa_abi(dylib_path, safetensors_path) {
                Ok((wp, cg, cache_pol)) => {
                    log::info!(
                        "Loaded compute graph from sfa_abi: {} functions",
                        cg.functions.len()
                    );
                    // SFCF version 0 signals "skip contract parsing".
                    (wp, cg, cache_pol, 0u32, 0usize, None)
                }
                Err(e) => {
                    log::warn!(
                        "sfa_abi not found, falling back to constants.bin: {}",
                        e
                    );
                    let data = executable.module_data();
                    let (registry, graph_pos, version) =
                        crate::weight_loader::parse_embedded(data)?;
                    let st_path = safetensors_path.map(std::path::Path::new);
                    let wp = WeightProvider::new(registry, st_path)?;
                    let mut pos = graph_pos;
                    let cg = if pos < data.len() {
                        ComputeGraph::parse(data, &mut pos, version)?
                    } else {
                        return Err(ExecutorError::MissingComputeGraph.into());
                    };
                    (wp, cg, None, version, pos, Some(data))
                }
            }
        } else {
            let data = executable.module_data();
            let (registry, graph_pos, version) =
                crate::weight_loader::parse_embedded(data)?;
            let st_path = safetensors_path.map(std::path::Path::new);
            let wp = WeightProvider::new(registry, st_path)?;
            let mut pos = graph_pos;
            let cg = if pos < data.len() {
                ComputeGraph::parse(data, &mut pos, version)?
            } else {
                return Err(ExecutorError::MissingComputeGraph.into());
            };
            (wp, cg, None, version, pos, Some(data))
        };

        // Parse contract section (v4+) for constants.bin path only.
        // SFA ABI path (version=0) skips contract parsing entirely —
        // the ABI encodes global input structure directly.
        if let Some(data) = data_for_contract {
            let contract = crate::weight_loader::parse_contract(data, &mut data_pos)?;
            if !contract.is_empty() {
                log::info!(
                    "SFCF v{} contract: num_global_inputs={:?}, global_input_names={:?}",
                    sfcf_version,
                    contract.get("num_global_inputs"),
                    contract.get("global_input_names"),
                );
                // Validate global input count: the runtime supports 1 input
                // (input_ids only — model computes positions internally) or
                // 2 inputs (input_ids + position_ids) for backward compat.
                if let Some(num_str) = contract.get("num_global_inputs") {
                    if let Ok(n) = num_str.parse::<usize>() {
                        if n != 1 && n != 2 {
                            panic!(
                                "model compiled with {} global inputs, runtime expects 1 or 2. \
                                 Recompile with: scripts/compile.py opt-125m",
                                n,
                            );
                        }
                    }
                }
            }
        }

        // Resolve cache policy: proto (from sfa_cache_policy symbol) first,
        // JSON metadata.json fallback second, none() as last resort.
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
            catalog: None,
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
        let stream = CpuStream;

        let result = crate::compute_graph_runner::run_function_graph(
            &self.compute_graph,
            &*self.executable,
            &self.weight_provider,
            &self.weight_cache,
            &mut func_outputs,
            input_ids,
            positions,
            &stream,
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
        let stream = CpuStream;

        let result = crate::compute_graph_runner::run_function_graph_with_kv_intercept(
            &self.compute_graph,
            &*self.executable,
            &self.weight_provider,
            &self.weight_cache,
            &mut func_outputs,
            input_ids,
            positions,
            &stream,
            block_manager,
            request_id,
            &self.cache_policy,
        )?;

        dump_layers(&func_outputs);

        Ok(result)
    }

    #[allow(dead_code)]
    pub fn forward_decode_cached(
        &self,
        input_ids: &[u32],
        position: u32,
        block_manager: &mut BlockManager,
        request_id: &str,
    ) -> Result<Tensor, anyhow::Error> {
        self.forward_with_kv(
            input_ids,
            &[position],
            Some(block_manager),
            Some(request_id),
        )
    }
}

fn dump_layers(func_outputs: &[Vec<Tensor>]) {
    let Ok(dump_dir) = std::env::var("DUMP_LAYERS") else { return };
    let _ = std::fs::create_dir_all(&dump_dir);
    for (fi, outputs) in func_outputs.iter().enumerate() {
        for (oi, t) in outputs.iter().enumerate() {
            let path = format!("{}/func_{}_{}.npy", dump_dir, fi, oi);
            let slice = t.as_slice();
            if !slice.is_empty() {
                let has_nan = slice.iter().any(|&x| x.is_nan());
                if has_nan {
                    log::warn!(
                        "DUMP_LAYERS: func[{}] output[{}] contains NaN — \
                         possible uninitialized buffer or dynamic shape sret issue",
                        fi, oi,
                    );
                }
                let all_same = slice.iter().all(|&x| x == slice[0]);
                if all_same {
                    log::warn!(
                        "DUMP_LAYERS: func[{}] output[{}] has ALL IDENTICAL \
                         values ({}) — possible read bug",
                        fi, oi, slice[0],
                    );
                }
                if has_nan || all_same {
                    continue;
                }
            } else {
                continue;
            }
            let _ = write_npy(&path, slice, &t.shape);
        }
    }
}

fn write_npy(path: &str, data: &[f32], shape: &[usize]) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)?;
    let shape_str = shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", ");
    let header = if shape.is_empty() {
        "{'descr': '<f4', 'fortran_order': False, 'shape': (), }".to_string()
    } else if shape.len() == 1 {
        format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({},), }}", shape_str)
    } else {
        format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({}), }}", shape_str)
    };
    let header_bytes = header.as_bytes();
    let header_len = header_bytes.len() as u16;
    let padding = (64 - ((10 + header_bytes.len()) % 64)) % 64;
    file.write_all(b"\x93NUMPY")?;
    file.write_all(&[1, 0])?;
    file.write_all(&header_len.to_le_bytes())?;
    file.write_all(header_bytes)?;
    for _ in 0..padding { file.write_all(b" ")?; }
    // SAFETY: `data` is a valid Vec<f32>; `size_of_val(data)` gives the
    // exact byte count, so the raw pointer cast produces a valid &[u8].
    let byte_slice = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    file.write_all(byte_slice)?;
    Ok(())
}

#[cfg(test)]
#[path = "executor_tests.rs"]
mod tests;
