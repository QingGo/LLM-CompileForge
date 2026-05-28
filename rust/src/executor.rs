use std::cell::RefCell;
use std::ffi::c_void;

use crate::block_manager::BlockManager;
use crate::compute_graph::ComputeGraph;
use crate::error::ExecutorError;
use crate::hal::cpu::CpuDevice;
use crate::kernel_catalog::KernelCatalog;
use crate::hal::cpu::Executable;
use crate::hal::traits::Device as DeviceTrait;
use crate::kv_cache::CachePolicy;
use crate::tensor::Tensor;
use crate::weight_loader::WeightProvider;

pub struct ModelExecutor {
    pub executable: Executable,
    pub weight_provider: WeightProvider,
    pub compute_graph: ComputeGraph,
    pub weight_cache: RefCell<std::collections::HashMap<String, Tensor<'static>>>,
    #[allow(dead_code)]
    pub catalog: Option<Box<dyn KernelCatalog>>,
    pub cache_policy: CachePolicy,
}

impl ModelExecutor {
    /// Load a model with the default CPU device.
    pub fn load(
        dylib_path: &str,
        safetensors_path: Option<&str>,
    ) -> Result<Self, anyhow::Error> {
        let device = CpuDevice::new();
        Self::load_with_device(&device, dylib_path, safetensors_path)
    }

    /// Load a model using a specific HAL device.  The device is used for
    /// compiling/loading the executable.
    pub fn load_with_device(
        device: &dyn DeviceTrait,
        dylib_path: &str,
        safetensors_path: Option<&str>,
    ) -> Result<Self, anyhow::Error> {
        let dylib_bytes = dylib_path.as_bytes();
        let _exec = device.compile(dylib_bytes)
            .map_err(|e| anyhow::anyhow!("Device rejected dylib '{}': {}", dylib_path, e))?;

        let executable = Executable::load(dylib_path)
            .map_err(|e| anyhow::anyhow!("Failed to load dylib '{}': {}", dylib_path, e))?;
        let lib = executable.lib();

        let data_ptr: *const u8 = {
            let sym: libloading::Symbol<*const c_void> = unsafe {
                lib.get(b"serveforge_constants_data")
                    .map_err(|e| anyhow::anyhow!("{}", e))?
            };
            *sym as *const u8
        };
        let size_val: u64 = {
            let sym = unsafe {
                lib.get::<*const u64>(b"serveforge_constants_size")
                    .map_err(|e| anyhow::anyhow!("{}", e))?
            };
            unsafe { *(*sym) }
        };
        let data: &[u8] =
            unsafe { std::slice::from_raw_parts(data_ptr, size_val as usize) };

        let (registry, graph_pos, sfcf_version) = crate::weight_loader::parse_embedded(data)?;
        let st_path = safetensors_path.map(std::path::Path::new);
        let weight_provider = WeightProvider::new(registry, st_path)?;

        let mut pos = graph_pos;
        let compute_graph = if pos < data.len() {
            ComputeGraph::parse(data, &mut pos, sfcf_version)?
        } else {
            return Err(ExecutorError::MissingComputeGraph.into());
        };

        // Parse contract section (v4+): key-value metadata appended after
        // the compute graph. Backward compat: v2/v3 has no contract section.
        let contract = crate::weight_loader::parse_contract(data, &mut pos)?;
        if !contract.is_empty() {
            log::info!(
                "SFCF v{} contract: num_global_inputs={:?}, global_input_names={:?}",
                sfcf_version,
                contract.get("num_global_inputs"),
                contract.get("global_input_names"),
            );
            // Validate that the model was compiled with exactly 2 global
            // inputs (input_ids, position_ids) as required by the runtime.
            if let Some(num_str) = contract.get("num_global_inputs") {
                if let Ok(n) = num_str.parse::<usize>() {
                    if n != 2 {
                        panic!(
                            "model compiled with {} global inputs, runtime expects 2. \
                             Recompile with: scripts/compile.py opt-125m",
                            n,
                        );
                    }
                }
            }
        }

        Ok(Self {
            executable,
            weight_provider,
            compute_graph,
            weight_cache: RefCell::new(std::collections::HashMap::new()),
            catalog: None,
            cache_policy: CachePolicy::none(),
        })
    }

    #[allow(dead_code)]
    pub fn forward(&self, input_ids: &[u32]) -> Result<Tensor<'static>, anyhow::Error> {
        // Default: use sequential positions [0, 1, ..., N-1] (full prefill)
        let positions: Vec<u32> = (0..input_ids.len() as u32).collect();
        self.forward_with_positions(input_ids, &positions)
    }

    /// Like forward() but accepts explicit positions for each token.
    /// positions[i] gives the position of input_ids[i] in the sequence.
    pub fn forward_with_positions(&self, input_ids: &[u32], positions: &[u32]) -> Result<Tensor<'static>, anyhow::Error> {
        let num_funcs = self.compute_graph.functions.len();
        let mut func_outputs: Vec<Vec<Tensor<'static>>> = vec![Vec::new(); num_funcs];

        let result = crate::compute_graph_runner::run_function_graph(
            &self.compute_graph,
            &self.executable,
            &self.weight_provider,
            &self.weight_cache,
            &mut func_outputs,
            input_ids,
            positions,
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
    ) -> Result<Tensor<'static>, anyhow::Error> {
        let num_funcs = self.compute_graph.functions.len();
        let mut func_outputs: Vec<Vec<Tensor<'static>>> = vec![Vec::new(); num_funcs];

        let result = crate::compute_graph_runner::run_function_graph_with_kv_intercept(
            &self.compute_graph,
            &self.executable,
            &self.weight_provider,
            &self.weight_cache,
            &mut func_outputs,
            input_ids,
            positions,
            block_manager,
            request_id,
        )?;

        dump_layers(&func_outputs);

        Ok(result)
    }

    pub fn forward_decode_cached(
        &self,
        input_ids: &[u32],
        position: u32,
        block_manager: &mut BlockManager,
        request_id: &str,
    ) -> Result<Tensor<'static>, anyhow::Error> {
        self.forward_with_kv(
            input_ids,
            &[position],
            Some(block_manager),
            Some(request_id),
        )
    }
}

fn dump_layers(func_outputs: &[Vec<Tensor<'static>>]) {
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
    let byte_slice = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * std::mem::size_of::<f32>())
    };
    file.write_all(byte_slice)?;
    Ok(())
}

#[cfg(test)]
#[path = "executor_tests.rs"]
mod tests;
