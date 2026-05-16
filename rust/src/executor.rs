//! Model executor — loads compiled .dylib and orchestrates inference.
//!
//! The executor is responsible for:
//! 1. Loading the compiled per-function .dylib via ``Executable``.
//! 2. Reading the embedded SFCF blob (weight registry + compute graph).
//! 3. Constructing a zero-copy ``WeightProvider``.
//! 4. Walking the compute graph and dispatching kernel calls via SSA.

use std::ffi::c_void;

use crate::compute_graph::{ComputeGraph, InputBinding};
use crate::hal_cpu::{Executable, MemRefDescAny};
use crate::tensor::{Dtype, Tensor};
use crate::weight_loader::{parse_embedded, WeightProvider};
use half::f16;

pub struct ModelExecutor {
    executable: Executable,
    pub weight_provider: WeightProvider,
    pub compute_graph: ComputeGraph,
}

impl ModelExecutor {
    pub fn load(
        dylib_path: &str,
        safetensors_path: Option<&str>,
    ) -> Result<Self, anyhow::Error> {
        let executable = Executable::load(dylib_path)
            .map_err(|e| anyhow::anyhow!("Failed to load dylib '{}': {}", dylib_path, e))?;
        let lib = executable.lib();

        // SAFETY: `serveforge_constants_data` is a symbol embedded at compile time
        // into the .dylib by _compile_embedded_data (llvm_backend.py). It points to
        // a static `const uint8_t[]` in the dylib's data section, which is valid
        // for the lifetime of the Library.
        let data_ptr: *const u8 = {
            let sym: libloading::Symbol<*const c_void> = unsafe {
                lib.get(b"serveforge_constants_data")
                    .map_err(|e| anyhow::anyhow!("{}", e))?
            };
            *sym as *const u8
        };
        // SAFETY: `serveforge_constants_size` is a `const uint64_t` symbol in the
        // dylib. `lib.get::<*const u64>` returns a pointer-to-pointer: the outer
        // pointer is the symbol address, and `*(*sym)` reads the u64 value at that
        // address. The symbol is static and valid for the Library lifetime.
        let size_val: u64 = {
            let sym = unsafe {
                lib.get::<*const u64>(b"serveforge_constants_size")
                    .map_err(|e| anyhow::anyhow!("{}", e))?
            };
            unsafe { *(*sym) }
        };
        // SAFETY: `data_ptr` points into the dylib's static data section.
        // `size_val` is read from the same dylib and bounds the data region.
        // Both values are embedded at compile time and immutable.
        let data: &[u8] =
            unsafe { std::slice::from_raw_parts(data_ptr, size_val as usize) };

        let (registry, graph_pos) = parse_embedded(data)?;
        let st_path = safetensors_path.map(std::path::Path::new);
        let weight_provider = WeightProvider::new(registry, st_path)?;

        let mut pos = graph_pos;
        let compute_graph = if pos < data.len() {
            ComputeGraph::parse(data, &mut pos)?
        } else {
            anyhow::bail!("SFCF data missing compute graph section");
        };

        Ok(Self {
            executable,
            weight_provider,
            compute_graph,
        })
    }

    pub fn executable(&self) -> &Executable {
        &self.executable
    }

    /// Run a full forward pass through the compute graph.
    ///
    /// ``input_ids`` supplies token IDs as the global model input.
    /// Returns the logits tensor (usually shape [batch, seq, vocab_size]).
    pub fn forward(&self, input_ids: &[u32]) -> Result<Tensor<'static>, anyhow::Error> {
        let num_funcs = self.compute_graph.functions.len();
        let mut func_outputs: Vec<Vec<Tensor<'static>>> = vec![Vec::new(); num_funcs];
        eprintln!(
            "[executor] forward start: {} funcs, {} input tokens, {} constants",
            num_funcs,
            input_ids.len(),
            self.weight_provider.constants().len(),
        );

        for func_def in &self.compute_graph.functions {
            let fi = func_def.index;
            let kernel = self
                .executable
                .lookup_typed(&func_def.symbol, func_def.total_args())?;

            let mut input_descs: Vec<MemRefDescAny> =
                Vec::with_capacity(func_def.num_inputs);
            let mut input_ptrs: Vec<*const c_void> =
                Vec::with_capacity(func_def.num_inputs);

            // Capture global input shape (batch, seq) for output dim inference
            let mut global_batch: usize = 1;
            let mut global_seq: usize = 1;

            for (bi, (binding, io_def)) in func_def.inputs.iter().enumerate() {
                eprintln!("[executor]  input[{}]", bi);
                eprintln!("[executor]  input[{}] io_def.shape={:?} rank={}", bi, io_def.shape, io_def.rank);
                let shape: Vec<usize> =
                    io_def.shape.iter().map(|&d| d as usize).collect();
                eprintln!("[executor]  input[{}] shape collected, rank={}", bi, shape.len());
                let tensor: Tensor = match binding {
                    InputBinding::GlobalInput => {
                        eprintln!("[executor]  input[{}] = GlobalInput shape={:?}", bi, shape);
                        let expected_numel: usize = shape.iter().product();
                        let data: Vec<f32> = if input_ids.len() >= expected_numel {
                            input_ids[..expected_numel].iter().map(|&id| id as f32).collect()
                        } else {
                            let mut padded = input_ids.iter().map(|&id| id as f32).collect::<Vec<_>>();
                            padded.resize(expected_numel, 0.0);
                            padded
                        };
                        if shape.len() >= 2 {
                            global_batch = shape[0];
                            global_seq = shape[1];
                        } else if shape.len() == 1 {
                            global_batch = shape[0];
                        }
                        Tensor::new_owned(shape, data, Dtype::F32)
                    }
                    InputBinding::Weight(key) => {
                        eprintln!("[executor]  input[{}] = Weight BEFORE get_weight {} (consts={})", bi, key, self.weight_provider.constants().len());
                        let t0 = std::time::Instant::now();
                        let desc = self
                            .weight_provider
                            .get_weight_memref(key)
                            .ok_or_else(|| {
                                anyhow::anyhow!("weight not found: {}", key)
                            })?;
                        eprintln!("[executor]  input[{}] = Weight got memref for {} (sizes={:?})", bi, key, desc.sizes);
                        // Use io_def.shape for correct rank (not desc.sizes which is always rank-2)
                        let n = desc.numel();
                        let data: Vec<f32> = unsafe {
                            let raw = desc.aligned as *const u16;
                            let slice = std::slice::from_raw_parts(raw, n);
                            slice.iter().map(|&h| f16::from_bits(h).to_f32()).collect()
                        };
                        let elapsed = t0.elapsed();
                        eprintln!(
                            "[executor]  input[{}] = Weight {} shape={:?} ({:.1}ms)",
                            bi,
                            key,
                            shape,
                            elapsed.as_secs_f64() * 1000.0,
                        );
                        Tensor::new_owned(shape, data, Dtype::F32)
                    }
                    InputBinding::Ssa {
                        producer_func,
                        output_idx,
                    } => {
                        eprintln!(
                            "[executor]  input[{}] = Ssa func[{}][{}]",
                            bi, producer_func, output_idx
                        );
                        let output_len = func_outputs[*producer_func].len();
                        eprintln!("[executor]  input[{}] Ssa func_outputs[{}] has {} entries", bi, producer_func, output_len);
                        let ref_tensor = &func_outputs[*producer_func][*output_idx];
                        ref_tensor.to_owned()
                    }
                };
                eprintln!("[executor]  input[{}] tensor done, shape={:?}", bi, tensor.shape);

                let desc = MemRefDescAny::from_f32(&tensor.shape, tensor.as_slice());
                input_ptrs.push(desc.as_input_ptr());
                input_descs.push(desc);
            }

            let mut output_descs: Vec<MemRefDescAny> =
                Vec::with_capacity(func_def.num_outputs);
            for io_def in &func_def.outputs {
                let mut shape: Vec<usize> =
                    io_def.shape.iter().map(|&d| d as usize).collect();
                // Replace 0-sentinel dims with inferred batch/seq from global input
                for d in shape.iter_mut() {
                    if *d == 0 {
                        *d = global_batch.max(1);
                    }
                }
                output_descs.push(MemRefDescAny::zeroed(&shape));
            }

            eprintln!(
                "[executor] calling func[{}] symbol={} outputs={} inputs={}",
                fi, func_def.symbol, output_descs.len(), input_ptrs.len()
            );
            // Collect output pointers for ciface ABI
            let output_ptrs: Vec<*mut c_void> =
                output_descs.iter().map(|od| od.as_output_ptr()).collect();
            unsafe {
                kernel.call(&output_ptrs, &input_ptrs);
            }
            eprintln!("[executor] func[{}] returned OK", fi);

            for od in &output_descs {
                let data = unsafe { od.read_output_f32() };
                let shape: Vec<usize> = od.sizes();
                func_outputs[fi].push(Tensor::new_owned(shape, data, Dtype::F32));
            }
        }

        let (g_func, g_idx) = self.compute_graph.global_output;
        let result = &func_outputs[g_func][g_idx];
        Ok(result.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_graph::{ComputeGraph, FuncDef, IOTensorDef, InputBinding};

    #[test]
    fn test_executor_load_nonexistent_fails() {
        let result = ModelExecutor::load("/nonexistent/lib.dylib", None);
        assert!(result.is_err());
    }
}
