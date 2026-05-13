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
        let executable = Executable::load(dylib_path)?;
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

        for func_def in &self.compute_graph.functions {
            let fi = func_def.index;
            let kernel = self
                .executable
                .lookup_typed(&func_def.symbol, func_def.total_args())?;

            // input_descs holds owned MemRefDescAny values that keep the
            // underlying data (Tensor slices) alive during the kernel call.
            let mut input_descs: Vec<MemRefDescAny> =
                Vec::with_capacity(func_def.num_inputs);
            let mut input_ptrs: Vec<*const c_void> =
                Vec::with_capacity(func_def.num_inputs);

            for (binding, io_def) in &func_def.inputs {
                let shape: Vec<usize> =
                    io_def.shape.iter().map(|&d| d as usize).collect();
                let tensor: Tensor = match binding {
                    InputBinding::GlobalInput => {
                        let data: Vec<f32> =
                            input_ids.iter().map(|&id| id as f32).collect();
                        Tensor::new_owned(shape, data, Dtype::F32)
                    }
                    InputBinding::Weight(key) => {
                        let desc = self
                            .weight_provider
                            .get_weight_memref(key)
                            .ok_or_else(|| {
                                anyhow::anyhow!("weight not found: {}", key)
                            })?;
                        let wshape: Vec<usize> =
                            desc.sizes.iter().map(|&s| s as usize).collect();
                        // SAFETY: `desc` was populated by `get_weight_memref`, which
                        // returns a descriptor pointing to data in either the mmap'd
                        // safetensors file or the embedded constants blob — both of
                        // which live as long as `self.weight_provider`.
                        let data = unsafe { desc.read_output_f32() };
                        Tensor::new_owned(wshape, data, Dtype::F32)
                    }
                    InputBinding::Ssa {
                        producer_func,
                        output_idx,
                    } => func_outputs[*producer_func][*output_idx].to_owned(),
                };

                let desc = MemRefDescAny::from_f32(&tensor.shape, tensor.as_slice());
                input_ptrs.push(desc.as_input_ptr());
                input_descs.push(desc);
            }

            let mut output_descs: Vec<MemRefDescAny> =
                Vec::with_capacity(func_def.num_outputs);
            for io_def in &func_def.outputs {
                let shape: Vec<usize> =
                    io_def.shape.iter().map(|&d| d as usize).collect();
                output_descs.push(MemRefDescAny::zeroed(&shape));
            }

            // SAFETY: The kernel call is safe because:
            // - `output_descs[0]` is a properly zeroed `MemRefDescAny` whose
            //   layout matches the MLIR strided memref descriptor ABI.
            // - `input_ptrs` points to `input_descs` descriptors that are
            //   constructed from valid tensor data kept alive by `input_descs`.
            // - The kernel function pointer was loaded via `lookup_typed` which
            //   uses `libloading::Symbol` with the correct `CifaceFn*` type.
            // - All descriptors remain on the stack for the call duration.
            unsafe {
                kernel.call(output_descs[0].as_output_ptr(), &input_ptrs);
            }

            for od in &output_descs {
                // SAFETY: The kernel was called immediately above and should have
                // populated `od.aligned` with a malloc'd output buffer. If the
                // kernel failed to do so, `od.aligned` stays null and
                // `read_output_f32` returns an empty Vec (graceful degradation).
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
