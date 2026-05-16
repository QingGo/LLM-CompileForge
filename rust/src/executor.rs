//! Model executor — loads compiled .dylib and orchestrates inference.
//!
//! The executor is responsible for:
//! 1. Loading the compiled per-function .dylib via ``Executable``.
//! 2. Reading the embedded SFCF blob (weight registry + compute graph).
//! 3. Constructing a zero-copy ``WeightProvider``.
//! 4. Walking the compute graph and dispatching kernel calls via SSA.

use std::ffi::c_void;

use crate::compute_graph::{ComputeGraph, InputBinding};
use crate::hal_cpu::{Executable, KernelFn, MemRefDesc2, MemRefDescAny};
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
            // Keep tensors alive until after kernel call (desc.aligned borrows tensor data)
            let mut _tensors: Vec<Tensor<'static>> = Vec::with_capacity(func_def.num_inputs);
            // Also keep raw byte buffers (e.g., for i64 global input)
            let mut _raw_buffers: Vec<Vec<u8>> = Vec::new();

            for (bi, (binding, io_def)) in func_def.inputs.iter().enumerate() {
                eprintln!("[executor]  bi={} start", bi);
                eprintln!("[executor]  input[{}]", bi);
                eprintln!("[executor]  input[{}] io_def.shape={:?} rank={}", bi, io_def.shape, io_def.rank);
                let shape: Vec<usize> =
                    io_def.shape.iter().map(|&d| d as usize).collect();
                eprintln!("[executor]  input[{}] shape collected, rank={}", bi, shape.len());
                let tensor: Tensor = match binding {
                    InputBinding::GlobalInput => {
                        eprintln!("[executor]  input[{}] = GlobalInput shape={:?}", bi, shape);
                        let expected_numel: usize = shape.iter().product();
                        let padded: Vec<i64> = if input_ids.len() >= expected_numel {
                            input_ids[..expected_numel].iter().map(|&id| id as i64).collect()
                        } else {
                            let mut p = input_ids.iter().map(|&id| id as i64).collect::<Vec<_>>();
                            p.resize(expected_numel, 0);
                            p
                        };
                        // Store i64 data as raw bytes, then build descriptor pointing to it.
                        let raw: Vec<u8> = padded.iter().flat_map(|&v| v.to_ne_bytes()).collect();
                        let p = raw.as_ptr();
                        let memref = MemRefDesc2 {
                            allocated: p as *mut c_void,
                            aligned: p as *mut c_void,
                            offset: 0,
                            sizes: [shape[0] as i64, shape.get(1).copied().unwrap_or(1) as i64],
                            strides: [shape.get(1).copied().unwrap_or(1) as i64, 1],
                        };
                        _raw_buffers.push(raw);
                        let desc = MemRefDescAny::R2(memref);
                        input_ptrs.push(desc.as_input_ptr());
                        input_descs.push(desc);
                        _tensors.push(Tensor::new_owned(vec![], vec![], Dtype::I64));
                        continue;  // skip the common push at loop end
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
                _tensors.push(tensor);
            }
            // Verify counts match — if not, pointers may be dangling
            assert_eq!(_tensors.len(), input_ptrs.len(), "tensor count mismatch");
            eprintln!("[executor]  {} inputs loaded, {} tensors kept alive", input_ptrs.len(), _tensors.len());

            // MLIR emit_c_interface convention: function returns output descriptors
            // as a struct via sret pointer (first argument).  The outputs' data
            // buffers are malloc'd inside the function.
            let mut sret: Vec<u8> = vec![0u8; 65536];
            let sret_ptr = sret.as_mut_ptr() as *mut c_void;

            // Build flat arg list: sret first, then all inputs
            let mut all_args: Vec<*const c_void> = Vec::with_capacity(1 + input_ptrs.len());
            all_args.push(sret_ptr);
            all_args.extend(input_ptrs.iter().copied());
            let t_kernel = std::time::Instant::now();
            unsafe {
                let raw_ptr = match &kernel {
                    KernelFn::HighArity(f) => f.0 as *const (),
                    _ => panic!("expected HighArity kernel"),
                };
                crate::ciface_high::call_high_arity(raw_ptr, &all_args);
            }
            let kernel_ms = t_kernel.elapsed().as_secs_f64() * 1000.0;
            eprintln!("[executor] func[{}] returned OK ({:.1}ms)", fi, kernel_ms);

            // Parse output descriptors from sret buffer.
            // Use io_def.shape (from compute graph) as fallback for 0-sentinel dims.
            let t_output = std::time::Instant::now();
                let mut sret_offset: usize = 0;
            for (oi, io_def) in func_def.outputs.iter().enumerate() {
                let r = io_def.rank as usize;
                let desc_size = 24 + 16 * r;
                let ptr_slice = &sret[sret_offset..sret_offset + desc_size];
                let (aligned, runtime_sizes) = unsafe { parse_sret_descriptor(ptr_slice, r) };
                // Use runtime sizes from sret, but replace suspicious values with safe ones
                let fallback: Vec<i64> = io_def.shape.iter().map(|&d|
                    if d == 0 { 1 } else { d as i64 }
                ).collect();
                // Validate runtime sizes
                for (di, (&runtime, &desired)) in runtime_sizes.iter().zip(fallback.iter()).enumerate() {
                    if runtime <= 0 || runtime > 10_000_000 {
                        eprintln!("[executor]  output[{}] dim[{}]: runtime={}, fallback={}", oi, di, runtime, desired);
                    }
                }
                if oi < 53 {
                    eprintln!("[executor]  output[{:>2}]: sizes={:?}", oi, runtime_sizes);
                }
                let sizes: Vec<i64> = runtime_sizes.iter().zip(fallback.iter()).map(|(&r, &f)|
                    if r <= 0 || r > 10_000_000 { f } else { r }
                ).collect();
                let sizes: Vec<i64> = runtime_sizes.iter().zip(fallback.iter()).map(|(&r, &f)|
                    if r <= 0 || r > 1_000_000_000 { f } else { r }
                ).collect();
                let shape: Vec<usize> = sizes.iter().map(|&s| s as usize).collect();
                if oi < 3 || aligned.is_null() {
                    eprintln!("[executor]  output[{}] aligned={:?} sizes={:?} null={}", oi, aligned, shape, aligned.is_null());
                }
                let data: Vec<f32> = if aligned.is_null() {
                    Vec::new()
                } else {
                    let n: usize = sizes.iter().map(|&s| s as usize).product();
                    unsafe {
                        let slice = std::slice::from_raw_parts(aligned as *const f32, n);
                        slice.to_vec()
                    }
                };
                let shape: Vec<usize> = sizes.iter().map(|&s| s as usize).collect();
                func_outputs[fi].push(Tensor::new_owned(shape, data, Dtype::F32));
                sret_offset += desc_size;
            }
            let output_ms = t_output.elapsed().as_secs_f64() * 1000.0;
            if output_ms > 1.0 {
                eprintln!("[executor] func[{}] output copy: {:.1}ms", fi, output_ms);
            }
        }

        let (g_func, g_idx) = self.compute_graph.global_output;
        let result = &func_outputs[g_func][g_idx];
        Ok(result.to_owned())
    }
}

/// Parse a single memref descriptor from an sret buffer byte slice.
/// LLVM struct layout: {ptr, ptr, i64, [N x i64], [N x i64]} where N = rank.
/// Returns (aligned_ptr, [size0, size1, ...]).
unsafe fn parse_sret_descriptor(slice: &[u8], rank: usize) -> (*mut u8, Vec<i64>) {
    use std::mem;
    let aligned = std::ptr::read_unaligned(slice.as_ptr().add(8) as *const *mut u8);
    let sizes: Vec<i64> = (0..rank).map(|i| {
        let offset = 24 + i * 8;  // after allocated(8) + aligned(8) + offset(8)
        std::ptr::read_unaligned(slice.as_ptr().add(offset) as *const i64)
    }).collect();
    (aligned, sizes)
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
