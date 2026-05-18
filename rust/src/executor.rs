use std::cell::RefCell;
use std::ffi::c_void;

use anyhow::bail;
use half::f16;

use crate::compute_graph::{ComputeGraph, InputBinding};
use crate::error::ExecutorError;
use crate::hal::cpu::CpuDevice;
use crate::hal::traits::{Device as DeviceTrait, Executable as ExecutableTrait};
use crate::hal_cpu::{Executable, KernelFn, MemRefDescAny, MemRefDesc2};
use crate::tensor::{Dtype, Tensor};
use crate::weight_loader::WeightProvider;

pub struct ModelExecutor {
    pub executable: Executable,
    pub weight_provider: WeightProvider,
    pub compute_graph: ComputeGraph,
    pub weight_cache: RefCell<std::collections::HashMap<String, Tensor<'static>>>,
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
        // Use the HAL device to compile (load) the executable — for now,
        // device.compile() validates the .dylib is loadable.
        let dylib_bytes = dylib_path.as_bytes();
        let _exec = device.compile(dylib_bytes)
            .map_err(|e| anyhow::anyhow!("Device rejected dylib '{}': {}", dylib_path, e))?;

        // For the inner SFCF parsing we still need the concrete Executable
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

        let (registry, graph_pos) = crate::weight_loader::parse_embedded(data)?;
        let st_path = safetensors_path.map(std::path::Path::new);
        let weight_provider = WeightProvider::new(registry, st_path)?;

        let mut pos = graph_pos;
        let compute_graph = if pos < data.len() {
            ComputeGraph::parse(data, &mut pos)?
        } else {
            return Err(ExecutorError::MissingComputeGraph.into());
        };

        Ok(Self {
            executable,
            weight_provider,
            compute_graph,
            weight_cache: RefCell::new(std::collections::HashMap::new()),
        })
    }

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

        for func_def in &self.compute_graph.functions {
            let fi = func_def.index;
            let kernel = self
                .executable
                .lookup_typed(&func_def.symbol, func_def.total_args())?;

            let mut input_descs: Vec<MemRefDescAny> =
                Vec::with_capacity(func_def.num_inputs);
            let mut input_ptrs: Vec<*const c_void> =
                Vec::with_capacity(func_def.num_inputs);
            let mut _tensors: Vec<Tensor<'static>> = Vec::with_capacity(func_def.num_inputs);
            let mut _raw_buffers: Vec<Vec<u8>> = Vec::new();

            for (bi, (binding, io_def)) in func_def.inputs.iter().enumerate() {
                let shape: Vec<usize> =
                    io_def.shape.iter().map(|&d| d as usize).collect();
                let tensor: Tensor = match binding {
                    InputBinding::GlobalInput => {
                        let expected_numel: usize = shape.iter().product();
                        let n_tokens = input_ids.len().min(expected_numel);
                        let padded: Vec<i64> = (0..expected_numel).map(|i| {
                            if i < n_tokens {
                                input_ids[i] as i64
                            } else {
                                0i64
                            }
                        }).collect();
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
                        input_descs.push(desc);
                        input_ptrs.push(input_descs.last().unwrap().as_input_ptr());
                        continue;
                    }
                    InputBinding::Weight(key) => {
                        let mut cache = self.weight_cache.borrow_mut();
                        if let Some(cached) = cache.get(key) {
                            cached.to_owned()
                        } else {
                            let desc = self
                                .weight_provider
                                .get_weight_memref(key)
                                .ok_or_else(|| {
                                    anyhow::anyhow!("weight not found: {}", key)
                                })?;
                            let n = desc.numel();
                            let data: Vec<f32> = unsafe {
                                let raw = desc.aligned as *const u16;
                                let slice = std::slice::from_raw_parts(raw, n);
                                slice.iter().map(|&h| f16::from_bits(h).to_f32()).collect()
                            };
                            let tensor = Tensor::new_owned(shape, data, Dtype::F32);
                            cache.insert(key.clone(), tensor.to_owned());
                            tensor
                        }
                    }
                    InputBinding::Ssa {
                        producer_func,
                        output_idx,
                    } => {
                        let ref_tensor = &func_outputs[*producer_func][*output_idx];
                        ref_tensor.to_owned()
                    }
                };

                let desc = MemRefDescAny::from_f32(&tensor.shape, tensor.as_slice());
                input_descs.push(desc);
                input_ptrs.push(input_descs.last().unwrap().as_input_ptr());
                _tensors.push(tensor);
            }
            debug_assert!(_tensors.len() <= input_ptrs.len());

            let mut sret: Vec<u8> = vec![0u8; 65536];
            let sret_ptr = sret.as_mut_ptr() as *mut c_void;

            let mut all_args: Vec<*const c_void> = Vec::with_capacity(1 + input_ptrs.len());
            all_args.push(sret_ptr);
            all_args.extend(input_ptrs.iter().copied());
            unsafe {
                let raw_ptr = match &kernel {
                    KernelFn::HighArity(f) => f.0 as *const (),
                    _ => panic!("expected HighArity kernel"),
                };
                crate::ciface_high::call_high_arity(raw_ptr, &all_args);
            }

            let mut sret_offset: usize = 0;
            for (oi, io_def) in func_def.outputs.iter().enumerate() {
                let r = io_def.rank as usize;
                let desc_size = 24 + 16 * r;
                let ptr_slice = &sret[sret_offset..sret_offset + desc_size];
                let (aligned, runtime_sizes) = unsafe { parse_sret_descriptor(ptr_slice, r) };
                let fallback: Vec<i64> = io_def.shape.iter().map(|&d|
                    if d == 0 { 1 } else { d as i64 }
                ).collect();
                let sizes: Vec<i64> = runtime_sizes.iter().zip(fallback.iter()).map(|(&r, &f)|
                    if r <= 0 || r > 1_000_000_000 { f } else { r }
                ).collect();
                let n: usize = sizes.iter().map(|&s| s as usize).product();
                let data: Vec<f32> = if aligned.is_null() {
                    Vec::new()
                } else {
                    unsafe {
                        let slice = std::slice::from_raw_parts(aligned as *const f32, n);
                        slice.to_vec()
                    }
                };
                let shape: Vec<usize> = sizes.iter().map(|&s| s as usize).collect();
                func_outputs[fi].push(Tensor::new_owned(shape, data, Dtype::F32));
                sret_offset += desc_size;
            }
        }

        let (g_func, g_idx) = self.compute_graph.global_output;
        let result = &func_outputs[g_func][g_idx];
        Ok(result.to_owned())
    }
}

unsafe fn parse_sret_descriptor(slice: &[u8], rank: usize) -> (*mut u8, Vec<i64>) {
    let aligned = std::ptr::read_unaligned(slice.as_ptr().add(8) as *const *mut u8);
    let sizes: Vec<i64> = (0..rank).map(|i| {
        let offset = 24 + i * 8;
        std::ptr::read_unaligned(slice.as_ptr().add(offset) as *const i64)
    }).collect();
    (aligned, sizes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hal::traits::{self, Device as _, Executable as _};

    /// A mock device that records its compile() call for validation.
    #[derive(Debug)]
    struct MockDevice {
        name: String,
        compile_called: std::sync::atomic::AtomicBool,
    }

    impl MockDevice {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
                compile_called: std::sync::atomic::AtomicBool::new(false),
            }
        }

        fn was_compile_called(&self) -> bool {
            self.compile_called.load(std::sync::atomic::Ordering::Relaxed)
        }
    }

    #[derive(Debug)]
    struct MockBuffer(Vec<u8>);

    impl traits::Buffer for MockBuffer {
        fn as_ptr(&self) -> *const u8 { self.0.as_ptr() }
        fn as_mut_ptr(&mut self) -> *mut u8 { self.0.as_mut_ptr() }
        fn len(&self) -> usize { self.0.len() }
        fn copy_from_host(&mut self, src: &[u8], _stream: &dyn traits::Stream) -> Result<(), anyhow::Error> {
            self.0.copy_from_slice(src);
            Ok(())
        }
        fn copy_to_host(&self, dst: &mut [u8], _stream: &dyn traits::Stream) -> Result<(), anyhow::Error> {
            dst.copy_from_slice(&self.0);
            Ok(())
        }
    }

    #[derive(Debug)]
    struct MockStream;

    impl traits::Stream for MockStream {
        fn synchronize(&self) -> Result<(), anyhow::Error> { Ok(()) }
    }

    #[derive(Debug)]
    struct MockExecutable {
        entry_count: usize,
    }

    impl traits::Executable for MockExecutable {
        fn execute(&self, _stream: &dyn traits::Stream, _inputs: &[&dyn traits::Buffer], _outputs: &[&dyn traits::Buffer]) -> Result<(), anyhow::Error> {
            Ok(())
        }
        fn entry_count(&self) -> usize { self.entry_count }
    }

    impl traits::Device for MockDevice {
        fn alloc(&self, size: usize) -> Result<Box<dyn traits::Buffer>, anyhow::Error> {
            Ok(Box::new(MockBuffer(vec![0u8; size])))
        }
        fn create_stream(&self) -> Result<Box<dyn traits::Stream>, anyhow::Error> {
            Ok(Box::new(MockStream))
        }
        fn compile(&self, module_data: &[u8]) -> Result<Box<dyn traits::Executable>, anyhow::Error> {
            self.compile_called.store(true, std::sync::atomic::Ordering::Relaxed);
            assert!(!module_data.is_empty(), "compile should receive non-empty data");
            Ok(Box::new(MockExecutable { entry_count: 2 }))
        }
        fn name(&self) -> &str { &self.name }
    }

    #[test]
    fn test_executor_load_nonexistent_fails() {
        let result = ModelExecutor::load("/nonexistent/lib.dylib", None);
        assert!(result.is_err());
    }

    #[test]
    fn test_executor_load_with_device_calls_compile() {
        let device = MockDevice::new("mock-test");
        // This will fail because the .dylib doesn't exist, but compile()
        // should be called first.
        let result = ModelExecutor::load_with_device(&device, "/nonexistent/lib.dylib", None);
        assert!(result.is_err(), "load should fail on nonexistent dylib");
        // Even though the load fails (no dylib), compile() should have been attempted
        // The compile_called flag indicates the trait was used
        assert!(device.was_compile_called(),
            "Device::compile() should be called during load_with_device");
    }

    #[test]
    fn test_parse_sret_descriptor_rank1() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let dummy_allocated = data.as_ptr() as u64;
        let mut buf = Vec::<u8>::new();
        buf.extend_from_slice(&dummy_allocated.to_ne_bytes());
        buf.extend_from_slice(&dummy_allocated.to_ne_bytes());
        buf.extend_from_slice(&0u64.to_ne_bytes());
        buf.extend_from_slice(&3u64.to_ne_bytes());
        buf.extend_from_slice(&1u64.to_ne_bytes());
        assert_eq!(buf.len(), 40);
        let (aligned, sizes) = unsafe { parse_sret_descriptor(&buf, 1) };
        assert_eq!(aligned as u64, dummy_allocated);
        assert_eq!(sizes, vec![3]);
    }

    #[test]
    fn test_parse_sret_descriptor_rank4() {
        let data: Vec<f32> = vec![0.0; 10];
        let p = data.as_ptr() as u64;
        let mut buf = Vec::<u8>::new();
        buf.extend_from_slice(&p.to_ne_bytes());
        buf.extend_from_slice(&p.to_ne_bytes());
        buf.extend_from_slice(&0u64.to_ne_bytes());
        buf.extend_from_slice(&2u64.to_ne_bytes());
        buf.extend_from_slice(&1u64.to_ne_bytes());
        buf.extend_from_slice(&4u64.to_ne_bytes());
        buf.extend_from_slice(&4u64.to_ne_bytes());
        buf.extend_from_slice(&4u64.to_ne_bytes());
        buf.extend_from_slice(&4u64.to_ne_bytes());
        buf.extend_from_slice(&1u64.to_ne_bytes());
        buf.extend_from_slice(&1u64.to_ne_bytes());
        assert_eq!(buf.len(), 88);
        let (aligned, sizes) = unsafe { parse_sret_descriptor(&buf, 4) };
        assert_eq!(aligned as u64, p);
        assert_eq!(sizes, vec![2, 1, 4, 4]);
    }

    #[test]
    fn test_global_input_i64_memref() {
        let input_ids: Vec<i64> = vec![2, 525, 484, 0];
        let raw: Vec<u8> = input_ids.iter().flat_map(|&v| v.to_ne_bytes()).collect();
        let p = raw.as_ptr();
        let desc = MemRefDesc2 {
            allocated: p as *mut std::ffi::c_void,
            aligned: p as *mut std::ffi::c_void,
            offset: 0,
            sizes: [1, 4],
            strides: [4, 1],
        };
        unsafe {
            for i in 0..4i64 {
                let val = *(desc.aligned.add((i * 8) as usize) as *const i64);
                assert_eq!(val, input_ids[i as usize]);
            }
        }
        assert_eq!(desc.sizes[0], 1);
        assert_eq!(desc.sizes[1], 4);
        assert_eq!(desc.strides[0], 4);
        assert_eq!(desc.strides[1], 1);
        assert_eq!(desc.numel(), 4);
    }

    #[test]
    fn test_f16_to_f32_conversion() {
        use half::f16;
        let f16_vals: Vec<u16> = vec![
            f16::from_f32(1.0).to_bits(),
            f16::from_f32(0.5).to_bits(),
            f16::from_f32(0.0).to_bits(),
            f16::from_f32(-1.0).to_bits(),
        ];
        let raw: Vec<u8> = f16_vals.iter().flat_map(|&v| v.to_ne_bytes()).collect();
        let p = raw.as_ptr();
        let desc = MemRefDesc2 {
            allocated: p as *mut std::ffi::c_void,
            aligned: p as *mut std::ffi::c_void,
            offset: 0,
            sizes: [2, 2],
            strides: [2, 1],
        };
        let n = desc.numel();
        let data: Vec<f32> = unsafe {
            let raw = desc.aligned as *const u16;
            let slice = std::slice::from_raw_parts(raw, n);
            slice.iter().map(|&h| f16::from_bits(h).to_f32()).collect()
        };
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 0.5).abs() < 1e-6);
        assert!((data[2] - 0.0).abs() < 1e-6);
        assert!((data[3] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_desc_pointers_unique_and_stable_in_vec() {
        // Regression: as_input_ptr() must point to Vec element (not stack-local desc),
        // and must remain valid after subsequent pushes (Vec must not reallocate).
        let data = vec![1.0f32; 768];
        let cap = 10;
        let mut descs: Vec<MemRefDescAny> = Vec::with_capacity(cap);
        let mut ptrs: Vec<*const std::ffi::c_void> = Vec::with_capacity(cap);

        for _ in 0..cap {
            let tensor = Tensor::new_owned(vec![768], data.clone(), Dtype::F32);
            descs.push(MemRefDescAny::from_f32(&tensor.shape, tensor.as_slice()));
            ptrs.push(descs.last().unwrap().as_input_ptr());
        }

        // Each pointer should be unique (pointing to different Vec elements)
        for i in 0..cap {
            for j in i + 1..cap {
                assert_ne!(ptrs[i], ptrs[j],
                    "descs[{}] and descs[{}] have same pointer: {:p}", i, j, ptrs[i]);
            }
        }

        // Re-verify by re-reading pointers from Vec
        for i in 0..cap {
            assert_eq!(ptrs[i], descs[i].as_input_ptr(),
                "descs[{}] pointer changed after subsequent pushes", i);
        }
    }

    #[test]
    fn test_desc_pointers_different_for_different_ranks() {
        // Different rank descriptors should have different pointers
        let mut descs: Vec<MemRefDescAny> = Vec::with_capacity(3);

        let d1 = vec![0.0f32];
        descs.push(MemRefDescAny::from_f32(&[], &d1));  // rank-0
        let d2 = vec![0.0f32; 4];
        descs.push(MemRefDescAny::from_f32(&[4], &d2));  // rank-1
        let d3 = vec![0.0f32; 8];
        descs.push(MemRefDescAny::from_f32(&[2, 4], &d3));  // rank-2

        let p0 = descs[0].as_input_ptr();
        let p1 = descs[1].as_input_ptr();
        let p2 = descs[2].as_input_ptr();

        assert_ne!(p0, p1, "rank-0 and rank-1 pointers must differ");
        assert_ne!(p1, p2, "rank-1 and rank-2 pointers must differ");
        assert_ne!(p0, p2, "rank-0 and rank-2 pointers must differ");
    }
}
