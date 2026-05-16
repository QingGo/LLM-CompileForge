use std::cell::RefCell;
use std::ffi::c_void;

use anyhow::bail;
use half::f16;

use crate::compute_graph::{ComputeGraph, InputBinding};
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
    pub fn load(
        dylib_path: &str,
        safetensors_path: Option<&str>,
    ) -> Result<Self, anyhow::Error> {
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
            anyhow::bail!("SFCF data missing compute graph section");
        };

        Ok(Self {
            executable,
            weight_provider,
            compute_graph,
            weight_cache: RefCell::new(std::collections::HashMap::new()),
        })
    }

    pub fn forward(&self, input_ids: &[u32]) -> Result<Tensor<'static>, anyhow::Error> {
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
                        let padded: Vec<i64> = if input_ids.len() >= expected_numel {
                            input_ids[..expected_numel].iter().map(|&id| id as i64).collect()
                        } else {
                            let mut p = input_ids.iter().map(|&id| id as i64).collect::<Vec<_>>();
                            p.resize(expected_numel, 0);
                            p
                        };
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
    use crate::compute_graph::{ComputeGraph, FuncDef, IOTensorDef, InputBinding};

    #[test]
    fn test_executor_load_nonexistent_fails() {
        let result = ModelExecutor::load("/nonexistent/lib.dylib", None);
        assert!(result.is_err());
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
}
