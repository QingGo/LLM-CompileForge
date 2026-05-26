use std::cell::RefCell;
use std::ffi::c_void;

use half::f16;

use crate::compute_graph::{ComputeGraph, InputBinding};
use crate::error::ExecutorError;
use crate::hal::cpu::CpuDevice;
use crate::hal::cpu::memref::MemRefDesc1;
use crate::hal::cpu::{Executable, MemRefDescAny, MemRefDesc2};
use crate::hal::traits::Device as DeviceTrait;
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

        let (registry, graph_pos, sfcf_version) = crate::weight_loader::parse_embedded(data)?;
        let st_path = safetensors_path.map(std::path::Path::new);
        let weight_provider = WeightProvider::new(registry, st_path)?;

        let mut pos = graph_pos;
        let compute_graph = if pos < data.len() {
            ComputeGraph::parse(data, &mut pos, sfcf_version)?
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

    #[allow(dead_code)]
    pub fn forward(&self, input_ids: &[u32]) -> Result<Tensor<'static>, anyhow::Error> {
        // Default: use sequential positions [0, 1, ..., N-1] (full prefill)
        let positions: Vec<u32> = (0..input_ids.len() as u32).collect();
        self.forward_with_positions(input_ids, &positions)
    }

    /// Like forward() but accepts explicit positions for each token.
    /// positions[i] gives the position of input_ids[i] in the sequence.
    pub fn forward_with_positions(&self, input_ids: &[u32], _positions: &[u32]) -> Result<Tensor<'static>, anyhow::Error> {
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

            for (_bi, (binding, io_def)) in func_def.inputs.iter().enumerate() {
                let shape: Vec<usize> =
                    io_def.shape.iter().map(|&d| d as usize).collect();
                let tensor: Tensor = match binding {
                    InputBinding::GlobalInput => {
                        // shape[i] == 0 is the SFCF dynamic sentinel
                        let is_dynamic = shape.iter().any(|&d| d == 0);
                        if is_dynamic {
                            let rank = io_def.rank as usize;
                            match rank {
                                1 => {
                                    let n_tokens = input_ids.len();
                                    let raw: Vec<u8> = input_ids.iter()
                                        .flat_map(|&v| (v as i64).to_ne_bytes())
                                        .collect();
                                    let p = raw.as_ptr();
                                    let memref = MemRefDesc1 {
                                        allocated: p as *mut c_void,
                                        aligned: p as *mut c_void,
                                        offset: 0,
                                        sizes: [n_tokens as i64],
                                        strides: [1],
                                    };
                                    _raw_buffers.push(raw);
                                    let desc = MemRefDescAny::R1(memref);
                                    input_descs.push(desc);
                                    input_ptrs.push(input_descs.last()
                                        .expect("input_descs has entry for GlobalInput")
                                        .as_input_ptr());
                                    continue;
                                }
                                2 => {
                                    let n_tokens = input_ids.len() as i64;
                                    let raw: Vec<u8> = input_ids.iter()
                                        .flat_map(|&v| (v as i64).to_ne_bytes())
                                        .collect();
                                    let p = raw.as_ptr();
                                    let memref = MemRefDesc2 {
                                        allocated: p as *mut c_void,
                                        aligned: p as *mut c_void,
                                        offset: 0,
                                        sizes: [1, n_tokens],
                                        strides: [n_tokens, 1],
                                    };
                                    _raw_buffers.push(raw);
                                    let desc = MemRefDescAny::R2(memref);
                                    input_descs.push(desc);
                                    input_ptrs.push(input_descs.last()
                                        .expect("input_descs has entry for GlobalInput")
                                        .as_input_ptr());
                                    continue;
                                }
                                r => anyhow::bail!(
                                    "forward_with_positions: unsupported rank {} for \
                                     dynamic GlobalInput (shape={:?})",
                                    r, shape,
                                ),
                            }
                        }
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
                        input_ptrs.push(input_descs.last()
                            .expect("input_descs has entry for GlobalInput").as_input_ptr());
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

                let desc = MemRefDescAny::from_f32(&tensor.shape, tensor.as_slice())
                    .map_err(|e| anyhow::anyhow!("weight desc: {}", e))?;
                input_descs.push(desc);
                input_ptrs.push(input_descs.last()
                    .expect("input_descs has entry for Weight/Ssa input").as_input_ptr());
                _tensors.push(tensor);
            }
            debug_assert!(_tensors.len() <= input_ptrs.len());

            const SRET_BUF_SIZE: usize = 131072;
            let mut sret: Vec<u8> = vec![0u8; SRET_BUF_SIZE];
            let sret_ptr = sret.as_mut_ptr() as *mut c_void;

            let mut all_args: Vec<*const c_void> = Vec::with_capacity(1 + input_ptrs.len());
            all_args.push(sret_ptr);
            all_args.extend(input_ptrs.iter().copied());
            // SAFETY: kernel was loaded from the compiled .dylib and validated
            // by Executable::lookup_typed().  sret_ptr and input_ptrs point to
            // writable/readable buffers of appropriate size.  The kernel is
            // _mlir_ciface_* — a C ABI function that reads MemRef descriptors
            // from input_ptrs and writes output descriptors to sret_ptr.
            unsafe {
                let raw_ptr = kernel.as_raw_ptr();
                crate::ciface_high::call_high_arity(raw_ptr, &all_args);
            }

            let mut sret_offset: usize = 0;
            for (oi, io_def) in func_def.outputs.iter().enumerate() {
                let r = io_def.rank as usize;
                let desc_size = 24 + 16 * r;
                let end = sret_offset + desc_size;
                if end > SRET_BUF_SIZE {
                    anyhow::bail!(
                        "sret overflow: func {} output {} desc_size={} offset={} exceeds {}",
                        fi, oi, desc_size, sret_offset, SRET_BUF_SIZE,
                    );
                }
                let ptr_slice = &sret[sret_offset..end];
                // SAFETY: parse_sret_descriptor reads structured binary data
                // from the sret buffer written by the MLIR ciface kernel.
                // desc_size was computed from the known rank r.  The slice
                // bounds are validated above (end <= SRET_BUF_SIZE).
                let (aligned, runtime_sizes) = match unsafe { parse_sret_descriptor(ptr_slice, r) } {
                    Ok(result) => result,
                    Err(e) => {
                        eprintln!("[executor] func_{} output_{}: {} — skipping", fi, oi, e);
                        sret_offset += desc_size;
                        continue;
                    }
                };
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

            // Dump layer outputs if DUMP_LAYERS is set
            if let Ok(dump_dir) = std::env::var("DUMP_LAYERS") {
                let _ = std::fs::create_dir_all(&dump_dir);
                for (oi, t) in func_outputs[fi].iter().enumerate() {
                    let path = format!("{}/func_{}_{}.npy", dump_dir, fi, oi);
                    let slice = t.as_slice();

                    // Skip outputs with dynamic shapes that couldn't be
                    // captured via sret (produces NaN / garbage data).
                    // DUMP_LAYERS only works reliably for functions whose
                    // output shapes are fully static (e.g. func_0).
                    if !slice.is_empty() {
                        let has_nan = slice.iter().any(|&x| x.is_nan());
                        if has_nan {
                            eprintln!(
                                "[WARN] DUMP_LAYERS: func[{}] output[{}] contains NaN — \
                                 possible uninitialized buffer or dynamic shape sret issue",
                                fi, oi,
                            );
                        }
                        let all_same = slice.iter().all(|&x| x == slice[0]);
                        if all_same {
                            eprintln!(
                                "[WARN] DUMP_LAYERS: func[{}] output[{}] has ALL IDENTICAL \
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

        let (g_func, g_idx) = self.compute_graph.global_output;
        let result = &func_outputs[g_func][g_idx];
        Ok(result.to_owned())
    }
}

unsafe fn parse_sret_descriptor(slice: &[u8], rank: usize) -> Result<(*mut u8, Vec<i64>), String> {
    let min_len = 24 + rank * 8;
    if slice.len() < min_len {
        return Err(format!(
            "slice too short: {} < {}",
            slice.len(), min_len,
        ));
    }
    // SAFETY: caller guarantees slice is long enough for the full descriptor
    // (rank-sized offset 24 + rank*8).  read_unaligned is safe for any aligned
    // or unaligned byte address on x86_64/aarch64.
    let aligned = std::ptr::read_unaligned(slice.as_ptr().add(8) as *const *mut u8);
    if aligned.is_null() {
        return Err("aligned pointer is null".to_string());
    }
    let sizes: Vec<i64> = (0..rank).map(|i| {
        let offset = 24 + i * 8;
        std::ptr::read_unaligned(slice.as_ptr().add(offset) as *const i64)
    }).collect();
    Ok((aligned, sizes))
}

// ---------------------------------------------------------------------------
// DUMP_LAYERS: write per-function output tensors as .npy files
// ---------------------------------------------------------------------------

/// Write an f32 tensor to a .npy file (NumPy format v1.0).
///
/// The format is:
///   - Magic: \x93NUMPY
///   - Version: major=1, minor=0
///   - Header length: u16 LE
///   - Header: ASCII Python dict literal, padded with spaces to 64-byte boundary
///   - Raw data: little-endian f32 values
fn write_npy(path: &str, data: &[f32], shape: &[usize]) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)?;

    // Build the header dict literal with proper NumPy 1.x shape syntax.
    //   rank 0: shape=()
    //   rank 1: shape=(N,)
    //   rank 2+: shape=(d0, d1, ...)
    let shape_str = shape
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let header = if shape.is_empty() {
        "{'descr': '<f4', 'fortran_order': False, 'shape': (), }".to_string()
    } else if shape.len() == 1 {
        format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({},), }}", shape_str)
    } else {
        format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({}), }}", shape_str)
    };

    // Pad header to a multiple of 64 bytes (from start of magic)
    let header_bytes = header.as_bytes();
    let header_len = header_bytes.len() as u16;
    let total_before_pad = 10 + header_bytes.len(); // 6 magic + 2 ver + 2 hdr_len + header
    let padding = (64 - (total_before_pad % 64)) % 64;

    // Write magic + version + header_len + header + padding
    file.write_all(b"\x93NUMPY")?;       // 6 bytes magic
    file.write_all(&[1, 0])?;            // 2 bytes version (v1.0)
    file.write_all(&header_len.to_le_bytes())?; // 2 bytes header length
    file.write_all(header_bytes)?;       // header dict
    for _ in 0..padding {
        file.write_all(b" ")?;
    }

    // Write raw f32 data (little-endian via <f4 descriptor)
    // SAFETY: f32 has no padding bits; reinterpreting &[f32] as &[u8] is valid.
    let byte_slice = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<f32>(),
        )
    };
    file.write_all(byte_slice)?;

    Ok(())
}

#[cfg(test)]
#[path = "executor_tests.rs"]
mod tests;
