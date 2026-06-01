//! CPU backend — top-level module.
//!
//! Sub-modules:
//! - ``device`` — ``CpuDevice`` (HAL Device impl), ``CpuStream``, ``CpuEvent``
//! - ``buffer`` — ``CpuBuffer`` (raw allocator-backed buffer)
//! - ``executable`` — ``CpuExecutable`` (dylib loader)
//! - ``kernel``, ``memref``, ``sret`` — MemRef descriptor helpers
//!
//! Re-exports ``CpuDevice``, ``CpuStream``, ``CpuEvent``, ``CpuExecutable``,
//! ``CpuBuffer``, and MemRef descriptor types for use by the rest of the crate.

use std::ffi::c_void;

use super::traits;

pub mod buffer;
pub mod device;
pub mod executable;
pub mod kernel;
pub mod memref;
pub mod sret;

use buffer::CpuBuffer as RawCpuBuffer;
#[allow(unused_imports)]
use device::RawCpuDevice;
use executable::CpuExecutable as RawCpuExecutable;

// ── CpuBuffer ─────────────────────────────────────────────────────────

#[derive(Debug)]
#[allow(dead_code)]
pub struct CpuBuffer {
    inner: RawCpuBuffer,
    /// Element size in bytes (4 for f32, 8 for i64).
    elem_size: usize,
    /// Logical shape for MemRef descriptor construction.
    dims: Vec<usize>,
    /// Tensor rank for output descriptor parsing.
    rank: u8,
}

impl CpuBuffer {
    pub fn new(inner: RawCpuBuffer) -> Self {
        let size = inner.size();
        Self { inner, elem_size: 4, dims: vec![size / 4], rank: 1 }
    }

    /// Create a buffer with explicit metadata for correct MemRef descriptor construction.
    /// Used for non-f32 inputs (e.g. i64 GlobalInputs with element_size=8).
    pub fn with_meta(inner: RawCpuBuffer, elem_size: usize, dims: Vec<usize>) -> Self {
        let rank = dims.len() as u8;
        Self { inner, elem_size, dims, rank }
    }

    /// Access the inner RawCpuBuffer (for HAL-internal use).
    #[allow(dead_code)]
    pub fn inner(&self) -> &RawCpuBuffer { &self.inner }
}

impl traits::Buffer for CpuBuffer {
    fn as_ptr(&self) -> *const u8 { self.inner.as_ptr() }
    fn as_mut_ptr(&mut self) -> *mut u8 { self.inner.as_mut_ptr() }
    fn len(&self) -> usize { self.inner.size() }

    fn copy_from_host(&mut self, src: &[u8], _stream: &dyn traits::Stream) -> Result<(), anyhow::Error> {
        let dst = self.inner.as_mut_slice();
        let n = dst.len().min(src.len());
        dst[..n].copy_from_slice(&src[..n]);
        Ok(())
    }

    fn copy_to_host(&self, dst: &mut [u8], _stream: &dyn traits::Stream) -> Result<(), anyhow::Error> {
        let src = self.inner.as_slice();
        let n = dst.len().min(src.len());
        dst[..n].copy_from_slice(&src[..n]);
        Ok(())
    }

    fn element_size(&self) -> usize {
        self.elem_size
    }

    fn shape(&self) -> Vec<usize> {
        self.dims.clone()
    }

    fn rank(&self) -> u8 {
        self.rank
    }
}

// ── CpuExecutable ─────────────────────────────────────────────────────

#[derive(Debug)]
pub struct CpuExecutable {
    inner: RawCpuExecutable,
    constants_data: Vec<u8>,
    /// Cached function pointer for `serveforge_free` exported by the dylib.
    /// Eliminates per-call libloading symbol lookup and memory leak.
    free_fn: unsafe extern "C" fn(*mut c_void),
}

impl CpuExecutable {
    #[allow(dead_code)]
    pub fn inner(&self) -> &RawCpuExecutable { &self.inner }

    /// Access the cached constants data embedded in the dylib.
    #[allow(dead_code)]
    pub fn module_data(&self) -> &[u8] {
        &self.constants_data
    }

    /// Construct a CpuExecutable from its raw parts (used by device.rs).
    #[allow(dead_code)]
    pub(crate) fn new(
        inner: RawCpuExecutable,
        constants_data: Vec<u8>,
        free_fn: unsafe extern "C" fn(*mut c_void),
    ) -> Self {
        Self { inner, constants_data, free_fn }
    }
}

impl traits::Executable for CpuExecutable {
    fn execute(
        &self,
        op_name: &str,
        _stream: &dyn traits::Stream,
        inputs: &[&dyn traits::Buffer],
        outputs: &[&dyn traits::Buffer],
    ) -> Result<Vec<Vec<i64>>, anyhow::Error> {
        // Step 1: Look up kernel by symbol name (arity = 1 sret + inputs)
        let arity = 1 + inputs.len();
        let kernel = self.inner.lookup_typed(op_name, arity)?;

        // Step 2: Construct MemRef descriptors from input Buffers (zero-copy)
        // IMPORTANT: Push to input_descs BEFORE taking the pointer, so the
        // descriptor lives on the heap (in the Vec) rather than on the stack.
        let mut input_descs: Vec<MemRefDescAny> = Vec::with_capacity(inputs.len());
        let mut input_ptrs: Vec<*const c_void> = Vec::with_capacity(inputs.len());
        // Collect input data pointers to detect pass-through tensors in sret.
        // When the dylib returns a weight or SSA wire as a pass-through output,
        // the sret's `allocated` pointer matches an input data pointer.
        // We must NOT free these — Rust owns them.
        let input_data_ptrs: Vec<*const u8> = inputs.iter()
            .map(|buf| buf.as_ptr())
            .collect();
        for &buf in inputs {
            let desc = make_memref_descriptor(buf)?;
            input_descs.push(desc);
            input_ptrs.push(input_descs.last().unwrap().as_input_ptr());
        }

        // Step 3: Allocate sret buffer for output descriptors
        const SRET_BUF_SIZE: usize = 131072;
        let mut sret: Vec<u8> = vec![0u8; SRET_BUF_SIZE];
        let sret_ptr = sret.as_mut_ptr() as *mut c_void;

        // Step 4: Build argument list and call the ciface kernel
        let mut all_args: Vec<*const c_void> = Vec::with_capacity(1 + input_ptrs.len());
        all_args.push(sret_ptr);
        all_args.extend(input_ptrs);
        let raw_ptr = kernel.as_raw_ptr();
        // SAFETY: kernel was loaded from a valid compiled .dylib.
        // sret_ptr and input_ptrs point to writable/readable buffers.
        // The kernel is a _mlir_ciface_* function that reads input descriptors
        // and writes output descriptors to the sret buffer.
        // SAFETY: kernel, sret_ptr, and input_ptrs are all verified above.
        unsafe {
            crate::ciface_high::call_high_arity(raw_ptr, &all_args);
        }
        // Step 5: Parse sret output descriptors, copy data to output
        // buffers, collect shapes and dylib-allocated pointers to free.
        //
        // Two-pass strategy prevents use-after-free and double-free:
        //   Pass 1: parse all descriptors, copy data from dylib buffers
        //           to pre-allocated Rust output buffers.
        //   Pass 2: deduplicate allocated pointers and free them via
        //           the dylib's own serveforge_free.
        let mut output_shapes: Vec<Vec<i64>> = Vec::with_capacity(outputs.len());
        if !outputs.is_empty() {
            let mut sret_offset: usize = 0;
            let mut to_free: Vec<*mut std::ffi::c_void> =
                Vec::with_capacity(outputs.len());

            // ── Pass 1: parse + copy ──────────────────────────────
            for (oi, output_buf) in outputs.iter().enumerate() {
                if sret_offset >= SRET_BUF_SIZE - 24 {
                    anyhow::bail!(
                        "sret overflow at output {} (offset {} >= {})",
                        oi, sret_offset, SRET_BUF_SIZE,
                    );
                }

                let out_rank = output_buf.rank() as usize;
                if !(1..=4).contains(&out_rank) {
                    anyhow::bail!(
                        "output {}: unsupported rank {} for sret parsing",
                        oi, out_rank,
                    );
                }
                let desc_size = 24 + 16 * out_rank;
                if sret_offset + desc_size > sret.len() {
                    anyhow::bail!(
                        "sret overflow at output {} (offset {} + {} > {})",
                        oi, sret_offset, desc_size, sret.len(),
                    );
                }
                let slice = &sret[sret_offset..sret_offset + desc_size];
                // SAFETY: `read_sret_descriptor` validates the slice length
                // and non-null pointers internally; we already verified the
                // slice is within bounds above.
                let (allocated, aligned, sizes) = unsafe {
                    read_sret_descriptor(slice, out_rank)?
                };
                sret_offset += desc_size;

                let n: usize = crate::hal::cpu::sret::checked_product_from_i64(&sizes)
                    .ok_or_else(|| anyhow::anyhow!(
                        "output {} sret sizes overflow: {:?}", oi, sizes
                    ))?;
                // When the dylib returns unresolved dynamic dimension
                // markers (negative values like -2, -3 in the sizes
                // array), checked_product clamps to 0 → n_bytes=0 →
                // no data would be copied.  Fall back to the pre-allocated
                // output buffer capacity: the dylib wrote actual data
                // there regardless of the metadata glitch.
                let has_negative = sizes.iter().any(|&s| s < 0);
                let n_bytes = if has_negative && n == 0 {
                    let fallback = output_buf.len();
                    log::trace!(
                        "execute: output[{}] sret has negative sizes {:?}, \
                         falling back to buf len={}",
                        oi, sizes, fallback,
                    );
                    fallback
                } else {
                    n * 4
                };

                log::trace!(
                    "execute: output[{}] sret rank={} sizes={:?} n={} n_bytes={} buf_cap={} \
                      allocated={:p} aligned={:p}",
                    oi, out_rank, sizes, n, n_bytes, output_buf.len(),
                    allocated, aligned,
                );

                if n_bytes == 0 {
                    output_shapes.push(sizes);
                    continue;
                }

                if n_bytes > output_buf.len() {
                    anyhow::bail!(
                        "output {}: dylib output {} bytes exceeds buffer capacity {} bytes",
                        oi, n_bytes, output_buf.len(),
                    );
                }

                debug_assert!(
                    n_bytes <= output_buf.len(),
                    "output {}: panic on n_bytes={} > buf.len()={}",
                    oi, n_bytes, output_buf.len(),
                );

                let dst = output_buf.as_ptr() as *mut u8;
                // SAFETY: `aligned` points to valid dylib output data.
                // `dst` is a pre-allocated Rust buffer. Regions are disjoint.
                unsafe {
                    std::ptr::copy_nonoverlapping(aligned, dst, n_bytes);
                }

                // Track for deferred free (Pass 2). Only enqueue heap
                // pointers; skip nullptr and MLIR sentinel 0xdeadbeef
                // (global constants from memref.get_global).
                let allocated_addr = allocated as usize;
                if allocated_addr != 0 && allocated_addr != 0xdeadbeef {
                    to_free.push(allocated as *mut std::ffi::c_void);
                }

                output_shapes.push(sizes);
            }

            // ── Pass 2: dedup + free ──────────────────────────────
            //
            // Only free pointers that are verifiably heap-allocated
            // AND not owned by Rust.
            //
            // Some MemRef descriptors in the sret have `allocated`
            // pointing to non-malloc memory (stack, globals, or
            // garbage from misaligned sret parsing). Use the system
            // allocator's own validation (malloc_zone_from_ptr) to
            // distinguish safe-to-free pointers from the rest.
            //
            // CRITICAL: The dylib's ciface calls pass through weight
            // and SSA wire tensors as outputs.  Their MemRef descriptors
            // have `allocated` pointing to Rust-owned memory (weight
            // cache, func_outputs Vecs).  Freeing these causes a
            // double-free crash (SIGABRT in Arc::drop_slow) because
            // Rust's allocator detects the double-free.
            // Use input_data_ptrs to skip these pass-through pointers.
            to_free.sort();
            to_free.dedup();
            for ptr in &to_free {
                let addr = *ptr as *const std::ffi::c_void;
                if addr.is_null() {
                    continue;
                }
                // Skip if this pointer matches an input buffer — it's
                // a pass-through weight or SSA wire owned by Rust.
                let addr_u8 = addr as *const u8;
                if input_data_ptrs.contains(&addr_u8) {
                    log::trace!(
                        "execute: skip free of pass-through input ptr {:p}",
                        addr,
                    );
                    continue;
                }
                // SAFETY: malloc_zone_from_ptr is thread-safe and
                // read-only. Returns null if addr is not in any
                // malloc zone (stack, global, or invalid pointer).
                #[link(name = "System")]
                extern "C" {
                    fn malloc_zone_from_ptr(ptr: *const std::ffi::c_void) -> *const std::ffi::c_void;
                }
                // SAFETY: `malloc_zone_from_ptr` is an Apple system call that
                // is thread-safe and accepts any pointer (returns null for
                // non-zone allocations).
                let zone = unsafe { malloc_zone_from_ptr(addr) };
                if !zone.is_null() {
                    // SAFETY: `self.free_fn` is the dylib's `serveforge_free`
                    // function pointer, valid for the lifetime of the executable.
                    // `*ptr` was allocated by the dylib and is safe to free.
                    unsafe { (self.free_fn)(*ptr); }
                }
            }
        }
        Ok(output_shapes)
    }

    fn function_count(&self) -> usize { 1 }

    fn module_data(&self) -> &[u8] {
        &self.constants_data
    }
}
// ── Re-exports (used by executor.rs, weight_loader.rs, device.rs) ─

#[allow(unused_imports)]
pub use device::{CpuDevice, CpuEvent, CpuStream};
#[allow(unused_imports)]
pub use executable::CpuExecutable as Executable;
pub use memref::{MemRefDesc2, MemRefDescAny};
pub use sret::{make_memref_descriptor, read_sret_descriptor};

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hal::traits::{Device as _, Stream as _};

    #[test]
    fn test_cpu_device_name() {
        let d = CpuDevice::new();
        assert!(!d.name().is_empty());
    }

    #[test]
    fn test_cpu_device_alloc_free() {
        let d = CpuDevice::new();
        let buf = d.alloc(64).expect("alloc 64 bytes");
        assert!(!buf.as_ptr().is_null());
        assert_eq!(buf.len(), 64);
    }

    #[test]
    fn test_cpu_stream_sync_noop() {
        let s = CpuStream;
        s.synchronize().expect("sync should be no-op");
    }

    #[test]
    fn test_cpu_buffer_copy_roundtrip() {
        let d = CpuDevice::new();
        let mut buf = d.alloc(8).expect("alloc");
        let stream = d.create_stream().expect("stream");
        let src = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        buf.copy_from_host(&src, &*stream).expect("copy_from_host");
        let mut dst = vec![0u8; 8];
        buf.copy_to_host(&mut dst, &*stream).expect("copy_to_host");
        assert_eq!(src, dst);
    }

    #[test]
    fn test_cpu_device_compile_loads_dylib() {
        let d = CpuDevice::new();
        let nonexistent = b"/nonexistent/libtest.dylib" as &[u8];
        let result = d.compile(nonexistent);
        assert!(result.is_err(), "loading nonexistent .dylib should fail");
    }

    #[test]
    fn test_trait_object_safety() {
        let d: Box<dyn traits::Device> = Box::new(CpuDevice::new());
        assert_eq!(d.name(), "CPU (Apple Silicon / x86-64)");
    }

    // ── MemRef tests ──────────────────────────────────────────────
    use memref::*;

    #[test]
    fn test_memref_desc2_layout() {
        assert_eq!(std::mem::size_of::<MemRefDesc2>(), 56);
        assert_eq!(std::mem::size_of::<MemRefDesc3>(), 72);
        assert_eq!(std::mem::size_of::<MemRefDesc1>(), 40);
    }

    #[test]
    fn test_memref_desc2_from_slice() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let desc = MemRefDesc2::from_f32_slice(&data, [2, 3]);
        assert_eq!(desc.sizes, [2, 3]);
        assert_eq!(desc.strides, [3, 1]);
        assert_eq!(desc.numel(), 6);
    }

    #[test]
    fn test_memref_desc2_zeroed() {
        let desc = MemRefDesc2::zeroed([4, 4]);
        assert_eq!(desc.sizes, [4, 4]);
        assert_eq!(desc.strides, [4, 1]);
        assert!(!desc.is_null());
    }

    #[test]
    fn test_memref_desc1() {
        let data = [10.0f32, 20.0, 30.0];
        let desc = MemRefDesc1::from_f32_slice(&data, [3]);
        assert_eq!(desc.sizes, [3]);
        assert_eq!(desc.strides, [1]);
        assert_eq!(desc.numel(), 3);
    }

    #[test]
    fn test_memref_desc3() {
        let data = vec![1.0f32; 24];
        let desc = MemRefDesc3::from_f32_slice(&data, [2, 3, 4]);
        assert_eq!(desc.sizes, [2, 3, 4]);
        assert_eq!(desc.strides, [12, 4, 1]);
        assert_eq!(desc.numel(), 24);
    }

    #[test]
    fn test_kernel_fn_arity() {
        // SAFETY: k3 is only used for arity() test (never called), so a
        // non-null placeholder is sufficient. `1usize` cast avoids
        // transmute_null_to_fn UB.
        let k3 = kernel::KernelFn::Arity3(unsafe {
            std::mem::transmute::<usize, kernel::CifaceFn3>(1usize)
        });
        assert_eq!(k3.arity(), 3);
    }

    #[test]
    fn test_buffer_as_mut_slice() {
        let mut d = RawCpuDevice::new();
        let mut buf = d.allocate(16);
        assert_eq!(buf.size(), 16);
        buf.as_mut_slice().fill(0u8);
    }

    #[test]
    fn test_memref_desc_any() {
        let data = vec![1.0f32; 6];
        let desc = MemRefDescAny::from_f32(&[2, 3], &data).unwrap();
        assert_eq!(desc.sizes(), vec![2, 3]);

        let desc2 = MemRefDescAny::zeroed(&[2, 3]).unwrap();
        assert!(!desc2.as_input_ptr().is_null());
    }

    #[test]
    fn test_memref_any_zeroed_with_0_dims() {
        let desc = MemRefDescAny::zeroed(&[0, 0, 50272]).unwrap();
        assert!(!desc.as_output_ptr().is_null());
        let sz = desc.sizes();
        assert_eq!(sz, vec![1, 1, 50272]);
    }

    #[test]
    fn test_memref_desc0_from_f32() {
        let data = [42.0f32];
        use memref::MemRefDesc0;
        let desc = MemRefDesc0::from_f32_dyn_slice(&data, &[]);
        assert_eq!(desc.sizes, [0i64; 0]);
        assert_eq!(desc.strides, [0i64; 0]);
        assert_eq!(desc.numel(), 1);
        // SAFETY: `desc.aligned` points to the scalar data of this rank-0
        // descriptor, initialized by `from_f32_dyn_slice`.
        unsafe {
            let val = *(desc.aligned as *const f32);
            assert!((val - 42.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_memref_desc0_zeroed() {
        use memref::MemRefDesc0;
        let desc = MemRefDesc0::zeroed_dyn(&[]);
        assert!(!desc.aligned.is_null());
        assert_eq!(desc.numel(), 1);
    }

    #[test]
    fn test_memref_any_zeroed_rank0() {
        let desc = MemRefDescAny::zeroed(&[]).unwrap();
        assert!(!desc.as_output_ptr().is_null());
        assert_eq!(desc.sizes(), Vec::<usize>::new());
    }

    #[test]
    fn test_memref_any_from_f32_rank0() {
        let data = [std::f32::consts::PI];
        let desc = MemRefDescAny::from_f32(&[], &data).unwrap();
        assert!(!desc.as_input_ptr().is_null());
        match &desc {
            MemRefDescAny::R0(d) => {
                // SAFETY: `d.aligned` points to valid f32 data initialized
                // by `from_f32` above.
                unsafe {
                    let val = *(d.aligned as *const f32);
                    assert!((val - std::f32::consts::PI).abs() < 1e-6);
                }
            }
            _ => panic!("expected R0 variant"),
        }
    }
}
