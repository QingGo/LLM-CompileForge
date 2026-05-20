//! CPU backend — HAL trait implementations and re-exports.
//!
//! ``CpuDevice`` allocates via ``std::alloc``.
//! ``CpuExecutable`` loads compiled .dylibs via ``libloading``.
//! ``CpuBuffer`` wraps heap memory with initialization tracking.
//! ``CpuStream`` is a no-op (CPU is synchronous).

use std::sync::atomic::{AtomicUsize, Ordering};

use super::traits;

pub mod buffer;
pub mod device;
pub mod executable;
pub mod kernel;
pub mod memref;

use buffer::CpuBuffer as RawCpuBuffer;
use device::CpuDevice as RawCpuDevice;
use executable::CpuExecutable as RawCpuExecutable;

// ── CpuDevice ──────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct CpuDevice {
    allocated: AtomicUsize,
}

impl CpuDevice {
    pub fn new() -> Self {
        Self { allocated: AtomicUsize::new(0) }
    }

    #[allow(dead_code)]
    pub fn total_allocated(&self) -> usize {
        self.allocated.load(Ordering::Relaxed)
    }
}

impl Default for CpuDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl traits::Device for CpuDevice {
    fn alloc(&self, size: usize) -> Result<Box<dyn traits::Buffer>, anyhow::Error> {
        let mut d = RawCpuDevice::new();
        let buf = d.allocate(size);
        self.allocated.fetch_add(d.total_allocated(), Ordering::Relaxed);
        Ok(Box::new(CpuBuffer(buf)))
    }

    fn create_stream(&self) -> Result<Box<dyn traits::Stream>, anyhow::Error> {
        Ok(Box::new(CpuStream))
    }

    fn compile(&self, module_data: &[u8]) -> Result<Box<dyn traits::Executable>, anyhow::Error> {
        let dylib_path = std::str::from_utf8(module_data)
            .map_err(|e| anyhow::anyhow!("module_data is not valid UTF-8 path: {}", e))?;
        let inner = RawCpuExecutable::load(dylib_path)?;
        Ok(Box::new(CpuExecutable { inner }))
    }

    fn name(&self) -> &str {
        "CPU (Apple Silicon / x86-64)"
    }
}

// ── CpuBuffer ─────────────────────────────────────────────────────────

#[derive(Debug)]
#[allow(dead_code)]
pub struct CpuBuffer(RawCpuBuffer);

impl traits::Buffer for CpuBuffer {
    fn as_ptr(&self) -> *const u8 { self.0.as_ptr() }
    fn as_mut_ptr(&mut self) -> *mut u8 { self.0.as_mut_ptr() }
    fn len(&self) -> usize { self.0.size() }

    fn copy_from_host(&mut self, src: &[u8], _stream: &dyn traits::Stream) -> Result<(), anyhow::Error> {
        let dst = self.0.as_mut_slice();
        let n = dst.len().min(src.len());
        dst[..n].copy_from_slice(&src[..n]);
        Ok(())
    }

    fn copy_to_host(&self, dst: &mut [u8], _stream: &dyn traits::Stream) -> Result<(), anyhow::Error> {
        let src = self.0.as_slice();
        let n = dst.len().min(src.len());
        dst[..n].copy_from_slice(&src[..n]);
        Ok(())
    }
}

// ── CpuExecutable ─────────────────────────────────────────────────────

#[derive(Debug)]
pub struct CpuExecutable {
    #[allow(dead_code)]
    inner: RawCpuExecutable,
}

impl CpuExecutable {
    #[allow(dead_code)]
    pub fn inner(&self) -> &RawCpuExecutable { &self.inner }

    #[allow(dead_code)]
    pub fn lookup_typed(&self, name: &str, arity: usize) -> Result<kernel::KernelFn, anyhow::Error> {
        self.inner.lookup_typed(name, arity)
    }
}

impl traits::Executable for CpuExecutable {
    fn execute(&self, _stream: &dyn traits::Stream, _inputs: &[&dyn traits::Buffer], _outputs: &[&dyn traits::Buffer]) -> Result<(), anyhow::Error> {
        anyhow::bail!("direct execute() not supported; use ModelExecutor::forward() instead")
    }
    fn entry_count(&self) -> usize { 1 }
}

// ── CpuStream ─────────────────────────────────────────────────────────

#[derive(Debug)]
#[allow(dead_code)]
pub struct CpuStream;

impl traits::Stream for CpuStream {
    fn synchronize(&self) -> Result<(), anyhow::Error> { Ok(()) }
}

// ── Re-exports (used by executor.rs, weight_loader.rs) ─

pub use executable::CpuExecutable as Executable;
pub use kernel::CifaceFn3;
pub use memref::{MemRefDesc1, MemRefDesc2, MemRefDescAny};

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hal::traits::{Device as _, Executable as _, Stream as _};

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
        let k3 = kernel::KernelFn::Arity3(unsafe {
            std::mem::transmute::<*const std::ffi::c_void, kernel::CifaceFn3>(std::ptr::null())
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
        let data = [3.14f32];
        let desc = MemRefDescAny::from_f32(&[], &data).unwrap();
        assert!(!desc.as_input_ptr().is_null());
        match &desc {
            MemRefDescAny::R0(d) => {
                unsafe {
                    let val = *(d.aligned as *const f32);
                    assert!((val - 3.14).abs() < 1e-6);
                }
            }
            _ => panic!("expected R0 variant"),
        }
    }
}
