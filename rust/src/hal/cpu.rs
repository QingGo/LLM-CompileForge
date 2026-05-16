//! CPU backend implementation of HAL traits.
//!
//! Wraps the existing ``hal_cpu`` types behind the new trait interface.
//! ``CpuDevice`` allocates via ``std::alloc``, compiles via libloading.
//! ``CpuExecutable`` wraps ``hal_cpu::Executable`` + ``KernelFn`` dispatch.
//!
//! For Phase A, ``compile()`` loads a pre-compiled `.dylib` (the MLIR→LLVM
//! compilation is done by the Python pipeline).  A future phase will move
//! ``llc + cc`` into ``compile()`` for self-contained Rust compilation.

use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::traits;
use crate::hal_cpu;

// ── CpuDevice ──────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct CpuDevice {
    allocated: AtomicUsize,
}

impl CpuDevice {
    pub fn new() -> Self {
        Self {
            allocated: AtomicUsize::new(0),
        }
    }

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
        let mut d = hal_cpu::Device::new();
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
        let inner = hal_cpu::Executable::load(dylib_path)?;
        Ok(Box::new(CpuExecutable { inner }))
    }

    fn name(&self) -> &str {
        "CPU (Apple Silicon / x86-64)"
    }
}

// ── CpuBuffer ──────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct CpuBuffer(hal_cpu::Buffer);

impl traits::Buffer for CpuBuffer {
    fn as_ptr(&self) -> *const u8 {
        self.0.as_ptr()
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.0.as_mut_ptr()
    }

    fn len(&self) -> usize {
        self.0.size()
    }

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

// ── CpuExecutable ──────────────────────────────────────────────────────

#[derive(Debug)]
pub struct CpuExecutable {
    inner: hal_cpu::Executable,
}

impl CpuExecutable {
    pub fn inner(&self) -> &hal_cpu::Executable {
        &self.inner
    }

    pub fn lookup_typed(
        &self,
        name: &str,
        arity: usize,
    ) -> Result<hal_cpu::KernelFn, anyhow::Error> {
        self.inner.lookup_typed(name, arity)
    }
}

impl traits::Executable for CpuExecutable {
    fn execute(
        &self,
        _stream: &dyn traits::Stream,
        inputs: &[&dyn traits::Buffer],
        outputs: &[&dyn traits::Buffer],
    ) -> Result<(), anyhow::Error> {
        // For CPU, execute() dispatches the ciface call.
        // We need a simpler interface for multi-function dispatch.
        // For now this is a placeholder — the full multi-function
        // dispatch lives in ModelExecutor::forward().
        anyhow::bail!("direct execute() not yet implemented; use ModelExecutor::forward() instead")
    }

    fn entry_count(&self) -> usize {
        1
    }
}

// ── CpuStream ──────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct CpuStream;

impl traits::Stream for CpuStream {
    fn synchronize(&self) -> Result<(), anyhow::Error> {
        Ok(()) // CPU is synchronous — no-op
    }
}

// ── Backward-compat re-exports ────────────────────────────────────────

pub use hal_cpu::{CifaceFn1, CifaceFn2, CifaceFn3, CifaceFn4, CifaceFn5, CifaceFn6, CifaceFn7, CifaceFn8};
pub use hal_cpu::{KernelFn, MemRefDesc, MemRefDesc0, MemRefDesc1, MemRefDesc2, MemRefDesc3, MemRefDesc4, MemRefDescAny};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hal::traits::{Buffer as _, Device as _, Executable as _, Stream as _};

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
    fn test_cpu_stream_debug() {
        let s = CpuStream;
        let _ = format!("{:?}", s);
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
        // Tests that compile() attempts to load a .dylib at the given path
        let d = CpuDevice::new();
        let nonexistent = b"/nonexistent/libtest.dylib" as &[u8];
        let result = d.compile(nonexistent);
        assert!(result.is_err(), "loading nonexistent .dylib should fail");
    }

    #[test]
    fn test_trait_object_safety() {
        // Verify that Box<dyn Device/Executable/Buffer/Stream> compiles
        let d: Box<dyn traits::Device> = Box::new(CpuDevice::new());
        assert_eq!(d.name(), "CPU (Apple Silicon / x86-64)");
    }
}
