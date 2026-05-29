//! CPU backend types: allocator, Device/Stream/Event trait implementations.

use std::alloc::{self, Layout};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::traits;

use super::CpuBuffer as HighCpuBuffer;

// ── RawCpuDevice (low-level allocator) ────────────────────────────────

#[derive(Debug)]
#[allow(dead_code)]
pub struct RawCpuDevice {
    allocated_bytes: usize,
}

impl Default for RawCpuDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl RawCpuDevice {
    pub fn new() -> Self {
        Self {
            allocated_bytes: 0,
        }
    }

    pub fn allocate(&mut self, size: usize) -> super::buffer::CpuBuffer {
        if size == 0 {
            return super::buffer::CpuBuffer::empty();
        }
        // SAFETY: `size > 0` per the check above. `Layout::from_size_align`
        // returns Err only on arithmetic overflow, which won't happen for
        // plausible allocations.
        let layout = Layout::from_size_align(size, 16).expect("invalid layout");
        // SAFETY: layout is valid (checked by from_size_align above) and size > 0.
        let ptr = unsafe { alloc::alloc(layout) };
        let Some(ptr) = NonNull::new(ptr) else {
            alloc::handle_alloc_error(layout);
        };
        self.allocated_bytes += size;
        super::buffer::CpuBuffer::from_raw(ptr, size, layout)
    }

    pub fn total_allocated(&self) -> usize {
        self.allocated_bytes
    }
}

// ── CpuDevice (high-level HAL Device) ──────────────────────────────────

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
        Ok(Box::new(HighCpuBuffer::new(buf)))
    }

    fn create_stream(&self) -> Result<Box<dyn traits::Stream>, anyhow::Error> {
        Ok(Box::new(CpuStream))
    }

    fn create_event(&self) -> Result<Box<dyn traits::Event>, anyhow::Error> {
        Ok(Box::new(CpuEvent))
    }

    fn compile(&self, module_data: &[u8]) -> Result<Box<dyn traits::Executable>, anyhow::Error> {
        let dylib_path = std::str::from_utf8(module_data)
            .map_err(|e| anyhow::anyhow!("module_data is not valid UTF-8 path: {}", e))?;
        let raw_exec =
            super::executable::CpuExecutable::load(dylib_path)?;
        let constants_data = raw_exec.load_constants()?;
        // Cache serveforge_free symbol. Load BEFORE moving `raw_exec` into
        // CpuExecutable to satisfy the borrow checker.
        let free_fn: unsafe extern "C" fn(*mut std::ffi::c_void) = {
            let sym: libloading::Symbol<unsafe extern "C" fn(*mut std::ffi::c_void)> =
                // SAFETY: lib.get() returns a valid symbol pointer if the dylib
                // was loaded successfully.
                unsafe { raw_exec.lib().get(b"serveforge_free")? };
            *sym
        };
        Ok(Box::new(super::CpuExecutable::new(raw_exec, constants_data, free_fn)))
    }

    fn name(&self) -> &str {
        "CPU (Apple Silicon / x86-64)"
    }
}

// ── CpuStream ──────────────────────────────────────────────────────────

#[derive(Debug)]
#[allow(dead_code)]
pub struct CpuStream;

impl traits::Stream for CpuStream {
    fn synchronize(&self) -> Result<(), anyhow::Error> { Ok(()) }
    fn wait_event(&self, _event: &dyn traits::Event) -> Result<(), anyhow::Error> { Ok(()) }
    fn record_event(&self, _event: &dyn traits::Event) -> Result<(), anyhow::Error> { Ok(()) }
}

// ── CpuEvent ───────────────────────────────────────────────────────────

/// CPU event — no-op (CPU is synchronous, all work completes immediately).
#[derive(Debug)]
#[allow(dead_code)]
pub struct CpuEvent;

impl traits::Event for CpuEvent {
    fn is_complete(&self) -> bool { true }
    fn synchronize(&self) -> Result<(), anyhow::Error> { Ok(()) }
}
