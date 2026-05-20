//! CPU Buffer — heap-allocated memory with initialization tracking.

use std::alloc::{self, Layout};
use std::ptr::NonNull;

#[derive(Debug)]
#[allow(dead_code)]
pub struct CpuBuffer {
    ptr: NonNull<u8>,
    size: usize,
    layout: Layout,
    initialized: bool,
}

#[allow(dead_code)]
impl CpuBuffer {
    pub fn empty() -> Self {
        Self {
            ptr: NonNull::dangling(),
            size: 0,
            layout: Layout::new::<u8>(),
            initialized: true,
        }
    }

    pub fn from_raw(ptr: NonNull<u8>, size: usize, layout: Layout) -> Self {
        Self {
            ptr,
            size,
            layout,
            initialized: false,
        }
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr().cast_const()
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    pub fn as_slice(&self) -> &[u8] {
        debug_assert!(
            self.initialized,
            "CpuBuffer::as_slice called on uninitialized memory"
        );
        if self.size == 0 {
            &[]
        } else {
            // SAFETY: `self.ptr` was allocated via `alloc::alloc` with
            // `self.layout`. The `initialized` flag guarantees contents
            // were written before reading.
            unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.size) }
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.initialized = true;
        if self.size == 0 {
            &mut []
        } else {
            // SAFETY: `self.ptr` is uniquely owned. `&mut self` ensures
            // exclusive access.
            unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size) }
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
}

impl Drop for CpuBuffer {
    fn drop(&mut self) {
        if self.size > 0 {
            // SAFETY: `self.ptr` was allocated with `self.layout` in
            // `CpuDevice::allocate`. No other references exist.
            unsafe { alloc::dealloc(self.ptr.as_ptr(), self.layout) };
        }
    }
}

// SAFETY: CpuBuffer owns a unique heap allocation with no interior
// mutability. Sending it transfers exclusive ownership.
unsafe impl Send for CpuBuffer {}

// SAFETY: CpuBuffer is read-only after initialization. Synchronization
// is the caller's responsibility.
unsafe impl Sync for CpuBuffer {}
