//! CPU Buffer — heap-allocated memory with initialization tracking.
//!
//! Supports three ownership modes:
//! - `borrowed: true`  — caller retains ownership; Drop is no-op
//! - `borrowed: false` — owned via `std::alloc`; Drop calls `alloc::dealloc`
//! - `size == 0`       — empty buffer; Drop is no-op

use std::alloc::{self, Layout};
use std::ffi::c_void;
use std::ptr::NonNull;

use crate::tensor::{Dtype, Tensor};

#[derive(Debug)]
#[allow(dead_code)]
pub struct CpuBuffer {
    ptr: NonNull<u8>,
    size: usize,
    layout: Layout,
    initialized: bool,
    borrowed: bool,
}

#[allow(dead_code)]
impl CpuBuffer {
    pub fn empty() -> Self {
        Self {
            ptr: NonNull::dangling(),
            size: 0,
            layout: Layout::new::<u8>(),
            initialized: true,
            borrowed: false,
        }
    }

    pub fn from_raw(ptr: NonNull<u8>, size: usize, layout: Layout) -> Self {
        Self {
            ptr,
            size,
            layout,
            initialized: false,
            borrowed: false,
        }
    }

    /// Create a CpuBuffer from a raw pointer and byte length.
    ///
    /// When `borrowed` is `true`, the caller retains ownership — Drop does
    /// nothing. When `borrowed` is `false`, Drop frees via `alloc::dealloc`.
    pub fn from_raw_parts(ptr: *mut u8, len: usize, borrowed: bool) -> Result<Self, String> {
        let ptr = NonNull::new(ptr).ok_or_else(|| "null pointer".to_string())?;
        Ok(Self {
            ptr,
            size: len,
            layout: Layout::new::<u8>(),
            initialized: true,
            borrowed,
        })
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

    /// Reinterpret the buffer as a slice of `f32`.
    ///
    /// # Safety
    ///
    /// The caller must ensure the buffer contains at least `n * 4` bytes
    /// of initialized f32 data. Panics in debug if `n * 4 > self.size`.
    pub fn view_as_f32(&self, n: usize) -> &[f32] {
        debug_assert!(
            self.initialized,
            "CpuBuffer::view_as_f32 called on uninitialized memory"
        );
        debug_assert!(
            n * 4 <= self.size,
            "CpuBuffer::view_as_f32: {} elements needs {} bytes, buffer has {}",
            n,
            n * 4,
            self.size
        );
        if n == 0 || self.size == 0 {
            &[]
        } else {
            // SAFETY: `self.ptr` points to `self.size` bytes of owned
            // memory. The caller guarantees `n * 4 <= self.size` and
            // the memory is initialized with valid f32 values.
            unsafe { std::slice::from_raw_parts(self.ptr.as_ptr() as *const f32, n) }
        }
    }

    /// Consume the buffer and transfer its data into a new Tensor.
    ///
    /// The data is copied (no zero-copy) because Tensor currently only
    /// supports `Vec<f32>` ownership. After calling this, the original
    /// heap allocation is freed via the appropriate deallocator.
    pub fn into_tensor(self, shape: Vec<usize>) -> Tensor {
        let n: usize = shape.iter().product();
        let data = if n == 0 {
            Vec::new()
        } else {
            // SAFETY: `view_as_f32` validates that the buffer has enough
            // bytes for `n` elements.
            self.view_as_f32(n).to_vec()
        };
        // `self` is dropped here, freeing the original allocation.
        Tensor::new_owned(shape, data, Dtype::F32)
    }
}

impl Drop for CpuBuffer {
    fn drop(&mut self) {
        if self.size > 0 && !self.borrowed {
            // SAFETY: `self.ptr` was allocated with `self.layout`
            // in `CpuDevice::allocate`. No other references exist.
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
