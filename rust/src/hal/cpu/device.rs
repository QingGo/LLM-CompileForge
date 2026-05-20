//! CPU Device — tracks allocated bytes.

use std::alloc::{self, Layout};
use std::ptr::NonNull;

use super::buffer::CpuBuffer;

#[derive(Debug)]
#[allow(dead_code)]
pub struct CpuDevice {
    allocated_bytes: usize,
}

#[allow(dead_code)]
impl CpuDevice {
    pub fn new() -> Self {
        Self {
            allocated_bytes: 0,
        }
    }

    pub fn allocate(&mut self, size: usize) -> CpuBuffer {
        if size == 0 {
            return CpuBuffer::empty();
        }
        // SAFETY: `size > 0` per the check above. `Layout::from_size_align`
        // returns Err only on arithmetic overflow, which won't happen for
        // plausible allocations.
        let layout = Layout::from_size_align(size, 16).expect("invalid layout");
        let ptr = unsafe { alloc::alloc(layout) };
        let Some(ptr) = NonNull::new(ptr) else {
            alloc::handle_alloc_error(layout);
        };
        self.allocated_bytes += size;
        CpuBuffer::from_raw(ptr, size, layout)
    }

    pub fn total_allocated(&self) -> usize {
        self.allocated_bytes
    }
}
