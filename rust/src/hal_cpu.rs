//! CPU HAL (Hardware Abstraction Layer).
//!
//! Provides the runtime primitives for executing compiled model functions
//! on CPU: memory allocation, buffer management, and dynamic library loading.
//!
//! Each compiled MLIR function becomes an ``extern "C"`` symbol in the
//! .dylib.  The ``Executable`` wraps ``libloading::Library`` and provides
//! typed function lookup.

use std::alloc::{self, Layout};
use std::ffi::c_void;
use std::ptr::NonNull;

// ── Device ────────────────────────────────────────────────────────

pub struct Device {
    /// Total bytes allocated (tracking for diagnostics).
    allocated_bytes: usize,
}

impl Device {
    pub fn new() -> Self {
        Self { allocated_bytes: 0 }
    }

    pub fn allocate(&mut self, size: usize) -> Buffer {
        if size == 0 {
            return Buffer::empty();
        }
        let layout = Layout::from_size_align(size, 16).expect("invalid layout");
        // Safety: layout has non-zero size, alloc succeeds or aborts.
        let ptr = unsafe { alloc::alloc(layout) };
        let Some(ptr) = NonNull::new(ptr) else {
            alloc::handle_alloc_error(layout);
        };
        self.allocated_bytes += size;
        Buffer {
            ptr,
            size,
            layout,
        }
    }

    pub fn total_allocated(&self) -> usize {
        self.allocated_bytes
    }
}

// ── Buffer ─────────────────────────────────────────────────────────

/// A heap-allocated CPU buffer.  Freed automatically on drop.
pub struct Buffer {
    ptr: NonNull<u8>,
    size: usize,
    layout: Layout,
}

impl Buffer {
    pub fn empty() -> Self {
        Self {
            ptr: NonNull::dangling(),
            size: 0,
            layout: Layout::new::<u8>(),
        }
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr().cast_const()
    }

    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    pub fn as_slice(&self) -> &[u8] {
        if self.size == 0 {
            &[]
        } else {
            // Safety: ptr is valid for self.size bytes.
            unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.size) }
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        if self.size == 0 {
            &mut []
        } else {
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

impl Drop for Buffer {
    fn drop(&mut self) {
        if self.size > 0 {
            // Safety: ptr was allocated with this layout.
            unsafe { alloc::dealloc(self.ptr.as_ptr(), self.layout) };
        }
    }
}

// Safety: Buffer owns its memory; Send is safe because we don't share
// mutable references across threads.
unsafe impl Send for Buffer {}

// ── Executable ─────────────────────────────────────────────────────

/// A loaded compiled model (.dylib) ready for function invocation.
///
/// Each ``_mlir_ciface_<func_name>`` symbol in the dylib is a C
/// wrapper function with signature ``void(void**)``.  The ``void**``
/// array contains pointers to memref descriptor structs (inputs first,
/// output struct as first element).
pub struct Executable {
    #[allow(dead_code)]
    lib: libloading::Library,
}

/// Type alias for the compiled function signature (void(void**)).
pub type KernelFn = unsafe extern "C" fn(*mut *mut c_void);

impl Executable {
    /// Load a compiled .dylib.
    pub fn load(path: &str) -> Result<Self, anyhow::Error> {
        // Safety: libloading opens the library; we trust the path.
        let lib = unsafe { libloading::Library::new(path)? };
        Ok(Self { lib })
    }

    /// Look up a kernel function by name.
    ///
    /// The symbol name is the ``_mlir_ciface_<name>`` C wrapper generated
    /// by the ``llvm.emit_c_interface`` attribute.
    pub fn lookup(&self, name: &str) -> Result<KernelFn, anyhow::Error> {
        // Safety: we cast the symbol to the known function signature.
        let sym = unsafe {
            let ptr = self.lib.get::<*mut c_void>(name.as_bytes())?;
            std::mem::transmute::<*mut c_void, KernelFn>(*ptr)
        };
        Ok(sym)
    }

    /// Convenience: invoke a kernel with an array of argument pointers.
    ///
    /// # Safety
    /// The caller must ensure ``args`` contains valid pointers for the
    /// expected function signature (result struct ptr + input desc ptrs).
    pub unsafe fn invoke(&self, name: &str, args: &mut [*mut c_void]) -> Result<(), anyhow::Error> {
        let func = self.lookup(name)?;
        func(args.as_mut_ptr());
        Ok(())
    }
}

// ── Memref descriptor (ranked, 2D) ────────────────────────────────
///
/// Binary layout matches the MLIR LLVM dialect's memref descriptor:
///   struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
///
/// This is the exact format expected by ``_mlir_ciface_*`` wrappers in
/// the compiled .dylib.

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct MemRefDescriptor {
    pub allocated: i64,
    pub aligned: *mut c_void,
    pub offset: i64,
    pub sizes: [i64; 2],
    pub strides: [i64; 2],
}

impl MemRefDescriptor {
    /// Build a descriptor from a ``&[f32]`` slice with given logical shape.
    pub fn from_f32_slice(data: &[f32], rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols);
        let p = data.as_ptr();
        Self {
            allocated: p as i64,
            aligned: p as *mut c_void,
            offset: 0,
            sizes: [rows as i64, cols as i64],
            strides: [cols as i64, 1],
        }
    }

    /// Build a zero-initialized descriptor (for output buffers where the
    /// compiled function writes the result via malloc).
    pub fn zeroed(rows: usize, cols: usize) -> Self {
        Self {
            allocated: 0,
            aligned: std::ptr::null_mut::<c_void>(),
            offset: 0,
            sizes: [rows as i64, cols as i64],
            strides: [cols as i64, 1],
        }
    }

    /// Read f32 output data from a descriptor that was written by the
    /// compiled function.  The descriptor's ``aligned`` and ``sizes``
    /// fields are overwritten by the function.
    ///
    /// # Safety
    /// The descriptor must have been populated by a compiled function
    /// that allocated the output buffer via malloc.
    pub unsafe fn read_output_f32(&self) -> Vec<f32> {
        let rows = self.sizes[0] as usize;
        let cols = self.sizes[1] as usize;
        let n = rows * cols;
        if n == 0 || self.aligned.is_null() {
            return Vec::new();
        }
        let slice = std::slice::from_raw_parts(self.aligned as *const f32, n);
        slice.to_vec()
    }
}

// ── Typed ciface function wrappers ────────────────────────────────

/// ``_mlir_ciface_*`` function with 3 pointer args: result + 2 inputs.
pub type CifaceFn3 = unsafe extern "C" fn(
    *mut MemRefDescriptor,
    *const MemRefDescriptor,
    *const MemRefDescriptor,
);

/// ``_mlir_ciface_*`` function with 4 pointer args: result + 3 inputs.
pub type CifaceFn4 = unsafe extern "C" fn(
    *mut MemRefDescriptor,
    *const MemRefDescriptor,
    *const MemRefDescriptor,
    *const MemRefDescriptor,
);

// Safety: MemRefDescriptor contains raw pointers but is only used
// within a synchronous call scope.
unsafe impl Send for MemRefDescriptor {}
