//! CPU HAL (Hardware Abstraction Layer).
//!
//! Provides the runtime primitives for executing compiled model functions
//! on CPU: memory allocation, buffer management, dynamic library loading,
//! and MLIR memref descriptor construction.
//!
//! Each lowered MLIR function becomes an ``_mlir_ciface_*`` symbol in the
//! .dylib.  The ``Executable`` wraps ``libloading::Library`` and provides
//! typed function lookup via ``KernelFn`` enum dispatch.

use std::alloc::{self, Layout};
use std::ffi::c_void;
use std::ptr::NonNull;

// ── Device ────────────────────────────────────────────────────────

pub struct Device {
    allocated_bytes: usize,
}

impl Device {
    pub fn new() -> Self {
        Self {
            allocated_bytes: 0,
        }
    }

    pub fn allocate(&mut self, size: usize) -> Buffer {
        if size == 0 {
            return Buffer::empty();
        }
        let layout = Layout::from_size_align(size, 16).expect("invalid layout");
        // SAFETY: `layout` has non-zero size. `alloc::alloc` returns
        // a pointer to uninitialized memory of `layout.size()` bytes,
        // or null on failure. `handle_alloc_error` aborts on null.
        let ptr = unsafe { alloc::alloc(layout) };
        let Some(ptr) = NonNull::new(ptr) else {
            alloc::handle_alloc_error(layout);
        };
        self.allocated_bytes += size;
        Buffer {
            ptr,
            size,
            layout,
            initialized: false,
        }
    }

    pub fn total_allocated(&self) -> usize {
        self.allocated_bytes
    }
}

// ── Buffer ─────────────────────────────────────────────────────────

pub struct Buffer {
    ptr: NonNull<u8>,
    size: usize,
    layout: Layout,
    initialized: bool,
}

impl Buffer {
    pub fn empty() -> Self {
        Self {
            ptr: NonNull::dangling(),
            size: 0,
            layout: Layout::new::<u8>(),
            initialized: true,
        }
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr().cast_const()
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Return the buffer contents as a byte slice.
    ///
    /// # Panics
    /// In debug builds, panics if the buffer has not been initialized
    /// (e.g., freshly allocated without any writes).
    pub fn as_slice(&self) -> &[u8] {
        debug_assert!(
            self.initialized,
            "Buffer::as_slice called on uninitialized memory"
        );
        if self.size == 0 {
            &[]
        } else {
            // SAFETY: `self.ptr` was allocated via `alloc::alloc` with
            // `self.layout`. It points to `self.size` bytes of initialized
            // memory (guarded by the `initialized` flag above).
            unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.size) }
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.initialized = true;
        if self.size == 0 {
            &mut []
        } else {
            // SAFETY: `self.ptr` was allocated via `alloc::alloc` with
            // `self.layout`. It points to `self.size` bytes of uniquely
            // owned mutable memory. `&mut self` ensures exclusive access.
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
            // SAFETY: `self.ptr` was allocated with `self.layout` in
            // `Device::allocate`. The allocation has not been freed yet,
            // and `&mut self` guarantees no other references exist.
            unsafe { alloc::dealloc(self.ptr.as_ptr(), self.layout) };
        }
    }
}

// SAFETY: Buffer owns a unique heap allocation with no interior
// mutability. Sending a Buffer to another thread transfers exclusive
// ownership of the allocation.
unsafe impl Send for Buffer {}

// ── Executable ─────────────────────────────────────────────────────

pub struct Executable {
    lib: libloading::Library,
}

impl Executable {
    /// Load a compiled .dylib.
    ///
    /// # Safety
    /// The caller must ensure `path` refers to a valid dynamic library.
    /// `libloading::Library::new` is unsafe because it may execute
    /// initializers in the loaded library.
    pub fn load(path: &str) -> Result<Self, anyhow::Error> {
        // SAFETY: The caller guarantees the path points to a trusted
        // compiled model .dylib. We control the compilation pipeline,
        // so the dylib has no malicious initializers.
        let lib = unsafe { libloading::Library::new(path)? };
        Ok(Self { lib })
    }

    pub fn lib(&self) -> &libloading::Library {
        &self.lib
    }

        /// Look up a kernel function by symbol name with a specific arity.
    pub fn lookup_typed(
        &self,
        name: &str,
        arity: usize,
    ) -> Result<KernelFn, anyhow::Error> {
        if arity > 300 {
            anyhow::bail!("unsupported kernel arity: {} (max 300)", arity);
        }
        // SAFETY: libloading::Symbol casts the symbol to the expected type.
        // For arities 1..8 we use typed CifaceFnN for direct calls.
        // For arities 9..300 we use a raw fn pointer + C trampoline.
        match arity {
            1 => { let sym: libloading::Symbol<CifaceFn1> = unsafe { self.lib.get(name.as_bytes()) }?; Ok(KernelFn::Arity1(*sym)) }
            2 => { let sym: libloading::Symbol<CifaceFn2> = unsafe { self.lib.get(name.as_bytes()) }?; Ok(KernelFn::Arity2(*sym)) }
            3 => { let sym: libloading::Symbol<CifaceFn3> = unsafe { self.lib.get(name.as_bytes()) }?; Ok(KernelFn::Arity3(*sym)) }
            4 => { let sym: libloading::Symbol<CifaceFn4> = unsafe { self.lib.get(name.as_bytes()) }?; Ok(KernelFn::Arity4(*sym)) }
            5 => { let sym: libloading::Symbol<CifaceFn5> = unsafe { self.lib.get(name.as_bytes()) }?; Ok(KernelFn::Arity5(*sym)) }
            6 => { let sym: libloading::Symbol<CifaceFn6> = unsafe { self.lib.get(name.as_bytes()) }?; Ok(KernelFn::Arity6(*sym)) }
            7 => { let sym: libloading::Symbol<CifaceFn7> = unsafe { self.lib.get(name.as_bytes()) }?; Ok(KernelFn::Arity7(*sym)) }
            8 => { let sym: libloading::Symbol<CifaceFn8> = unsafe { self.lib.get(name.as_bytes()) }?; Ok(KernelFn::Arity8(*sym)) }
             _ => {
                let sym: libloading::Symbol<unsafe extern "C" fn()> = unsafe { self.lib.get(name.as_bytes()) }?;
                Ok(KernelFn::HighArity(crate::ciface_high::FnPtr(*sym)))
            }
        }
    }
}

// ── Kernel function dispatch ───────────────────────────────────────

pub enum KernelFn {
    Arity1(CifaceFn1),
    Arity2(CifaceFn2),
    Arity3(CifaceFn3),
    Arity4(CifaceFn4),
    Arity5(CifaceFn5),
    Arity6(CifaceFn6),
    Arity7(CifaceFn7),
    Arity8(CifaceFn8),
    HighArity(crate::ciface_high::FnPtr),
}

impl KernelFn {
    pub fn arity(&self) -> usize {
        match self {
            KernelFn::Arity1(_) => 1,
            KernelFn::Arity2(_) => 2,
            KernelFn::Arity3(_) => 3,
            KernelFn::Arity4(_) => 4,
            KernelFn::Arity5(_) => 5,
            KernelFn::Arity6(_) => 6,
            KernelFn::Arity7(_) => 7,
            KernelFn::Arity8(_) => 8,
            KernelFn::HighArity(_) => 0,
        }
    }

    pub unsafe fn call(&self, outputs: &[*mut c_void], inputs: &[*const c_void]) {
        let total = outputs.len() + inputs.len();
        match (self, outputs.len(), inputs.len()) {
            (KernelFn::Arity1(f), 0, 0) => f(outputs[0]),
            (KernelFn::Arity2(f), 0, 1) => f(outputs[0], inputs[0]),
            (KernelFn::Arity3(f), 0, 2) => f(outputs[0], inputs[0], inputs[1]),
            (KernelFn::Arity4(f), 0, 3) => f(outputs[0], inputs[0], inputs[1], inputs[2]),
            (KernelFn::Arity5(f), 0, 4) => f(outputs[0], inputs[0], inputs[1], inputs[2], inputs[3]),
            (KernelFn::Arity6(f), 0, 5) => f(outputs[0], inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]),
            (KernelFn::Arity7(f), 0, 6) => f(outputs[0], inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5]),
            (KernelFn::Arity8(f), 0, 7) => f(outputs[0], inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6]),
            (KernelFn::HighArity(f), _, _) if total >= 1 && total <= 300 => {
                let ptr = f.0 as *const ();
                crate::ciface_high::call_high_arity(ptr, outputs, inputs);
            }
            _ => panic!(
                "kernel arity mismatch: outputs={}, inputs={}",
                outputs.len(),
                inputs.len(),
            ),
        }
    }
}

// ── Typed ciface function pointers ─────────────────────────────────

pub type CifaceFn1 = unsafe extern "C" fn(*mut c_void);
pub type CifaceFn2 = unsafe extern "C" fn(*mut c_void, *const c_void);
pub type CifaceFn3 = unsafe extern "C" fn(*mut c_void, *const c_void, *const c_void);
pub type CifaceFn4 =
    unsafe extern "C" fn(*mut c_void, *const c_void, *const c_void, *const c_void);
pub type CifaceFn5 = unsafe extern "C" fn(
    *mut c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *const c_void,
);
pub type CifaceFn6 = unsafe extern "C" fn(
    *mut c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *const c_void,
);
pub type CifaceFn7 = unsafe extern "C" fn(
    *mut c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *const c_void,
);
pub type CifaceFn8 = unsafe extern "C" fn(
    *mut c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *const c_void,
);

// ── Rank-generic MemRef descriptor ─────────────────────────────────
///
/// Binary layout matches the MLIR LLVM dialect memref descriptor:
///   struct<(ptr, ptr, i64, array<RANK x i64>, array<RANK x i64>)>
///
/// ``allocated``/``aligned`` are typeless pointers; ``sizes`` and
/// ``strides`` are rank-specific arrays.  The const generic ``RANK``
/// must match the tensor rank declared in the compiled function's
/// interface.

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MemRefDesc<const RANK: usize> {
    pub allocated: *mut c_void,
    pub aligned: *mut c_void,
    pub offset: i64,
    pub sizes: [i64; RANK],
    pub strides: [i64; RANK],
}

impl<const RANK: usize> Default for MemRefDesc<RANK> {
    fn default() -> Self {
        Self {
            allocated: std::ptr::null_mut(),
            aligned: std::ptr::null_mut(),
            offset: 0,
            sizes: [0i64; RANK],
            strides: [1i64; RANK],
        }
    }
}

pub type MemRefDesc0 = MemRefDesc<0>;
pub type MemRefDesc1 = MemRefDesc<1>;
pub type MemRefDesc2 = MemRefDesc<2>;
pub type MemRefDesc3 = MemRefDesc<3>;
pub type MemRefDesc4 = MemRefDesc<4>;

impl<const RANK: usize> MemRefDesc<RANK> {
    /// Build a descriptor from a ``&[f32]`` slice with given logical shape.
    pub fn from_f32_slice(data: &[f32], shape: [usize; RANK]) -> Self {
        let numel: usize = shape.iter().product();
        assert_eq!(data.len(), numel);
        let p = data.as_ptr();
        let mut sizes = [0i64; RANK];
        let mut strides_arr = [0i64; RANK];
        let mut stride = 1i64;
        for i in (0..RANK).rev() {
            sizes[i] = shape[i] as i64;
            strides_arr[i] = stride;
            stride *= shape[i] as i64;
        }
        Self {
            allocated: p as *mut c_void,
            aligned: p as *mut c_void,
            offset: 0,
            sizes,
            strides: strides_arr,
        }
    }

    /// Build a descriptor from a dynamic shape slice (runtime rank check).
    pub fn from_f32_dyn_slice(data: &[f32], shape: &[usize]) -> Self {
        assert_eq!(shape.len(), RANK, "shape rank mismatch");
        let mut sizes = [0i64; RANK];
        let mut strides_arr = [0i64; RANK];
        let p = data.as_ptr();
        let mut stride = 1i64;
        for i in (0..RANK).rev() {
            sizes[i] = shape[i] as i64;
            strides_arr[i] = stride;
            stride *= shape[i] as i64;
        }
        Self {
            allocated: p as *mut c_void,
            aligned: p as *mut c_void,
            offset: 0,
            sizes,
            strides: strides_arr,
        }
    }

    /// Build a zero-initialized output descriptor.
    /// The kernel will overwrite ``aligned`` with a malloc'd buffer.
    pub fn zeroed(shape: [usize; RANK]) -> Self {
        let mut sizes = [0i64; RANK];
        let mut strides_arr = [0i64; RANK];
        let mut stride = 1i64;
        for i in (0..RANK).rev() {
            sizes[i] = shape[i] as i64;
            strides_arr[i] = stride;
            stride *= shape[i] as i64;
        }
        Self {
            allocated: std::ptr::null_mut(),
            aligned: std::ptr::null_mut(),
            offset: 0,
            sizes,
            strides: strides_arr,
        }
    }

    /// Build a zero-initialized output descriptor from a dynamic shape.
    /// 0 dims (kDynamic sentinel) are replaced with 1 to avoid null allocations.
    pub fn zeroed_dyn(shape: &[usize]) -> Self {
        assert_eq!(shape.len(), RANK, "shape rank mismatch");
        // Replace 0 sentinel dims with 1 so kernel has a buffer to write to.
        let safe_shape: Vec<usize> = shape.iter().map(|&d| if d == 0 { 1 } else { d }).collect();
        let numel: usize = safe_shape.iter().product();
        let layout = std::alloc::Layout::array::<f32>(numel.max(1)).expect("invalid layout");
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) as *mut c_void };
        let mut sizes = [0i64; RANK];
        let mut strides_arr = [0i64; RANK];
        let mut stride = 1i64;
        for i in (0..RANK).rev() {
            sizes[i] = safe_shape[i] as i64;
            strides_arr[i] = stride;
            stride *= safe_shape[i] as i64;
        }
        Self {
            allocated: ptr,
            aligned: ptr,
            offset: 0,
            sizes,
            strides: strides_arr,
        }
    }

    pub fn from_raw_ptr(data: *const u8, shape: &[usize]) -> Self {
        assert_eq!(shape.len(), RANK, "shape rank mismatch");
        let mut sizes = [0i64; RANK];
        let mut strides_arr = [0i64; RANK];
        let mut stride = 1i64;
        for i in (0..RANK).rev() {
            sizes[i] = shape[i] as i64;
            strides_arr[i] = stride;
            stride *= shape[i] as i64;
        }
        Self {
            allocated: data as *mut c_void,
            aligned: data as *mut c_void,
            offset: 0,
            sizes,
            strides: strides_arr,
        }
    }

    pub fn numel(&self) -> usize {
        self.sizes.iter().map(|&s| s as usize).product()
    }

    pub fn is_null(&self) -> bool {
        self.aligned.is_null()
    }

    pub fn as_ptr(&self) -> *const c_void {
        self.aligned.cast_const()
    }

    /// Read f32 output data from a descriptor populated by a kernel.
    ///
    /// # Safety
    /// ``self.aligned`` must point to a valid buffer of ``numel()`` f32
    /// elements. After a kernel call, the kernel should have set
    /// ``aligned`` to a malloc'd buffer. If the kernel did not write
    /// ``aligned``, this returns an empty Vec.
    pub unsafe fn read_output_f32(&self) -> Vec<f32> {
        let n = self.numel();
        if n == 0 || self.aligned.is_null() {
            return Vec::new();
        }
        // SAFETY: The caller guarantees `self.aligned` points to a valid
        // buffer of at least `n` f32 elements.
        let slice = unsafe { std::slice::from_raw_parts(self.aligned as *const f32, n) };
        slice.to_vec()
    }
}

// SAFETY: MemRefDesc contains raw pointers, but they are only valid
// within the scope of a single synchronous kernel call. After the call,
// data is copied out via `read_output_f32`. The descriptor is not
// retained across threads.
unsafe impl<const RANK: usize> Send for MemRefDesc<RANK> {}

// ── Rank-erased descriptor enum ────────────────────────────────────

pub enum MemRefDescAny {
    R0(MemRefDesc0),
    R1(MemRefDesc1),
    R2(MemRefDesc2),
    R3(MemRefDesc3),
    R4(MemRefDesc4),
}

impl MemRefDescAny {
    pub fn from_f32(shape: &[usize], data: &[f32]) -> Self {
        match shape.len() {
            0 => Self::R0(MemRefDesc0::from_f32_dyn_slice(data, shape)),
            1 => Self::R1(MemRefDesc1::from_f32_dyn_slice(data, shape)),
            2 => Self::R2(MemRefDesc2::from_f32_dyn_slice(data, shape)),
            3 => Self::R3(MemRefDesc3::from_f32_dyn_slice(data, shape)),
            4 => Self::R4(MemRefDesc4::from_f32_dyn_slice(data, shape)),
            r => panic!("unsupported rank {}", r),
        }
    }

    pub fn zeroed(shape: &[usize]) -> Self {
        match shape.len() {
            0 => Self::R0(MemRefDesc0::zeroed_dyn(shape)),
            1 => Self::R1(MemRefDesc1::zeroed_dyn(shape)),
            2 => Self::R2(MemRefDesc2::zeroed_dyn(shape)),
            3 => Self::R3(MemRefDesc3::zeroed_dyn(shape)),
            4 => Self::R4(MemRefDesc4::zeroed_dyn(shape)),
            r => panic!("unsupported rank {}", r),
        }
    }

    pub fn as_output_ptr(&self) -> *mut c_void {
        match self {
            MemRefDescAny::R0(d) => d as *const MemRefDesc0 as *mut c_void,
            MemRefDescAny::R1(d) => d as *const MemRefDesc1 as *mut c_void,
            MemRefDescAny::R2(d) => d as *const MemRefDesc2 as *mut c_void,
            MemRefDescAny::R3(d) => d as *const MemRefDesc3 as *mut c_void,
            MemRefDescAny::R4(d) => d as *const MemRefDesc4 as *mut c_void,
        }
    }

    pub fn as_input_ptr(&self) -> *const c_void {
        match self {
            MemRefDescAny::R0(d) => d as *const MemRefDesc0 as *const c_void,
            MemRefDescAny::R1(d) => d as *const MemRefDesc1 as *const c_void,
            MemRefDescAny::R2(d) => d as *const MemRefDesc2 as *const c_void,
            MemRefDescAny::R3(d) => d as *const MemRefDesc3 as *const c_void,
            MemRefDescAny::R4(d) => d as *const MemRefDesc4 as *const c_void,
        }
    }

    pub unsafe fn read_output_f32(&self) -> Vec<f32> {
        match self {
            MemRefDescAny::R0(d) => d.read_output_f32(),
            MemRefDescAny::R1(d) => d.read_output_f32(),
            MemRefDescAny::R2(d) => d.read_output_f32(),
            MemRefDescAny::R3(d) => d.read_output_f32(),
            MemRefDescAny::R4(d) => d.read_output_f32(),
        }
    }

    pub fn sizes(&self) -> Vec<usize> {
        match self {
            MemRefDescAny::R0(d) => d.sizes.iter().map(|&x| x as usize).collect(),
            MemRefDescAny::R1(d) => d.sizes.iter().map(|&x| x as usize).collect(),
            MemRefDescAny::R2(d) => d.sizes.iter().map(|&x| x as usize).collect(),
            MemRefDescAny::R3(d) => d.sizes.iter().map(|&x| x as usize).collect(),
            MemRefDescAny::R4(d) => d.sizes.iter().map(|&x| x as usize).collect(),
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(desc.is_null());
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
        let k3 = KernelFn::Arity3(unsafe {
            std::mem::transmute::<*const c_void, CifaceFn3>(std::ptr::null())
        });
        assert_eq!(k3.arity(), 3);
    }

    #[test]
    fn test_buffer_as_mut_slice() {
        let mut d = Device::new();
        let mut buf = d.allocate(16);
        assert_eq!(buf.size(), 16);
        buf.as_mut_slice().fill(0u8);
    }

    #[test]
    fn test_memref_desc_any() {
        let data = vec![1.0f32; 6];
        let desc = MemRefDescAny::from_f32(&[2, 3], &data);
        assert!(!desc.as_input_ptr().is_null());
        let ptr = desc.as_input_ptr();

        let desc2 = MemRefDescAny::zeroed(&[2, 3]);
        assert!(desc2.as_input_ptr().is_null() == false); // struct exists
    }
}
