//! MLIR MemRef descriptors — matches the LLVM dialect memref struct layout.
//!
//! Binary layout (per MLIR LLVM dialect):
//!   struct<(ptr, ptr, i64, array<RANK x i64>, array<RANK x i64>)>
//!
//! ``allocated``/``aligned`` are typeless pointers; ``sizes`` and
//! ``strides`` are rank-specific arrays.  The const generic ``RANK``
//! must match the tensor rank declared in the compiled function.

use std::ffi::c_void;

// ── Rank-parameterized MemRef descriptor ──────────────────────────

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

#[allow(dead_code)]
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

    /// Build a zero-initialized output descriptor (allocates memory).
    pub fn zeroed(shape: [usize; RANK]) -> Self {
        let numel: usize = shape.iter().product();
        let layout = std::alloc::Layout::array::<f32>(numel.max(1)).expect("invalid layout");
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) as *mut c_void };
        let mut sizes = [0i64; RANK];
        let mut strides_arr = [0i64; RANK];
        let mut stride = 1i64;
        for i in (0..RANK).rev() {
            sizes[i] = shape[i] as i64;
            strides_arr[i] = stride;
            stride *= shape[i] as i64;
        }
        Self {
            allocated: ptr,
            aligned: ptr,
            offset: 0,
            sizes,
            strides: strides_arr,
        }
    }

    /// Build zeroed descriptor from dynamic shape (0 dims → 1 to avoid null alloc).
    pub fn zeroed_dyn(shape: &[usize]) -> Self {
        assert_eq!(shape.len(), RANK, "shape rank mismatch");
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
    /// ``self.aligned`` must point to a valid buffer of ``numel()`` f32 elements.
    pub unsafe fn read_output_f32(&self) -> Vec<f32> {
        let n = self.numel();
        if n == 0 || self.aligned.is_null() {
            return Vec::new();
        }
        let slice = unsafe { std::slice::from_raw_parts(self.aligned as *const f32, n) };
        slice.to_vec()
    }
}

// SAFETY: MemRefDesc contains raw pointers that are only valid within
// a single synchronous kernel call scope.
unsafe impl<const RANK: usize> Send for MemRefDesc<RANK> {}

// ── Rank-erased descriptor enum ───────────────────────────────────

pub enum MemRefDescAny {
    R0(MemRefDesc0),
    R1(MemRefDesc1),
    R2(MemRefDesc2),
    R3(MemRefDesc3),
    R4(MemRefDesc4),
}

#[allow(dead_code)]
impl MemRefDescAny {
    pub fn from_f32(shape: &[usize], data: &[f32]) -> Result<Self, anyhow::Error> {
        match shape.len() {
            0 => Ok(Self::R0(MemRefDesc0::from_f32_dyn_slice(data, shape))),
            1 => Ok(Self::R1(MemRefDesc1::from_f32_dyn_slice(data, shape))),
            2 => Ok(Self::R2(MemRefDesc2::from_f32_dyn_slice(data, shape))),
            3 => Ok(Self::R3(MemRefDesc3::from_f32_dyn_slice(data, shape))),
            4 => Ok(Self::R4(MemRefDesc4::from_f32_dyn_slice(data, shape))),
            r => anyhow::bail!(
                "MemRefDescAny::from_f32: unsupported rank {} (shape={:?})", r, shape
            ),
        }
    }

    pub fn zeroed(shape: &[usize]) -> Result<Self, anyhow::Error> {
        match shape.len() {
            0 => Ok(Self::R0(MemRefDesc0::zeroed_dyn(shape))),
            1 => Ok(Self::R1(MemRefDesc1::zeroed_dyn(shape))),
            2 => Ok(Self::R2(MemRefDesc2::zeroed_dyn(shape))),
            3 => Ok(Self::R3(MemRefDesc3::zeroed_dyn(shape))),
            4 => Ok(Self::R4(MemRefDesc4::zeroed_dyn(shape))),
            r => anyhow::bail!(
                "MemRefDescAny::zeroed: unsupported rank {} (shape={:?})", r, shape
            ),
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
