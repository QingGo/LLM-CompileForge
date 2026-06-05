//! SFA MemRef type — unified tensor descriptor for HAL and dylib interfaces.
//!
//! ``SfaMemRef`` wraps per-rank raw descriptors (binary-compatible with
//! ``include/sfa.h`` and MLIR memref structs) plus element-size metadata.
//!
//! The raw structs are imported from ``crate::model::sfa_tensor`` where they are
//! defined as ``#[repr(C)]`` types matching the LLVM dialect memref layout:
//!
//!   struct<(ptr, ptr, i64, array<RANK x i64>, array<RANK x i64>)>
//!
//! Per-rank sizes:
//!   R1: 40 bytes  |  R2: 56 bytes  |  R3: 72 bytes  |  R4: 88 bytes

use std::ffi::c_void;

use crate::hal::cpu::memref::{MemRefDescAny, MemRefDesc1, MemRefDesc2, MemRefDesc3, MemRefDesc4};
use crate::model::sfa_tensor::{SFATensorRaw1, SFATensorRaw2, SFATensorRaw3, SFATensorRaw4};

// ── Type aliases (matching include/sfa.h naming) ────────────────────

pub type SfaMemRef1 = SFATensorRaw1;
pub type SfaMemRef2 = SFATensorRaw2;
pub type SfaMemRef3 = SFATensorRaw3;
pub type SfaMemRef4 = SFATensorRaw4;

// ── Rank-erased raw enum ────────────────────────────────────────────

/// Rank-erased raw descriptor enum.
#[derive(Debug, Clone, Copy)]
pub enum SfaMemRefRaw {
    R1(SfaMemRef1),
    R2(SfaMemRef2),
    R3(SfaMemRef3),
    R4(SfaMemRef4),
}

// ── SfaMemRef ───────────────────────────────────────────────────────

/// Unified tensor descriptor for HAL operations.
///
/// Combines a binary-compatible raw descriptor (matching the MLIR memref
/// layout) with element-size metadata.  Represents a buffer that the
/// runtime can pass to compiled kernels or dispatch as HAL operations.
#[derive(Debug, Clone, Copy)]
pub struct SfaMemRef {
    /// Rank-specific raw descriptor (binary-compatible with C/MLIR).
    pub raw: SfaMemRefRaw,
    /// Element size in bytes (4 for f32, 8 for i64).
    pub elem_size: usize,
}

impl SfaMemRef {
    // ── Constructors ───────────────────────────────────────────────

    /// Build a rank-1 ``SfaMemRef`` from raw parts.
    pub fn r1(allocated: *mut c_void, sizes: [i64; 1], strides: [i64; 1], elem_size: usize) -> Self {
        Self {
            raw: SfaMemRefRaw::R1(SfaMemRef1 {
                allocated,
                aligned: allocated,
                offset: 0,
                sizes,
                strides,
            }),
            elem_size,
        }
    }

    /// Build a rank-2 ``SfaMemRef`` from raw parts.
    pub fn r2(allocated: *mut c_void, sizes: [i64; 2], strides: [i64; 2], elem_size: usize) -> Self {
        Self {
            raw: SfaMemRefRaw::R2(SfaMemRef2 {
                allocated,
                aligned: allocated,
                offset: 0,
                sizes,
                strides,
            }),
            elem_size,
        }
    }

    /// Build a rank-3 ``SfaMemRef`` from raw parts.
    pub fn r3(allocated: *mut c_void, sizes: [i64; 3], strides: [i64; 3], elem_size: usize) -> Self {
        Self {
            raw: SfaMemRefRaw::R3(SfaMemRef3 {
                allocated,
                aligned: allocated,
                offset: 0,
                sizes,
                strides,
            }),
            elem_size,
        }
    }

    /// Build a rank-4 ``SfaMemRef`` from raw parts.
    pub fn r4(allocated: *mut c_void, sizes: [i64; 4], strides: [i64; 4], elem_size: usize) -> Self {
        Self {
            raw: SfaMemRefRaw::R4(SfaMemRef4 {
                allocated,
                aligned: allocated,
                offset: 0,
                sizes,
                strides,
            }),
            elem_size,
        }
    }

    /// Build from a shape slice (dynamic rank).
    pub fn from_shape(ptr: *mut c_void, shape: &[usize], elem_size: usize) -> Result<Self, anyhow::Error> {
        let rank = shape.len();
        if rank < 1 || rank > 4 {
            anyhow::bail!("SfaMemRef::from_shape: unsupported rank {}", rank);
        }
        let mut sizes = vec![0i64; rank];
        let mut strides = vec![1i64; rank];
        for i in (0..rank).rev() {
            sizes[i] = shape[i] as i64;
            if i < rank - 1 {
                strides[i] = strides[i + 1] * shape[i + 1] as i64;
            }
        }
        match rank {
            1 => Ok(Self::r1(ptr, [sizes[0]], [strides[0]], elem_size)),
            2 => Ok(Self::r2(ptr, [sizes[0], sizes[1]], [strides[0], strides[1]], elem_size)),
            3 => Ok(Self::r3(ptr, [sizes[0], sizes[1], sizes[2]], [strides[0], strides[1], strides[2]], elem_size)),
            4 => Ok(Self::r4(ptr, [sizes[0], sizes[1], sizes[2], sizes[3]], [strides[0], strides[1], strides[2], strides[3]], elem_size)),
            _ => unreachable!(),
        }
    }

    // ── Accessors ──────────────────────────────────────────────────

    /// Pointer for ciface argument list (points to the raw struct on the heap/stack).
    pub fn as_input_ptr(&self) -> *const c_void {
        self.raw.as_input_ptr()
    }

    /// Mutable pointer for output descriptors (points to the raw struct).
    pub fn as_mut_ptr(&mut self) -> *mut c_void {
        self.raw.as_mut_ptr()
    }

    /// Data pointer (== allocated field).
    pub fn data_ptr(&self) -> *mut u8 {
        self.raw.data_ptr()
    }

    /// Tensor rank.
    pub fn rank(&self) -> u8 {
        self.raw.rank()
    }

    /// Shape dimensions as i64.
    pub fn sizes_i64(&self) -> Vec<i64> {
        self.raw.sizes_i64()
    }

    /// Shape dimensions as usize.
    pub fn sizes(&self) -> Vec<usize> {
        self.sizes_i64().into_iter().map(|s| s as usize).collect()
    }

    /// Strides as i64.
    pub fn strides_i64(&self) -> Vec<i64> {
        self.raw.strides_i64()
    }

    /// Total number of elements (product of sizes).
    pub fn numel(&self) -> usize {
        self.raw.numel()
    }

    /// Total size in bytes (numel × elem_size).
    pub fn byte_len(&self) -> usize {
        self.numel() * self.elem_size
    }

    /// Element size in bytes.
    pub fn element_size(&self) -> usize {
        self.elem_size
    }

    // ── Conversion ─────────────────────────────────────────────────

    /// Convert to a rank-erased MemRef descriptor for dylib ciface calls.
    ///
    /// The underlying raw structs have the same binary layout as
    /// ``MemRefDesc<RANK>``, so this conversion is a field-by-field copy.
    pub(crate) fn to_memref_desc_any(&self) -> MemRefDescAny {
        match &self.raw {
            SfaMemRefRaw::R1(r) => MemRefDescAny::R1(MemRefDesc1 {
                allocated: r.allocated,
                aligned: r.aligned,
                offset: r.offset,
                sizes: r.sizes,
                strides: r.strides,
            }),
            SfaMemRefRaw::R2(r) => MemRefDescAny::R2(MemRefDesc2 {
                allocated: r.allocated,
                aligned: r.aligned,
                offset: r.offset,
                sizes: r.sizes,
                strides: r.strides,
            }),
            SfaMemRefRaw::R3(r) => MemRefDescAny::R3(MemRefDesc3 {
                allocated: r.allocated,
                aligned: r.aligned,
                offset: r.offset,
                sizes: r.sizes,
                strides: r.strides,
            }),
            SfaMemRefRaw::R4(r) => MemRefDescAny::R4(MemRefDesc4 {
                allocated: r.allocated,
                aligned: r.aligned,
                offset: r.offset,
                sizes: r.sizes,
                strides: r.strides,
            }),
        }
    }
}

// ── SfaMemRefRaw impls ──────────────────────────────────────────────

impl SfaMemRefRaw {
    pub fn as_input_ptr(&self) -> *const c_void {
        match self {
            SfaMemRefRaw::R1(d) => d as *const SfaMemRef1 as *const c_void,
            SfaMemRefRaw::R2(d) => d as *const SfaMemRef2 as *const c_void,
            SfaMemRefRaw::R3(d) => d as *const SfaMemRef3 as *const c_void,
            SfaMemRefRaw::R4(d) => d as *const SfaMemRef4 as *const c_void,
        }
    }

    pub fn as_mut_ptr(&mut self) -> *mut c_void {
        match self {
            SfaMemRefRaw::R1(d) => d as *mut SfaMemRef1 as *mut c_void,
            SfaMemRefRaw::R2(d) => d as *mut SfaMemRef2 as *mut c_void,
            SfaMemRefRaw::R3(d) => d as *mut SfaMemRef3 as *mut c_void,
            SfaMemRefRaw::R4(d) => d as *mut SfaMemRef4 as *mut c_void,
        }
    }

    pub fn data_ptr(&self) -> *mut u8 {
        match self {
            SfaMemRefRaw::R1(r) => r.allocated as *mut u8,
            SfaMemRefRaw::R2(r) => r.allocated as *mut u8,
            SfaMemRefRaw::R3(r) => r.allocated as *mut u8,
            SfaMemRefRaw::R4(r) => r.allocated as *mut u8,
        }
    }

    pub fn rank(&self) -> u8 {
        match self {
            SfaMemRefRaw::R1(_) => 1,
            SfaMemRefRaw::R2(_) => 2,
            SfaMemRefRaw::R3(_) => 3,
            SfaMemRefRaw::R4(_) => 4,
        }
    }

    pub fn sizes_i64(&self) -> Vec<i64> {
        match self {
            SfaMemRefRaw::R1(r) => r.sizes.to_vec(),
            SfaMemRefRaw::R2(r) => r.sizes.to_vec(),
            SfaMemRefRaw::R3(r) => r.sizes.to_vec(),
            SfaMemRefRaw::R4(r) => r.sizes.to_vec(),
        }
    }

    pub fn strides_i64(&self) -> Vec<i64> {
        match self {
            SfaMemRefRaw::R1(r) => r.strides.to_vec(),
            SfaMemRefRaw::R2(r) => r.strides.to_vec(),
            SfaMemRefRaw::R3(r) => r.strides.to_vec(),
            SfaMemRefRaw::R4(r) => r.strides.to_vec(),
        }
    }

    pub fn numel(&self) -> usize {
        let sizes = self.sizes_i64();
        crate::hal::cpu::sret::checked_product_from_i64(&sizes).unwrap_or(usize::MAX)
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::{alloc_zeroed, Layout};

    #[test]
    fn test_sfa_memref_r2_construction() {
        let layout = Layout::array::<f32>(8).unwrap();
        let ptr = unsafe { alloc_zeroed(layout) as *mut c_void };
        let sfa = SfaMemRef::r2(ptr, [2, 4], [4, 1], 4);
        assert_eq!(sfa.rank(), 2);
        assert_eq!(sfa.sizes(), vec![2, 4]);
        assert_eq!(sfa.numel(), 8);
        assert_eq!(sfa.byte_len(), 32);
        assert_eq!(sfa.element_size(), 4);
        assert!(!sfa.as_input_ptr().is_null());
        assert_eq!(sfa.data_ptr(), ptr as *mut u8);
        unsafe { std::alloc::dealloc(ptr as *mut u8, layout) };
    }

    #[test]
    fn test_sfa_memref_r1_i64() {
        let layout = Layout::array::<i64>(3).unwrap();
        let ptr = unsafe { alloc_zeroed(layout) as *mut c_void };
        let sfa = SfaMemRef::r1(ptr, [3], [1], 8);
        assert_eq!(sfa.rank(), 1);
        assert_eq!(sfa.sizes(), vec![3]);
        assert_eq!(sfa.numel(), 3);
        assert_eq!(sfa.byte_len(), 24);
        assert_eq!(sfa.element_size(), 8);
        unsafe { std::alloc::dealloc(ptr as *mut u8, layout) };
    }

    #[test]
    fn test_sfa_memref_from_shape() {
        let layout = Layout::array::<f32>(120).unwrap();
        let ptr = unsafe { alloc_zeroed(layout) as *mut c_void };
        let sfa = SfaMemRef::from_shape(ptr, &[2, 3, 4, 5], 4).unwrap();
        assert_eq!(sfa.rank(), 4);
        assert_eq!(sfa.sizes(), vec![2, 3, 4, 5]);
        assert_eq!(sfa.strides_i64(), vec![60, 20, 5, 1]);
        assert_eq!(sfa.numel(), 120);
        unsafe { std::alloc::dealloc(ptr as *mut u8, layout) };
    }

    #[test]
    fn test_sfa_memref_to_memref_desc_any() {
        let layout = Layout::array::<f32>(8).unwrap();
        let ptr = unsafe { alloc_zeroed(layout) as *mut c_void };
        let sfa = SfaMemRef::r2(ptr, [2, 4], [4, 1], 4);
        let desc = sfa.to_memref_desc_any();
        assert_eq!(desc.sizes(), vec![2, 4]);
        match desc {
            MemRefDescAny::R2(d) => {
                assert_eq!(d.sizes, [2, 4]);
                assert_eq!(d.strides, [4, 1]);
                assert_eq!(d.allocated, ptr);
                assert_eq!(d.aligned, ptr);
            }
            _ => panic!("expected R2 variant"),
        }
        unsafe { std::alloc::dealloc(ptr as *mut u8, layout) };
    }

    #[test]
    fn test_sfa_memref_raw_sizes() {
        assert_eq!(std::mem::size_of::<SfaMemRef1>(), 40);
        assert_eq!(std::mem::size_of::<SfaMemRef2>(), 56);
        assert_eq!(std::mem::size_of::<SfaMemRef3>(), 72);
        assert_eq!(std::mem::size_of::<SfaMemRef4>(), 88);
    }
}
