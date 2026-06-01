//! SFA Tensor — Stable Function ABI tensor descriptors with Ownership/Device semantics.
//!
//! Binary layout of per-rank structs matches MLIR LLVM dialect memref descriptor:
//!   struct<(ptr, ptr, i64, array<RANK x i64>, array<RANK x i64>)>
//!
//! Per-rank sizes (first 3 fields = 24-byte header):
//!   SFATensorRaw1: 24 + 8 + 8  = 40 bytes
//!   SFATensorRaw2: 24 + 16 + 16 = 56 bytes
//!   SFATensorRaw3: 24 + 24 + 24 = 72 bytes
//!   SFATensorRaw4: 24 + 32 + 32 = 88 bytes

use std::alloc::{self, Layout};
use std::ffi::c_void;
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

// ── Drop tracking for tests ────────────────────────────────────────

/// Global counter: total number of Owned SFATensor drops across all threads.
pub(crate) static OWNED_DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

// Thread-local flag: set to true when an Owned tensor is dropped on this thread.
thread_local! {
    static DROP_TRIGGERED: Cell<bool> = const { Cell::new(false) };
}

// ── Per-rank tensor descriptors (matches include/sfa.h) ────────────

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SFATensorRaw1 {
    pub allocated: *mut c_void,
    pub aligned: *mut c_void,
    pub offset: i64,
    pub sizes: [i64; 1],
    pub strides: [i64; 1],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SFATensorRaw2 {
    pub allocated: *mut c_void,
    pub aligned: *mut c_void,
    pub offset: i64,
    pub sizes: [i64; 2],
    pub strides: [i64; 2],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SFATensorRaw3 {
    pub allocated: *mut c_void,
    pub aligned: *mut c_void,
    pub offset: i64,
    pub sizes: [i64; 3],
    pub strides: [i64; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SFATensorRaw4 {
    pub allocated: *mut c_void,
    pub aligned: *mut c_void,
    pub offset: i64,
    pub sizes: [i64; 4],
    pub strides: [i64; 4],
}

// ── Rank-erased union ──────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub enum SFATensorRawAny {
    R1(SFATensorRaw1),
    R2(SFATensorRaw2),
    R3(SFATensorRaw3),
    R4(SFATensorRaw4),
}

impl SFATensorRawAny {
    /// Get the data pointer (== allocated field) from any rank variant.
    fn data_ptr(&self) -> *mut u8 {
        match self {
            SFATensorRawAny::R1(r) => r.allocated as *mut u8,
            SFATensorRawAny::R2(r) => r.allocated as *mut u8,
            SFATensorRawAny::R3(r) => r.allocated as *mut u8,
            SFATensorRawAny::R4(r) => r.allocated as *mut u8,
        }
    }
}

// ── Device & Ownership enums ───────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    CPU,
    #[allow(dead_code)]
    CUDA(u32),
}

/// Ownership mode for tensor memory.
///
/// - `Owned`: SFATensor owns the memory; it is freed on Drop.
/// - `Borrowed`: Caller retains ownership; Drop is a no-op.
/// - `Managed`: Custom deleter is called on Drop with the data pointer.
pub enum Ownership {
    Owned,
    Borrowed,
    #[allow(dead_code)]
    Managed {
        deleter: Box<dyn Fn(*mut u8) + Send + Sync>,
    },
}

impl std::fmt::Debug for Ownership {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Ownership::Owned => write!(f, "Owned"),
            Ownership::Borrowed => write!(f, "Borrowed"),
            Ownership::Managed { .. } => write!(f, "Managed(..)"),
        }
    }
}

// ── Internal owned-buffer wrapper ──────────────────────────────────

/// Type-erased buffer that deallocates with the correct `Layout` on Drop.
#[derive(Debug)]
struct OwnedBuf {
    ptr: *mut u8,
    layout: Layout,
}

// SAFETY: OwnedBuf owns a unique heap allocation; moving it across
// threads is safe as long as the caller ensures no concurrent access.
unsafe impl Send for OwnedBuf {}

impl Drop for OwnedBuf {
    fn drop(&mut self) {
        if self.layout.size() > 0 {
            unsafe { alloc::dealloc(self.ptr, self.layout) };
        }
    }
}

// ── SFATensor ──────────────────────────────────────────────────────

/// High-level tensor wrapper around a per-rank raw descriptor.
///
/// Combines the binary-compatible `SFATensorRaw<RANK>` descriptor with
/// device placement, ownership semantics, and element-size metadata.
#[derive(Debug)]
pub struct SFATensor {
    /// Binary-compatible raw descriptor (abi-ready).
    pub raw: SFATensorRawAny,
    /// Device where the tensor data resides.
    pub device: Device,
    /// Memory ownership mode.
    pub ownership: Ownership,
    /// Size of a single element in bytes (e.g. 4 for f32).
    pub elem_size: usize,
    /// True rank (0 for scalars, 1..4 for tensors).
    rank_val: usize,
    /// For Owned tensors: holds the backing allocation (deallocated on Drop).
    #[allow(dead_code)]
    owned_buf: Option<OwnedBuf>,
}

impl SFATensor {
    // ── Constructors ───────────────────────────────────────────────

    /// Create an owned CPU tensor from `Vec<f32>`.
    ///
    /// # Panics
    /// Panics if `data.len() != product(shape)` or `shape.len()` not in 1..=4.
    pub fn from_vec_f32(mut data: Vec<f32>, shape: Vec<usize>) -> Self {
        let rank = shape.len();
        assert!(rank >= 1 && rank <= 4, "rank must be 1..=4, got {}", rank);
        let numel: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            numel,
            "data length {} != numel {}",
            data.len(),
            numel
        );

        let elem_size = std::mem::size_of::<f32>();
        let layout = Layout::array::<f32>(data.capacity()).expect("valid layout");
        let ptr = data.as_ptr() as *mut c_void;

        // Transfer ownership: take pointer + layout, forget the Vec.
        let raw_ptr = data.as_mut_ptr() as *mut u8;
        let owned_buf = OwnedBuf {
            ptr: raw_ptr,
            layout,
        };
        std::mem::forget(data);

        let raw = build_raw_any(rank, ptr, &shape, elem_size);

        SFATensor {
            raw,
            device: Device::CPU,
            ownership: Ownership::Owned,
            elem_size,
            rank_val: rank,
            owned_buf: Some(owned_buf),
        }
    }

    /// Create an owned CPU tensor from `Vec<i64>`.
    ///
    /// # Panics
    /// Panics if `data.len() != product(shape)` or `shape.len()` not in 1..=4.
    #[allow(dead_code)]
    pub fn from_vec_i64(mut data: Vec<i64>, shape: Vec<usize>) -> Self {
        let rank = shape.len();
        assert!(rank >= 1 && rank <= 4, "rank must be 1..=4, got {}", rank);
        let numel: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            numel,
            "data length {} != numel {}",
            data.len(),
            numel
        );

        let elem_size = std::mem::size_of::<i64>();
        let layout = Layout::array::<i64>(data.capacity()).expect("valid layout");
        let ptr = data.as_ptr() as *mut c_void;

        let raw_ptr = data.as_mut_ptr() as *mut u8;
        let owned_buf = OwnedBuf {
            ptr: raw_ptr,
            layout,
        };
        std::mem::forget(data);

        let raw = build_raw_any(rank, ptr, &shape, elem_size);

        SFATensor {
            raw,
            device: Device::CPU,
            ownership: Ownership::Owned,
            elem_size,
            rank_val: rank,
            owned_buf: Some(owned_buf),
        }
    }

    /// Generic constructor from raw parts.
    ///
    /// The caller is responsible for ensuring `raw` pointers remain valid
    /// for the lifetime of this tensor (if `Ownership::Borrowed`) or that
    /// the `OwnedBuf` / deleter correctly frees the memory.
    #[allow(dead_code)]
    pub fn from_raw_parts(
        raw: SFATensorRawAny,
        device: Device,
        ownership: Ownership,
        elem_size: usize,
    ) -> Self {
        let rank = match &raw {
            SFATensorRawAny::R1(_) => 1,
            SFATensorRawAny::R2(_) => 2,
            SFATensorRawAny::R3(_) => 3,
            SFATensorRawAny::R4(_) => 4,
        };

        SFATensor {
            raw,
            device,
            ownership,
            elem_size,
            rank_val: rank,
            owned_buf: None,
        }
    }

    /// Create a scalar (rank-0) tensor from a single `f32` value.
    #[allow(dead_code)]
    pub fn scalar_f32(value: f32) -> Self {
        let data = vec![value];
        let mut t = Self::from_vec_f32(data, vec![1]);
        t.rank_val = 0;
        t
    }

    // ── Accessors ──────────────────────────────────────────────────

    /// Tensor rank. 0 for scalars, 1..4 for regular tensors.
    pub fn rank(&self) -> usize {
        self.rank_val
    }

    /// Total number of elements (product of shape dimensions).
    pub fn numel(&self) -> usize {
        self.shape().iter().product()
    }

    /// Shape as `Vec<usize>`. Empty for scalars.
    pub fn shape(&self) -> Vec<usize> {
        match &self.raw {
            SFATensorRawAny::R1(r) => {
                if self.rank_val == 0 {
                    vec![]
                } else {
                    vec![r.sizes[0] as usize]
                }
            }
            SFATensorRawAny::R2(r) => vec![r.sizes[0] as usize, r.sizes[1] as usize],
            SFATensorRawAny::R3(r) => {
                vec![
                    r.sizes[0] as usize,
                    r.sizes[1] as usize,
                    r.sizes[2] as usize,
                ]
            }
            SFATensorRawAny::R4(r) => {
                vec![
                    r.sizes[0] as usize,
                    r.sizes[1] as usize,
                    r.sizes[2] as usize,
                    r.sizes[3] as usize,
                ]
            }
        }
    }

    /// Get a raw pointer to the tensor data.
    #[allow(dead_code)]
    fn data_ptr(&self) -> *mut u8 {
        self.raw.data_ptr()
    }
}

// ── Drop ───────────────────────────────────────────────────────────

impl Drop for SFATensor {
    fn drop(&mut self) {
        match &self.ownership {
            Ownership::Owned => {
                OWNED_DROP_COUNT.fetch_add(1, Ordering::SeqCst);
                DROP_TRIGGERED.with(|f| f.set(true));
                // owned_buf is dropped automatically when self is dropped,
                // which deallocates the backing memory.
            }
            Ownership::Borrowed => {
                // Caller retains ownership — do nothing.
            }
            Ownership::Managed { deleter } => {
                let ptr = self.raw.data_ptr();
                if !ptr.is_null() {
                    deleter(ptr);
                }
            }
        }
        // owned_buf drops here (if Some), freeing memory.
    }
}

// ── Helpers ────────────────────────────────────────────────────────

/// Build an `SFATensorRawAny` for the given rank with row-major strides.
fn build_raw_any(
    rank: usize,
    ptr: *mut c_void,
    shape: &[usize],
    _elem_size: usize,
) -> SFATensorRawAny {
    // Compute row-major strides (last dim stride = 1)
    let mut strides = vec![1i64; rank];
    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1] as i64;
    }

    match rank {
        1 => SFATensorRawAny::R1(SFATensorRaw1 {
            allocated: ptr,
            aligned: ptr,
            offset: 0,
            sizes: [shape[0] as i64],
            strides: [strides[0]],
        }),
        2 => SFATensorRawAny::R2(SFATensorRaw2 {
            allocated: ptr,
            aligned: ptr,
            offset: 0,
            sizes: [shape[0] as i64, shape[1] as i64],
            strides: [strides[0], strides[1]],
        }),
        3 => SFATensorRawAny::R3(SFATensorRaw3 {
            allocated: ptr,
            aligned: ptr,
            offset: 0,
            sizes: [shape[0] as i64, shape[1] as i64, shape[2] as i64],
            strides: [strides[0], strides[1], strides[2]],
        }),
        4 => SFATensorRawAny::R4(SFATensorRaw4 {
            allocated: ptr,
            aligned: ptr,
            offset: 0,
            sizes: [
                shape[0] as i64,
                shape[1] as i64,
                shape[2] as i64,
                shape[3] as i64,
            ],
            strides: [strides[0], strides[1], strides[2], strides[3]],
        }),
        _ => panic!("unsupported rank: {}", rank),
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sfa_tensor_from_vec_f32() {
        let t = SFATensor::from_vec_f32(vec![1.0f32, 2.0, 3.0], vec![3]);
        assert_eq!(t.rank(), 1);
        assert_eq!(t.numel(), 3);
        assert_eq!(t.shape(), vec![3]);
        assert_eq!(t.device, Device::CPU);
        assert_eq!(t.elem_size, 4);

        // Verify data integrity via raw pointer.
        let ptr = t.data_ptr() as *const f32;
        let data = unsafe { std::slice::from_raw_parts(ptr, 3) };
        assert_eq!(data, &[1.0f32, 2.0, 3.0]);
    }

    #[test]
    fn test_sfa_tensor_owned_drop() {
        DROP_TRIGGERED.with(|f| f.set(false));
        {
            let _t = SFATensor::from_vec_f32(vec![1.0f32, 2.0, 3.0], vec![3]);
            // Drop not yet called while tensor is alive.
            DROP_TRIGGERED.with(|f| assert!(!f.get()));
        } // _t drops here.
        DROP_TRIGGERED.with(|f| assert!(f.get()));
    }

    #[test]
    fn test_sfa_tensor_borrowed_no_drop() {
        DROP_TRIGGERED.with(|f| f.set(false));

        // Stack-allocated data that outlives the tensor wrapper.
        let mut stack_data: [f32; 3] = [4.0, 5.0, 6.0];
        let raw = SFATensorRawAny::R1(SFATensorRaw1 {
            allocated: stack_data.as_mut_ptr() as *mut c_void,
            aligned: stack_data.as_mut_ptr() as *mut c_void,
            offset: 0,
            sizes: [3],
            strides: [1],
        });

        {
            let t = SFATensor::from_raw_parts(
                raw,
                Device::CPU,
                Ownership::Borrowed,
                4, // elem_size for f32
            );
            assert_eq!(t.rank(), 1);
            assert_eq!(t.numel(), 3);
            // Verify data is accessible.
            let ptr = t.data_ptr() as *const f32;
            let data = unsafe { std::slice::from_raw_parts(ptr, 3) };
            assert_eq!(data, &[4.0f32, 5.0, 6.0]);
        } // t drops here — should be no-op.

        // Drop must NOT have triggered (borrowed => no dealloc).
        DROP_TRIGGERED.with(|f| assert!(!f.get()));

        // Stack data still intact.
        assert_eq!(stack_data, [4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_sfa_tensor_device_cpu() {
        let t = SFATensor::from_vec_f32(vec![1.0f32], vec![1]);
        assert_eq!(t.device, Device::CPU);
        assert_ne!(t.device, Device::CUDA(0));
    }

    #[test]
    fn test_sfa_tensor_scalar_rank0() {
        let t = SFATensor::scalar_f32(std::f32::consts::PI);
        assert_eq!(t.rank(), 0);
        assert_eq!(t.numel(), 1);
        assert!(t.shape().is_empty());

        // Data must be accessible.
        let ptr = t.data_ptr() as *const f32;
        let val = unsafe { *ptr };
        assert!((val - std::f32::consts::PI).abs() < 1e-7);
    }
}
