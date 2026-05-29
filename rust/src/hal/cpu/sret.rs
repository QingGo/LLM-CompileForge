//! sret (struct-return) descriptor parsing for MLIR ciface call convention.
//!
//! Provides helpers to construct MemRef descriptors from HAL Buffers (for
//! kernel inputs) and to read back output descriptors from the sret buffer
//! after a ciface kernel call.
//!
//! See also: ``super::memref`` for the MemRef descriptor types.

use crate::hal::traits;

/// Construct a MemRef descriptor from a Buffer, respecting its element
/// size and shape metadata.
///
/// Zero-copy: the descriptor's `aligned` pointer points directly into the
/// buffer's backing memory. The buffer must remain valid for the kernel call.
pub fn make_memref_descriptor(buf: &dyn traits::Buffer) -> Result<super::MemRefDescAny, anyhow::Error> {
    let ptr = buf.as_ptr();
    let shape = buf.shape();
    match shape.len() {
        1 => Ok(super::MemRefDescAny::R1(super::memref::MemRefDesc1::from_raw_ptr(ptr, &shape))),
        2 => Ok(super::MemRefDescAny::R2(super::memref::MemRefDesc2::from_raw_ptr(ptr, &shape))),
        3 => Ok(super::MemRefDescAny::R3(super::memref::MemRefDesc3::from_raw_ptr(ptr, &shape))),
        4 => Ok(super::MemRefDescAny::R4(super::memref::MemRefDesc4::from_raw_ptr(ptr, &shape))),
        r => anyhow::bail!(
            "make_memref_descriptor: unsupported rank {} for input buffer (shape={:?})",
            r, shape,
        ),
    }
}

/// Read a single sret output descriptor at a KNOWN rank.
///
/// Reads exactly the bytes for the given rank. The caller must supply the
/// correct rank (from the compute graph) to avoid misaligned reads when
/// consecutive descriptors have different ranks.
///
/// The MemRef descriptor layout (MLIR LLVM dialect convention):
///
/// ```text
///   offset 0:  allocated ptr   — raw malloc return; safe to free
///   offset 8:  aligned  ptr   — aligned data pointer; use for reads
///   offset 16: offset   i64   — byte offset within allocation
///   offset 24: sizes    [i64; RANK]
///   offset 24+8*RANK: strides [i64; RANK]
/// ```
///
/// When no alignment is requested (default for f32), allocated == aligned.
/// When alignment IS requested, aligned may differ from allocated — only
/// `allocated` is safe to pass to `free()`.
///
/// Returns `(allocated, aligned, sizes)`.
///
/// # Safety
///
/// `slice` must be at least `24 + 16 * rank` bytes long and contain valid
/// MemRef descriptor data written by a ciface kernel.
pub unsafe fn read_sret_descriptor(
    slice: &[u8],
    rank: usize,
) -> Result<(*mut u8, *mut u8, Vec<i64>), anyhow::Error> {
    let min_len = 24 + rank * 16;
    if slice.len() < min_len {
        anyhow::bail!(
            "sret slice too short: {} < {} (rank {})",
            slice.len(),
            min_len,
            rank,
        );
    }
    let allocated = std::ptr::read_unaligned(slice.as_ptr() as *const *mut u8);
    let aligned = std::ptr::read_unaligned(slice.as_ptr().add(8) as *const *mut u8);
    if aligned.is_null() {
        anyhow::bail!("sret aligned pointer is null (rank {})", rank);
    }
    let sizes: Vec<i64> = (0..rank)
        .map(|i| std::ptr::read_unaligned(slice.as_ptr().add(24 + i * 8) as *const i64))
        .collect();
    Ok((allocated, aligned, sizes))
}

/// Compute checked product of i64 dimensions. Negative dims clamped to 0.
/// Returns None on overflow.
pub fn checked_product_from_i64(sizes: &[i64]) -> Option<usize> {
    sizes.iter().try_fold(1usize, |acc, &s| {
        let dim = std::cmp::max(0, s) as usize;
        acc.checked_mul(dim)
    })
}

/// Compute checked product of usize dimensions. Returns None on overflow.
pub fn checked_product_usize(sizes: &[usize]) -> Option<usize> {
    sizes.iter().try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hal::cpu::MemRefDescAny;
    use crate::hal::traits;

    /// A minimal buffer for testing descriptor construction.
    #[derive(Debug)]
    struct TestBuffer {
        data: Vec<u8>,
        elem_size: usize,
        dims: Vec<usize>,
    }

    impl TestBuffer {
        fn new(data: Vec<u8>, elem_size: usize, dims: Vec<usize>) -> Self {
            Self { data, elem_size, dims }
        }
    }

    impl traits::Buffer for TestBuffer {
        fn as_ptr(&self) -> *const u8 { self.data.as_ptr() }
        fn as_mut_ptr(&mut self) -> *mut u8 { self.data.as_mut_ptr() }
        fn len(&self) -> usize { self.data.len() }
        fn copy_from_host(&mut self, src: &[u8], _stream: &dyn traits::Stream) -> Result<(), anyhow::Error> {
            self.data.copy_from_slice(src);
            Ok(())
        }
        fn copy_to_host(&self, dst: &mut [u8], _stream: &dyn traits::Stream) -> Result<(), anyhow::Error> {
            dst.copy_from_slice(&self.data);
            Ok(())
        }
        fn element_size(&self) -> usize { self.elem_size }
        fn shape(&self) -> Vec<usize> { self.dims.clone() }
        fn rank(&self) -> u8 { self.dims.len() as u8 }
    }

    #[test]
    fn test_make_memref_descriptor_rank1() {
        let data = vec![0u8; 64];
        let buf = TestBuffer::new(data, 4, vec![16]);
        let desc = make_memref_descriptor(&buf).expect("rank-1 descriptor");
        assert_eq!(desc.sizes(), vec![16]);
        match desc {
            MemRefDescAny::R1(_) => {}
            _ => panic!("expected R1 variant"),
        }
    }

    #[test]
    fn test_make_memref_descriptor_rank2() {
        let data = vec![0u8; 96];
        let buf = TestBuffer::new(data, 4, vec![4, 6]);
        let desc = make_memref_descriptor(&buf).expect("rank-2 descriptor");
        assert_eq!(desc.sizes(), vec![4, 6]);
        match desc {
            MemRefDescAny::R2(_) => {}
            _ => panic!("expected R2 variant"),
        }
    }

    #[test]
    fn test_make_memref_descriptor_rank3() {
        let data = vec![0u8; 120];
        let buf = TestBuffer::new(data, 4, vec![2, 3, 5]);
        let desc = make_memref_descriptor(&buf).expect("rank-3 descriptor");
        assert_eq!(desc.sizes(), vec![2, 3, 5]);
        match desc {
            MemRefDescAny::R3(_) => {}
            _ => panic!("expected R3 variant"),
        }
    }

    #[test]
    fn test_make_memref_descriptor_rank4() {
        let data = vec![0u8; 168];
        let buf = TestBuffer::new(data, 4, vec![2, 3, 4, 7]);
        let desc = make_memref_descriptor(&buf).expect("rank-4 descriptor");
        assert_eq!(desc.sizes(), vec![2, 3, 4, 7]);
        match desc {
            MemRefDescAny::R4(_) => {}
            _ => panic!("expected R4 variant"),
        }
    }

    #[test]
    fn test_make_memref_descriptor_unsupported_rank() {
        let data = vec![0u8; 32];
        let buf = TestBuffer::new(data, 4, vec![1, 2, 3, 4, 5]);
        let result = make_memref_descriptor(&buf);
        assert!(result.is_err(), "rank 5 should be unsupported");
    }

    #[test]
    fn test_read_sret_descriptor_rank1() {
        // Build a valid sret descriptor manually for rank-1
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let allocated = data.as_ptr() as *mut u8;
        let aligned = data.as_ptr() as *mut u8;
        let mut sret_buf = vec![0u8; 24 + 16];
        // SAFETY: `sret_buf` is a properly-sized buffer (24 + 16 bytes)
        // for a rank-1 sret descriptor. Writes are to aligned offsets.
        unsafe {
            std::ptr::write(sret_buf.as_mut_ptr() as *mut *mut u8, allocated);
            std::ptr::write(sret_buf.as_mut_ptr().add(8) as *mut *mut u8, aligned);
            std::ptr::write(sret_buf.as_mut_ptr().add(24) as *mut i64, 4i64);
        }
        // SAFETY: `sret_buf` contains a valid rank-1 descriptor written above.
        let (got_alloc, got_aligned, sizes) = unsafe {
            read_sret_descriptor(&sret_buf, 1).expect("valid rank-1 sret")
        };
        assert_eq!(got_alloc, allocated);
        assert_eq!(got_aligned, aligned);
        assert_eq!(sizes, vec![4]);
    }

    #[test]
    fn test_read_sret_descriptor_slice_too_short() {
        let slice = vec![0u8; 10];
        // SAFETY: `read_sret_descriptor` validates the slice length internally
        // and returns Err for short slices.
        let result = unsafe { read_sret_descriptor(&slice, 2) };
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too short"));
    }

    #[test]
    fn test_read_sret_descriptor_null_aligned() {
        let sret_buf = vec![0u8; 40];
        // SAFETY: `read_sret_descriptor` validates non-null pointers internally
        // and returns Err for null aligned pointers.
        let result = unsafe { read_sret_descriptor(&sret_buf, 1) };
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("null"));
    }

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn checked_product_from_i64_doesnt_panic(
            sizes in proptest::collection::vec(any::<i64>(), 0..10)
        ) {
            let _ = checked_product_from_i64(&sizes);
        }

        #[test]
        fn checked_product_usize_doesnt_panic(
            sizes in proptest::collection::vec(any::<usize>(), 0..10)
        ) {
            let _ = checked_product_usize(&sizes);
        }

        #[test]
        fn checked_product_from_i64_matches_manual(
            a in 0i64..1000,
            b in 0i64..1000,
        ) {
            let result = checked_product_from_i64(&[a, b]);
            let expected = (a as usize).checked_mul(b as usize);
            assert_eq!(result, expected);
        }
    }
}
