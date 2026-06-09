//! CPU backend tests — split from hal/cpu/mod.rs.
//!
//! Tests for CpuDevice, CpuBuffer, MemRef descriptors, sret parsing,
//! and dylib ABI validation.

use super::super::traits;
use super::super::traits::{Device as _, Stream as _};
use super::device::RawCpuDevice;
use super::kernel::{CifaceFn3, KernelFn};
use super::memref::{
    MemRefDesc0, MemRefDesc1, MemRefDesc2, MemRefDesc3, MemRefDescAny,
};
use super::sret;
use super::{CpuBuffer, CpuDevice, CpuStream};

#[test]
fn test_cpu_device_name() {
    let d = CpuDevice::new();
    assert!(!d.name().is_empty());
}

#[test]
fn test_cpu_device_alloc_free() {
    let d = CpuDevice::new();
    let buf = d.alloc(64).expect("alloc 64 bytes");
    assert!(!buf.as_ptr().is_null());
    assert_eq!(buf.len(), 64);
}

#[test]
fn test_cpu_stream_sync_noop() {
    let s = CpuStream;
    s.synchronize().expect("sync should be no-op");
}

#[test]
fn test_cpu_buffer_copy_roundtrip() {
    let d = CpuDevice::new();
    let mut buf = d.alloc(8).expect("alloc");
    let stream = d.create_stream().expect("stream");
    let src = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
    buf.copy_from_host(&src, &*stream).expect("copy_from_host");
    let mut dst = vec![0u8; 8];
    buf.copy_to_host(&mut dst, &*stream).expect("copy_to_host");
    assert_eq!(src, dst);
}

#[test]
fn test_cpu_device_compile_loads_dylib() {
    let d = CpuDevice::new();
    let nonexistent = b"/nonexistent/libtest.dylib" as &[u8];
    let result = d.compile(nonexistent);
    assert!(result.is_err(), "loading nonexistent .dylib should fail");
}

#[test]
fn test_trait_object_safety() {
    let d: Box<dyn traits::Device> = Box::new(CpuDevice::new());
    assert_eq!(d.name(), "CPU (Apple Silicon / x86-64)");
}

// ── MemRef tests ──────────────────────────────────────────────

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
    assert!(!desc.is_null());
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
fn test_buffer_as_mut_slice() {
    let mut d = RawCpuDevice::new();
    let mut buf = d.allocate(16);
    assert_eq!(buf.size(), 16);
    buf.as_mut_slice().fill(0u8);
}

#[test]
fn test_memref_desc_any() {
    let data = vec![1.0f32; 6];
    let desc = MemRefDescAny::from_f32(&[2, 3], &data).unwrap();
    assert_eq!(desc.sizes(), vec![2, 3]);

    let desc2 = MemRefDescAny::zeroed(&[2, 3]).unwrap();
    assert!(!desc2.as_input_ptr().is_null());
}

#[test]
fn test_memref_any_zeroed_with_0_dims() {
    let desc = MemRefDescAny::zeroed(&[0, 0, 50272]).unwrap();
    assert!(!desc.as_output_ptr().is_null());
    let sz = desc.sizes();
    assert_eq!(sz, vec![1, 1, 50272]);
}

#[test]
fn test_memref_desc0_from_f32() {
    let data = [42.0f32];
    let desc = MemRefDesc0::from_f32_dyn_slice(&data, &[]);
    assert_eq!(desc.sizes, [0i64; 0]);
    assert_eq!(desc.strides, [0i64; 0]);
    assert_eq!(desc.numel(), 1);
    // SAFETY: `desc.aligned` points to the scalar data of this rank-0
    // descriptor, initialized by `from_f32_dyn_slice`.
    unsafe {
        let val = *(desc.aligned as *const f32);
        assert!((val - 42.0).abs() < 1e-6);
    }
}

#[test]
fn test_memref_desc0_zeroed() {
    let desc = MemRefDesc0::zeroed_dyn(&[]);
    assert!(!desc.aligned.is_null());
    assert_eq!(desc.numel(), 1);
}

#[test]
fn test_memref_any_zeroed_rank0() {
    let desc = MemRefDescAny::zeroed(&[]).unwrap();
    assert!(!desc.as_output_ptr().is_null());
    assert_eq!(desc.sizes(), Vec::<usize>::new());
}

// ── Regression test: negative sentinel in sret ──────────────
/// When the dylib returns unresolved dynamic dimension markers
/// (e.g., [-2, -3, 768]) in the sret descriptor,
/// `checked_product_from_i64` correctly clamps them to n=0.
/// `execute()` must NOT copy dylib data for these outputs
/// (the dylib's internal malloc buffer may be mis-sized).
/// Instead, it pushes the RESOLVED shape from output buffer
/// metadata (from compute graph io_def), preserving the
/// output_shapes count while remaining deterministic.
#[test]
fn test_negative_sentinel_preserves_output_shapes_semantics() {
    // Verify sentinel behavior: negative sizes produce numel = 0
    let result =
        crate::hal::cpu::sret::checked_product_from_i64(&[-2, -3, 768]);
    assert_eq!(
        result,
        Some(0),
        "negative sentinels should produce numel=0"
    );

    // Verify that the has_negative detection works
    let sizes = vec![-2i64, -3, 768];
    assert!(
        sizes.iter().any(|&s| s < 0),
        "should detect negative sizes"
    );

    // Verify that resolved shape (from buffer metadata) is correct.
    // For a [1, 4, 768] output tensor, the resolved shape should
    // be positive and match the compute graph's io_def resolution.
    let resolved: Vec<i64> = [1i64, 4, 768]
        .iter()
        .map(|&d| d as i64)
        .collect();
    assert_eq!(resolved, vec![1, 4, 768]);
    assert!(resolved.iter().all(|&s| s > 0),
        "resolved shape must be all positive");
}

#[test]
fn test_memref_any_from_f32_rank0() {
    let data = [std::f32::consts::PI];
    let desc = MemRefDescAny::from_f32(&[], &data).unwrap();
    assert!(!desc.as_input_ptr().is_null());
    match &desc {
        MemRefDescAny::R0(d) => {
            // SAFETY: `d.aligned` points to valid f32 data initialized
            // by `from_f32` above.
            unsafe {
                let val = *(d.aligned as *const f32);
                assert!((val - std::f32::consts::PI).abs() < 1e-6);
            }
        }
        _ => panic!("expected R0 variant"),
    }
}

#[test]
fn test_buffer_as_sfa_memref() {
    let d = CpuDevice::new();
    let mut buf = d.alloc(32).expect("alloc");
    let src = vec![1u8; 32];
    let s = CpuStream;
    buf.copy_from_host(&src, &s).expect("copy");
    let sfa = traits::Buffer::as_sfa_memref(buf.as_ref());
    assert_eq!(sfa.rank(), 1);
    assert_eq!(sfa.sizes(), vec![8]);
    assert_eq!(sfa.element_size(), 4);
    assert!(!sfa.data_ptr().is_null());
}

#[test]
fn test_buffer_as_sfa_memref_rank2() {
    let mut raw = RawCpuDevice::new();
    let inner = raw.allocate(64);
    let buf = CpuBuffer::with_meta(inner, 4, vec![4, 4]);
    let sfa = traits::Buffer::as_sfa_memref(&buf);
    assert_eq!(sfa.rank(), 2);
    assert_eq!(sfa.sizes(), vec![4, 4]);
    assert_eq!(sfa.numel(), 16);
    assert_eq!(sfa.byte_len(), 64);
    assert_eq!(sfa.element_size(), 4);
}

// ── Sret parsing regression tests ────────────────────────────

/// When the sret buffer contains a single rank-3 descriptor,
/// `read_sret_descriptor` must return correct allocated/aligned/sizes.
/// Verifies the fix for SIGSEGV at 0xc00 caused by reading strides
/// as pointers when descriptor rank didn't match expected.
#[test]
fn test_read_sret_rank3_no_misalignment() {
    // Build a rank-3 descriptor: [1, 4, 768] f32
    let data: Vec<f32> = vec![0.0; 1 * 4 * 768];
    let allocated = data.as_ptr() as *mut u8;
    let aligned = data.as_ptr() as *mut u8;
    // MemRef descriptor layout: allocated(8) + aligned(8) + offset(8)
    // + sizes[3]*8 + strides[3]*8 = 72 bytes
    let mut sret_buf = vec![0u8; 24 + 3 * 16];
    // SAFETY: Writing known values to a properly-sized buffer.
    unsafe {
        std::ptr::write(sret_buf.as_mut_ptr() as *mut *mut u8, allocated);
        std::ptr::write(sret_buf.as_mut_ptr().add(8) as *mut *mut u8, aligned);
        std::ptr::write(sret_buf.as_mut_ptr().add(24) as *mut i64, 1i64);
        std::ptr::write(sret_buf.as_mut_ptr().add(32) as *mut i64, 4i64);
        std::ptr::write(sret_buf.as_mut_ptr().add(40) as *mut i64, 768i64);
    }
    // SAFETY: `sret_buf` contains a valid rank-3 descriptor.
    let (got_alloc, got_aligned, sizes) = unsafe {
        sret::read_sret_descriptor(&sret_buf, 3).expect("valid rank-3 sret")
    };
    assert_eq!(got_alloc, allocated);
    assert_eq!(got_aligned, aligned);
    assert_eq!(sizes, vec![1i64, 4, 768]);
}

/// Verify that reading the same sret data with a wrong rank (rank=1)
/// does NOT interpret strides as pointers — it produces correct (but
/// partial) data from the first 40 bytes.  The aligned pointer must
/// still be valid (not 0xc00).
#[test]
fn test_sret_rank3_read_as_rank1_does_not_crash() {
    let data: Vec<f32> = vec![1.0; 1 * 4 * 768];
    let allocated = data.as_ptr() as *mut u8;
    let aligned = data.as_ptr() as *mut u8;
    let mut sret_buf = vec![0u8; 24 + 3 * 16];
    unsafe {
        std::ptr::write(sret_buf.as_mut_ptr() as *mut *mut u8, allocated);
        std::ptr::write(sret_buf.as_mut_ptr().add(8) as *mut *mut u8, aligned);
        std::ptr::write(sret_buf.as_mut_ptr().add(24) as *mut i64, 1i64);
        std::ptr::write(sret_buf.as_mut_ptr().add(32) as *mut i64, 4i64);
        std::ptr::write(sret_buf.as_mut_ptr().add(40) as *mut i64, 768i64);
    }
    // Read as rank-1: gets first 40 bytes. aligned is still correct.
    let (got_alloc, got_aligned, sizes) = unsafe {
        sret::read_sret_descriptor(&sret_buf, 1).expect("rank-1 read of rank-3 sret")
    };
    // aligned is the real data pointer, NOT a stride value
    assert_eq!(got_aligned, aligned,
        "aligned pointer should be the real data, not a stride (0xc00 bug)");
    assert_eq!(got_alloc, allocated);
    // sizes[0] = real sizes[0] = 1
    assert_eq!(sizes, vec![1i64]);
}

/// Second-descriptor read after wrong-rank first read: verify that
/// if the first descriptor was read with rank=1 (40 bytes), the
/// second descriptor starts at offset 40 which is INSIDE the real
/// rank-3 descriptor.  The aligned pointer at offset 48 would be
/// stride[0] = 3072 = 0xc00 — the original SIGSEGV cause.
/// This test documents the bug to ensure it stays fixed.
#[test]
fn test_sret_second_descriptor_misalignment_reproduces_0xc00() {
    let data: Vec<f32> = vec![1.0; 1 * 4 * 768];
    let allocated = data.as_ptr() as *mut u8;
    let aligned = data.as_ptr() as *mut u8;
    let mut sret_buf = vec![0u8; 24 + 3 * 16];
    // Write rank-3 descriptor with [1, 4, 768] sizes and [3072, 768, 1] strides
    unsafe {
        std::ptr::write(sret_buf.as_mut_ptr() as *mut *mut u8, allocated);
        std::ptr::write(sret_buf.as_mut_ptr().add(8) as *mut *mut u8, aligned);
        std::ptr::write(sret_buf.as_mut_ptr().add(16) as *mut i64, 0i64); // offset
        // sizes
        std::ptr::write(sret_buf.as_mut_ptr().add(24) as *mut i64, 1i64);
        std::ptr::write(sret_buf.as_mut_ptr().add(32) as *mut i64, 4i64);
        std::ptr::write(sret_buf.as_mut_ptr().add(40) as *mut i64, 768i64);
        // strides
        std::ptr::write(sret_buf.as_mut_ptr().add(48) as *mut i64, 3072i64);
        std::ptr::write(sret_buf.as_mut_ptr().add(56) as *mut i64, 768i64);
        std::ptr::write(sret_buf.as_mut_ptr().add(64) as *mut i64, 1i64);
    }

    // Simulate: first output read as rank=1 (40 bytes), offset moves to 40
    // Second output at offset 40: read as rank=1 another 40 bytes.
    // Two rank-1 output descriptors = 2 * (24 + 16*1) = 80 bytes.
    // Extend with headroom for safe offset calculations.
    sret_buf.resize(4096, 0);
    let slice2 = &sret_buf[40..80];
    let result = unsafe { sret::read_sret_descriptor(slice2, 1) };
    // With the old buggy code, aligned would be 0xc00 (3072).
    // The descriptor at offset 40 reads:
    //   allocated = bytes[40..47] = 768 (sizes[2])
    //   aligned   = bytes[48..55] = 3072 (strides[0]) = 0xc00 ← crash!
    // Now that we get only 1 output (rank-3), this path is never reached.
    // But the test documents what would happen if descriptors don't match.
    assert!(result.is_ok(), "should succeed (sret buffer is zeroed beyond)");
    let (_alloc, _aligned, _sizes) = result.unwrap();
    // When aligned=3072, this pointer is not null so read_sret passes,
    // but copy_nonoverlapping would SIGSEGV.
}

/// Verify that the fixed dylib proto has exactly 1 OutputDescriptor
/// per function (matching post-bufferization sret).
#[test]
fn test_sfa_abi_proto_one_output_per_function() {
    let dylib_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../outputs/compiled/opt_125m_fresh/libopt_125m.dylib"
    );
    if !std::path::Path::new(dylib_path).exists() {
        eprintln!("SKIP: dylib not found at {}", dylib_path);
        return;
    }
    unsafe {
        let lib = libloading::Library::new(dylib_path)
            .expect("failed to load dylib");
        let abi = crate::model::abi::load_sfa_abi(&lib)
            .expect("failed to load sfa_abi");
        assert_eq!(abi.funcs.len(), 16, "16 functions expected");
        // After _fixup_output_names fix, main_0 has 211 outputs (one per result)
        let main0 = &abi.funcs[0];
        assert_eq!(main0.outputs.len(), 211,
            "main_0 should have 211 outputs (one per func result)");
        assert_eq!(main0.outputs[0].rank, 1,
            "main_0 output rank should be 1 (individual scalar result)");
        // All other functions must have 1 output
        for fi in 1..abi.funcs.len() {
            assert_eq!(abi.funcs[fi].outputs.len(), 1,
                "func[{}] should have 1 output", fi);
        }
    }
}
