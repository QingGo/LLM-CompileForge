//! Golden dylib contract tests — verify runtime FFI against hand-written
//! C kernels.  These tests do NOT depend on the compiler pipeline — only on
//! the memref descriptor layout contract (include/sfa.h).
//!
//! Each golden .dylib exports a _mlir_ciface_<name> function with the same
//! calling convention as MLIR-compiled dylibs.  The test allocates a large
//! enough sret buffer (descriptor + output data), constructs input
//! descriptors, calls the function, and checks the result against a reference
//! computed in pure Rust.
//!
//! Golden dylib sources: tests/data/golden/*.c
//! Build: make -f tests/data/golden/Makefile

#[cfg(test)]
mod golden_tests {
    use crate::hal::cpu::kernel::{CifaceFn1, CifaceFn3, CifaceFn4};
    use crate::hal::cpu::memref::{MemRefDesc1, MemRefDesc2, MemRefDesc3};

    const GOLDEN_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../tests/data/golden");

    /// Allocate a heap buffer for sret: descriptor bytes + output data bytes.
    /// Returns (buffer, descriptor pointer cast from the buffer start).
    fn sret_buf_2d(output_rows: usize, output_cols: usize) -> (Vec<u8>, *mut MemRefDesc2) {
        let desc_size = std::mem::size_of::<MemRefDesc2>();
        let data_size = output_rows * output_cols * std::mem::size_of::<f32>();
        let buf = vec![0u8; desc_size + data_size];
        let ptr = buf.as_ptr() as *mut MemRefDesc2;
        (buf, ptr)
    }

    fn sret_buf_3d(d0: usize, d1: usize, d2: usize) -> (Vec<u8>, *mut MemRefDesc3) {
        let desc_size = std::mem::size_of::<MemRefDesc3>();
        let data_size = d0 * d1 * d2 * std::mem::size_of::<f32>();
        let buf = vec![0u8; desc_size + data_size];
        let ptr = buf.as_ptr() as *mut MemRefDesc3;
        (buf, ptr)
    }

    // ── Matmul ────────────────────────────────────────────────────

    #[test]
    fn test_golden_matmul_2x3_times_3x2() {
        let dylib = format!("{}/matmul.dylib", GOLDEN_DIR);
        let lib = unsafe { libloading::Library::new(&dylib).expect("load matmul.dylib") };
        let func: libloading::Symbol<CifaceFn3> = unsafe {
            lib.get(b"_mlir_ciface_matmul_f32").expect("symbol")
        };

        let (_sret_buf, sret_ptr) = sret_buf_2d(2, 2);
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b: Vec<f32> = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let a_desc = MemRefDesc2::from_f32_slice(&a, [2, 3]);
        let b_desc = MemRefDesc2::from_f32_slice(&b, [3, 2]);

        unsafe {
            func(
                sret_ptr as *mut std::ffi::c_void,
                &a_desc as *const MemRefDesc2 as *const std::ffi::c_void,
                &b_desc as *const MemRefDesc2 as *const std::ffi::c_void,
            )
        };

        let out = unsafe { (*sret_ptr).read_output_f32() };
        let expected: [f32; 4] = [58.0, 64.0, 139.0, 154.0];
        assert_eq!(out.len(), 4);
        for (i, (&val, &exp)) in out.iter().zip(expected.iter()).enumerate() {
            assert!((val - exp).abs() < 1e-4, "matmul mismatch at [{}]", i);
        }
    }

    #[test]
    fn test_golden_matmul_identity() {
        let dylib = format!("{}/matmul.dylib", GOLDEN_DIR);
        let lib = unsafe { libloading::Library::new(&dylib).expect("load matmul.dylib") };
        let func: libloading::Symbol<CifaceFn3> = unsafe {
            lib.get(b"_mlir_ciface_matmul_f32").expect("symbol")
        };

        let (_sret_buf, sret_ptr) = sret_buf_2d(2, 2);
        let a: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
        let b: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
        let a_desc = MemRefDesc2::from_f32_slice(&a, [2, 2]);
        let b_desc = MemRefDesc2::from_f32_slice(&b, [2, 2]);

        unsafe {
            func(
                sret_ptr as *mut std::ffi::c_void,
                &a_desc as *const MemRefDesc2 as *const std::ffi::c_void,
                &b_desc as *const MemRefDesc2 as *const std::ffi::c_void,
            )
        };

        let out = unsafe { (*sret_ptr).read_output_f32() };
        let expected: [f32; 4] = [5.0, 6.0, 7.0, 8.0];
        for (i, (&val, &exp)) in out.iter().zip(expected.iter()).enumerate() {
            assert!((val - exp).abs() < 1e-5, "identity matmul mismatch at [{}]", i);
        }
    }

    // ── Layer Norm ────────────────────────────────────────────────

    #[test]
    fn test_golden_layer_norm_basic() {
        let dylib = format!("{}/layer_norm.dylib", GOLDEN_DIR);
        let lib = unsafe { libloading::Library::new(&dylib).expect("load layer_norm.dylib") };
        let func: libloading::Symbol<CifaceFn4> = unsafe {
            lib.get(b"_mlir_ciface_layer_norm_f32").expect("symbol")
        };

        let (_sret_buf, sret_ptr) = sret_buf_3d(2, 3, 2);
        let x: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let w: Vec<f32> = vec![1.0, 1.0];
        let b: Vec<f32> = vec![0.0, 0.0];

        let x_desc = MemRefDesc3::from_f32_slice(&x, [2, 3, 2]);
        let w_desc = MemRefDesc1::from_f32_slice(&w, [2]);
        let b_desc = MemRefDesc1::from_f32_slice(&b, [2]);

        unsafe {
            func(
                sret_ptr as *mut std::ffi::c_void,
                &x_desc as *const MemRefDesc3 as *const std::ffi::c_void,
                &w_desc as *const MemRefDesc1 as *const std::ffi::c_void,
                &b_desc as *const MemRefDesc1 as *const std::ffi::c_void,
            )
        };

        let out = unsafe { (*sret_ptr).read_output_f32() };
        assert_eq!(out.len(), 12);

        let eps = 1e-5f32;
        for bs in 0..2 {
            for s in 0..3 {
                let base = (bs * 3 + s) * 2;
                let mean = (x[base] + x[base + 1]) / 2.0;
                let var = ((x[base] - mean).powi(2) + (x[base + 1] - mean).powi(2)) / 2.0;
                for h in 0..2 {
                    let ref_val = (x[base + h] - mean) / (var + eps).sqrt();
                    let got = out[base + h];
                    assert!(
                        (got - ref_val).abs() < 1e-4,
                        "layer_norm mismatch at [{},{},{}]: expected {}, got {}",
                        bs, s, h, ref_val, got
                    );
                }
            }
        }
    }

    #[test]
    fn test_golden_layer_norm_with_scale() {
        let dylib = format!("{}/layer_norm.dylib", GOLDEN_DIR);
        let lib = unsafe { libloading::Library::new(&dylib).expect("load layer_norm.dylib") };
        let func: libloading::Symbol<CifaceFn4> = unsafe {
            lib.get(b"_mlir_ciface_layer_norm_f32").expect("symbol")
        };

        let (_sret_buf, sret_ptr) = sret_buf_3d(1, 1, 2);
        let x: Vec<f32> = vec![0.0, 4.0];
        let w: Vec<f32> = vec![2.0, 1.0];
        let b: Vec<f32> = vec![1.0, 0.0];

        let x_desc = MemRefDesc3::from_f32_slice(&x, [1, 1, 2]);
        let w_desc = MemRefDesc1::from_f32_slice(&w, [2]);
        let b_desc = MemRefDesc1::from_f32_slice(&b, [2]);

        unsafe {
            func(
                sret_ptr as *mut std::ffi::c_void,
                &x_desc as *const MemRefDesc3 as *const std::ffi::c_void,
                &w_desc as *const MemRefDesc1 as *const std::ffi::c_void,
                &b_desc as *const MemRefDesc1 as *const std::ffi::c_void,
            )
        };

        let out = unsafe { (*sret_ptr).read_output_f32() };
        assert!((out[0] + 1.0).abs() < 1e-4, "expected -1.0, got {}", out[0]);
        assert!((out[1] - 1.0).abs() < 1e-4, "expected 1.0, got {}", out[1]);
    }

    // ── Contract: sret round-trip (C writes → Rust reads) ────────

    #[test]
    fn test_sret_round_trip() {
        let dylib = format!("{}/sret_write.dylib", GOLDEN_DIR);
        let lib = unsafe { libloading::Library::new(&dylib).expect("load sret_write.dylib") };
        let func: libloading::Symbol<CifaceFn1> = unsafe {
            lib.get(b"_mlir_ciface_sret_write_f32").expect("symbol")
        };

        let (_sret_buf, sret_ptr) = sret_buf_2d(3, 4);

        unsafe {
            func(sret_ptr as *mut std::ffi::c_void);
        }

        let desc = unsafe { &*sret_ptr };
        assert!(!desc.allocated.is_null(), "allocated must be non-null");
        assert!(!desc.aligned.is_null(), "aligned must be non-null");
        assert_eq!(desc.sizes, [3, 4], "sizes");
        assert_eq!(desc.strides, [4, 1], "strides");

        // read_output_f32 uses aligned + offset to find the data start
        let out = unsafe { desc.read_output_f32() };
        assert_eq!(out.len(), 12, "output data length");
        for i in 0..12 {
            assert!(
                (out[i] - (i + 1) as f32).abs() < 1e-6,
                "sret data[{}]: expected {}, got {}", i, i + 1, out[i]
            );
        }
    }

    // ── Contract: MemRef descriptor layout ────────────────────────

    #[test]
    fn test_memref_desc2_sizeof() {
        assert_eq!(
            std::mem::size_of::<MemRefDesc2>(),
            56,
            "MemRefDesc2 must be 56 bytes"
        );
    }

    #[test]
    fn test_memref_desc3_sizeof() {
        assert_eq!(
            std::mem::size_of::<MemRefDesc3>(),
            72,
            "MemRefDesc3 must be 72 bytes"
        );
    }
}
