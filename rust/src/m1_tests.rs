//! M1 integration test: load a compiled .dylib and call a compute function.
//!
//! Verifies the Rust → compiled MLIR kernel FFI path works end-to-end:
//!   1. ``libloading`` opens the .dylib
//!   2. ``_mlir_ciface_add_two`` is looked up
//!   3. Input memref descriptors are constructed
//!   4. The function is called via the C ABI
//!   5. Output is read back from the result descriptor

#[cfg(test)]
mod m1_tests {
    use crate::hal_cpu::{CifaceFn3, MemRefDescriptor};

    const DYLIB_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../tests/data/test_m1.dylib");

    #[test]
    fn test_m1_add_two_via_dylib() {
        let lib = unsafe {
            libloading::Library::new(DYLIB_PATH).expect("failed to load test_m1.dylib")
        };

        let func: libloading::Symbol<CifaceFn3> = unsafe {
            lib.get(b"_mlir_ciface_add_two")
                .expect("symbol _mlir_ciface_add_two not found")
        };

        // Input data: a = all 1.0, b = all 2.0, expected = all 3.0
        let rows = 2usize;
        let cols = 64usize;
        let n = rows * cols;
        let a: Vec<f32> = vec![1.0; n];
        let b: Vec<f32> = vec![2.0; n];

        let a_desc = MemRefDescriptor::from_f32_slice(&a, rows, cols);
        let b_desc = MemRefDescriptor::from_f32_slice(&b, rows, cols);
        let mut out_desc = MemRefDescriptor::zeroed(rows, cols);

        // Call: result_ptr, in0_ptr, in1_ptr
        unsafe { func(&mut out_desc, &a_desc, &b_desc) };

        let out = unsafe { out_desc.read_output_f32() };
        assert_eq!(out.len(), n, "output size mismatch");
        for (i, &val) in out.iter().enumerate() {
            assert!(
                (val - 3.0).abs() < 1e-5,
                "mismatch at index {}: expected 3.0, got {}", i, val
            );
        }
    }
}
