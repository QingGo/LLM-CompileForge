//! Precision contract tests — validate golden dylib ciface output against
//! ``precision_cases.pb`` element-wise.
//!
//! Each test case in the proto defines input tensors, weight tensors, and
//! expected output with a per-element ``max_abs_error`` threshold.  The test
//! loads the appropriate golden .dylib, constructs MemRef descriptors, calls
//! ciface directly, and compares every element.
//!
//! Golden dylib sources: tests/data/golden/*.c
//! Fixture:             tests/contract/fixtures/precision_cases.pb
//! Build dylibs:        make -f tests/data/golden/Makefile

#[cfg(test)]
mod precision_contract {
    use crate::hal::cpu::kernel::{CifaceFn3, CifaceFn4};
    use crate::hal::cpu::memref::{MemRefDesc1, MemRefDesc2, MemRefDesc3};
    use crate::model::abi::proto::{NumericalTestCase, PrecisionContract};
    use prost::Message;
    use std::io::Read;

    const FIXTURE_PATH: &str =
        concat!(env!("CARGO_MANIFEST_DIR"), "/../tests/contract/fixtures/precision_cases.pb");
    const GOLDEN_DIR: &str =
        concat!(env!("CARGO_MANIFEST_DIR"), "/../tests/data/golden");

    /// Load the precision contract from the shared protobuf fixture.
    fn load_contract() -> PrecisionContract {
        let mut file = std::fs::File::open(FIXTURE_PATH)
            .expect("precision_cases.pb not found — run: python tests/contract/fixtures/generate.py");
        let mut buf = Vec::new();
        file.read_to_end(&mut buf).expect("read precision_cases.pb");
        PrecisionContract::decode(buf.as_slice()).expect("decode PrecisionContract")
    }

    /// Find a specific test case by name in the contract.
    fn find_case<'a>(contract: &'a PrecisionContract, name: &str) -> &'a NumericalTestCase {
        contract
            .cases
            .iter()
            .find(|c| c.name == name)
            .unwrap_or_else(|| panic!("test case '{}' not found in precision_cases.pb", name))
    }

    /// Allocate sret buffer: descriptor bytes + output data bytes.
    /// Returns (buffer, descriptor raw pointer).
    fn sret_buf_2d(rows: usize, cols: usize) -> (Vec<u8>, *mut MemRefDesc2) {
        let desc_size = std::mem::size_of::<MemRefDesc2>();
        let data_size = rows * cols * std::mem::size_of::<f32>();
        let buf = vec![0u8; desc_size + data_size];
        let ptr = buf.as_ptr() as *mut MemRefDesc2;
        (buf, ptr)
    }

    /// Allocate sret buffer for multi-output rank-3: N descriptors + N data blocks.
    fn sret_buf_multi_3d(
        num_outputs: usize,
        d0: usize,
        d1: usize,
        d2: usize,
    ) -> (Vec<u8>, *mut std::ffi::c_void) {
        let desc_size = std::mem::size_of::<MemRefDesc3>();
        let data_size_per = d0 * d1 * d2 * std::mem::size_of::<f32>();
        let total = desc_size * num_outputs + data_size_per * num_outputs;
        let buf = vec![0u8; total];
        let ptr = buf.as_ptr() as *mut std::ffi::c_void;
        (buf, ptr)
    }

    /// Read a single rank-3 output descriptor from a given offset within the sret buffer.
    /// Returns (aligned_ptr, sizes).
    unsafe fn read_sret_desc3(buf: &[u8], offset: usize) -> (*mut f32, [i64; 3]) {
        let aligned =
            std::ptr::read_unaligned(buf.as_ptr().add(offset + 8) as *const *mut f32);
        assert!(!aligned.is_null(), "sret aligned ptr is null at offset {}", offset);
        let sizes: [i64; 3] = [
            std::ptr::read_unaligned(buf.as_ptr().add(offset + 24) as *const i64),
            std::ptr::read_unaligned(buf.as_ptr().add(offset + 32) as *const i64),
            std::ptr::read_unaligned(buf.as_ptr().add(offset + 40) as *const i64),
        ];
        (aligned, sizes)
    }

    /// Read f32 slice from a descriptor's aligned pointer.
    unsafe fn read_desc_f32(ptr: *mut f32, sizes: [i64; 3]) -> Vec<f32> {
        let numel: usize = sizes.iter().map(|&s| s.max(0) as usize).product();
        let slice = unsafe { std::slice::from_raw_parts(ptr, numel) };
        slice.to_vec()
    }

    // ── matmul_2x2_f32 ─────────────────────────────────────────────

    #[test]
    fn test_contract_matmul_2x2_f32() {
    let _dylib_guard = crate::dylib_lock::lock();
        let contract = load_contract();
        let case = find_case(&contract, "matmul_2x2_f32");

        let dylib_path = format!("{}/matmul.dylib", GOLDEN_DIR);
        let lib =
            unsafe { libloading::Library::new(&dylib_path).expect("load matmul.dylib") };
        let func: libloading::Symbol<CifaceFn3> =
            unsafe { lib.get(b"_mlir_ciface_matmul_f32").expect("symbol") };

        let input_rows = case.input_shape[0] as usize;
        let input_cols = case.input_shape[1] as usize;
        let weight_rows = case.weight_shape[0] as usize;
        let weight_cols = case.weight_shape[1] as usize;

        let (_sret_buf, sret_ptr) = sret_buf_2d(input_rows, weight_cols);

        let a_desc = MemRefDesc2::from_f32_slice(
            &case.input_data,
            [input_rows, input_cols],
        );
        let b_desc = MemRefDesc2::from_f32_slice(
            &case.weight_data,
            [weight_rows, weight_cols],
        );

        unsafe {
            func(
                sret_ptr as *mut std::ffi::c_void,
                &a_desc as *const MemRefDesc2 as *const std::ffi::c_void,
                &b_desc as *const MemRefDesc2 as *const std::ffi::c_void,
            );
        }

        let out = unsafe { (*sret_ptr).read_output_f32() };
        assert_eq!(
            out.len(),
            case.expected_output.len(),
            "output length mismatch"
        );

        let max_err = case.max_abs_error as f32;
        for (i, (&actual, &expected)) in
            out.iter().zip(case.expected_output.iter()).enumerate()
        {
            let diff = (actual - expected).abs();
            assert!(
                diff <= max_err,
                "matmul_2x2_f32 index {}: actual={}, expected={}, diff={}, max_err={}",
                i,
                actual,
                expected,
                diff,
                max_err
            );
        }
    }

    // ── rms_norm_2x4_f32 ───────────────────────────────────────────

    #[test]
    fn test_contract_rms_norm_2x4_f32() {
    let _dylib_guard = crate::dylib_lock::lock();
        let contract = load_contract();
        let case = find_case(&contract, "rms_norm_2x4_f32");

        let dylib_path = format!("{}/rms_norm.dylib", GOLDEN_DIR);
        let lib =
            unsafe { libloading::Library::new(&dylib_path).expect("load rms_norm.dylib") };
        let func: libloading::Symbol<CifaceFn3> =
            unsafe { lib.get(b"_mlir_ciface_rms_norm_f32").expect("symbol") };

        let input_dim0 = case.input_shape[0] as usize;
        let input_dim1 = case.input_shape[1] as usize;
        let weight_dim0 = case.weight_shape[0] as usize;

        let (_sret_buf, sret_ptr) = sret_buf_2d(input_dim0, input_dim1);

        let x_desc = MemRefDesc2::from_f32_slice(&case.input_data, [input_dim0, input_dim1]);
        let w_desc = MemRefDesc1::from_f32_slice(&case.weight_data, [weight_dim0]);

        unsafe {
            func(
                sret_ptr as *mut std::ffi::c_void,
                &x_desc as *const MemRefDesc2 as *const std::ffi::c_void,
                &w_desc as *const MemRefDesc1 as *const std::ffi::c_void,
            );
        }

        let out = unsafe { (*sret_ptr).read_output_f32() };
        assert_eq!(
            out.len(),
            case.expected_output.len(),
            "output length mismatch"
        );

        let max_err = case.max_abs_error as f32;
        for (i, (&actual, &expected)) in
            out.iter().zip(case.expected_output.iter()).enumerate()
        {
            let diff = (actual - expected).abs();
            assert!(
                diff <= max_err,
                "rms_norm_2x4_f32 index {}: actual={}, expected={}, diff={}, max_err={}",
                i,
                actual,
                expected,
                diff,
                max_err
            );
        }
    }

    // ── multi_out_ln_f32 ───────────────────────────────────────────

    #[test]
    fn test_contract_multi_out_ln_f32() {
    let _dylib_guard = crate::dylib_lock::lock();
        let contract = load_contract();
        let case = find_case(&contract, "multi_out_ln_f32");

        let dylib_path = format!("{}/layer_norm_multi.dylib", GOLDEN_DIR);
        let lib =
            unsafe { libloading::Library::new(&dylib_path).expect("load layer_norm_multi.dylib") };
        let func: libloading::Symbol<CifaceFn4> = unsafe {
            lib.get(b"_mlir_ciface_layer_norm_f32").expect("symbol")
        };

        // expected_shape = [2, 2, 4, 64] → 2 outputs of shape (2, 4, 64)
        let num_outputs = case.expected_shape[0] as usize;
        let out_d0 = case.expected_shape[1] as usize;
        let out_d1 = case.expected_shape[2] as usize;
        let out_d2 = case.expected_shape[3] as usize;

        // weight_shape = [2, 64] → gamma(64) + beta(64)
        let weight_dim1 = case.weight_shape[1] as usize;
        let gamma: Vec<f32> = case.weight_data[..weight_dim1].to_vec();
        let beta: Vec<f32> =
            case.weight_data[weight_dim1..weight_dim1 * 2].to_vec();

        let input_d0 = case.input_shape[0] as usize;
        let input_d1 = case.input_shape[1] as usize;
        let input_d2 = case.input_shape[2] as usize;

        let (_sret_buf, sret_ptr) = sret_buf_multi_3d(num_outputs, out_d0, out_d1, out_d2);

        let x_desc =
            MemRefDesc3::from_f32_slice(&case.input_data, [input_d0, input_d1, input_d2]);
        let w_desc = MemRefDesc1::from_f32_slice(&gamma, [weight_dim1]);
        let b_desc = MemRefDesc1::from_f32_slice(&beta, [weight_dim1]);

        unsafe {
            func(
                sret_ptr,
                &x_desc as *const MemRefDesc3 as *const std::ffi::c_void,
                &w_desc as *const MemRefDesc1 as *const std::ffi::c_void,
                &b_desc as *const MemRefDesc1 as *const std::ffi::c_void,
            );
        }

        // Read both outputs from the multi-output sret buffer
        let desc3_size = std::mem::size_of::<MemRefDesc3>(); // 72 bytes
        let mut combined_out = Vec::new();

        for out_idx in 0..num_outputs {
            let offset = out_idx * desc3_size;
            let (aligned, sizes) =
                unsafe { read_sret_desc3(&_sret_buf, offset) };
            let data = unsafe { read_desc_f32(aligned, sizes) };
            assert_eq!(
                data.len(),
                out_d0 * out_d1 * out_d2,
                "output {} length mismatch",
                out_idx
            );
            combined_out.extend_from_slice(&data);
        }

        assert_eq!(
            combined_out.len(),
            case.expected_output.len(),
            "combined output length mismatch"
        );

        let max_err = case.max_abs_error as f32;
        for (i, (&actual, &expected)) in
            combined_out.iter().zip(case.expected_output.iter()).enumerate()
        {
            let diff = (actual - expected).abs();
            assert!(
                diff <= max_err,
                "multi_out_ln_f32 index {}: actual={}, expected={}, diff={}, max_err={}",
                i,
                actual,
                expected,
                diff,
                max_err
            );
        }
    }
}
