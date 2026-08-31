//! Internal end-to-end tests that validate multi-function graph execution
//! (SSA wiring) using ONLY golden dylibs. No compiler, no HF, no safetensors.
//!
//! Each test manually sequences ciface calls to simulate the SSA edge from
//! one function's sret output to another function's input.
//!
//! Golden dylib sources: tests/data/golden/ssa_chain.c
//! Build: cc -O0 -fPIC -shared -o tests/data/golden/ssa_chain.dylib tests/data/golden/ssa_chain.c

#[cfg(test)]
mod internal_e2e_tests {
    use crate::hal::cpu::kernel::{CifaceFn2, CifaceFn3};
    use crate::hal::cpu::memref::MemRefDesc2;

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

    /// Test: two-function SSA chain — func_0 matmul output wires to func_1 add input.
    ///
    /// func_0 (_mlir_ciface_matmul_2x2): hard-coded 2×2 @ 2×2 → [[19,22],[43,50]]
    /// func_1 (_mlir_ciface_add_constant): adds 10.0 per element
    ///
    /// SSA edge: func_0's sret descriptor → func_1's input descriptor
    /// Expected final output: [29, 32, 53, 60]
    #[test]
    fn test_ssa_chain_matmul_to_add() {
    let _dylib_guard = crate::dylib_lock::lock();
        let dylib_path = format!("{}/ssa_chain.dylib", GOLDEN_DIR);

        // SAFETY: loading a golden dylib compiled from known-safe C source.
        let lib =
            unsafe { libloading::Library::new(&dylib_path).expect("load ssa_chain.dylib") };

        // SAFETY: symbol names are correct — verified via `nm`.
        let func_0: libloading::Symbol<CifaceFn3> = unsafe {
            lib.get(b"_mlir_ciface_matmul_2x2")
                .expect("symbol _mlir_ciface_matmul_2x2")
        };
        let func_1: libloading::Symbol<CifaceFn2> = unsafe {
            lib.get(b"_mlir_ciface_add_constant")
                .expect("symbol _mlir_ciface_add_constant")
        };

        // ── Step 1: Call func_0 ──────────────────────────────────────
        // Input A = [[1,2],[3,4]], Input B = [[5,6],[7,8]]
        // (the golden function ignores actual values, but we provide correct
        // slice data for descriptor construction)
        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let b_data: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
        let a_desc = MemRefDesc2::from_f32_slice(&a_data, [2, 2]);
        let b_desc = MemRefDesc2::from_f32_slice(&b_data, [2, 2]);

        // Allocate sret for func_0 output (56-byte descriptor + 4×f32 = 16 bytes data)
        let (_sret0_buf, sret0_ptr) = sret_buf_2d(2, 2);

        // SAFETY: calling ciface function with correctly-constructed sret + input descriptors.
        unsafe {
            func_0(
                sret0_ptr as *mut std::ffi::c_void,
                &a_desc as *const MemRefDesc2 as *const std::ffi::c_void,
                &b_desc as *const MemRefDesc2 as *const std::ffi::c_void,
            )
        };

        // Verify func_0 intermediate output
        let c_out = unsafe { (*sret0_ptr).read_output_f32() };
        let expected_c: [f32; 4] = [19.0, 22.0, 43.0, 50.0];
        assert_eq!(c_out.len(), 4, "func_0 output length");
        for (i, (&val, &exp)) in c_out.iter().zip(expected_c.iter()).enumerate() {
            assert!(
                (val - exp).abs() < 1e-5,
                "func_0 matmul mismatch at [{}]: expected {}, got {}",
                i,
                exp,
                val
            );
        }

        // ── Step 2: SSA wire — func_0 output → func_1 input ─────────
        // sret0_ptr holds the descriptor that func_0 wrote (allocated, aligned,
        // sizes, strides). Pass it directly as func_1's input descriptor.
        let (_sret1_buf, sret1_ptr) = sret_buf_2d(2, 2);

        // SAFETY: sret0_ptr contains a valid MemRefDesc2 written by func_0.
        // sret1_ptr is a zeroed buffer ready to receive func_1's output.
        unsafe {
            func_1(
                sret1_ptr as *mut std::ffi::c_void,
                sret0_ptr as *const std::ffi::c_void,
            )
        };

        // ── Step 3: Verify final output ──────────────────────────────
        let d_out = unsafe { (*sret1_ptr).read_output_f32() };
        let expected_d: [f32; 4] = [29.0, 32.0, 53.0, 60.0];
        assert_eq!(d_out.len(), 4, "func_1 output length");
        for (i, (&val, &exp)) in d_out.iter().zip(expected_d.iter()).enumerate() {
            assert!(
                (val - exp).abs() < 1e-5,
                "SSA chain mismatch at [{}]: expected {}, got {}",
                i,
                exp,
                val
            );
        }

        // _sret0_buf and _sret1_buf dropped here — dylib unloaded after.
        drop(lib);
    }

    /// Test: three-function SSA chain — matmul → add → relu.
    ///
    /// func_0 (_mlir_ciface_chain3_matmul): hard-coded [[1,0,0],[0,1,0]] @ [[1,2],[3,4],[5,6]]
    ///   → [[1,2],[3,4]]
    /// func_1 (_mlir_ciface_chain3_add): adds bias [[10,-10],[10,-10]] → [[11,-8],[13,-6]]
    /// func_2 (_mlir_ciface_chain3_relu): max(0, x) → [[11,0],[13,0]]
    ///
    /// Expected final output: [11.0, 0.0, 13.0, 0.0]
    #[test]
    fn test_ssa_chain_3func_matches_expected() {
    let _dylib_guard = crate::dylib_lock::lock();
        let dylib_path = format!("{}/ssa_chain_3func.dylib", GOLDEN_DIR);

        // SAFETY: loading a golden dylib compiled from known-safe C source.
        let lib =
            unsafe { libloading::Library::new(&dylib_path).expect("load ssa_chain_3func.dylib") };

        // SAFETY: symbol names are correct — verified via `nm`.
        let func_0: libloading::Symbol<CifaceFn3> = unsafe {
            lib.get(b"_mlir_ciface_chain3_matmul")
                .expect("symbol _mlir_ciface_chain3_matmul")
        };
        let func_1: libloading::Symbol<CifaceFn3> = unsafe {
            lib.get(b"_mlir_ciface_chain3_add")
                .expect("symbol _mlir_ciface_chain3_add")
        };
        let func_2: libloading::Symbol<CifaceFn2> = unsafe {
            lib.get(b"_mlir_ciface_chain3_relu")
                .expect("symbol _mlir_ciface_chain3_relu")
        };

        // ── Step 1: Call func_0 (matmul) ────────────────────────────
        let a_data: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let b_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a_desc = MemRefDesc2::from_f32_slice(&a_data, [2, 3]);
        let b_desc = MemRefDesc2::from_f32_slice(&b_data, [3, 2]);

        let (_sret0_buf, sret0_ptr) = sret_buf_2d(2, 2);

        // SAFETY: calling ciface function with correctly-constructed sret + input descriptors.
        unsafe {
            func_0(
                sret0_ptr as *mut std::ffi::c_void,
                &a_desc as *const MemRefDesc2 as *const std::ffi::c_void,
                &b_desc as *const MemRefDesc2 as *const std::ffi::c_void,
            )
        };

        let matmul_out = unsafe { (*sret0_ptr).read_output_f32() };
        let expected_matmul: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(matmul_out.len(), 4, "func_0 output length");
        for (i, (&val, &exp)) in matmul_out.iter().zip(expected_matmul.iter()).enumerate() {
            assert!(
                (val - exp).abs() < 1e-5,
                "func_0 matmul mismatch at [{}]: expected {}, got {}",
                i,
                exp,
                val
            );
        }

        // ── Step 2: SSA wire — func_0 output → func_1 input ─────────
        let (_sret1_buf, sret1_ptr) = sret_buf_2d(2, 2);
        let bias_desc = MemRefDesc2::from_f32_slice(&[10.0f32, -10.0, 10.0, -10.0], [2, 2]);

        // SAFETY: sret0_ptr contains a valid MemRefDesc2 written by func_0.
        unsafe {
            func_1(
                sret1_ptr as *mut std::ffi::c_void,
                sret0_ptr as *const std::ffi::c_void,
                &bias_desc as *const MemRefDesc2 as *const std::ffi::c_void,
            )
        };

        let add_out = unsafe { (*sret1_ptr).read_output_f32() };
        let expected_add: [f32; 4] = [11.0, -8.0, 13.0, -6.0];
        assert_eq!(add_out.len(), 4, "func_1 output length");
        for (i, (&val, &exp)) in add_out.iter().zip(expected_add.iter()).enumerate() {
            assert!(
                (val - exp).abs() < 1e-5,
                "func_1 add mismatch at [{}]: expected {}, got {}",
                i,
                exp,
                val
            );
        }

        // ── Step 3: SSA wire — func_1 output → func_2 input ─────────
        let (_sret2_buf, sret2_ptr) = sret_buf_2d(2, 2);

        // SAFETY: sret1_ptr contains a valid MemRefDesc2 written by func_1.
        unsafe {
            func_2(
                sret2_ptr as *mut std::ffi::c_void,
                sret1_ptr as *const std::ffi::c_void,
            )
        };

        let relu_out = unsafe { (*sret2_ptr).read_output_f32() };
        let expected_relu: [f32; 4] = [11.0, 0.0, 13.0, 0.0];
        assert_eq!(relu_out.len(), 4, "func_2 output length");
        for (i, (&val, &exp)) in relu_out.iter().zip(expected_relu.iter()).enumerate() {
            assert!(
                (val - exp).abs() < 1e-5,
                "SSA chain 3func mismatch at [{}]: expected {}, got {}",
                i,
                exp,
                val
            );
        }

        drop(lib);
    }

    /// Test: branch chain — func_0 output consumed by TWO downstream functions.
    ///
    /// func_0 (_mlir_ciface_fork_matmul): hard-coded [[1,2],[3,4]] @ [[0.1,0.2],[0.3,0.4]]
    ///   → [[0.7, 1.0], [1.5, 2.2]]
    /// func_1 (_mlir_ciface_fork_relu): passes through (all positive) → same as func_0
    /// func_2 (_mlir_ciface_fork_sigmoid): element-wise sigmoid
    ///
    /// Key property: func_0's sret descriptor is passed to BOTH func_1 AND func_2
    /// (fork SSA pattern with multiple consumers).
    #[test]
    fn test_ssa_chain_fork_matches_expected() {
    let _dylib_guard = crate::dylib_lock::lock();
        let dylib_path = format!("{}/ssa_chain_fork.dylib", GOLDEN_DIR);

        // SAFETY: loading a golden dylib compiled from known-safe C source.
        let lib =
            unsafe { libloading::Library::new(&dylib_path).expect("load ssa_chain_fork.dylib") };

        // SAFETY: symbol names are correct — verified via `nm`.
        let func_0: libloading::Symbol<CifaceFn3> = unsafe {
            lib.get(b"_mlir_ciface_fork_matmul")
                .expect("symbol _mlir_ciface_fork_matmul")
        };
        let func_1: libloading::Symbol<CifaceFn2> = unsafe {
            lib.get(b"_mlir_ciface_fork_relu")
                .expect("symbol _mlir_ciface_fork_relu")
        };
        let func_2: libloading::Symbol<CifaceFn2> = unsafe {
            lib.get(b"_mlir_ciface_fork_sigmoid")
                .expect("symbol _mlir_ciface_fork_sigmoid")
        };

        // ── Step 1: Call func_0 (matmul) ────────────────────────────
        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let b_data: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];
        let a_desc = MemRefDesc2::from_f32_slice(&a_data, [2, 2]);
        let b_desc = MemRefDesc2::from_f32_slice(&b_data, [2, 2]);

        let (_sret0_buf, sret0_ptr) = sret_buf_2d(2, 2);

        // SAFETY: calling ciface function with correctly-constructed sret + input descriptors.
        unsafe {
            func_0(
                sret0_ptr as *mut std::ffi::c_void,
                &a_desc as *const MemRefDesc2 as *const std::ffi::c_void,
                &b_desc as *const MemRefDesc2 as *const std::ffi::c_void,
            )
        };

        // Verify func_0 intermediate output
        let matmul_out = unsafe { (*sret0_ptr).read_output_f32() };
        let expected_matmul: [f32; 4] = [0.7, 1.0, 1.5, 2.2];
        assert_eq!(matmul_out.len(), 4, "func_0 output length");
        for (i, (&val, &exp)) in matmul_out.iter().zip(expected_matmul.iter()).enumerate() {
            assert!(
                (val - exp).abs() < 1e-5,
                "func_0 matmul mismatch at [{}]: expected {}, got {}",
                i,
                exp,
                val
            );
        }

        // ── Step 2: SSA fork — func_0 output → func_1 (relu) ────────
        let (_sret1_buf, sret1_ptr) = sret_buf_2d(2, 2);

        // SAFETY: sret0_ptr contains a valid MemRefDesc2 written by func_0.
        unsafe {
            func_1(
                sret1_ptr as *mut std::ffi::c_void,
                sret0_ptr as *const std::ffi::c_void,
            )
        };

        let relu_out = unsafe { (*sret1_ptr).read_output_f32() };
        // All values positive, ReLU is identity
        let expected_relu: [f32; 4] = [0.7, 1.0, 1.5, 2.2];
        assert_eq!(relu_out.len(), 4, "func_1 relu output length");
        for (i, (&val, &exp)) in relu_out.iter().zip(expected_relu.iter()).enumerate() {
            assert!(
                (val - exp).abs() < 1e-5,
                "func_1 relu mismatch at [{}]: expected {}, got {}",
                i,
                exp,
                val
            );
        }

        // ── Step 3: SSA fork — func_0 output → func_2 (sigmoid) ─────
        let (_sret2_buf, sret2_ptr) = sret_buf_2d(2, 2);

        // SAFETY: sret0_ptr contains a valid MemRefDesc2 written by func_0.
        // This is the same descriptor passed to func_1 — fork SSA pattern.
        unsafe {
            func_2(
                sret2_ptr as *mut std::ffi::c_void,
                sret0_ptr as *const std::ffi::c_void,
            )
        };

        let sigmoid_out = unsafe { (*sret2_ptr).read_output_f32() };
        let expected_sigmoid: [f32; 4] = [
            0.66818777,
            0.73105858,
            0.81757447,
            0.90024951,
        ];
        assert_eq!(sigmoid_out.len(), 4, "func_2 sigmoid output length");
        for (i, (&val, &exp)) in sigmoid_out.iter().zip(expected_sigmoid.iter()).enumerate() {
            assert!(
                (val - exp).abs() < 1e-5,
                "func_2 sigmoid mismatch at [{}]: expected {}, got {}",
                i,
                exp,
                val
            );
        }

        drop(lib);
    }

    /// Test: dynamic dimension SSA chain — verify SSA wiring works with
    /// descriptors that have resolved dynamic dimensions.
    ///
    /// func_0 (_mlir_ciface_dynamic_matmul): validates input shapes are [2,2],
    ///   then produces hard-coded output [[10,20],[30,40]].
    ///   This simulates dynamic dims being resolved before the call.
    /// func_1 (_mlir_ciface_dynamic_scale): multiplies by 2.5f
    ///   → [[25,50],[75,100]]
    ///
    /// Expected final output: [25.0, 50.0, 75.0, 100.0]
    #[test]
    fn test_ssa_chain_dynamic_matches_expected() {
    let _dylib_guard = crate::dylib_lock::lock();
        let dylib_path = format!("{}/ssa_chain_dynamic.dylib", GOLDEN_DIR);

        // SAFETY: loading a golden dylib compiled from known-safe C source.
        let lib =
            unsafe { libloading::Library::new(&dylib_path).expect("load ssa_chain_dynamic.dylib") };

        // SAFETY: symbol names are correct — verified via `nm`.
        let func_0: libloading::Symbol<CifaceFn3> = unsafe {
            lib.get(b"_mlir_ciface_dynamic_matmul")
                .expect("symbol _mlir_ciface_dynamic_matmul")
        };
        let func_1: libloading::Symbol<CifaceFn2> = unsafe {
            lib.get(b"_mlir_ciface_dynamic_scale")
                .expect("symbol _mlir_ciface_dynamic_scale")
        };

        // ── Step 1: Create descriptors with resolved shapes ──────────
        // Simulate dynamic dim resolution: initially dims were unknown (0),
        // now resolved to actual values [2,2] by the runtime.
        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let b_data: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
        let a_desc = MemRefDesc2::from_f32_slice(&a_data, [2, 2]);
        let b_desc = MemRefDesc2::from_f32_slice(&b_data, [2, 2]);

        // Verify the descriptors carry the resolved shape
        assert_eq!(
            a_desc.sizes,
            [2, 2],
            "Input A should have resolved dynamic dims to [2,2]"
        );
        assert_eq!(
            b_desc.sizes,
            [2, 2],
            "Input B should have resolved dynamic dims to [2,2]"
        );

        // ── Step 2: Call func_0 (matmul) ────────────────────────────
        let (_sret0_buf, sret0_ptr) = sret_buf_2d(2, 2);

        // SAFETY: calling ciface function with resolved dynamic dim descriptors.
        unsafe {
            func_0(
                sret0_ptr as *mut std::ffi::c_void,
                &a_desc as *const MemRefDesc2 as *const std::ffi::c_void,
                &b_desc as *const MemRefDesc2 as *const std::ffi::c_void,
            )
        };

        let matmul_out = unsafe { (*sret0_ptr).read_output_f32() };
        let expected_matmul: [f32; 4] = [10.0, 20.0, 30.0, 40.0];
        assert_eq!(matmul_out.len(), 4, "func_0 output length");
        for (i, (&val, &exp)) in matmul_out.iter().zip(expected_matmul.iter()).enumerate() {
            assert!(
                (val - exp).abs() < 1e-5,
                "func_0 matmul mismatch at [{}]: expected {}, got {}",
                i,
                exp,
                val
            );
        }

        // ── Step 3: SSA wire — func_0 output → func_1 input ─────────
        let (_sret1_buf, sret1_ptr) = sret_buf_2d(2, 2);

        // SAFETY: sret0_ptr contains a valid MemRefDesc2 written by func_0.
        unsafe {
            func_1(
                sret1_ptr as *mut std::ffi::c_void,
                sret0_ptr as *const std::ffi::c_void,
            )
        };

        let scale_out = unsafe { (*sret1_ptr).read_output_f32() };
        let expected_scale: [f32; 4] = [25.0, 50.0, 75.0, 100.0];
        assert_eq!(scale_out.len(), 4, "func_1 output length");
        for (i, (&val, &exp)) in scale_out.iter().zip(expected_scale.iter()).enumerate() {
            assert!(
                (val - exp).abs() < 1e-5,
                "SSA chain dynamic mismatch at [{}]: expected {}, got {}",
                i,
                exp,
                val
            );
        }

        drop(lib);
    }
}
