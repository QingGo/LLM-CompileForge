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
}
