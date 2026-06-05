//! Contract test: HAL execution pipeline precision.
//!
//! Verifies that the Executable trait + SFATensor infrastructure
//! correctly propagates tensor data through the execution pipeline.
//! Uses a MockExecutable to return known values.
//!
//! Independent of compiler — no compiled dylib required.

use crate::hal::traits;
use crate::hal::traits::Executable;
use crate::hal::sfa::SfaMemRef;

#[derive(Debug)]
struct MockExecutable {
    output_data: Vec<f32>,
    output_shape: Vec<i64>,
}

// SAFETY: MockExecutable is a testing-only struct with no internal state
// shared across threads. Send + Sync are safe to implement.
unsafe impl Send for MockExecutable {}
unsafe impl Sync for MockExecutable {}

impl traits::Executable for MockExecutable {
    fn execute(
        &self,
        _op_name: &str,
        _stream: &dyn traits::Stream,
        _inputs: &[SfaMemRef],
        outputs: &mut [SfaMemRef],
    ) -> Result<Vec<Vec<i64>>, anyhow::Error> {
        for output_sfa in outputs.iter_mut() {
            let n_bytes = output_sfa.byte_len();
            let n_floats = n_bytes / 4;
            let copy_len = self.output_data.len().min(n_floats);
            if copy_len > 0 {
                let dst = output_sfa.data_ptr() as *mut f32;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        self.output_data.as_ptr(),
                        dst,
                        copy_len,
                    );
                }
            }
        }
        Ok(vec![self.output_shape.clone()])
    }

    fn function_count(&self) -> usize { 1 }
}

#[test]
fn test_mock_executable_preserves_f32_data() {
    let expected = vec![1.5_f32, 3.5_f32, 5.5_f32, 7.5_f32];
    let mock = MockExecutable {
        output_data: expected.clone(),
        output_shape: vec![1, 4],
    };

    let buf = vec![0.0_f32; 4];
    let mut output = SfaMemRef::from_shape(
        buf.as_ptr() as *mut std::ffi::c_void,
        &[1_usize, 4],
        4,
    )
    .expect("create output sfa");

    let shapes = mock
        .execute("test_op", &crate::hal::cpu::CpuStream, &[], &mut [output])
        .expect("mock execute");

    assert_eq!(shapes.len(), 1);
    assert_eq!(shapes[0], vec![1, 4]);

    for (i, &expected_val) in expected.iter().enumerate() {
        let actual = buf[i];
        assert!(
            (actual - expected_val).abs() < 1e-6,
            "precision violation at index {}: actual={} expected={}",
            i, actual, expected_val,
        );
    }
}

#[test]
fn test_mock_executable_preserves_larger_buffer() {
    let expected: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
    let mock = MockExecutable {
        output_data: expected.clone(),
        output_shape: vec![10, 10],
    };

    let buf = vec![0.0_f32; 100];
    let mut output = SfaMemRef::from_shape(
        buf.as_ptr() as *mut std::ffi::c_void,
        &[10_usize, 10],
        4,
    )
    .expect("create output sfa");

    mock
        .execute("test_op", &crate::hal::cpu::CpuStream, &[], &mut [output])
        .expect("mock execute");

    for (i, &expected_val) in expected.iter().enumerate() {
        assert!(
            (buf[i] - expected_val).abs() < 1e-6,
            "precision violation at index {}: actual={} expected={}",
            i, buf[i], expected_val,
        );
    }
}
