//! Global input assembly — builds MemRef descriptors for `input_ids` and
//! `positions` buffers.
//!
//! Extracted from `compute_graph_runner.rs` and `executor.rs` to eliminate
//! ~80 lines of duplicate code that was identical in both files.

use std::ffi::c_void;

use crate::compute_graph::IOTensorDef;
use crate::hal::cpu::memref::MemRefDesc1;
use crate::hal::cpu::{MemRefDescAny, MemRefDesc2};
use crate::sfa_tensor::SFATensor;

/// Build a MemRef descriptor and raw buffer for a GlobalInput binding.
///
/// `bi` is the binding index (0 = input_ids, 1 = position_ids).
/// Returns `(desc, raw_buffer)` ready to be pushed into input_descs/_raw_buffers.
///
/// # Dynamic shapes
///
/// When any dimension in `io_def.shape` is 0 (the SFCF dynamic sentinel),
/// the descriptor size follows the actual number of tokens:
///
///   - Rank 1 → `MemRefDesc1` with `sizes=[n_tokens]`, `strides=[1]`
///   - Rank 2 → `MemRefDesc2` with `sizes=[1, n_tokens]`, `strides=[n_tokens, 1]`
///
/// # Static / padded shapes
///
/// When all dimensions are positive, the descriptor uses the fixed shape
/// from `io_def`. Input data is padded with zeros if shorter than the
/// expected number of elements.
pub fn fill_global_input(
    input_ids: &[u32],
    positions: &[u32],
    io_def: &IOTensorDef,
    bi: usize,
) -> Result<SFATensor, anyhow::Error> {
    let shape: Vec<usize> = io_def.shape.iter().map(|&d| d as usize).collect();
    let is_dynamic = shape.contains(&0);
    if is_dynamic {
        let rank = io_def.rank as usize;
        let data_source: &[u32] = if bi == 1 { positions } else { input_ids };
        match rank {
            1 => {
                let n_tokens = data_source.len();
                let data: Vec<i64> = data_source.iter().map(|&v| v as i64).collect();
                return Ok(SFATensor::from_vec_i64(data, vec![n_tokens]));
            }
            2 => {
                let n_tokens = data_source.len();
                let data: Vec<i64> = data_source.iter().map(|&v| v as i64).collect();
                return Ok(SFATensor::from_vec_i64(data, vec![1, n_tokens]));
            }
            r => anyhow::bail!(
                "fill_global_input: unsupported rank {} for dynamic \
                 GlobalInput (shape={:?})",
                r,
                shape,
            ),
        }
    }
    let data_source: &[u32] = if bi == 1 { positions } else { input_ids };
    let expected_numel: usize = crate::hal::cpu::sret::checked_product_usize(&shape)
        .ok_or_else(|| anyhow::anyhow!(
            "global_input shape overflow: {:?}", shape
        ))?;
    let n_tokens = data_source.len().min(expected_numel);
    let padded: Vec<i64> = (0..expected_numel)
        .map(|i| {
            if i < n_tokens {
                data_source[i] as i64
            } else {
                0i64
            }
        })
        .collect();
    let static_shape: Vec<usize> = vec![
        shape[0],
        shape.get(1).copied().unwrap_or(1),
    ];
    Ok(SFATensor::from_vec_i64(padded, static_shape))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_graph::IOTensorDef;
    use crate::sfa_tensor::SFATensorRawAny;

    #[test]
    fn test_fill_global_input_returns_sfa_tensor() {
        let io_def = IOTensorDef {
            rank: 1,
            shape: vec![3],
            consumed_internally: false,
        };
        let result = fill_global_input(&[1u32, 2, 3], &[0u32, 1, 2], &io_def, 0);
        assert!(result.is_ok());
        let tensor = result.unwrap();
        // Static path always uses rank 2 (matching old MemRefDesc2 behavior):
        // sizes=[shape[0], max(shape[1], 1)] for dylib compatibility.
        assert_eq!(tensor.rank(), 2);
        assert_eq!(tensor.elem_size, 8);
        assert_eq!(tensor.shape(), vec![3, 1]);
    }

    #[test]
    fn test_fill_global_input_i64_data() {
        let io_def = IOTensorDef {
            rank: 1,
            shape: vec![3],
            consumed_internally: false,
        };
        let result = fill_global_input(&[1u32, 2, 3], &[0u32, 1, 2], &io_def, 0);
        assert!(result.is_ok());
        let tensor = result.unwrap();
        match &tensor.raw {
            SFATensorRawAny::R2(r) => {
                let ptr = r.allocated as *const i64;
                let data = unsafe { std::slice::from_raw_parts(ptr, 3) };
                assert_eq!(data, &[1i64, 2, 3]);
            }
            _ => panic!("expected R2 tensor"),
        }
    }

    #[test]
    fn test_fill_global_input_dynamic_shape() {
        let io_def = IOTensorDef {
            rank: 1,
            shape: vec![0], // dynamic sentinel
            consumed_internally: false,
        };
        let result = fill_global_input(&[1u32, 2, 3, 4], &[0u32, 1, 2, 3], &io_def, 0);
        assert!(result.is_ok());
        let tensor = result.unwrap();
        assert_eq!(tensor.rank(), 1);
        assert_eq!(tensor.shape(), vec![4]);
    }
}
