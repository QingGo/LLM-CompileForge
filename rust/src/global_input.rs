//! Global input assembly — builds MemRef descriptors for `input_ids` and
//! `positions` buffers.
//!
//! Extracted from `compute_graph_runner.rs` and `executor.rs` to eliminate
//! ~80 lines of duplicate code that was identical in both files.

use std::ffi::c_void;

use crate::compute_graph::IOTensorDef;
use crate::hal::cpu::memref::MemRefDesc1;
use crate::hal::cpu::{MemRefDescAny, MemRefDesc2};

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
) -> Result<(MemRefDescAny, Vec<u8>), anyhow::Error> {
    let shape: Vec<usize> = io_def.shape.iter().map(|&d| d as usize).collect();
    let is_dynamic = shape.iter().any(|&d| d == 0);
    if is_dynamic {
        let rank = io_def.rank as usize;
        let data_source: &[u32] = if bi == 1 { positions } else { input_ids };
        match rank {
            1 => {
                let n_tokens = data_source.len();
                let raw: Vec<u8> = data_source
                    .iter()
                    .flat_map(|&v| (v as i64).to_ne_bytes())
                    .collect();
                let p = raw.as_ptr();
                let memref = MemRefDesc1 {
                    allocated: p as *mut c_void,
                    aligned: p as *mut c_void,
                    offset: 0,
                    sizes: [n_tokens as i64],
                    strides: [1],
                };
                return Ok((MemRefDescAny::R1(memref), raw));
            }
            2 => {
                let n_tokens = data_source.len() as i64;
                let raw: Vec<u8> = data_source
                    .iter()
                    .flat_map(|&v| (v as i64).to_ne_bytes())
                    .collect();
                let p = raw.as_ptr();
                let memref = MemRefDesc2 {
                    allocated: p as *mut c_void,
                    aligned: p as *mut c_void,
                    offset: 0,
                    sizes: [1, n_tokens],
                    strides: [n_tokens, 1],
                };
                return Ok((MemRefDescAny::R2(memref), raw));
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
    let raw: Vec<u8> = padded.iter().flat_map(|&v| v.to_ne_bytes()).collect();
    let p = raw.as_ptr();
    let memref = MemRefDesc2 {
        allocated: p as *mut c_void,
        aligned: p as *mut c_void,
        offset: 0,
        sizes: [shape[0] as i64, shape.get(1).copied().unwrap_or(1) as i64],
        strides: [shape.get(1).copied().unwrap_or(1) as i64, 1],
    };
    Ok((MemRefDescAny::R2(memref), raw))
}
