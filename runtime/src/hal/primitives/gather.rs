//! Gather operation — embedding lookup / indexed load.

use crate::model::tensor::Dtype;

/// Gather rows from a weight table by indices stored as raw bytes.
///
/// Supports both i64 and f32 indices based on dtype parameter.
pub fn gather_from_bytes(
    weight_table: &[f32],
    indices_bytes: &[u8],
    output: &mut [f32],
    embed_dim: usize,
    index_dtype: Dtype,
) -> Result<(), String> {
    let num_indices = match index_dtype {
        Dtype::I64 => indices_bytes.len() / 8,
        Dtype::F32 => indices_bytes.len() / 4,
        _ => return Err(format!("gather: unsupported index dtype {:?}", index_dtype)),
    };

    for i in 0..num_indices {
        let idx: i64 = match index_dtype {
            Dtype::I64 => {
                let bytes: [u8; 8] = indices_bytes[i * 8..(i + 1) * 8]
                    .try_into()
                    .map_err(|_| "gather: invalid i64 bytes")?;
                i64::from_le_bytes(bytes)
            }
            Dtype::F32 => {
                let bytes: [u8; 4] = indices_bytes[i * 4..(i + 1) * 4]
                    .try_into()
                    .map_err(|_| "gather: invalid f32 bytes")?;
                (f32::from_le_bytes(bytes) as i32).max(0) as i64
            }
            _ => unreachable!(),
        };

        let src_start = (idx as usize) * embed_dim;
        let dst_start = i * embed_dim;
        if src_start + embed_dim > weight_table.len() {
            return Err(format!("gather: index {} out of bounds", idx));
        }
        if dst_start + embed_dim > output.len() {
            return Err(format!("gather: output overflow at index {}", i));
        }
        output[dst_start..dst_start + embed_dim]
            .copy_from_slice(&weight_table[src_start..src_start + embed_dim]);
    }
    Ok(())
}

/// Gather rows from a weight table by f32 indices.
///
/// Each f32 index value is cast to i32 then clamped to >= 0 before
/// being used as a row offset into the weight table.
pub fn gather_f32(
    weight_table: &[f32],
    indices_f32: &[f32],
    output: &mut [f32],
    embed_dim: usize,
) -> Result<(), String> {
    let num_indices = indices_f32.len();
    for i in 0..num_indices {
        let idx: usize = (indices_f32[i] as i32).max(0) as usize;
        let src_start = idx * embed_dim;
        let dst_start = i * embed_dim;
        if src_start + embed_dim > weight_table.len() {
            return Err(format!("gather: index {} out of bounds", idx));
        }
        if dst_start + embed_dim > output.len() {
            return Err(format!("gather: output overflow at index {}", i));
        }
        output[dst_start..dst_start + embed_dim]
            .copy_from_slice(&weight_table[src_start..src_start + embed_dim]);
    }
    Ok(())
}

/// Gather with explicit i64 indices (for when indices are stored as i64 bytes).
pub fn gather_i64(
    weight_table: &[f32],
    indices: &[i64],
    output: &mut [f32],
    embed_dim: usize,
) -> Result<(), String> {
    for (i, &idx) in indices.iter().enumerate() {
        let src_start = (idx as usize) * embed_dim;
        let dst_start = i * embed_dim;
        if src_start + embed_dim > weight_table.len() {
            return Err(format!(
                "gather: index {} out of bounds",
                idx,
            ));
        }
        output[dst_start..dst_start + embed_dim]
            .copy_from_slice(&weight_table[src_start..src_start + embed_dim]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gather_f32_basic() {
        // 3x2 weight table, gather indices [1, 0]
        let weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let indices = [1.0, 0.0];
        let mut out = [0.0; 4];
        gather_f32(&weights, &indices, &mut out, 2).unwrap();
        assert_eq!(out, [3.0, 4.0, 1.0, 2.0]);
    }

    #[test]
    fn test_gather_f32_single() {
        let weights = [10.0, 20.0, 30.0, 40.0];
        let indices = [1.0];
        let mut out = [0.0; 2];
        gather_f32(&weights, &indices, &mut out, 2).unwrap();
        assert_eq!(out, [30.0, 40.0]);
    }

    #[test]
    fn test_gather_f32_out_of_bounds() {
        let weights = [1.0, 2.0, 3.0, 4.0];
        let indices = [5.0];
        let mut out = [0.0; 2];
        let result = gather_f32(&weights, &indices, &mut out, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_gather_i64_basic() {
        let weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let indices = [2i64, 0];
        let mut out = [0.0; 4];
        gather_i64(&weights, &indices, &mut out, 2).unwrap();
        assert_eq!(out, [5.0, 6.0, 1.0, 2.0]);
    }
}
