//! Gather operation — embedding lookup / indexed load.

/// Gather rows from a weight table by i64 indices.
///
/// `weight_table` shape: `[vocab_size, embed_dim]`
/// `indices` as f32 bytes (reinterpreted as i64)
/// `output` shape: `[num_indices, embed_dim]`
pub fn gather_f32(
    weight_table: &[f32],
    indices_f32: &[f32],
    output: &mut [f32],
    embed_dim: usize,
) -> Result<(), String> {
    // Reinterpret f32 bytes as i64 indices
    let num_indices = indices_f32.len();
    for i in 0..num_indices {
        let idx = indices_f32[i] as i64;
        let src_start = (idx as usize) * embed_dim;
        let dst_start = i * embed_dim;
        if src_start + embed_dim > weight_table.len() {
            return Err(format!(
                "gather: index {} out of bounds (vocab_size={})",
                idx,
                weight_table.len() / embed_dim,
            ));
        }
        if dst_start + embed_dim > output.len() {
            return Err(format!(
                "gather: output overflow at index {} (output len={})",
                i,
                output.len(),
            ));
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
