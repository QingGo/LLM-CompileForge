//! Transpose operation — generic N-dimensional transpose.

/// Generic N-dimensional transpose.
///
/// `perm` is the permutation of dimensions. For example, `perm = [0, 2, 1]`
/// swaps dimensions 1 and 2.
pub fn transpose_nd(
    inp: &[f32],
    out: &mut [f32],
    input_shape: &[i64],
    output_shape: &[i64],
    perm: &[usize],
) -> Result<(), String> {
    let rank = input_shape.len();
    if rank < 2 {
        return Err(format!("transpose: expected rank >= 2, got {}", rank));
    }
    if perm.len() != rank {
        return Err(format!(
            "transpose: permutation length {} != rank {}",
            perm.len(),
            rank,
        ));
    }

    // Precompute strides for input
    let mut in_strides: Vec<usize> = vec![1; rank];
    for i in (0..rank - 1).rev() {
        in_strides[i] = in_strides[i + 1] * input_shape[i + 1] as usize;
    }

    // Precompute strides for output
    let mut out_strides: Vec<usize> = vec![1; rank];
    for i in (0..rank - 1).rev() {
        out_strides[i] = out_strides[i + 1] * output_shape[i + 1] as usize;
    }

    // Iterate over flat output and compute source index
    for flat_idx in 0..out.len() {
        let mut idx = flat_idx;
        let mut o_idx = vec![0usize; rank];
        for d in 0..rank {
            o_idx[d] = idx / out_strides[d];
            idx %= out_strides[d];
        }

        let mut i_idx = vec![0usize; rank];
        for d in 0..rank {
            i_idx[perm[d]] = o_idx[d];
        }

        let in_flat: usize = i_idx
            .iter()
            .zip(in_strides.iter())
            .map(|(&i, &s)| i * s)
            .sum();

        out[flat_idx] = inp[in_flat];
    }

    Ok(())
}

/// 2D matrix transpose: `out[j,i] = inp[i,j]`
pub fn transpose_2d(inp: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = inp[i * cols + j];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose_2d() {
        // [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
        let inp = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = [0.0; 6];
        transpose_2d(&inp, &mut out, 2, 3);
        assert_eq!(out, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_nd_swap_last_two() {
        // Shape [2, 3, 4], perm [0, 2, 1] -> output [2, 4, 3]
        let inp: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let mut out = vec![0.0f32; 24];
        transpose_nd(&inp, &mut out, &[2, 3, 4], &[2, 4, 3], &[0, 2, 1]).unwrap();

        // inp[b,s,h] -> out[b,h,s]
        // out[0,0,0] = inp[0,0,0] = 0
        assert_eq!(out[0], 0.0);
        // out[0,0,1] = inp[0,1,0] = 4
        assert_eq!(out[1], 4.0);
        // out[0,1,0] = inp[0,0,1] = 1
        assert_eq!(out[3], 1.0);
    }

    #[test]
    fn test_transpose_nd_identity() {
        let inp = [1.0, 2.0, 3.0, 4.0];
        let mut out = [0.0; 4];
        transpose_nd(&inp, &mut out, &[2, 2], &[2, 2], &[0, 1]).unwrap();
        assert_eq!(out, inp);
    }
}
