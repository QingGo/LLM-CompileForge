//! Reduce operations — sum, mean, max along axes.

/// Reduce sum along the last dimension.
/// `out[i] = sum(inp[i*last_dim .. (i+1)*last_dim])`
pub fn reduce_sum_last_dim(inp: &[f32], out: &mut [f32], last_dim: usize) {
    if last_dim == 0 {
        return;
    }
    let n = inp.len() / last_dim;
    for i in 0..n {
        let start = i * last_dim;
        let end = start + last_dim;
        out[i] = inp[start..end].iter().sum();
    }
}

/// Reduce mean along the last dimension.
pub fn reduce_mean_last_dim(inp: &[f32], out: &mut [f32], last_dim: usize) {
    if last_dim == 0 {
        return;
    }
    let n = inp.len() / last_dim;
    let inv = 1.0 / last_dim as f32;
    for i in 0..n {
        let start = i * last_dim;
        let end = start + last_dim;
        out[i] = inp[start..end].iter().sum::<f32>() * inv;
    }
}

/// Reduce max along the last dimension.
pub fn reduce_max_last_dim(inp: &[f32], out: &mut [f32], last_dim: usize) {
    if last_dim == 0 {
        return;
    }
    let n = inp.len() / last_dim;
    for i in 0..n {
        let start = i * last_dim;
        let end = start + last_dim;
        out[i] = inp[start..end].iter().copied().fold(f32::NEG_INFINITY, f32::max);
    }
}

/// Generic reduce along an arbitrary axis.
///
/// `outer` = product of dims before the axis
/// `reduce_size` = size of the axis being reduced
/// `inner` = product of dims after the axis
pub fn reduce_along_axis(
    inp: &[f32],
    out: &mut [f32],
    outer: usize,
    reduce_size: usize,
    inner: usize,
    kind: &str,
) {
    for oi in 0..outer {
        for ii in 0..inner {
            let base = oi * reduce_size * inner + ii;
            let val: f32 = match kind {
                "mean" => {
                    let sum: f32 = (0..reduce_size)
                        .map(|r| inp[base + r * inner])
                        .sum();
                    sum / reduce_size as f32
                }
                "max" => (0..reduce_size)
                    .map(|r| inp[base + r * inner])
                    .fold(f32::NEG_INFINITY, f32::max),
                _ => {
                    // Default: sum
                    (0..reduce_size)
                        .map(|r| inp[base + r * inner])
                        .sum()
                }
            };
            out[oi * inner + ii] = val;
        }
    }
}

/// Full reduction to scalar.
pub fn reduce_to_scalar(inp: &[f32], kind: &str) -> f32 {
    match kind {
        "mean" => inp.iter().sum::<f32>() / inp.len() as f32,
        "max" => inp.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        _ => inp.iter().sum(),
    }
}

/// Shape of operation — writes input shape dims as f32 values.
pub fn shape_of(input_shape: &[i64], output: &mut [f32]) {
    for (i, &dim) in input_shape.iter().enumerate() {
        if i < output.len() {
            output[i] = dim as f32;
        }
    }
}

/// Shape of operation — extracts a single dimension.
pub fn shape_of_with_dim(input_shape: &[i64], output: &mut [f32], dim: usize) {
    if dim < input_shape.len() {
        output[0] = input_shape[dim] as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduce_sum_last_dim() {
        let inp = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = [0.0; 2];
        reduce_sum_last_dim(&inp, &mut out, 3);
        assert_eq!(out, [6.0, 15.0]);
    }

    #[test]
    fn test_reduce_mean_last_dim() {
        let inp = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = [0.0; 2];
        reduce_mean_last_dim(&inp, &mut out, 3);
        assert!((out[0] - 2.0).abs() < 1e-6);
        assert!((out[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_max_last_dim() {
        let inp = [1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let mut out = [0.0; 2];
        reduce_max_last_dim(&inp, &mut out, 3);
        assert_eq!(out, [5.0, 6.0]);
    }

    #[test]
    fn test_reduce_along_axis_sum() {
        // 2x3 matrix, reduce along axis 1 (size 3)
        let inp = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = [0.0; 2];
        reduce_along_axis(&inp, &mut out, 2, 3, 1, "sum");
        assert_eq!(out, [6.0, 15.0]);
    }

    #[test]
    fn test_reduce_along_axis_mean() {
        let inp = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = [0.0; 2];
        reduce_along_axis(&inp, &mut out, 2, 3, 1, "mean");
        assert!((out[0] - 2.0).abs() < 1e-6);
        assert!((out[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_to_scalar_sum() {
        let inp = [1.0, 2.0, 3.0, 4.0];
        assert!((reduce_to_scalar(&inp, "sum") - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_to_scalar_mean() {
        let inp = [1.0, 2.0, 3.0, 4.0];
        assert!((reduce_to_scalar(&inp, "mean") - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_to_scalar_max() {
        let inp = [1.0, 5.0, 3.0, 2.0];
        assert!((reduce_to_scalar(&inp, "max") - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_shape_of() {
        let shape = [1i64, 4, 768];
        let mut out = [0.0; 3];
        shape_of(&shape, &mut out);
        assert_eq!(out, [1.0, 4.0, 768.0]);
    }

    #[test]
    fn test_shape_of_extracts_single_dim() {
        let shape = [1i64, 4, 768];
        let mut out = [0.0; 1];
        shape_of_with_dim(&shape, &mut out, 0);
        assert_eq!(out, [1.0]);
    }

    #[test]
    fn test_shape_of_extracts_seq_dim() {
        let shape = [1i64, 4, 768];
        let mut out = [0.0; 1];
        shape_of_with_dim(&shape, &mut out, 1);
        assert_eq!(out, [4.0]);
    }
}
