//! Matrix multiplication — BLAS wrapper.

// BLAS FFI declarations (same as hal_ops_cpu.rs)
#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemm(
        order: i32, transA: i32, transB: i32,
        m: i32, n: i32, k: i32,
        alpha: f32, a: *const f32, lda: i32,
        b: *const f32, ldb: i32,
        beta: f32, c: *mut f32, ldc: i32,
    );
}

#[cfg(not(target_os = "macos"))]
extern "C" {
    fn cblas_sgemm(
        order: i32, transA: i32, transB: i32,
        m: i32, n: i32, k: i32,
        alpha: f32, a: *const f32, lda: i32,
        b: *const f32, ldb: i32,
        beta: f32, c: *mut f32, ldc: i32,
    );
}

const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_NO_TRANS: i32 = 111;
const CBLAS_TRANS: i32 = 112;

/// Matrix multiplication: `C[M,N] = A[M,K] @ B[K,N]` (or `@ B.T[K,N]` when `transpose_b`).
///
/// All matrices are row-major. `a_shape` and `b_shape` are the full tensor shapes.
/// Batched dimensions (everything before the last two) are iterated automatically.
///
/// When `transpose_b` is true, B is stored as `[N, K]` and the computation is
/// `C[M,N] = A[M,K] @ B.T[K,N]` — i.e., the "N" dimension is read from `b_shape[-2]`.
pub fn matmul_blas(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    a_shape: &[i64],
    b_shape: &[i64],
    transpose_b: bool,
) -> Result<(), String> {
    if a_shape.len() < 2 || b_shape.len() < 2 {
        return Err(format!(
            "matmul: expected rank >= 2, got a={:?} b={:?}",
            a_shape, b_shape,
        ));
    }

    let m = a_shape[a_shape.len() - 2] as i32;
    let k = a_shape[a_shape.len() - 1] as i32;
    let (n, trans_b, ldb) = if transpose_b {
        let k_b = b_shape[b_shape.len() - 1] as i32;
        if k_b != k {
            (k_b, CBLAS_NO_TRANS, k_b)
        } else {
            (b_shape[b_shape.len() - 2] as i32, CBLAS_TRANS, k)
        }
    } else {
        (b_shape[b_shape.len() - 1] as i32, CBLAS_NO_TRANS, b_shape[b_shape.len() - 1] as i32)
    };

    if k == 0 || m == 0 || n == 0 {
        return Ok(());
    }

    let lda = k;
    let ldc = n;

    // Compute batch size: product of all leading dims (before last 2).
    // Use the maximum of A and B leading-dims product to handle rank promotion
    // where one operand has extra leading dims. When one operand has fewer
    // leading dims than the other, its data is broadcast (reused) across the
    // extra batches.
    let a_batch: usize = a_shape[..a_shape.len().saturating_sub(2)]
        .iter()
        .map(|&d| d as usize)
        .product();
    let b_batch: usize = b_shape[..b_shape.len().saturating_sub(2)]
        .iter()
        .map(|&d| d as usize)
        .product();
    let batch = a_batch.max(b_batch);

    let a_stride: usize = (m * k) as usize;
    // b_stride: per-batch stride for B. The BLAS call accesses exactly k*n
    // elements from B regardless of transpose flag, so stride is k*n.
    let b_stride: usize = (k * n) as usize;
    let out_stride: usize = (m * n) as usize;
    let fallback_to_naive = ldb < k.max(1);

    for batch_idx in 0..batch {
        // Broadcasting: when one operand has fewer batches, cycle through
        // its data (batch_idx % batch_size). Batching matches standard
        // tensor broadcasting semantics (NumPy/PyTorch).
        let a_offset = (batch_idx % a_batch.max(1)) * a_stride;
        let b_offset = (batch_idx % b_batch.max(1)) * b_stride;
        let out_offset = batch_idx * out_stride;

        // Bounds check: verify each operand has enough elements for at least
        // one full matrix slice at its wrapped offset. Broadcast offsets cycle
        // within their allocation, so they only need to hold one slice.
        if a_offset + a_stride > a.len() {
            return Err(format!(
                "matmul_blas: A has insufficient data (offset {} + stride {} > len {}); a_shape={:?}",
                a_offset, a_stride, a.len(), a_shape,
            ));
        }
        if b_offset + b_stride > b.len() {
            return Err(format!(
                "matmul_blas: B has insufficient data (offset {} + stride {} > len {}); b_shape={:?}",
                b_offset, b_stride, b.len(), b_shape,
            ));
        }
        if out_offset + out_stride > out.len() {
            return Err(format!(
                "matmul_blas: Out has insufficient data (offset {} + stride {} > len {})",
                out_offset, out_stride, out.len(),
            ));
        }

        if fallback_to_naive {
            matmul_naive(
                &a[a_offset..],
                &b[b_offset..],
                &mut out[out_offset..],
                m as usize, k as usize, n as usize, false,
            );
        } else {
            // SAFETY: cblas_sgemm reads exactly m*k elements from A starting
            // at a[a_offset], k*n elements from B starting at b[b_offset], and
            // writes m*n elements to C starting at out[out_offset]. Bounds are
            // verified above: a_offset+a_stride <= a.len(), same for B and out.
            unsafe {
                cblas_sgemm(
                    CBLAS_ROW_MAJOR,
                    CBLAS_NO_TRANS,
                    trans_b,
                    m, n, k,
                    1.0,
                    a[a_offset..].as_ptr(), lda,
                    b[b_offset..].as_ptr(), ldb,
                    0.0,
                    out[out_offset..].as_mut_ptr(), ldc,
                );
            }
        }
    }

    Ok(())
}

/// Pure-Rust matrix multiplication (for testing without BLAS).
/// `C[M,N] = A[M,K] @ B[K,N]`
pub fn matmul_naive(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize, trans_b: bool) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                let b_val = if trans_b {
                    b[j * k + p]  // B is stored as [N, K], access B[j, p]
                } else {
                    b[p * n + j]  // B is stored as [K, N]
                };
                sum += a[i * k + p] * b_val;
            }
            out[i * n + j] = sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_naive_2x2() {
        // [1,2; 3,4] @ [5,6; 7,8] = [19,22; 43,50]
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut out = [0.0; 4];
        matmul_naive(&a, &b, &mut out, 2, 2, 2, false);
        assert_eq!(out, [19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matmul_naive_identity() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let eye = [1.0, 0.0, 0.0, 1.0];
        let mut out = [0.0; 4];
        matmul_naive(&a, &eye, &mut out, 2, 2, 2, false);
        assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_matmul_naive_1x4_times_4x1() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut out = [0.0; 1];
        matmul_naive(&a, &b, &mut out, 1, 4, 1, false);
        assert!((out[0] - 70.0).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_blas_basic() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut out = [0.0; 4];
        matmul_blas(&a, &b, &mut out, &[2, 2], &[2, 2], false).unwrap();
        assert_eq!(out, [19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matmul_blas_transpose_b() {
        // A = [[1,2],[3,4]], B = [[5,6],[7,8]] stored as [2,2]
        // C = A @ B^T = [[1*5+2*6, 1*7+2*8],[3*5+4*6, 3*7+4*8]] = [[17,23],[39,53]]
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut out = [0.0; 4];
        matmul_blas(&a, &b, &mut out, &[2, 2], &[2, 2], true).unwrap();
        let expected = [17.0, 23.0, 39.0, 53.0];
        for (i, (o, e)) in out.iter().zip(expected.iter()).enumerate() {
            assert!((o - e).abs() < 1e-5, "out[{}]={}, expected={}", i, o, e);
        }
    }

    #[test]
    fn test_matmul_blas_transpose_b_3x2_times_2x3() {
        // A[3,2] @ B^T[3,2] where B is stored as [3,2]
        // A = [[1,2],[3,4],[5,6]]  B = [[7,8],[9,10],[11,12]]
        // B^T = [[7,9,11],[8,10,12]]
        // C = A @ B^T = [[1*7+2*8, 1*9+2*10, 1*11+2*12], [3*7+4*8, 3*9+4*10, 3*11+4*12], [5*7+6*8, 5*9+6*10, 5*11+6*12]]
        //   = [[23, 29, 35], [53, 67, 81], [83, 105, 127]]
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut out = [0.0; 9];
        matmul_blas(&a, &b, &mut out, &[3, 2], &[3, 2], true).unwrap();
        let expected = [23.0, 29.0, 35.0, 53.0, 67.0, 81.0, 83.0, 105.0, 127.0];
        for (i, (o, e)) in out.iter().zip(expected.iter()).enumerate() {
            assert!((o - e).abs() < 1e-5, "out[{}]={}, expected={}", i, o, e);
        }
    }

    #[test]
    fn test_matmul_blas_attention_narrow() {
        // Attention matmul: Q[4, 64] @ K^T[4, 64] = [4, 4]
        // B stored as NxK=[4,64] for transpose_b=true.
        // A[i,k] = (i+1) as f32, B[j,k] = (j+1) as f32
        // C[i,j] = 64 * (i+1) * (j+1)
        let m: usize = 4;
        let k: usize = 64;
        let n: usize = 4;

        let mut a = vec![0.0f32; m * k];
        let mut b = vec![0.0f32; n * k];
        for i in 0..m {
            for kk in 0..k {
                a[i * k + kk] = (i + 1) as f32;
            }
        }
        for j in 0..n {
            for kk in 0..k {
                b[j * k + kk] = (j + 1) as f32;
            }
        }

        let mut out = vec![0.0f32; m * n];
        matmul_blas(&a, &b, &mut out, &[4, 64], &[4, 64], true).unwrap();

        // Verify against naive matmul as oracle
        let mut expected = vec![0.0f32; m * n];
        matmul_naive(&a, &b, &mut expected, m, k, n, true);
        for idx in 0..(m * n) {
            let diff = (out[idx] - expected[idx]).abs();
            assert!(
                diff < 1e-4,
                "out[{}] = {}, expected = {}, diff = {}",
                idx, out[idx], expected[idx], diff
            );
        }
    }

    #[test]
    fn test_matmul_blas_transpose_64x4() {
        // Q[4, 64] @ K[64, 4] — simulates attention with K stored as [head_dim, seq].
        // K is already in [K=64, N=4] layout, so k_b≠k detection fires.
        let m: usize = 4;
        let k: usize = 64;
        let n: usize = 4;

        let mut q = vec![0.0f32; m * k];
        let mut k_mat = vec![0.0f32; k * n];
        for i in 0..m {
            for kk in 0..k {
                q[i * k + kk] = (i * k + kk + 1) as f32;
            }
        }
        for kk in 0..k {
            for j in 0..n {
                k_mat[kk * n + j] = (kk * n + j + 1) as f32;
            }
        }

        let mut out = vec![0.0f32; m * n];
        matmul_blas(&q, &k_mat, &mut out, &[4, 64], &[64, 4], true).unwrap();

        let mut expected = vec![0.0f32; m * n];
        matmul_naive(&q, &k_mat, &mut expected, m, k, n, false);
        for idx in 0..(m * n) {
            let diff = (out[idx] - expected[idx]).abs();
            assert!(
                diff < 1e-4,
                "out[{}] = {}, expected = {}, diff = {}",
                idx, out[idx], expected[idx], diff
            );
        }
    }
}
