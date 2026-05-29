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

/// Matrix multiplication: `C[M,N] = A[M,K] @ B[K,N]`
///
/// All matrices are row-major. `a_shape` and `b_shape` are the full
/// tensor shapes; the last two dims are used for M, K, N.
pub fn matmul_blas(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    a_shape: &[i64],
    b_shape: &[i64],
) -> Result<(), String> {
    if a_shape.len() < 2 || b_shape.len() < 2 {
        return Err(format!(
            "matmul: expected rank >= 2, got a={:?} b={:?}",
            a_shape, b_shape,
        ));
    }

    let m = a_shape[a_shape.len() - 2] as i32;
    let k = a_shape[a_shape.len() - 1] as i32;
    let n = b_shape[b_shape.len() - 1] as i32;

    if k == 0 || m == 0 || n == 0 {
        return Ok(());
    }

    let lda = k;
    let ldb = n;
    let ldc = n;

    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            m, n, k,
            1.0,
            a.as_ptr(), lda,
            b.as_ptr(), ldb,
            0.0,
            out.as_mut_ptr(), ldc,
        );
    }

    Ok(())
}

/// Pure-Rust matrix multiplication (for testing without BLAS).
/// `C[M,N] = A[M,K] @ B[K,N]`
pub fn matmul_naive(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
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
        matmul_naive(&a, &b, &mut out, 2, 2, 2);
        assert_eq!(out, [19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matmul_naive_identity() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let eye = [1.0, 0.0, 0.0, 1.0];
        let mut out = [0.0; 4];
        matmul_naive(&a, &eye, &mut out, 2, 2, 2);
        assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_matmul_naive_1x4_times_4x1() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut out = [0.0; 1];
        matmul_naive(&a, &b, &mut out, 1, 4, 1);
        assert!((out[0] - 70.0).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_blas_basic() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut out = [0.0; 4];
        matmul_blas(&a, &b, &mut out, &[2, 2], &[2, 2]).unwrap();
        assert_eq!(out, [19.0, 22.0, 43.0, 50.0]);
    }
}
