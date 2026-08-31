//! Shared Accelerate/BLAS bridge.
//!
//! Exactly one module owns the ``cblas_sgemm`` FFI declaration so the
//! fused path (Phase 4) and the op-plan kernels (Phase 5) cannot drift on
//! the dominant arithmetic entry point.

#[cfg_attr(target_os = "macos", link(name = "Accelerate", kind = "framework"))]
extern "C" {
    #[link_name = "cblas_sgemm"]
    fn cblas_sgemm(
        order: i32,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

pub(crate) const CBLAS_ROW_MAJOR: i32 = 101;
pub(crate) const CBLAS_NO_TRANS: i32 = 111;
pub(crate) const CBLAS_TRANS: i32 = 112;

/// Safe wrapper over ``cblas_sgemm`` with the exact row-major contract the
/// dylib BLAS bridge uses.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sgemm(
    order: i32,
    transa: i32,
    transb: i32,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) {
    debug_assert_eq!(transa, CBLAS_NO_TRANS, "only NoTrans A is supported");
    if m == 0 || n == 0 || k == 0 {
        if beta == 0.0 {
            for v in c.iter_mut() {
                *v = 0.0;
            }
        } else {
            for v in c.iter_mut() {
                *v *= beta;
            }
        }
        return;
    }
    // SAFETY: a/b/c are contiguous f32 slices that outlive this call; the
    // caller passes leading dimensions that satisfy the CBLAS row-major
    // contract for the requested shapes.
    unsafe {
        cblas_sgemm(
            order,
            transa,
            transb,
            m as i32,
            n as i32,
            k as i32,
            alpha,
            a.as_ptr(),
            lda as i32,
            b.as_ptr(),
            ldb as i32,
            beta,
            c.as_mut_ptr(),
            ldc as i32,
        );
    }
}

/// Vec-growing variant of [`sgemm_transb`] for the fused layer workspace.
pub(crate) fn sgemm_transb_into(
    a: &[f32],
    m: usize,
    n: usize,
    k: usize,
    b: &[f32],
    out: &mut Vec<f32>,
) {
    out.clear();
    out.resize(m * n, 0.0);
    sgemm_transb(a, m, n, k, b, out);
}

/// `C = A @ B^T` where `A` is `[m,k]` row-major and `B` is stored as
/// `[n,k]` row-major.  Mirrors ``sfa_sgemm_transb`` exactly.
pub(crate) fn sgemm_transb(a: &[f32], m: usize, n: usize, k: usize, b: &[f32], out: &mut [f32]) {
    debug_assert!(a.len() >= m * k && b.len() >= n * k && out.len() >= m * n);
    sgemm(
        CBLAS_ROW_MAJOR,
        CBLAS_NO_TRANS,
        CBLAS_TRANS,
        m,
        n,
        k,
        1.0,
        a,
        k,
        b,
        k,
        0.0,
        out,
        n,
    );
}

/// `C = A @ B` for contiguous row-major matrices.
pub(crate) fn sgemm_nn(
    a: &[f32],
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    b: &[f32],
    ldb: usize,
    out: &mut [f32],
) {
    debug_assert!(a.len() >= m.saturating_sub(1).saturating_mul(lda) + k);
    debug_assert!(b.len() >= k.saturating_sub(1).saturating_mul(ldb) + n);
    debug_assert!(out.len() >= m * n);
    sgemm(
        CBLAS_ROW_MAJOR,
        CBLAS_NO_TRANS,
        CBLAS_NO_TRANS,
        m,
        n,
        k,
        1.0,
        a,
        lda,
        b,
        ldb,
        0.0,
        out,
        n,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sgemm_transb_matches_reference() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0, 7.0, 8.0, 9.0]; // [2,3]
        let mut out = [0.0f32; 2];
        sgemm_transb(&a, 1, 2, 3, &b, &mut out);
        assert_eq!(out, [32.0, 50.0]);
    }

    #[test]
    fn sgemm_nn_matches_reference() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2,3]
        let b = [1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0]; // [3,2]
        let mut out = [0.0f32; 4];
        sgemm_nn(&a, 2, 2, 3, 3, &b, 2, &mut out);
        assert_eq!(out, [4.0, 5.0, 10.0, 11.0]);
    }
}
