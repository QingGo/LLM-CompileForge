//! Production F16/BF16 weight GEMV kernels.
//!
//! Contract: `out = x @ W^T` for a single row `x` (`m == 1`), where
//! activations and outputs are F32 and the weight `W` is row-major
//! `[n, k]` stored in the source dtype.
//!
//! F16 uses AVX2+F16C+FMA convert-on-read; BF16 has no native ISA on this
//! machine and is emulated by widening the high 16 bits to float bits with
//! AVX2 (`u16 << 16`).  Both have the same scalar fallback shape, and the
//! threaded helpers split disjoint output-row ranges.

use half::f16;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// `out = x @ W^T`, F16 weights, scalar fallback.
pub(crate) fn gemv_f16_scalar_into(x: &[f32], n: usize, k: usize, w: &[u16], out: &mut [f32]) {
    assert_eq!(x.len(), k, "x must have k elements");
    assert!(w.len() >= n * k, "weight too small");
    assert!(out.len() >= n, "out too small");
    for (row, dst) in out.iter_mut().enumerate().take(n) {
        let mut acc = 0.0f32;
        for (kk, &xv) in x.iter().enumerate() {
            acc += xv * f16::from_bits(w[row * k + kk]).to_f32();
        }
        *dst = acc;
    }
}

/// `out = x @ W^T`, F16 weights, AVX2+F16C+FMA when available.
pub(crate) fn gemv_f16_into(x: &[f32], n: usize, k: usize, w: &[u16], out: &mut [f32]) {
    assert_eq!(x.len(), k, "x must have k elements");
    assert!(w.len() >= n * k, "weight too small");
    assert!(out.len() >= n, "out too small");

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("f16c")
            && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: feature detection above guarantees AVX2+F16C+FMA.
            unsafe { gemv_f16_avx2_inner(x, n, k, w, out) };
            return;
        }
    }
    gemv_f16_scalar_into(x, n, k, w, out);
}

/// `out = x @ W^T`, BF16 weights, scalar fallback.
pub(crate) fn gemv_bf16_scalar_into(x: &[f32], n: usize, k: usize, w: &[u16], out: &mut [f32]) {
    assert_eq!(x.len(), k, "x must have k elements");
    assert!(w.len() >= n * k, "weight too small");
    assert!(out.len() >= n, "out too small");
    for (row, dst) in out.iter_mut().enumerate().take(n) {
        let mut acc = 0.0f32;
        for (kk, &xv) in x.iter().enumerate() {
            acc += xv * bf16_to_f32(w[row * k + kk]);
        }
        *dst = acc;
    }
}

/// `out = x @ W^T`, BF16 weights, AVX2 bit-widening when available.
pub(crate) fn gemv_bf16_into(x: &[f32], n: usize, k: usize, w: &[u16], out: &mut [f32]) {
    assert_eq!(x.len(), k, "x must have k elements");
    assert!(w.len() >= n * k, "weight too small");
    assert!(out.len() >= n, "out too small");

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: feature detection above guarantees AVX2.
            unsafe { gemv_bf16_avx2_inner(x, n, k, w, out) };
            return;
        }
    }
    gemv_bf16_scalar_into(x, n, k, w, out);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn gemv_f16_avx2_inner(x: &[f32], n: usize, k: usize, w: &[u16], out: &mut [f32]) {
    // SAFETY: caller checked AVX2+F16C+FMA and slice lengths.
    unsafe {
        let w_bytes = w.as_ptr() as *const u8;
        let mut rb = 0usize;
        while rb + 8 <= n {
            let mut acc = [_mm256_setzero_ps(); 8];
            let mut kk = 0usize;
            while kk + 8 <= k {
                let xv = _mm256_loadu_ps(x.as_ptr().add(kk));
                for r in 0..8 {
                    let off = (rb + r) * k * 2 + kk * 2;
                    let wv = _mm256_cvtph_ps(_mm_loadu_si128(w_bytes.add(off) as *const __m128i));
                    acc[r] = _mm256_fmadd_ps(wv, xv, acc[r]);
                }
                kk += 8;
            }
            for r in 0..8 {
                let lo = acc[r];
                let hi = _mm256_extractf128_ps(lo, 1);
                let mut lo128 = _mm256_castps256_ps128(lo);
                lo128 = _mm_add_ps(lo128, hi);
                lo128 = _mm_hadd_ps(lo128, lo128);
                lo128 = _mm_hadd_ps(lo128, lo128);
                *out.get_unchecked_mut(rb + r) = _mm_cvtss_f32(lo128);
                for (kkk, &xv) in x.iter().enumerate().skip(kk) {
                    *out.get_unchecked_mut(rb + r) +=
                        xv * f16::from_bits(*w.get_unchecked((rb + r) * k + kkk)).to_f32();
                }
            }
            rb += 8;
        }
        for row in rb..n {
            let mut acc = 0.0f32;
            for (kkk, &xv) in x.iter().enumerate() {
                acc += xv * f16::from_bits(w[row * k + kkk]).to_f32();
            }
            *out.get_unchecked_mut(row) = acc;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn gemv_bf16_avx2_inner(x: &[f32], n: usize, k: usize, w: &[u16], out: &mut [f32]) {
    // SAFETY: caller checked AVX2 and slice lengths.
    unsafe {
        let w_bytes = w.as_ptr() as *const u8;
        let mut rb = 0usize;
        while rb + 8 <= n {
            let mut acc = [_mm256_setzero_ps(); 8];
            let mut kk = 0usize;
            while kk + 8 <= k {
                let xv = _mm256_loadu_ps(x.as_ptr().add(kk));
                for r in 0..8 {
                    let off = (rb + r) * k * 2 + kk * 2;
                    let epi = _mm256_cvtepu16_epi32(_mm_loadu_si128(w_bytes.add(off) as *const __m128i));
                    let shifted = _mm256_slli_epi32(epi, 16);
                    let wv = _mm256_castsi256_ps(shifted);
                    acc[r] = _mm256_add_ps(_mm256_mul_ps(wv, xv), acc[r]);
                }
                kk += 8;
            }
            for r in 0..8 {
                let lo = acc[r];
                let hi = _mm256_extractf128_ps(lo, 1);
                let mut lo128 = _mm256_castps256_ps128(lo);
                lo128 = _mm_add_ps(lo128, hi);
                lo128 = _mm_hadd_ps(lo128, lo128);
                lo128 = _mm_hadd_ps(lo128, lo128);
                *out.get_unchecked_mut(rb + r) = _mm_cvtss_f32(lo128);
                for (kkk, &xv) in x.iter().enumerate().skip(kk) {
                    *out.get_unchecked_mut(rb + r) +=
                        xv * bf16_to_f32(*w.get_unchecked((rb + r) * k + kkk));
                }
            }
            rb += 8;
        }
        for row in rb..n {
            let mut acc = 0.0f32;
            for (kkk, &xv) in x.iter().enumerate() {
                acc += xv * bf16_to_f32(w[row * k + kkk]);
            }
            *out.get_unchecked_mut(row) = acc;
        }
    }
}

/// Threaded F16/BF16 GEMV.  `threads` row chunks write disjoint ranges of
/// `out`, so no post-pass reduction is required.
pub(crate) fn gemv_threaded_into(
    x: &[f32],
    n: usize,
    k: usize,
    w: &[u16],
    dtype: crate::model::tensor::Dtype,
    threads: usize,
    out: &mut [f32],
) {
    let threads = threads.clamp(1, 32);
    if threads == 1 || n < 64 {
        match dtype {
            crate::model::tensor::Dtype::F16 => gemv_f16_into(x, n, k, w, out),
            crate::model::tensor::Dtype::BF16 => gemv_bf16_into(x, n, k, w, out),
            other => panic!("gemv_threaded_into: unsupported weight dtype {other:?}"),
        }
        return;
    }

    let chunk = n.div_ceil(threads);
    let x_slice = &x[..];
    let w_slice = &w[..];
    std::thread::scope(|scope| {
        for (idx, dst) in out[..n].chunks_mut(chunk).enumerate() {
            let start = idx * chunk;
            let count = dst.len();
            scope.spawn(move || match dtype {
                crate::model::tensor::Dtype::F16 => {
                    gemv_f16_into(x_slice, count, k, &w_slice[start * k..], dst)
                }
                crate::model::tensor::Dtype::BF16 => {
                    gemv_bf16_into(x_slice, count, k, &w_slice[start * k..], dst)
                }
                other => panic!("gemv_threaded_into: unsupported weight dtype {other:?}"),
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::tensor::Dtype;

    fn naive_f16(x: &[f32], n: usize, k: usize, w: &[u16]) -> Vec<f32> {
        let mut out = vec![0.0; n];
        gemv_f16_scalar_into(x, n, k, w, &mut out);
        out
    }

    fn naive_bf16(x: &[f32], n: usize, k: usize, w: &[u16]) -> Vec<f32> {
        let mut out = vec![0.0; n];
        gemv_bf16_scalar_into(x, n, k, w, &mut out);
        out
    }

    #[test]
    fn f16_avx_matches_scalar_on_non_multiple_shapes() {
        let n = 17usize;
        let k = 40usize;
        let x: Vec<f32> = (0..k).map(|i| (i as f32 - 19.0) * 0.03).collect();
        let w: Vec<u16> = (0..n * k)
            .map(|i| f16::from_f32(((i % 13) as f32 - 6.0) * 0.2).to_bits())
            .collect();
        let want = naive_f16(&x, n, k, &w);
        let mut got = vec![0.0; n];
        gemv_f16_into(&x, n, k, &w, &mut got);
        for (g, &s) in got.iter().zip(want.iter()) {
            let ok = (g.is_nan() && s.is_nan()) || (g - s).abs() <= 1e-3f32.max(s.abs() * 1e-3);
            assert!(ok, "got {g} want {s}");
        }
    }

    #[test]
    fn bf16_avx_matches_scalar_on_non_multiple_shapes() {
        let n = 19usize;
        let k = 33usize;
        let x: Vec<f32> = (0..k).map(|i| (i as f32 - 15.0) * 0.05).collect();
        let w: Vec<u16> = (0..n * k)
            .map(|i| ((i * 131) & 0xFFFF) as u16)
            .collect();
        let want = naive_bf16(&x, n, k, &w);
        let mut got = vec![0.0; n];
        gemv_bf16_into(&x, n, k, &w, &mut got);
        for (g, &s) in got.iter().zip(want.iter()) {
            let ok = (g.is_nan() && s.is_nan()) || (g - s).abs() <= 1e-3f32.max(s.abs() * 1e-3);
            assert!(ok, "got {g} want {s}");
        }
    }

    #[test]
    fn threaded_matches_single_thread_for_f16_and_bf16() {
        let n = 1024usize;
        let k = 64usize;
        let x: Vec<f32> = (0..k).map(|i| (i as f32 - 31.0) * 0.02).collect();
        let w16: Vec<u16> = (0..n * k)
            .map(|i| f16::from_f32(((i % 17) as f32 - 8.0) * 0.1).to_bits())
            .collect();
        let wbf: Vec<u16> = (0..n * k).map(|i| (i * 31) as u16).collect();

        for dtype in [Dtype::F16, Dtype::BF16] {
            let w = if dtype == Dtype::F16 { &w16 } else { &wbf };
            let mut single = vec![0.0f32; n];
            gemv_threaded_into(&x, n, k, w, dtype, 1, &mut single);
            for threads in [2usize, 4] {
                let mut multi = vec![0.0f32; n];
                gemv_threaded_into(&x, n, k, w, dtype, threads, &mut multi);
                for (idx, (&a, &b)) in single.iter().zip(multi.iter()).enumerate() {
                    let ok = a == b
                        || (a.is_nan() && b.is_nan())
                        || (a - b).abs() <= 1e-2f32.max(a.abs() * 1e-3);
                    assert!(
                        ok,
                        "dtype={dtype:?} threads={threads} mismatch at {idx}: {a} vs {b}"
                    );
                }
            }
        }
    }
}
