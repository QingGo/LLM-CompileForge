//! Bounded fp16 GEMV spike (Phase S0).
//!
//! Answers exactly one question: with fp16 weights (F16C convert-on-read)
//! on this machine and the existing KV contract, is 125 tok/s reachable?
//!
//! This module is **not** wired into `serveforge`'s inference path.  It
//! only provides the measured kernels for the standalone spike binary
//! (`fp16_gemv_spike`).  No ABI, CLI flag, or runtime contract changes.
//!
//! Kernel contract: `y = x @ W^T` with
//!   * `x`: row-major f32 `[m, k]` (the spike measures m=1),
//!   * `w`: row-major f16 `[n, k]`, read straight from a safetensors mmap,
//!   * `y`: row-major f32 `[m, n]`.
//!
//! The AVX2+F16C kernel widens 8 f16 weights at a time to f32 and uses
//! FMA.  Three independent accumulators cover 24 columns of `k` per
//! iteration (all real OPT shapes are divisible by 8).

use std::path::Path;

use half::f16;

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

const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_NO_TRANS: i32 = 111;
const CBLAS_TRANS: i32 = 112;

impl std::fmt::Display for F16TensorView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}, {}] f16", self.n, self.k)
    }
}

/// AVX2+F16C fp16-weight GEMV for a single row (`m == 1`).
///
/// Thin spike wrapper over the production [`crate::engine::gemv`] kernel so
/// standalone measurements and inference share exactly one implementation.
pub fn fp16_gemv_f16c(x: &[f32], n: usize, k: usize, w: &[u16]) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    crate::engine::gemv::gemv_f16_into(x, n, k, w, &mut out);
    out
}

/// Scalar reference used by tests and by very small shapes.
pub fn fp16_gemv_scalar(x: &[f32], n: usize, k: usize, w: &[u16]) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    crate::engine::gemv::gemv_f16_scalar_into(x, n, k, w, &mut out);
    out
}

/// f32 control group: the same `y = x @ W^T` through Accelerate SGEMM.
pub fn f32_gemv_blas(x: &[f32], n: usize, k: usize, w_f32: &[f32]) -> Vec<f32> {
    assert_eq!(x.len(), k, "x must have k elements");
    assert!(w_f32.len() >= n * k, "weight too small");
    let mut out = vec![0.0f32; n];
    if n == 0 || k == 0 {
        return out;
    }
    // SAFETY: x/w_f32/out are valid contiguous f32 slices that outlive the
    // call; the dimensions and leading dimensions satisfy the CBLAS
    // row-major contract for `[1,k] @ [n,k]^T`.
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            1,
            n as i32,
            k as i32,
            1.0f32,
            x.as_ptr(),
            k as i32,
            w_f32.as_ptr(),
            k as i32,
            0.0f32,
            out.as_mut_ptr(),
            n as i32,
        );
    }
    out
}

/// Multithreaded fp16 GEMV.  Disjoint row ranges write directly into the
/// shared output slice, so no reduction pass is needed.
pub fn fp16_gemv_threaded(
    x: &[f32],
    n: usize,
    k: usize,
    w: &[u16],
    threads: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    crate::engine::gemv::gemv_threaded_into(
        x,
        n,
        k,
        w,
        crate::model::tensor::Dtype::F16,
        threads,
        &mut out,
    );
    out
}

/// Threaded f32 control GEMV for vocabulary-size rows.
pub fn f32_gemv_blas_threaded(
    x: &[f32],
    n: usize,
    k: usize,
    w_f32: &[f32],
    threads: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    let threads = threads.clamp(1, 32);
    if threads == 1 || n < 64 {
        out.copy_from_slice(&f32_gemv_blas(x, n, k, w_f32));
        return out;
    }

    let chunk = n.div_ceil(threads);
    let out_slice = &mut out[..];
    let w_slice = &w_f32[..];
    let x_slice = &x[..];
    std::thread::scope(|scope| {
        for (idx, dst) in out_slice.chunks_mut(chunk).enumerate() {
            let start = idx * chunk;
            let count = dst.len();
            scope.spawn(move || {
                let partial = f32_gemv_blas(x_slice, count, k, &w_slice[start * k..]);
                dst.copy_from_slice(&partial);
            });
        }
    });
    out
}

// ── Safetensors f16 mmap reader (header parse only) ─────────────────

pub struct F16TensorView {
    pub data: Vec<u16>,
    pub n: usize,
    pub k: usize,
}

/// Load one `[n, k]` F16 tensor from a safetensors file into owned u16
/// storage.  The spike intentionally owns its data: it must be usable as a
/// standalone benchmark without borrowing the runtime's model plumbing.
pub fn load_f16_tensor(path: &Path, key: &str) -> Result<F16TensorView, anyhow::Error> {
    let file = std::fs::File::open(path)?;
    // SAFETY: read-only file mmap; immutable access for the lifetime of the
    // returned data (we copy the requested tensor before the mmap drops).
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    if mmap.len() < 8 {
        anyhow::bail!("safetensors file too short");
    }
    let header_len = u64::from_le_bytes(mmap[..8].try_into()?) as usize;
    if mmap.len() < 8 + header_len {
        anyhow::bail!("safetensors header truncated");
    }
    let header: serde_json::Value = serde_json::from_slice(&mmap[8..8 + header_len])?;
    let info = header
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("tensor key not found: {}", key))?;
    let shape: Vec<usize> = info
        .get("shape")
        .and_then(|s| s.as_array())
        .map(|a| a.iter().map(|v| v.as_u64().unwrap_or(1) as usize).collect())
        .unwrap_or_default();
    anyhow::ensure!(shape.len() == 2, "expected 2-D weight, got {:?}", shape);
    let dtype = info.get("dtype").and_then(|d| d.as_str()).unwrap_or("");
    anyhow::ensure!(dtype == "F16", "expected F16 tensor, got {}", dtype);
    let offsets = info
        .get("data_offsets")
        .and_then(|o| o.as_array())
        .ok_or_else(|| anyhow::anyhow!("missing data_offsets for {}", key))?;
    let start = offsets[0].as_u64().unwrap_or(0) as usize + 8 + header_len;
    let end = offsets[1].as_u64().unwrap_or(0) as usize + 8 + header_len;
    anyhow::ensure!(end <= mmap.len() && end >= start, "bad data range");
    let (n, k) = (shape[0], shape[1]);
    let numel = n.checked_mul(k).ok_or_else(|| anyhow::anyhow!("shape overflow"))?;
    anyhow::ensure!(
        end - start >= numel * 2,
        "tensor {} has {} bytes, expected {}",
        key,
        end - start,
        numel * 2
    );

    let mut data = vec![0u16; numel];
    for (idx, src) in mmap[start..end].chunks_exact(2).enumerate() {
        data[idx] = u16::from_le_bytes([src[0], src[1]]);
    }
    Ok(F16TensorView { data, n, k })
}

/// Convert an f16 `[n, k]` weight view to f32 for the BLAS control group.
pub fn f16_weight_to_f32(w: &[u16]) -> Vec<f32> {
    w.iter().map(|&h| f16::from_bits(h).to_f32()).collect()
}

/// Thin timing helper: returns the minimum wall time of `iters` runs.
pub fn min_elapsed_ms(
    mut f: impl FnMut(),
    warmups: usize,
    iters: usize,
) -> Result<f64, anyhow::Error> {
    anyhow::ensure!(iters > 0, "iters must be > 0");
    for _ in 0..warmups {
        f();
    }
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t0 = std::time::Instant::now();
        f();
        best = best.min(t0.elapsed().as_secs_f64() * 1e3);
    }
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp16_gemv_f16c_matches_scalar() {
        let k = 40usize;
        let n = 7usize;
        let x: Vec<f32> = (0..k).map(|i| (i as f32) * 0.25 - 3.0).collect();
        let w: Vec<u16> = (0..n * k)
            .map(|i| f16::from_f32(((i % 13) as f32) * 0.1 - 0.7).to_bits())
            .collect();
        let got = fp16_gemv_f16c(&x, n, k, &w);
        let want = fp16_gemv_scalar(&x, n, k, &w);
        for (g, wv) in got.iter().zip(want.iter()) {
            let tol = 1e-3f32.max(wv.abs() * 1e-3);
            assert!((g - wv).abs() <= tol, "got {g} want {wv}");
        }
    }

    #[test]
    fn fp16_gemv_threaded_matches_single_thread() {
        let k = 64usize;
        let n = 1024usize;
        let x: Vec<f32> = (0..k).map(|i| (i as f32) * 0.125).collect();
        let w: Vec<u16> = (0..n * k)
            .map(|i| f16::from_f32(((i % 17) as f32) - 8.0).to_bits())
            .collect();
        let single = fp16_gemv_f16c(&x, n, k, &w);
        for threads in [2usize, 4, 6] {
            let multi = fp16_gemv_threaded(&x, n, k, &w, threads);
            assert_eq!(multi.len(), single.len());
            for (m, s) in multi.iter().zip(single.iter()) {
                assert!((m - s).abs() <= 1e-2f32.max(s.abs() * 1e-3));
            }
        }
    }

    #[test]
    fn f32_blas_control_has_expected_shape_and_finite_values() {
        let k = 24usize;
        let n = 16usize;
        let x: Vec<f32> = vec![1.0; k];
        let w: Vec<f32> = vec![0.5; n * k];
        let out = f32_gemv_blas(&x, n, k, &w);
        assert_eq!(out.len(), n);
        for v in out {
            assert!((v - 12.0).abs() < 1e-3);
        }
    }
}
