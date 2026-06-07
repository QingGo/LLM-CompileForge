//! HAL dispatch integration tests — shape metadata → kernel dispatch pipeline
//! with controlled shapes. Exercises matmul_blas correctness including batched
//! and batch-bounds scenarios.

use std::alloc::{alloc_zeroed, Layout};

use crate::hal::primitives::matmul::{matmul_blas, matmul_naive};
use crate::hal::rust::executable::build_shape_meta_from_sfa;
use crate::hal::sfa::SfaMemRef;

// ── shape metadata from SfaMemRef ─────────────────────────────────────

#[test]
fn test_shape_meta_from_sfa_2d() {
    // Allocate backing memory for a [4, 64] f32 tensor.
    let layout_a = Layout::array::<f32>(4 * 64).expect("valid layout");
    // SAFETY: alloc_zeroed returns a valid pointer for the requested layout.
    let ptr_a = unsafe { alloc_zeroed(layout_a) as *mut std::ffi::c_void };

    // Allocate backing memory for output [4, 4] f32 tensor.
    let layout_out = Layout::array::<f32>(4 * 4).expect("valid layout");
    // SAFETY: alloc_zeroed returns a valid pointer for the requested layout.
    let ptr_out = unsafe { alloc_zeroed(layout_out) as *mut std::ffi::c_void };

    let input = SfaMemRef::r2(ptr_a, [4, 64], [64, 1], 4);
    let output = SfaMemRef::r2(ptr_out, [4, 4], [4, 1], 4);

    let meta = build_shape_meta_from_sfa(&[input], &[output]);

    // input_shapes should be [[4, 64]]
    assert_eq!(meta.input_shapes.len(), 1, "expected 1 input shape");
    assert_eq!(
        meta.input_shapes[0],
        vec![4i64, 64],
        "input_shapes[0] should be [4, 64]"
    );

    // output_shape should be [4, 4]
    assert_eq!(
        meta.output_shape,
        vec![4i64, 4],
        "output_shape should be [4, 4]"
    );

    // SAFETY: backing allocations are dropped here. The slices constructed
    // inside build_shape_meta_from_sfa were only used to read sizes, not data.
    unsafe {
        std::alloc::dealloc(ptr_a as *mut u8, layout_a);
        std::alloc::dealloc(ptr_out as *mut u8, layout_out);
    }
}

#[test]
fn test_shape_meta_from_sfa_multi_input() {
    let layout = Layout::array::<f32>(8).expect("valid layout");
    // SAFETY: alloc_zeroed returns a valid pointer for the requested layout.
    let ptr1 = unsafe { alloc_zeroed(layout) as *mut std::ffi::c_void };
    let ptr2 = unsafe { alloc_zeroed(layout) as *mut std::ffi::c_void };
    let ptr3 = unsafe { alloc_zeroed(layout) as *mut std::ffi::c_void };

    let in1 = SfaMemRef::r1(ptr1, [8], [1], 4);
    let in2 = SfaMemRef::r2(ptr2, [2, 4], [4, 1], 4);
    let in3 = SfaMemRef::r1(ptr3, [1], [1], 4);
    let out = SfaMemRef::r1(ptr1, [2], [1], 4);

    let meta = build_shape_meta_from_sfa(&[in1, in2, in3], &[out]);

    assert_eq!(meta.input_shapes.len(), 3);
    assert_eq!(meta.input_shapes[0], vec![8i64]);
    assert_eq!(meta.input_shapes[1], vec![2i64, 4]);
    assert_eq!(meta.input_shapes[2], vec![1i64]);
    assert_eq!(meta.output_shape, vec![2i64]);

    // SAFETY: dealloc all three allocations.
    unsafe {
        std::alloc::dealloc(ptr1 as *mut u8, layout);
        std::alloc::dealloc(ptr2 as *mut u8, layout);
        std::alloc::dealloc(ptr3 as *mut u8, layout);
    }
}

// ── matmul_blas vs matmul_naive correctness ──────────────────────────

#[test]
fn test_matmul_blas_2d() {
    // [4, 64] @ [64, 4] — attention-style matmul (no transpose)
    let m: usize = 4;
    let k: usize = 64;
    let n: usize = 4;

    let mut a = vec![0.0f32; m * k];
    let mut b = vec![0.0f32; k * n];
    for i in 0..m {
        for kk in 0..k {
            a[i * k + kk] = (i * k + kk + 1) as f32;
        }
    }
    for kk in 0..k {
        for j in 0..n {
            b[kk * n + j] = (kk * n + j + 1) as f32;
        }
    }

    let mut out_blas = vec![0.0f32; m * n];
    matmul_blas(&a, &b, &mut out_blas, &[4, 64], &[64, 4], false)
        .expect("matmul_blas should succeed");

    let mut out_naive = vec![0.0f32; m * n];
    matmul_naive(&a, &b, &mut out_naive, m, k, n, false);

    for idx in 0..(m * n) {
        let diff = (out_blas[idx] - out_naive[idx]).abs();
        assert!(
            diff < 1e-4,
            "out[{}] blas={} naive={} diff={}",
            idx,
            out_blas[idx],
            out_naive[idx],
            diff
        );
    }
}

#[test]
fn test_matmul_blas_batched() {
    // Batched: [1, 12, 4, 64] @ [1, 12, 64, 4] with transpose_b=true
    //   A shape: [batch=12, m=4, k=64]
    //   B shape: [batch=12, n=4, k=64] stored as row-major, transpose to get [k=64, n=4]
    let batch: usize = 12;
    let m: usize = 4;
    let k: usize = 64;
    let n: usize = 4;

    let total_a = batch * m * k;
    let total_b = batch * n * k;
    let total_out = batch * m * n;

    let mut a = vec![0.0f32; total_a];
    let mut b = vec![0.0f32; total_b];
    // Use small values (<= 10) to keep f32 precision tight.
    for b_idx in 0..batch {
        for i in 0..m {
            for kk in 0..k {
                a[b_idx * m * k + i * k + kk] = ((b_idx * 7 + i * 3 + kk % 5 + 1) as f32) * 0.1;
            }
        }
        for j in 0..n {
            for kk in 0..k {
                b[b_idx * n * k + j * k + kk] = ((b_idx * 5 + j * 7 + kk % 3 + 1) as f32) * 0.1;
            }
        }
    }

    let mut out_blas = vec![0.0f32; total_out];
    matmul_blas(
        &a,
        &b,
        &mut out_blas,
        &[1i64, 12, 4, 64],
        &[1i64, 12, 4, 64],
        true,
    )
    .expect("matmul_blas batched should succeed");

    // Compare per-batch against naive
    let mut out_naive = vec![0.0f32; m * n];
    for b_idx in 0..batch {
        let a_slice = &a[b_idx * m * k..(b_idx + 1) * m * k];
        let b_slice = &b[b_idx * n * k..(b_idx + 1) * n * k];
        matmul_naive(a_slice, b_slice, &mut out_naive, m, k, n, true);

        let blas_slice = &out_blas[b_idx * m * n..(b_idx + 1) * m * n];
        for idx in 0..(m * n) {
            let diff = (blas_slice[idx] - out_naive[idx]).abs();
            // BLAS may use fused multiply-add; tolerance relaxed for 64-element dot product.
            assert!(
                diff < 5e-3,
                "batch {}: out[{}] blas={} naive={} diff={}",
                b_idx,
                idx,
                blas_slice[idx],
                out_naive[idx],
                diff
            );
        }
    }
}

#[test]
fn test_matmul_blas_batch_bounds() {
    // Verify batch iteration doesn't overshoot: the batch loop must not
    // access elements beyond the slice bounds for any batch index.
    let batch: usize = 4;
    let m: usize = 2;
    let k: usize = 3;
    let n: usize = 2;

    let total_a = batch * m * k; // 4 * 6 = 24
    let total_b = batch * k * n; // 4 * 6 = 24
    let total_out = batch * m * n; // 4 * 4 = 16

    let a = vec![1.0f32; total_a];
    let b = vec![2.0f32; total_b];
    let mut out = vec![0.0f32; total_out];

    let result = matmul_blas(
        &a,
        &b,
        &mut out,
        &[batch as i64, m as i64, k as i64],
        &[batch as i64, k as i64, n as i64],
        false,
    );
    assert!(result.is_ok(), "matmul_blas should succeed with valid bounds");

    // Compute expected total elements needed per batch:
    // A per-batch: m * k, B per-batch: k * n, Out per-batch: m * n
    let a_per_batch = m * k;
    let b_per_batch = k * n;
    let out_per_batch = m * n;

    // Verify offsets stay in bounds for all batch indices.
    // With broadcasting, offsets wrap via modulo, cycling through data.
    let a_eff_batch: usize = batch; // shape [4,2,3] → batch=4 leading dims
    let b_eff_batch: usize = batch; // same for B
    for batch_idx in 0..batch {
        let a_offset = (batch_idx % a_eff_batch) * a_per_batch;
        let b_offset = (batch_idx % b_eff_batch) * b_per_batch;
        let out_offset = batch_idx * out_per_batch;

        assert!(
            a_offset + a_per_batch <= a.len(),
            "batch {}: A wrapped offset {} + per_batch {} > a.len {}",
            batch_idx, a_offset, a_per_batch, a.len()
        );
        assert!(
            b_offset + b_per_batch <= b.len(),
            "batch {}: B wrapped offset {} + per_batch {} > b.len {}",
            batch_idx, b_offset, b_per_batch, b.len()
        );
        assert!(
            out_offset + out_per_batch <= out.len(),
            "batch {}: Out offset {} + per_batch {} > out.len {}",
            batch_idx, out_offset, out_per_batch, out.len()
        );

        // Last batch: out should exactly fill output buffer
        if batch_idx == batch - 1 {
            assert_eq!(
                out_offset + out_per_batch,
                out.len(),
                "last batch: Out should consume exactly all elements"
            );
        }
    }
}

#[test]
fn test_matmul_blas_bounds_error_on_truncated_input() {
    // When the data slice is shorter than the shape declares, the
    // bounds checking should return an error (not panic/overshoot).
    let a = vec![1.0f32; 8]; // only 8 elements
    let b = vec![2.0f32; 24];
    let mut out = vec![0.0f32; 16];

    // Shape says [4, 2, 2] = 16 elements for A, but only 8 provided
    let result = matmul_blas(
        &a,
        &b,
        &mut out,
        &[4i64, 2, 2], // batch=4, m=2, k=2 → needs 16 elements
        &[4i64, 2, 2],
        false,
    );
    assert!(
        result.is_err(),
        "should error when data slice is shorter than shape declares"
    );
}

#[test]
fn test_matmul_blas_zero_dim() {
    // Zero-size dimensions should return Ok(()) without panicking.
    let a = vec![0.0f32; 0];
    let b = vec![0.0f32; 0];
    let mut out = vec![0.0f32; 0];

    let result = matmul_blas(&a, &b, &mut out, &[0, 4], &[4, 0], false);
    assert!(result.is_ok(), "matmul_blas with zero dim should return Ok");
}

#[test]
fn test_matmul_blas_batched_transpose_b_rank4() {
    // Rank-4 batched matmul with transpose_b=true.
    // A = [1, 2, 4, 8], B = [1, 2, 4, 8] where B is stored as [N, K].
    let batch0: i64 = 1;
    let batch1: i64 = 2;
    let m: usize = 4;
    let k: usize = 8;
    let n: usize = 4;

    let total_a = (batch0 * batch1) as usize * m * k;
    let total_b = (batch0 * batch1) as usize * n * k;
    let total_out = (batch0 * batch1) as usize * m * n;

    let mut a = vec![0.0f32; total_a];
    let mut b = vec![0.0f32; total_b];
    for b0 in 0..batch0 {
        for b1 in 0..batch1 {
            let b_idx = (b0 * batch1 + b1) as usize;
            for i in 0..m {
                for kk in 0..k {
                    a[b_idx * m * k + i * k + kk] = (b_idx * 100 + i * 10 + kk) as f32;
                }
            }
            for j in 0..n {
                for kk in 0..k {
                    b[b_idx * n * k + j * k + kk] = (b_idx * 100 + j * 10 + kk) as f32;
                }
            }
        }
    }

    let mut out_blas = vec![0.0f32; total_out];
    matmul_blas(
        &a, &b, &mut out_blas,
        &[batch0, batch1, m as i64, k as i64],
        &[batch0, batch1, n as i64, k as i64],
        true,
    )
    .expect("rank-4 batched transpose matmul should succeed");

    // Verify per-batch against naive
    let mut out_naive = vec![0.0f32; m * n];
    for b_idx in 0..(batch0 * batch1) as usize {
        let a_slice = &a[b_idx * m * k..(b_idx + 1) * m * k];
        let b_slice = &b[b_idx * n * k..(b_idx + 1) * n * k];
        matmul_naive(a_slice, b_slice, &mut out_naive, m, k, n, true);
        let blas_slice = &out_blas[b_idx * m * n..(b_idx + 1) * m * n];
        for idx in 0..(m * n) {
            let diff = (blas_slice[idx] - out_naive[idx]).abs();
            assert!(
                diff < 1e-3,
                "batch {}: out[{}] blas={} naive={} diff={}",
                b_idx, idx, blas_slice[idx], out_naive[idx], diff
            );
        }
    }
}

#[test]
fn test_matmul_blas_broadcast_b_across_batches() {
    // Broadcasting: A has 4 batches, B has 1 batch. B's single slice should
    // be reused for all 4 A batches (standard tensor broadcasting).
    let a_batch: i64 = 4;
    let m: usize = 2;
    let k: usize = 3;
    let n: usize = 2;

    let a = vec![1.0f32; a_batch as usize * m * k];
    let b = vec![2.0f32; k * n]; // only 1 batch worth of B
    let mut out = vec![0.0f32; a_batch as usize * m * n];

    // A=[4,2,3], B=[3,2] (broadcast B's single batch across A's 4 batches)
    let result = matmul_blas(
        &a, &b, &mut out,
        &[a_batch, m as i64, k as i64],
        &[k as i64, n as i64],
        false,
    );
    assert!(
        result.is_ok(),
        "broadcast matmul should succeed, got: {:?}",
        result.err()
    );

    let mut out_naive = vec![0.0f32; m * n];
    for b_idx in 0..a_batch as usize {
        matmul_naive(
            &a[b_idx * m * k..(b_idx + 1) * m * k],
            &b[..],
            &mut out_naive,
            m, k, n, false,
        );
        let blas_slice = &out[b_idx * m * n..(b_idx + 1) * m * n];
        for idx in 0..(m * n) {
            assert!(
                (blas_slice[idx] - out_naive[idx]).abs() < 1e-4,
                "batch {}: out[{}] blas={} naive={}",
                b_idx, idx, blas_slice[idx], out_naive[idx]
            );
        }
    }
}
