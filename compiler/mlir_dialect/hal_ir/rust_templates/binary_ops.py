"""Rust template constants for binary/unary/comparison ops.

Contains matmul, element_wise, and compare CPU function templates.
"""

from __future__ import annotations

OP_MATMUL = """\
// ── matmul ──────────────────────────────────────────────────────────

pub fn matmul_cpu(
    inputs: &[&[f32]],
    output: &mut [f32],
    shape_meta: &OpShapeMeta,
) -> Result<(), String> {
    let a = inputs[0];
    let b = inputs[1];

    let a_shape = shape_meta.input_shapes.get(0)
        .ok_or_else(|| "matmul: missing input shape 0".to_string())?;
    let b_shape = shape_meta.input_shapes.get(1)
        .ok_or_else(|| "matmul: missing input shape 1".to_string())?;

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

    // Row-major: A[M,K] @ B[K,N] = C[M,N]
    // lda = K, ldb = N, ldc = N
    let lda = k;
    let ldb = n;
    let ldc = n;

    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,   // order
            CBLAS_NO_TRANS,    // transA
            CBLAS_NO_TRANS,    // transB
            m, n, k,           // M, N, K
            1.0,               // alpha
            a.as_ptr(), lda,   // A, lda
            b.as_ptr(), ldb,   // B, ldb
            0.0,               // beta
            output.as_mut_ptr(), ldc, // C, ldc
        );
    }

    Ok(())
}
"""

OP_ELEMENT_WISE = """\
// ── element_wise ────────────────────────────────────────────────────

pub fn element_wise_cpu(
    inputs: &[&[f32]],
    output: &mut [f32],
    shape_meta: &OpShapeMeta,
) -> Result<(), String> {
    match shape_meta.kind.as_deref() {
        Some("add") => {
            for (o, (&a, &b)) in output.iter_mut().zip(inputs[0].iter().zip(inputs[1].iter())) {
                *o = a + b;
            }
        }
        Some("sub") => {
            for (o, (&a, &b)) in output.iter_mut().zip(inputs[0].iter().zip(inputs[1].iter())) {
                *o = a - b;
            }
        }
        Some("mul") => {
            for (o, (&a, &b)) in output.iter_mut().zip(inputs[0].iter().zip(inputs[1].iter())) {
                *o = a * b;
            }
        }
        Some("div") => {
            for (o, (&a, &b)) in output.iter_mut().zip(inputs[0].iter().zip(inputs[1].iter())) {
                *o = a / b;
            }
        }
        Some("relu") => {
            for (o, &a) in output.iter_mut().zip(inputs[0].iter()) {
                *o = if a > 0.0 { a } else { 0.0 };
            }
        }
        Some("rsqrt") => {
            for (o, &a) in output.iter_mut().zip(inputs[0].iter()) {
                *o = 1.0 / a.sqrt();
            }
        }
        Some("silu") => {
            for (o, &a) in output.iter_mut().zip(inputs[0].iter()) {
                // SiLU(x) = x * sigmoid(x)
                // sigmoid(x) = 1 / (1 + exp(-x))
                let sig = 1.0 / ((-a).exp() + 1.0);
                *o = a * sig;
            }
        }
        Some("gelu") => {
            for (o, &a) in output.iter_mut().zip(inputs[0].iter()) {
                // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                let x3 = a * a * a;
                let inner = (0.7978845608028654 * (a + 0.044715 * x3)).tanh();
                *o = 0.5 * a * (1.0 + inner);
            }
        }
        Some("neg") => {
            for (o, &a) in output.iter_mut().zip(inputs[0].iter()) {
                *o = -a;
            }
        }
        Some("exp") => {
            for (o, &a) in output.iter_mut().zip(inputs[0].iter()) {
                *o = a.exp();
            }
        }
        Some("tanh") => {
            for (o, &a) in output.iter_mut().zip(inputs[0].iter()) {
                *o = a.tanh();
            }
        }
        Some("sqrt") => {
            for (o, &a) in output.iter_mut().zip(inputs[0].iter()) {
                *o = a.sqrt();
            }
        }
        Some("sigmoid") => {
            for (o, &a) in output.iter_mut().zip(inputs[0].iter()) {
                *o = 1.0 / (1.0 + (-a).exp());
            }
        }
        Some("cos") => {
            for (o, &a) in output.iter_mut().zip(inputs[0].iter()) {
                *o = a.cos();
            }
        }
        Some("sin") => {
            for (o, &a) in output.iter_mut().zip(inputs[0].iter()) {
                *o = a.sin();
            }
        }
        Some("softplus") => {
            for (o, &a) in output.iter_mut().zip(inputs[0].iter()) {
                // softplus(x) = ln(1 + exp(x))
                *o = (1.0 + a.exp()).ln();
            }
        }
        Some("pow") => {
            // pow is binary: inputs[0]^inputs[1]
            for (o, (&a, &b)) in output.iter_mut().zip(inputs[0].iter().zip(inputs[1].iter())) {
                *o = a.powf(b);
            }
        }
        Some("max") => {
            for (o, (&a, &b)) in output.iter_mut().zip(inputs[0].iter().zip(inputs[1].iter())) {
                *o = a.max(b);
            }
        }
        other => {
            return Err(format!(
                "element_wise: unimplemented kind {:?}",
                other,
            ));
        }
    }
    Ok(())
}
"""

OP_COMPARE = """\
// ── compare ────────────────────────────────────────────────────────

pub fn compare_cpu(
    inputs: &[&[f32]],
    output: &mut [f32],
    shape_meta: &OpShapeMeta,
) -> Result<(), String> {
    match shape_meta.kind.as_deref() {
        Some("le") => {
            for (o, (&a, &b)) in output.iter_mut().zip(inputs[0].iter().zip(inputs[1].iter())) {
                *o = if a <= b { 1.0 } else { 0.0 };
            }
        }
        Some("lt") => {
            for (o, (&a, &b)) in output.iter_mut().zip(inputs[0].iter().zip(inputs[1].iter())) {
                *o = if a < b { 1.0 } else { 0.0 };
            }
        }
        Some("gt") => {
            for (o, (&a, &b)) in output.iter_mut().zip(inputs[0].iter().zip(inputs[1].iter())) {
                *o = if a > b { 1.0 } else { 0.0 };
            }
        }
        Some("ge") => {
            for (o, (&a, &b)) in output.iter_mut().zip(inputs[0].iter().zip(inputs[1].iter())) {
                *o = if a >= b { 1.0 } else { 0.0 };
            }
        }
        Some("eq") => {
            for (o, (&a, &b)) in output.iter_mut().zip(inputs[0].iter().zip(inputs[1].iter())) {
                *o = if (a - b).abs() < 1e-6 { 1.0 } else { 0.0 };
            }
        }
        Some("ne") => {
            for (o, (&a, &b)) in output.iter_mut().zip(inputs[0].iter().zip(inputs[1].iter())) {
                *o = if (a - b).abs() >= 1e-6 { 1.0 } else { 0.0 };
            }
        }
        Some("logical_and") => {
            for (o, (&a, &b)) in output.iter_mut().zip(inputs[0].iter().zip(inputs[1].iter())) {
                *o = if a != 0.0 && b != 0.0 { 1.0 } else { 0.0 };
            }
        }
        other => {
            return Err(format!(
                "compare: unimplemented kind {:?}",
                other,
            ));
        }
    }
    Ok(())
}
"""
