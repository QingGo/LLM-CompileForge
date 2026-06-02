//! Vector operations — element-wise functions on f32 slices.

/// Element-wise addition: `out[i] = a[i] + b[i]`
pub fn vec_add(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len().min(b.len()).min(out.len());
    for i in 0..n {
        out[i] = a[i] + b[i];
    }
}

/// Element-wise subtraction: `out[i] = a[i] - b[i]`
pub fn vec_sub(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len().min(b.len()).min(out.len());
    for i in 0..n {
        out[i] = a[i] - b[i];
    }
}

/// Element-wise multiplication: `out[i] = a[i] * b[i]`
pub fn vec_mul(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len().min(b.len()).min(out.len());
    for i in 0..n {
        out[i] = a[i] * b[i];
    }
}

/// Element-wise division: `out[i] = a[i] / b[i]`
pub fn vec_div(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len().min(b.len()).min(out.len());
    for i in 0..n {
        out[i] = a[i] / b[i];
    }
}

/// Element-wise exp: `out[i] = exp(a[i])`
pub fn vec_exp(a: &[f32], out: &mut [f32]) {
    let n = a.len().min(out.len());
    for i in 0..n {
        out[i] = a[i].exp();
    }
}

/// Element-wise sqrt: `out[i] = sqrt(a[i])`
pub fn vec_sqrt(a: &[f32], out: &mut [f32]) {
    let n = a.len().min(out.len());
    for i in 0..n {
        out[i] = a[i].sqrt();
    }
}

/// Element-wise rsqrt: `out[i] = 1 / sqrt(a[i])`
pub fn vec_rsqrt(a: &[f32], out: &mut [f32]) {
    let n = a.len().min(out.len());
    for i in 0..n {
        out[i] = 1.0 / a[i].sqrt();
    }
}

/// Element-wise tanh: `out[i] = tanh(a[i])`
pub fn vec_tanh(a: &[f32], out: &mut [f32]) {
    let n = a.len().min(out.len());
    for i in 0..n {
        out[i] = a[i].tanh();
    }
}

/// Element-wise sigmoid: `out[i] = 1 / (1 + exp(-a[i]))`
pub fn vec_sigmoid(a: &[f32], out: &mut [f32]) {
    let n = a.len().min(out.len());
    for i in 0..n {
        out[i] = 1.0 / (1.0 + (-a[i]).exp());
    }
}

/// Element-wise ReLU: `out[i] = max(a[i], 0)`
pub fn vec_relu(a: &[f32], out: &mut [f32]) {
    let n = a.len().min(out.len());
    for i in 0..n {
        out[i] = if a[i] > 0.0 { a[i] } else { 0.0 };
    }
}

/// Element-wise negation: `out[i] = -a[i]`
pub fn vec_neg(a: &[f32], out: &mut [f32]) {
    let n = a.len().min(out.len());
    for i in 0..n {
        out[i] = -a[i];
    }
}

/// Element-wise cos: `out[i] = cos(a[i])`
pub fn vec_cos(a: &[f32], out: &mut [f32]) {
    let n = a.len().min(out.len());
    for i in 0..n {
        out[i] = a[i].cos();
    }
}

/// Element-wise sin: `out[i] = sin(a[i])`
pub fn vec_sin(a: &[f32], out: &mut [f32]) {
    let n = a.len().min(out.len());
    for i in 0..n {
        out[i] = a[i].sin();
    }
}

/// Element-wise softplus: `out[i] = ln(1 + exp(a[i]))`
pub fn vec_softplus(a: &[f32], out: &mut [f32]) {
    let n = a.len().min(out.len());
    for i in 0..n {
        out[i] = (1.0 + a[i].exp()).ln();
    }
}

/// Element-wise SiLU: `out[i] = a[i] * sigmoid(a[i])`
pub fn vec_silu(a: &[f32], out: &mut [f32]) {
    let n = a.len().min(out.len());
    for i in 0..n {
        let sig = 1.0 / (1.0 + (-a[i]).exp());
        out[i] = a[i] * sig;
    }
}

/// Element-wise GELU: `out[i] = 0.5 * a[i] * (1 + tanh(sqrt(2/pi) * (a[i] + 0.044715 * a[i]^3)))`
pub fn vec_gelu(a: &[f32], out: &mut [f32]) {
    let n = a.len().min(out.len());
    for i in 0..n {
        let x = a[i];
        let x3 = x * x * x;
        let inner = 0.7978845608028654 * (x + 0.044715 * x3);
        out[i] = 0.5 * x * (1.0 + inner.tanh());
    }
}

/// Element-wise pow: `out[i] = a[i] ^ b[i]`
pub fn vec_pow(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len().min(b.len()).min(out.len());
    for i in 0..n {
        out[i] = a[i].powf(b[i]);
    }
}

/// Element-wise max: `out[i] = max(a[i], b[i])`
pub fn vec_max(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len().min(b.len()).min(out.len());
    for i in 0..n {
        out[i] = a[i].max(b[i]);
    }
}

/// Scalar division in-place: `a[i] /= scalar`
pub fn vec_div_scalar_inplace(a: &mut [f32], scalar: f32) {
    let inv = 1.0 / scalar;
    for v in a.iter_mut() {
        *v *= inv;
    }
}

/// Copy: `out = a`
pub fn vec_copy(a: &[f32], out: &mut [f32]) {
    let n = a.len().min(out.len());
    out[..n].copy_from_slice(&a[..n]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_add() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let mut out = [0.0; 3];
        vec_add(&a, &b, &mut out);
        assert_eq!(out, [5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_vec_sub() {
        let a = [5.0, 7.0, 9.0];
        let b = [1.0, 2.0, 3.0];
        let mut out = [0.0; 3];
        vec_sub(&a, &b, &mut out);
        assert_eq!(out, [4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_vec_mul() {
        let a = [2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0];
        let mut out = [0.0; 3];
        vec_mul(&a, &b, &mut out);
        assert_eq!(out, [10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_vec_div() {
        let a = [10.0, 18.0, 28.0];
        let b = [2.0, 3.0, 4.0];
        let mut out = [0.0; 3];
        vec_div(&a, &b, &mut out);
        assert_eq!(out, [5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_vec_exp() {
        let a = [0.0, 1.0, 2.0];
        let mut out = [0.0; 3];
        vec_exp(&a, &mut out);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - std::f32::consts::E).abs() < 1e-5);
        assert!((out[2] - 7.389).abs() < 0.01);
    }

    #[test]
    fn test_vec_sqrt() {
        let a = [4.0, 9.0, 16.0];
        let mut out = [0.0; 3];
        vec_sqrt(&a, &mut out);
        assert_eq!(out, [2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_vec_rsqrt() {
        let a = [4.0, 9.0, 16.0];
        let mut out = [0.0; 3];
        vec_rsqrt(&a, &mut out);
        assert!((out[0] - 0.5).abs() < 1e-6);
        assert!((out[1] - 1.0 / 3.0).abs() < 1e-6);
        assert!((out[2] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_vec_tanh() {
        let a = [0.0, 1.0, -1.0];
        let mut out = [0.0; 3];
        vec_tanh(&a, &mut out);
        assert!((out[0]).abs() < 1e-6);
        assert!((out[1] - 0.7616).abs() < 0.01);
        assert!((out[2] + 0.7616).abs() < 0.01);
    }

    #[test]
    fn test_vec_sigmoid() {
        let a = [0.0, 1.0, -1.0];
        let mut out = [0.0; 3];
        vec_sigmoid(&a, &mut out);
        assert!((out[0] - 0.5).abs() < 1e-6);
        assert!((out[1] - 0.7311).abs() < 0.01);
        assert!((out[2] - 0.2689).abs() < 0.01);
    }

    #[test]
    fn test_vec_relu() {
        let a = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut out = [0.0; 5];
        vec_relu(&a, &mut out);
        assert_eq!(out, [0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_vec_neg() {
        let a = [1.0, -2.0, 3.0];
        let mut out = [0.0; 3];
        vec_neg(&a, &mut out);
        assert_eq!(out, [-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_vec_silu() {
        let a = [0.0, 1.0, -1.0];
        let mut out = [0.0; 3];
        vec_silu(&a, &mut out);
        assert!((out[0]).abs() < 1e-6);
        assert!((out[1] - 0.7311).abs() < 0.01);
        assert!((out[2] + 0.2689).abs() < 0.01);
    }

    #[test]
    fn test_vec_gelu() {
        let a = [0.0, 1.0, -1.0];
        let mut out = [0.0; 3];
        vec_gelu(&a, &mut out);
        assert!((out[0]).abs() < 1e-6);
        assert!((out[1] - 0.8413).abs() < 0.01);
        assert!((out[2] + 0.1587).abs() < 0.01);
    }

    #[test]
    fn test_vec_pow() {
        let a = [2.0, 3.0, 4.0];
        let b = [3.0, 2.0, 0.5];
        let mut out = [0.0; 3];
        vec_pow(&a, &b, &mut out);
        assert!((out[0] - 8.0).abs() < 1e-6);
        assert!((out[1] - 9.0).abs() < 1e-6);
        assert!((out[2] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_vec_max() {
        let a = [1.0, 5.0, 3.0];
        let b = [4.0, 2.0, 6.0];
        let mut out = [0.0; 3];
        vec_max(&a, &b, &mut out);
        assert_eq!(out, [4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_vec_div_scalar_inplace() {
        let mut a = [10.0, 20.0, 30.0];
        vec_div_scalar_inplace(&mut a, 10.0);
        assert_eq!(a, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_vec_copy() {
        let a = [1.0, 2.0, 3.0];
        let mut out = [0.0; 3];
        vec_copy(&a, &mut out);
        assert_eq!(out, [1.0, 2.0, 3.0]);
    }
}
