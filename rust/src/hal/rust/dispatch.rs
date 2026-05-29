//! Dispatch functions for ``HalRustExecutable``.
//!
//! Each function handles a specific HAL operation type, converting buffers
//! to ``&[f32]``/``&mut [f32]`` slices and calling the corresponding
//! ``*_cpu`` function from ``hal_ops_cpu``.

use crate::hal::traits;

use super::executable::HalRustExecutable;

/// Dispatch a matmul operation.
pub fn dispatch_matmul(
    inputs: &[&dyn traits::Buffer],
    outputs: &[&dyn traits::Buffer],
) -> Result<Vec<Vec<i64>>, anyhow::Error> {
    let input_slices: Vec<&[f32]> = inputs
        .iter()
        .map(|b| {
            // SAFETY: buf_as_f32_slice requires f32 data (element_size == 4).
            // Matmul inputs are f32 tensors from weights or SSA wires, guaranteed
            // by the compute graph runner.
            unsafe { HalRustExecutable::buf_as_f32_slice(*b) }
        })
        .collect();
    let output_shapes: Vec<Vec<i64>> = outputs
        .iter()
        .map(|b| b.shape().into_iter().map(|s| s as i64).collect())
        .collect();
    let meta = HalRustExecutable::build_shape_meta(inputs, outputs);

    if let Some(out_buf) = outputs.first() {
        // SAFETY: buf_as_f32_mut requires f32 data and write access. The output
        // buffer is pre-allocated as f32 by the compute graph runner and valid
        // for the matmul result size.
        let out_slice = unsafe { HalRustExecutable::buf_as_f32_mut(*out_buf) };
        crate::hal::hal_ops_cpu::matmul_cpu(&input_slices, out_slice, &meta)
            .map_err(|e| anyhow::anyhow!("matmul_cpu: {}", e))?;
    }
    Ok(output_shapes)
}

/// Dispatch an element_wise operation.
pub fn dispatch_element_wise(
    inputs: &[&dyn traits::Buffer],
    outputs: &[&dyn traits::Buffer],
    kind: &str,
) -> Result<Vec<Vec<i64>>, anyhow::Error> {
    let input_slices: Vec<&[f32]> = inputs
        .iter()
        .map(|b| {
            // SAFETY: buf_as_f32_slice requires f32 data. Element-wise
            // inputs are f32 tensors guaranteed by the compute graph runner.
            unsafe { HalRustExecutable::buf_as_f32_slice(*b) }
        })
        .collect();
    let output_shapes: Vec<Vec<i64>> = outputs
        .iter()
        .map(|b| b.shape().into_iter().map(|s| s as i64).collect())
        .collect();
    let mut meta = HalRustExecutable::build_shape_meta(inputs, outputs);
    meta.kind = Some(kind.to_string());

    if let Some(out_buf) = outputs.first() {
        // SAFETY: buf_as_f32_mut requires f32 data and write access.
        // The output buffer is pre-allocated as f32 by the compute graph runner.
        let out_slice = unsafe { HalRustExecutable::buf_as_f32_mut(*out_buf) };
        crate::hal::hal_ops_cpu::element_wise_cpu(&input_slices, out_slice, &meta)
            .map_err(|e| anyhow::anyhow!("element_wise_cpu: {}", e))?;
    }
    Ok(output_shapes)
}

/// Dispatch a softmax operation.
pub fn dispatch_softmax(
    inputs: &[&dyn traits::Buffer],
    outputs: &[&dyn traits::Buffer],
) -> Result<Vec<Vec<i64>>, anyhow::Error> {
    let input_slices: Vec<&[f32]> = inputs
        .iter()
        .map(|b| {
            // SAFETY: buf_as_f32_slice requires f32 data. Softmax inputs
            // are f32 tensors guaranteed by the compute graph runner.
            unsafe { HalRustExecutable::buf_as_f32_slice(*b) }
        })
        .collect();
    let output_shapes: Vec<Vec<i64>> = outputs
        .iter()
        .map(|b| b.shape().into_iter().map(|s| s as i64).collect())
        .collect();
    let meta = HalRustExecutable::build_shape_meta(inputs, outputs);

    if let Some(out_buf) = outputs.first() {
        // SAFETY: buf_as_f32_mut requires f32 data and write access.
        // The output buffer is pre-allocated as f32 by the compute graph runner.
        let out_slice = unsafe { HalRustExecutable::buf_as_f32_mut(*out_buf) };
        crate::hal::hal_ops_cpu::softmax_cpu(&input_slices, out_slice, &meta)
            .map_err(|e| anyhow::anyhow!("softmax_cpu: {}", e))?;
    }
    Ok(output_shapes)
}

/// Dispatch a reshape operation.
pub fn dispatch_reshape(
    inputs: &[&dyn traits::Buffer],
    outputs: &[&dyn traits::Buffer],
) -> Result<Vec<Vec<i64>>, anyhow::Error> {
    let input_slices: Vec<&[f32]> = inputs
        .iter()
        .map(|b| {
            // SAFETY: buf_as_f32_slice requires f32 data. Reshape inputs
            // are f32 tensors guaranteed by the compute graph runner.
            unsafe { HalRustExecutable::buf_as_f32_slice(*b) }
        })
        .collect();
    let output_shapes: Vec<Vec<i64>> = outputs
        .iter()
        .map(|b| b.shape().into_iter().map(|s| s as i64).collect())
        .collect();
    let meta = HalRustExecutable::build_shape_meta(inputs, outputs);

    if let Some(out_buf) = outputs.first() {
        // SAFETY: buf_as_f32_mut requires f32 data and write access.
        // The output buffer is pre-allocated as f32 by the compute graph runner.
        let out_slice = unsafe { HalRustExecutable::buf_as_f32_mut(*out_buf) };
        crate::hal::hal_ops_cpu::reshape_cpu(&input_slices, out_slice, &meta)
            .map_err(|e| anyhow::anyhow!("reshape_cpu: {}", e))?;
    }
    Ok(output_shapes)
}

/// Dispatch a transpose operation.
pub fn dispatch_transpose(
    inputs: &[&dyn traits::Buffer],
    outputs: &[&dyn traits::Buffer],
    perm: Option<&str>,
) -> Result<Vec<Vec<i64>>, anyhow::Error> {
    let input_slices: Vec<&[f32]> = inputs
        .iter()
        .map(|b| {
            // SAFETY: buf_as_f32_slice requires f32 data. Transpose inputs
            // are f32 tensors guaranteed by the compute graph runner.
            unsafe { HalRustExecutable::buf_as_f32_slice(*b) }
        })
        .collect();
    let output_shapes: Vec<Vec<i64>> = outputs
        .iter()
        .map(|b| b.shape().into_iter().map(|s| s as i64).collect())
        .collect();
    let mut meta = HalRustExecutable::build_shape_meta(inputs, outputs);
    meta.kind = perm.map(|s| s.to_string());

    if let Some(out_buf) = outputs.first() {
        // SAFETY: buf_as_f32_mut requires f32 data and write access.
        // The output buffer is pre-allocated as f32 by the compute graph runner.
        let out_slice = unsafe { HalRustExecutable::buf_as_f32_mut(*out_buf) };
        crate::hal::hal_ops_cpu::transpose_cpu(&input_slices, out_slice, &meta)
            .map_err(|e| anyhow::anyhow!("transpose_cpu: {}", e))?;
    }
    Ok(output_shapes)
}

/// Dispatch a reduce operation.
pub fn dispatch_reduce(
    inputs: &[&dyn traits::Buffer],
    outputs: &[&dyn traits::Buffer],
    kind: &str,
) -> Result<Vec<Vec<i64>>, anyhow::Error> {
    let input_slices: Vec<&[f32]> = inputs
        .iter()
        .map(|b| {
            // SAFETY: buf_as_f32_slice requires f32 data. Reduce inputs
            // are f32 tensors guaranteed by the compute graph runner.
            unsafe { HalRustExecutable::buf_as_f32_slice(*b) }
        })
        .collect();
    let output_shapes: Vec<Vec<i64>> = outputs
        .iter()
        .map(|b| b.shape().into_iter().map(|s| s as i64).collect())
        .collect();
    let mut meta = HalRustExecutable::build_shape_meta(inputs, outputs);
    meta.kind = Some(kind.to_string());

    if let Some(out_buf) = outputs.first() {
        // SAFETY: buf_as_f32_mut requires f32 data and write access.
        // The output buffer is pre-allocated as f32 by the compute graph runner.
        let out_slice = unsafe { HalRustExecutable::buf_as_f32_mut(*out_buf) };
        crate::hal::hal_ops_cpu::reduce_cpu(&input_slices, out_slice, &meta)
            .map_err(|e| anyhow::anyhow!("reduce_cpu: {}", e))?;
    }
    Ok(output_shapes)
}

/// Dispatch a gather operation.
pub fn dispatch_gather(
    inputs: &[&dyn traits::Buffer],
    outputs: &[&dyn traits::Buffer],
) -> Result<Vec<Vec<i64>>, anyhow::Error> {
    let input_slices: Vec<&[f32]> = inputs
        .iter()
        .map(|b| {
            // SAFETY: buf_as_f32_slice requires f32 data. Gather inputs
            // are f32 tensors guaranteed by the compute graph runner.
            unsafe { HalRustExecutable::buf_as_f32_slice(*b) }
        })
        .collect();
    let output_shapes: Vec<Vec<i64>> = outputs
        .iter()
        .map(|b| b.shape().into_iter().map(|s| s as i64).collect())
        .collect();
    let meta = HalRustExecutable::build_shape_meta(inputs, outputs);

    if let Some(out_buf) = outputs.first() {
        // SAFETY: buf_as_f32_mut requires f32 data and write access.
        // The output buffer is pre-allocated as f32 by the compute graph runner.
        let out_slice = unsafe { HalRustExecutable::buf_as_f32_mut(*out_buf) };
        crate::hal::hal_ops_cpu::gather_cpu(&input_slices, out_slice, &meta)
            .map_err(|e| anyhow::anyhow!("gather_cpu: {}", e))?;
    }
    Ok(output_shapes)
}

/// Dispatch a fill operation.
pub fn dispatch_fill(
    inputs: &[&dyn traits::Buffer],
    outputs: &[&dyn traits::Buffer],
    kind: Option<&str>,
    value: Option<f64>,
) -> Result<Vec<Vec<i64>>, anyhow::Error> {
    let input_slices: Vec<&[f32]> = inputs
        .iter()
        .map(|b| {
            // SAFETY: buf_as_f32_slice requires f32 data. Fill inputs
            // are f32 tensors guaranteed by the compute graph runner.
            unsafe { HalRustExecutable::buf_as_f32_slice(*b) }
        })
        .collect();
    let output_shapes: Vec<Vec<i64>> = outputs
        .iter()
        .map(|b| b.shape().into_iter().map(|s| s as i64).collect())
        .collect();
    let mut meta = HalRustExecutable::build_shape_meta(inputs, outputs);
    meta.kind = kind.map(|s| s.to_string());
    meta.value = value;

    if let Some(out_buf) = outputs.first() {
        // SAFETY: buf_as_f32_mut requires f32 data and write access.
        // The output buffer is pre-allocated as f32 by the compute graph runner.
        let out_slice = unsafe { HalRustExecutable::buf_as_f32_mut(*out_buf) };
        crate::hal::hal_ops_cpu::fill_cpu(&input_slices, out_slice, &meta)
            .map_err(|e| anyhow::anyhow!("fill_cpu: {}", e))?;
    }
    Ok(output_shapes)
}

/// Dispatch a shape_of operation.
pub fn dispatch_shape_of(
    inputs: &[&dyn traits::Buffer],
    outputs: &[&dyn traits::Buffer],
) -> Result<Vec<Vec<i64>>, anyhow::Error> {
    let input_slices: Vec<&[f32]> = inputs
        .iter()
        .map(|b| {
            // SAFETY: buf_as_f32_slice requires f32 data. Shape-of inputs
            // are f32 tensors guaranteed by the compute graph runner.
            unsafe { HalRustExecutable::buf_as_f32_slice(*b) }
        })
        .collect();
    let output_shapes: Vec<Vec<i64>> = outputs
        .iter()
        .map(|b| b.shape().into_iter().map(|s| s as i64).collect())
        .collect();
    let meta = HalRustExecutable::build_shape_meta(inputs, outputs);

    if let Some(out_buf) = outputs.first() {
        // SAFETY: buf_as_f32_mut requires f32 data and write access.
        // The output buffer is pre-allocated as f32 by the compute graph runner.
        let out_slice = unsafe { HalRustExecutable::buf_as_f32_mut(*out_buf) };
        crate::hal::hal_ops_cpu::shape_of_cpu(&input_slices, out_slice, &meta)
            .map_err(|e| anyhow::anyhow!("shape_of_cpu: {}", e))?;
    }
    Ok(output_shapes)
}

/// Dispatch a slice operation.
pub fn dispatch_slice(
    inputs: &[&dyn traits::Buffer],
    outputs: &[&dyn traits::Buffer],
) -> Result<Vec<Vec<i64>>, anyhow::Error> {
    let input_slices: Vec<&[f32]> = inputs
        .iter()
        .map(|b| {
            // SAFETY: buf_as_f32_slice requires f32 data. Slice inputs
            // are f32 tensors guaranteed by the compute graph runner.
            unsafe { HalRustExecutable::buf_as_f32_slice(*b) }
        })
        .collect();
    let output_shapes: Vec<Vec<i64>> = outputs
        .iter()
        .map(|b| b.shape().into_iter().map(|s| s as i64).collect())
        .collect();
    let meta = HalRustExecutable::build_shape_meta(inputs, outputs);

    if let Some(out_buf) = outputs.first() {
        // SAFETY: buf_as_f32_mut requires f32 data and write access.
        // The output buffer is pre-allocated as f32 by the compute graph runner.
        let out_slice = unsafe { HalRustExecutable::buf_as_f32_mut(*out_buf) };
        crate::hal::hal_ops_cpu::slice_cpu(&input_slices, out_slice, &meta)
            .map_err(|e| anyhow::anyhow!("slice_cpu: {}", e))?;
    }
    Ok(output_shapes)
}

/// Dispatch an unsqueeze operation.
pub fn dispatch_unsqueeze(
    inputs: &[&dyn traits::Buffer],
    outputs: &[&dyn traits::Buffer],
) -> Result<Vec<Vec<i64>>, anyhow::Error> {
    let input_slices: Vec<&[f32]> = inputs
        .iter()
        .map(|b| {
            // SAFETY: buf_as_f32_slice requires f32 data. Unsqueeze inputs
            // are f32 tensors guaranteed by the compute graph runner.
            unsafe { HalRustExecutable::buf_as_f32_slice(*b) }
        })
        .collect();
    let output_shapes: Vec<Vec<i64>> = outputs
        .iter()
        .map(|b| b.shape().into_iter().map(|s| s as i64).collect())
        .collect();
    let meta = HalRustExecutable::build_shape_meta(inputs, outputs);

    if let Some(out_buf) = outputs.first() {
        // SAFETY: buf_as_f32_mut requires f32 data and write access.
        // The output buffer is pre-allocated as f32 by the compute graph runner.
        let out_slice = unsafe { HalRustExecutable::buf_as_f32_mut(*out_buf) };
        crate::hal::hal_ops_cpu::unsqueeze_cpu(&input_slices, out_slice, &meta)
            .map_err(|e| anyhow::anyhow!("unsqueeze_cpu: {}", e))?;
    }
    Ok(output_shapes)
}

/// Dispatch a compare operation.
pub fn dispatch_compare(
    inputs: &[&dyn traits::Buffer],
    outputs: &[&dyn traits::Buffer],
    kind: &str,
) -> Result<Vec<Vec<i64>>, anyhow::Error> {
    let input_slices: Vec<&[f32]> = inputs
        .iter()
        .map(|b| {
            // SAFETY: buf_as_f32_slice requires f32 data. Compare inputs
            // are f32 tensors guaranteed by the compute graph runner.
            unsafe { HalRustExecutable::buf_as_f32_slice(*b) }
        })
        .collect();
    let output_shapes: Vec<Vec<i64>> = outputs
        .iter()
        .map(|b| b.shape().into_iter().map(|s| s as i64).collect())
        .collect();
    let mut meta = HalRustExecutable::build_shape_meta(inputs, outputs);
    meta.kind = Some(kind.to_string());

    if let Some(out_buf) = outputs.first() {
        // SAFETY: buf_as_f32_mut requires f32 data and write access.
        // The output buffer is pre-allocated as f32 by the compute graph runner.
        let out_slice = unsafe { HalRustExecutable::buf_as_f32_mut(*out_buf) };
        crate::hal::hal_ops_cpu::compare_cpu(&input_slices, out_slice, &meta)
            .map_err(|e| anyhow::anyhow!("compare_cpu: {}", e))?;
    }
    Ok(output_shapes)
}

/// Dispatch a concat operation (not yet implemented in Rust backend).
pub fn dispatch_concat(
    inputs: &[&dyn traits::Buffer],
    outputs: &[&dyn traits::Buffer],
) -> Result<Vec<Vec<i64>>, anyhow::Error> {
    let output_shapes: Vec<Vec<i64>> = outputs
        .iter()
        .map(|b| b.shape().into_iter().map(|s| s as i64).collect())
        .collect();
    // Simple flat copy (works if no actual concat needed — metadata-only)
    if let (Some(inp), Some(out)) = (inputs.first(), outputs.first()) {
        // SAFETY: buf_as_f32_slice requires f32 data. Concat inputs
        // are f32 tensors guaranteed by the compute graph runner.
        let in_slice = unsafe { HalRustExecutable::buf_as_f32_slice(*inp) };
        // SAFETY: buf_as_f32_mut requires f32 data and write access.
        // The output buffer is pre-allocated as f32 by the compute graph runner.
        let out_slice = unsafe { HalRustExecutable::buf_as_f32_mut(*out) };
        let n = in_slice.len().min(out_slice.len());
        out_slice[..n].copy_from_slice(&in_slice[..n]);
    }
    Ok(output_shapes)
}
