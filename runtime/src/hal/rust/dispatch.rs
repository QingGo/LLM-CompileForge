//! Dispatch functions for ``HalRustExecutable``.
//!
//! Each function handles a specific HAL operation type, converting SfaMemRef
//! descriptors to ``&[f32]``/``&mut [f32]`` slices and calling the corresponding
//! ``*_cpu`` function from ``hal_ops_cpu``.
//!
//! Wave 3 / Task 10: ``dispatch_via_trait()`` replaces string-based dispatch
//! with trait-object dispatch via ``KernelOp`` registry. Each kernel type
//! satisfies the ``KernelOp`` contract checked at compile time.

use crate::hal::sfa::SfaMemRef;

use super::executable::HalRustExecutable;

/// Unified dispatch through the ``KernelOp`` trait registry.
///
/// Looks up ``op_name`` in ``kernel_registry()`` and delegates to
/// ``KernelOp::execute_typed``. This is the canonical dispatch path
/// for all HAL operations.
pub fn dispatch_via_trait(
    op_name: &str,
    inputs: &[SfaMemRef],
    outputs: &mut [SfaMemRef],
) -> Result<Vec<Vec<i64>>, anyhow::Error> {
    let input_slices: Vec<&[f32]> = inputs
        .iter()
        // SAFETY: All inputs are f32 buffers (element_size == 4).
        .map(|m| unsafe { HalRustExecutable::sfa_as_f32_slice(m) })
        .collect();
    let output_shapes: Vec<Vec<i64>> = outputs
        .iter()
        .map(|m| m.sizes_i64())
        .collect();
    let meta = HalRustExecutable::build_shape_meta_from_sfa(inputs, outputs);

    let registry = crate::hal::primitives::traits::kernel_registry();
    let kernel = registry
        .get(op_name)
        .ok_or_else(|| anyhow::anyhow!("unknown op: {}", op_name))?;

    if let Some(out_sfa) = outputs.first() {
        // SAFETY: Output buffer is pre-allocated as f32 by the runner.
        let out_slice = unsafe { HalRustExecutable::sfa_as_f32_mut(out_sfa) };
        kernel
            .execute_typed(&input_slices, out_slice, &meta)
            .map_err(|e| anyhow::anyhow!("{}: {}", op_name, e))?;
    } else {
        kernel
            .execute_typed(&input_slices, &mut [], &meta)
            .map_err(|e| anyhow::anyhow!("{}: {}", op_name, e))?;
    }
    Ok(output_shapes)
}

/// Dispatch a matmul operation.
#[allow(dead_code)]
pub fn dispatch_matmul(
    inputs: &[SfaMemRef],
    outputs: &mut [SfaMemRef],
) -> Result<Vec<Vec<i64>>, anyhow::Error> {
    let input_slices: Vec<&[f32]> = inputs
        .iter()
        .map(|m| unsafe { HalRustExecutable::sfa_as_f32_slice(m) })
        .collect();
    let output_shapes: Vec<Vec<i64>> = outputs
        .iter()
        .map(|m| m.sizes_i64())
        .collect();
    let meta = HalRustExecutable::build_shape_meta_from_sfa(inputs, outputs);

    if let Some(out_sfa) = outputs.first() {
        let out_slice = unsafe { HalRustExecutable::sfa_as_f32_mut(out_sfa) };
        crate::hal::hal_ops_cpu::matmul_cpu(&input_slices, out_slice, &meta)
            .map_err(|e| anyhow::anyhow!("matmul_cpu: {}", e))?;
    }
    Ok(output_shapes)
}

/// Dispatch an element_wise operation.
#[allow(dead_code)]
pub fn dispatch_element_wise(
    inputs: &[SfaMemRef],
    outputs: &mut [SfaMemRef],
    kind: &str,
) -> Result<Vec<Vec<i64>>, anyhow::Error> {
    let input_slices: Vec<&[f32]> = inputs
        .iter()
        .map(|m| unsafe { HalRustExecutable::sfa_as_f32_slice(m) })
        .collect();
    let output_shapes: Vec<Vec<i64>> = outputs
        .iter()
        .map(|m| m.sizes_i64())
        .collect();
    let mut meta = HalRustExecutable::build_shape_meta_from_sfa(inputs, outputs);
    meta.kind = Some(kind.to_string());

    if let Some(out_sfa) = outputs.first() {
        let out_slice = unsafe { HalRustExecutable::sfa_as_f32_mut(out_sfa) };
        crate::hal::hal_ops_cpu::element_wise_cpu(&input_slices, out_slice, &meta)
            .map_err(|e| anyhow::anyhow!("element_wise_cpu: {}", e))?;
    }
    Ok(output_shapes)
}

/// Dispatch a softmax operation.
#[allow(dead_code)]
pub fn dispatch_softmax(
    inputs: &[SfaMemRef],
    outputs: &mut [SfaMemRef],
) -> Result<Vec<Vec<i64>>, anyhow::Error> {
    let input_slices: Vec<&[f32]> = inputs
        .iter()
        .map(|m| unsafe { HalRustExecutable::sfa_as_f32_slice(m) })
        .collect();
    let output_shapes: Vec<Vec<i64>> = outputs
        .iter()
        .map(|m| m.sizes_i64())
        .collect();
    let meta = HalRustExecutable::build_shape_meta_from_sfa(inputs, outputs);

    if let Some(out_sfa) = outputs.first() {
        let out_slice = unsafe { HalRustExecutable::sfa_as_f32_mut(out_sfa) };
        crate::hal::hal_ops_cpu::softmax_cpu(&input_slices, out_slice, &meta)
            .map_err(|e| anyhow::anyhow!("softmax_cpu: {}", e))?;
    }
    Ok(output_shapes)
}

/// Dispatch a reshape operation.
#[allow(dead_code)]
pub fn dispatch_reshape(
    inputs: &[SfaMemRef],
    outputs: &mut [SfaMemRef],
) -> Result<Vec<Vec<i64>>, anyhow::Error> {
    let input_slices: Vec<&[f32]> = inputs
        .iter()
        .map(|m| unsafe { HalRustExecutable::sfa_as_f32_slice(m) })
        .collect();
    let output_shapes: Vec<Vec<i64>> = outputs
        .iter()
        .map(|m| m.sizes_i64())
        .collect();
    let meta = HalRustExecutable::build_shape_meta_from_sfa(inputs, outputs);

    if let Some(out_sfa) = outputs.first() {
        let out_slice = unsafe { HalRustExecutable::sfa_as_f32_mut(out_sfa) };
        crate::hal::hal_ops_cpu::reshape_cpu(&input_slices, out_slice, &meta)
            .map_err(|e| anyhow::anyhow!("reshape_cpu: {}", e))?;
    }
    Ok(output_shapes)
}

/// Dispatch a transpose operation.
#[allow(dead_code)]
pub fn dispatch_transpose(
    inputs: &[SfaMemRef],
    outputs: &mut [SfaMemRef],
    perm: Option<&str>,
) -> Result<Vec<Vec<i64>>, anyhow::Error> {
    let input_slices: Vec<&[f32]> = inputs
        .iter()
        .map(|m| unsafe { HalRustExecutable::sfa_as_f32_slice(m) })
        .collect();
    let output_shapes: Vec<Vec<i64>> = outputs
        .iter()
        .map(|m| m.sizes_i64())
        .collect();
    let mut meta = HalRustExecutable::build_shape_meta_from_sfa(inputs, outputs);
    meta.kind = perm.map(|s| s.to_string());

    if let Some(out_sfa) = outputs.first() {
        let out_slice = unsafe { HalRustExecutable::sfa_as_f32_mut(out_sfa) };
        crate::hal::hal_ops_cpu::transpose_cpu(&input_slices, out_slice, &meta)
            .map_err(|e| anyhow::anyhow!("transpose_cpu: {}", e))?;
    }
    Ok(output_shapes)
}

/// Dispatch a reduce operation.
#[allow(dead_code)]
pub fn dispatch_reduce(
    inputs: &[SfaMemRef],
    outputs: &mut [SfaMemRef],
    kind: &str,
) -> Result<Vec<Vec<i64>>, anyhow::Error> {
    let input_slices: Vec<&[f32]> = inputs
        .iter()
        .map(|m| unsafe { HalRustExecutable::sfa_as_f32_slice(m) })
        .collect();
    let output_shapes: Vec<Vec<i64>> = outputs
        .iter()
        .map(|m| m.sizes_i64())
        .collect();
    let mut meta = HalRustExecutable::build_shape_meta_from_sfa(inputs, outputs);
    meta.kind = Some(kind.to_string());

    if let Some(out_sfa) = outputs.first() {
        let out_slice = unsafe { HalRustExecutable::sfa_as_f32_mut(out_sfa) };
        crate::hal::hal_ops_cpu::reduce_cpu(&input_slices, out_slice, &meta)
            .map_err(|e| anyhow::anyhow!("reduce_cpu: {}", e))?;
    }
    Ok(output_shapes)
}

/// Dispatch a gather operation.
#[allow(dead_code)]
pub fn dispatch_gather(
    inputs: &[SfaMemRef],
    outputs: &mut [SfaMemRef],
) -> Result<Vec<Vec<i64>>, anyhow::Error> {
    let input_slices: Vec<&[f32]> = inputs
        .iter()
        .map(|m| unsafe { HalRustExecutable::sfa_as_f32_slice(m) })
        .collect();
    let output_shapes: Vec<Vec<i64>> = outputs
        .iter()
        .map(|m| m.sizes_i64())
        .collect();
    let meta = HalRustExecutable::build_shape_meta_from_sfa(inputs, outputs);

    if let Some(out_sfa) = outputs.first() {
        let out_slice = unsafe { HalRustExecutable::sfa_as_f32_mut(out_sfa) };
        crate::hal::hal_ops_cpu::gather_cpu(&input_slices, out_slice, &meta)
            .map_err(|e| anyhow::anyhow!("gather_cpu: {}", e))?;
    }
    Ok(output_shapes)
}

/// Dispatch a fill operation.
#[allow(dead_code)]
pub fn dispatch_fill(
    inputs: &[SfaMemRef],
    outputs: &mut [SfaMemRef],
    kind: Option<&str>,
    value: Option<f64>,
) -> Result<Vec<Vec<i64>>, anyhow::Error> {
    let input_slices: Vec<&[f32]> = inputs
        .iter()
        .map(|m| unsafe { HalRustExecutable::sfa_as_f32_slice(m) })
        .collect();
    let output_shapes: Vec<Vec<i64>> = outputs
        .iter()
        .map(|m| m.sizes_i64())
        .collect();
    let mut meta = HalRustExecutable::build_shape_meta_from_sfa(inputs, outputs);
    meta.kind = kind.map(|s| s.to_string());
    meta.value = value;

    if let Some(out_sfa) = outputs.first() {
        let out_slice = unsafe { HalRustExecutable::sfa_as_f32_mut(out_sfa) };
        crate::hal::hal_ops_cpu::fill_cpu(&input_slices, out_slice, &meta)
            .map_err(|e| anyhow::anyhow!("fill_cpu: {}", e))?;
    }
    Ok(output_shapes)
}

/// Dispatch a shape_of operation.
#[allow(dead_code)]
pub fn dispatch_shape_of(
    inputs: &[SfaMemRef],
    outputs: &mut [SfaMemRef],
) -> Result<Vec<Vec<i64>>, anyhow::Error> {
    let input_slices: Vec<&[f32]> = inputs
        .iter()
        .map(|m| unsafe { HalRustExecutable::sfa_as_f32_slice(m) })
        .collect();
    let output_shapes: Vec<Vec<i64>> = outputs
        .iter()
        .map(|m| m.sizes_i64())
        .collect();
    let meta = HalRustExecutable::build_shape_meta_from_sfa(inputs, outputs);

    if let Some(out_sfa) = outputs.first() {
        let out_slice = unsafe { HalRustExecutable::sfa_as_f32_mut(out_sfa) };
        crate::hal::hal_ops_cpu::shape_of_cpu(&input_slices, out_slice, &meta)
            .map_err(|e| anyhow::anyhow!("shape_of_cpu: {}", e))?;
    }
    Ok(output_shapes)
}

/// Dispatch a slice operation.
#[allow(dead_code)]
pub fn dispatch_slice(
    inputs: &[SfaMemRef],
    outputs: &mut [SfaMemRef],
) -> Result<Vec<Vec<i64>>, anyhow::Error> {
    let input_slices: Vec<&[f32]> = inputs
        .iter()
        .map(|m| unsafe { HalRustExecutable::sfa_as_f32_slice(m) })
        .collect();
    let output_shapes: Vec<Vec<i64>> = outputs
        .iter()
        .map(|m| m.sizes_i64())
        .collect();
    let meta = HalRustExecutable::build_shape_meta_from_sfa(inputs, outputs);

    if let Some(out_sfa) = outputs.first() {
        let out_slice = unsafe { HalRustExecutable::sfa_as_f32_mut(out_sfa) };
        crate::hal::hal_ops_cpu::slice_cpu(&input_slices, out_slice, &meta)
            .map_err(|e| anyhow::anyhow!("slice_cpu: {}", e))?;
    }
    Ok(output_shapes)
}

/// Dispatch an unsqueeze operation.
#[allow(dead_code)]
pub fn dispatch_unsqueeze(
    inputs: &[SfaMemRef],
    outputs: &mut [SfaMemRef],
) -> Result<Vec<Vec<i64>>, anyhow::Error> {
    let input_slices: Vec<&[f32]> = inputs
        .iter()
        .map(|m| unsafe { HalRustExecutable::sfa_as_f32_slice(m) })
        .collect();
    let output_shapes: Vec<Vec<i64>> = outputs
        .iter()
        .map(|m| m.sizes_i64())
        .collect();
    let meta = HalRustExecutable::build_shape_meta_from_sfa(inputs, outputs);

    if let Some(out_sfa) = outputs.first() {
        let out_slice = unsafe { HalRustExecutable::sfa_as_f32_mut(out_sfa) };
        crate::hal::hal_ops_cpu::unsqueeze_cpu(&input_slices, out_slice, &meta)
            .map_err(|e| anyhow::anyhow!("unsqueeze_cpu: {}", e))?;
    }
    Ok(output_shapes)
}

/// Dispatch a compare operation.
#[allow(dead_code)]
pub fn dispatch_compare(
    inputs: &[SfaMemRef],
    outputs: &mut [SfaMemRef],
    kind: &str,
) -> Result<Vec<Vec<i64>>, anyhow::Error> {
    let input_slices: Vec<&[f32]> = inputs
        .iter()
        .map(|m| unsafe { HalRustExecutable::sfa_as_f32_slice(m) })
        .collect();
    let output_shapes: Vec<Vec<i64>> = outputs
        .iter()
        .map(|m| m.sizes_i64())
        .collect();
    let mut meta = HalRustExecutable::build_shape_meta_from_sfa(inputs, outputs);
    meta.kind = Some(kind.to_string());

    if let Some(out_sfa) = outputs.first() {
        let out_slice = unsafe { HalRustExecutable::sfa_as_f32_mut(out_sfa) };
        crate::hal::hal_ops_cpu::compare_cpu(&input_slices, out_slice, &meta)
            .map_err(|e| anyhow::anyhow!("compare_cpu: {}", e))?;
    }
    Ok(output_shapes)
}

/// Dispatch a concat operation.
#[allow(dead_code)]
pub fn dispatch_concat(
    inputs: &[SfaMemRef],
    outputs: &mut [SfaMemRef],
) -> Result<Vec<Vec<i64>>, anyhow::Error> {
    let output_shapes: Vec<Vec<i64>> = outputs
        .iter()
        .map(|m| m.sizes_i64())
        .collect();
    if let (Some(inp), Some(out)) = (inputs.first(), outputs.first()) {
        let in_slice = unsafe { HalRustExecutable::sfa_as_f32_slice(inp) };
        let out_slice = unsafe { HalRustExecutable::sfa_as_f32_mut(out) };
        let n = in_slice.len().min(out_slice.len());
        out_slice[..n].copy_from_slice(&in_slice[..n]);
    }
    Ok(output_shapes)
}
