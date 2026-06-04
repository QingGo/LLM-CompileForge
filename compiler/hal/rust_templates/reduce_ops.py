"""Rust template constants for reduction ops.

Contains reduce, softmax, and shape_of CPU function templates.
"""

from __future__ import annotations

OP_REDUCE = """\
// ── reduce ──────────────────────────────────────────────────────────

pub fn reduce_cpu(
    inputs: &[&[f32]],
    output: &mut [f32],
    shape_meta: &OpShapeMeta,
) -> Result<(), String> {
    let inp = inputs[0];
    let input_shape = shape_meta.input_shapes.get(0)
        .ok_or_else(|| "reduce: missing input shape".to_string())?;
    let output_shape = &shape_meta.output_shape;

    let kind = shape_meta.kind.as_deref().unwrap_or("sum");

    // Determine reduction axis and sizes.
    // If output rank < input rank, find the reduced dim.
    // If output rank == input rank and last dim == 1, reduce last dim.
    if output_shape.len() == input_shape.len() {
        // Same rank — one dimension is reduced to 1.
        let reduce_dim = input_shape
            .iter()
            .zip(output_shape.iter())
            .position(|(&i, &o)| i != o && o == 1)
            .unwrap_or(input_shape.len() - 1);

        let outer: usize = input_shape[..reduce_dim].iter().map(|&d| d as usize).product();
        let reduce_size: usize = input_shape[reduce_dim] as usize;
        let inner: usize = input_shape[reduce_dim + 1..].iter().map(|&d| d as usize).product();

        for oi in 0..outer {
            for ii in 0..inner {
                let base = oi * reduce_size * inner + ii;
                let sum: f32 = (0..reduce_size)
                    .map(|r| inp[base + r * inner])
                    .sum();
                let out_idx = oi * inner + ii;
                if kind == "mean" {
                    output[out_idx] = sum / reduce_size as f32;
                } else {
                    output[out_idx] = sum;
                }
            }
        }
    } else {
        // Full reduction to scalar
        let sum: f32 = inp.iter().copied().sum();
        if kind == "mean" {
            output[0] = sum / inp.len() as f32;
        } else {
            output[0] = sum;
        }
    }

    Ok(())
}
"""

OP_SOFTMAX = """\
// ── softmax ─────────────────────────────────────────────────────────

pub fn softmax_cpu(
    inputs: &[&[f32]],
    output: &mut [f32],
    shape_meta: &OpShapeMeta,
) -> Result<(), String> {
    let inp = inputs[0];

    // Determine last dimension size from output_shape or input_shapes.
    let last_dim = shape_meta
        .output_shape
        .last()
        .copied()
        .or_else(|| shape_meta.input_shapes.get(0)?.last().copied())
        .ok_or_else(|| "softmax: cannot determine last dimension".to_string())? as usize;

    if last_dim == 0 {
        return Ok(());
    }

    let n = inp.len();
    for chunk in output.chunks_mut(last_dim) {
        // Find max for numerical stability
        let max_val = chunk
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, |a, b| a.max(b));

        // Compute exp and sum
        let mut sum = 0.0f32;
        for v in chunk.iter_mut() {
            let e = (*v - max_val).exp();
            *v = e;
            sum += e;
        }

        // Normalize
        let inv_sum = 1.0 / sum;
        for v in chunk.iter_mut() {
            *v *= inv_sum;
        }
    }

    Ok(())
}
"""

OP_SHAPE_OF = """\
// ── shape_of ───────────────────────────────────────────────────────

pub fn shape_of_cpu(
    inputs: &[&[f32]],
    output: &mut [f32],
    shape_meta: &OpShapeMeta,
) -> Result<(), String> {
    let input_shape = shape_meta.input_shapes.get(0)
        .ok_or_else(|| "shape_of: missing input shape".to_string())?;

    if output.len() != input_shape.len() {
        return Err(format!(
            "shape_of: output len {} != input rank {}",
            output.len(),
            input_shape.len(),
        ));
    }

    for (i, &dim) in input_shape.iter().enumerate() {
        output[i] = dim as f32;
    }
    Ok(())
}
"""
