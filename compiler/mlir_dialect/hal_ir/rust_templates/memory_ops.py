"""Rust template constants for memory/movement ops.

Contains reshape, transpose, slice, gather, fill, and unsqueeze
CPU function templates.
"""

from __future__ import annotations


def _flat_copy_template(op_name: str) -> str:
    """Shared template for ops that are metadata-only flat copies."""
    return f"""\
// ── {op_name} (metadata only — flat copy) ────────────────────────────

pub fn {op_name}_cpu(
    inputs: &[&[f32]],
    output: &mut [f32],
    shape_meta: &OpShapeMeta,
) -> Result<(), String> {{
    let inp = inputs[0];
    if inp.len() != output.len() {{
        return Err(format!(
            "{op_name}: element count mismatch: input {{}} != output {{}}",
            inp.len(),
            output.len(),
        ));
    }}
    output.copy_from_slice(inp);
    Ok(())
}}
"""


OP_RESHAPE = _flat_copy_template("reshape")

OP_TRANSPOSE = """\
// ── transpose (generic nd transpose) ───────────────────────────────

pub fn transpose_cpu(
    inputs: &[&[f32]],
    output: &mut [f32],
    shape_meta: &OpShapeMeta,
) -> Result<(), String> {
    let inp = inputs[0];
    let input_shape = shape_meta.input_shapes.get(0)
        .ok_or_else(|| "transpose: missing input shape".to_string())?;
    let output_shape = &shape_meta.output_shape;

    let rank = input_shape.len();
    if rank < 2 {
        return Err(format!("transpose: expected rank >= 2, got {}", rank));
    }

    // The permutation is stored in shape_meta.kind as "dim1,dim2" or we
    // default to swapping the last two dimensions (common for attention).
    let perm: Vec<usize> = if let Some(kind) = &shape_meta.kind {
        kind.split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .collect()
    } else {
        // Default: swap last two dims
        let mut p: Vec<usize> = (0..rank).collect();
        let last = rank - 1;
        p.swap(last - 1, last);
        p
    };

    if perm.len() != rank {
        return Err(format!(
            "transpose: permutation length {} != rank {}",
            perm.len(),
            rank,
        ));
    }

    // Precompute strides for input
    let mut in_strides: Vec<usize> = vec![1; rank];
    for i in (0..rank - 1).rev() {
        in_strides[i] = in_strides[i + 1] * input_shape[i + 1] as usize;
    }

    // Precompute strides for output
    let mut out_strides: Vec<usize> = vec![1; rank];
    for i in (0..rank - 1).rev() {
        out_strides[i] = out_strides[i + 1] * output_shape[i + 1] as usize;
    }

    // Iterate over flat output and compute source index
    for flat_idx in 0..output.len() {
        // Compute output n-dimensional index
        let mut idx = flat_idx;
        let mut o_idx = vec![0usize; rank];
        for d in 0..rank {
            o_idx[d] = idx / out_strides[d];
            idx %= out_strides[d];
        }

        // Map to input index via inverse permutation
        let mut i_idx = vec![0usize; rank];
        for d in 0..rank {
            i_idx[perm[d]] = o_idx[d];
        }

        // Compute flat input index
        let in_flat: usize = i_idx
            .iter()
            .zip(in_strides.iter())
            .map(|(&i, &s)| i * s)
            .sum();

        output[flat_idx] = inp[in_flat];
    }

    Ok(())
}
"""

OP_SLICE = """\
// ── slice ───────────────────────────────────────────────────────────

pub fn slice_cpu(
    inputs: &[&[f32]],
    output: &mut [f32],
    shape_meta: &OpShapeMeta,
) -> Result<(), String> {
    let inp = inputs[0];
    let input_shape = shape_meta.input_shapes.get(0)
        .ok_or_else(|| "slice: missing input shape".to_string())?;
    let output_shape = &shape_meta.output_shape;

    if input_shape.is_empty() {
        return Err("slice: empty input shape".to_string());
    }

    // Determine slice dimension from shape delta.
    // The sliced dimension has input_size > output_size.
    // We copy a contiguous sub-tensor.
    let slice_dim = input_shape
        .iter()
        .zip(output_shape.iter())
        .position(|(&i, &o)| i != o)
        .unwrap_or(0);

    let outer: usize = input_shape[..slice_dim].iter().map(|&d| d as usize).product();
    let slice_size: usize = output_shape[slice_dim] as usize;
    let inner: usize = input_shape[slice_dim + 1..].iter().map(|&d| d as usize).product();

    let src_stride = inner * input_shape[slice_dim] as usize;
    let dst_stride = inner * slice_size;

    for i in 0..outer {
        let src_base = i * src_stride;
        let dst_base = i * dst_stride;
        for j in 0..slice_size {
            let src_off = src_base + j * inner;
            let dst_off = dst_base + j * inner;
            for k in 0..inner {
                output[dst_off + k] = inp[src_off + k];
            }
        }
    }

    Ok(())
}
"""

OP_GATHER = """\
// ── gather (embedding / indexed load) ──────────────────────────────

pub fn gather_cpu(
    inputs: &[&[f32]],
    output: &mut [f32],
    shape_meta: &OpShapeMeta,
) -> Result<(), String> {
    let weight_table = inputs[0];

    let weight_shape = shape_meta.input_shapes.get(0)
        .ok_or_else(|| "gather: missing weight shape".to_string())?;
    let indices_shape = shape_meta.input_shapes.get(1)
        .ok_or_else(|| "gather: missing indices shape".to_string())?;

    if weight_shape.len() < 1 {
        return Err("gather: weight must be rank >= 1".to_string());
    }

    let vocab_size = weight_shape[0] as usize;
    let embed_dim: usize = weight_shape.iter().skip(1).map(|&d| d as usize).product();

    // Indices are stored as f32 in the flat buffer, but logically are i64 or i32.
    // Reinterpret the bytes.
    let indices_bytes = inputs[1].len() * 4; // f32 size
    let num_indices = indices_bytes / 8; // i64 size
    let indices: &[i64] = unsafe {
        std::slice::from_raw_parts(inputs[1].as_ptr() as *const i64, num_indices)
    };

    let num_output = output.len();
    let expected = num_indices * embed_dim;
    if num_output != expected {
        return Err(format!(
            "gather: output len {} != indices {} * embed_dim {}",
            num_output, num_indices, embed_dim,
        ));
    }

    for (i, &idx) in indices.iter().enumerate() {
        let src_start = (idx as usize) * embed_dim;
        let dst_start = i * embed_dim;
        if src_start + embed_dim > weight_table.len() {
            return Err(format!(
                "gather: index {} out of bounds (vocab_size={})",
                idx, vocab_size,
            ));
        }
        output[dst_start..dst_start + embed_dim]
            .copy_from_slice(&weight_table[src_start..src_start + embed_dim]);
    }

    Ok(())
}
"""

OP_FILL = """\
// ── fill (splat constant / arange) ─────────────────────────────────

pub fn fill_cpu(
    inputs: &[&[f32]],
    output: &mut [f32],
    shape_meta: &OpShapeMeta,
) -> Result<(), String> {
    match shape_meta.kind.as_deref() {
        Some("arange") => {
            for (i, o) in output.iter_mut().enumerate() {
                *o = i as f32;
            }
        }
        _ => {
            let value = shape_meta.value.unwrap_or(1.0) as f32;
            output.fill(value);
        }
    }
    Ok(())
}
"""

OP_UNSQUEEZE = _flat_copy_template("unsqueeze")
