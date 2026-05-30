"""EmitRust backend — generate hal_ops_cpu.rs from hal_ir.json.

Reads a compiled model's ``hal_ir.json``, collects all unique op types,
and emits a Rust source file that dispatches to the primitives library
(``crate::hal::primitives::*``).

Usage::

    from compiler.mlir_dialect.hal_ir.emit_rust import emit_rust

    path = emit_rust("compiled/opt_125m_kv/hal_ir.json")
    print(f"Generated: {path}")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


# ── Op → primitive dispatch mapping ────────────────────────────────────

# Maps (op_type, kind) → Rust expression that calls the primitive.
# The expression receives: inputs: &[&[f32]], output: &mut [f32], meta: &OpShapeMeta
OP_DISPATCH: dict[tuple[str, str | None], str] = {
    # Binary ops
    ("element_wise", "add"): "{ crate::hal::primitives::vec_add(inputs[0], inputs[1], output); Ok(()) }",
    ("element_wise", "sub"): "{ crate::hal::primitives::vec_sub(inputs[0], inputs[1], output); Ok(()) }",
    ("element_wise", "mul"): "{ crate::hal::primitives::vec_mul(inputs[0], inputs[1], output); Ok(()) }",
    ("element_wise", "div"): "{ crate::hal::primitives::vec_div(inputs[0], inputs[1], output); Ok(()) }",
    ("element_wise", "max"): "{ crate::hal::primitives::vec_max(inputs[0], inputs[1], output); Ok(()) }",
    ("element_wise", "pow"): "{ crate::hal::primitives::vec_pow(inputs[0], inputs[1], output); Ok(()) }",
    # Unary ops
    ("element_wise", "relu"): "{ crate::hal::primitives::vec_relu(inputs[0], output); Ok(()) }",
    ("element_wise", "exp"): "{ crate::hal::primitives::vec_exp(inputs[0], output); Ok(()) }",
    ("element_wise", "sqrt"): "{ crate::hal::primitives::vec_sqrt(inputs[0], output); Ok(()) }",
    ("element_wise", "rsqrt"): "{ crate::hal::primitives::vec_rsqrt(inputs[0], output); Ok(()) }",
    ("element_wise", "tanh"): "{ crate::hal::primitives::vec_tanh(inputs[0], output); Ok(()) }",
    ("element_wise", "sigmoid"): "{ crate::hal::primitives::vec_sigmoid(inputs[0], output); Ok(()) }",
    ("element_wise", "neg"): "{ crate::hal::primitives::vec_neg(inputs[0], output); Ok(()) }",
    ("element_wise", "cos"): "{ crate::hal::primitives::vec_cos(inputs[0], output); Ok(()) }",
    ("element_wise", "sin"): "{ crate::hal::primitives::vec_sin(inputs[0], output); Ok(()) }",
    ("element_wise", "softplus"): "{ crate::hal::primitives::vec_softplus(inputs[0], output); Ok(()) }",
    ("element_wise", "silu"): "{ crate::hal::primitives::vec_silu(inputs[0], output); Ok(()) }",
    ("element_wise", "gelu"): "{ crate::hal::primitives::vec_gelu(inputs[0], output); Ok(()) }",
    # Reduce ops
    ("reduce", "sum"): "dispatch_reduce_sum(inputs, output, shape_meta)",
    ("reduce", "mean"): "dispatch_reduce_mean(inputs, output, shape_meta)",
    ("reduce", "max"): "dispatch_reduce_max(inputs, output, shape_meta)",
    # Matmul
    ("matmul", None): "dispatch_matmul(inputs, output, shape_meta)",
    # Softmax (fused)
    ("softmax", None): "dispatch_softmax(inputs, output, shape_meta)",
    # Gather
    ("gather", None): "crate::hal::primitives::gather_f32(inputs[0], inputs[1], output, embed_dim_from_shape(shape_meta)).map_err(|e| e.to_string())",
    # Transpose
    ("transpose", None): "dispatch_transpose(inputs, output, shape_meta)",
    # Memory ops (flat copy)
    ("reshape", None): "{ output.copy_from_slice(inputs[0]); Ok(()) }",
    ("unsqueeze", None): "{ output.copy_from_slice(inputs[0]); Ok(()) }",
    # Fill
    ("fill", None): "dispatch_fill(output, shape_meta)",
    ("fill", "arange"): "dispatch_fill(output, shape_meta)",
    # Shape of
    ("shape_of", None): "{ crate::hal::primitives::shape_of(shape_meta.input_shapes.get(0).unwrap_or(&vec![]), output); Ok(()) }",
    # Slice
    ("slice", None): "dispatch_slice(inputs, output, shape_meta)",
    # Compare
    ("compare", "le"): "dispatch_compare_le(inputs, output)",
    ("compare", "lt"): "dispatch_compare_lt(inputs, output)",
    ("compare", "gt"): "dispatch_compare_gt(inputs, output)",
    ("compare", "ge"): "dispatch_compare_ge(inputs, output)",
    ("compare", "eq"): "dispatch_compare_eq(inputs, output)",
    ("compare", "ne"): "dispatch_compare_ne(inputs, output)",
    ("compare", "logical_and"): "dispatch_compare_logical_and(inputs, output)",
    # Cache stubs
    ("cache_read", None): "Ok(())",
    ("cache_write", None): "Ok(())",
}


# ── Generated dispatch helpers ─────────────────────────────────────────

DISPATCH_HELPERS = """\
// ── Dispatch helpers (generated) ──────────────────────────────────────

fn embed_dim_from_shape(meta: &OpShapeMeta) -> usize {
    meta.input_shapes.get(0)
        .map(|s| s.iter().skip(1).map(|&d| d as usize).product())
        .unwrap_or(768)
}

fn dispatch_reduce_sum(inputs: &[&[f32]], output: &mut [f32], meta: &OpShapeMeta) -> Result<(), String> {
    let last_dim = meta.output_shape.last().copied().unwrap_or(1) as usize;
    crate::hal::primitives::reduce_sum_last_dim(inputs[0], output, last_dim);
    Ok(())
}

fn dispatch_reduce_mean(inputs: &[&[f32]], output: &mut [f32], meta: &OpShapeMeta) -> Result<(), String> {
    let last_dim = meta.output_shape.last().copied().unwrap_or(1) as usize;
    crate::hal::primitives::reduce_mean_last_dim(inputs[0], output, last_dim);
    Ok(())
}

fn dispatch_reduce_max(inputs: &[&[f32]], output: &mut [f32], meta: &OpShapeMeta) -> Result<(), String> {
    let last_dim = meta.output_shape.last().copied().unwrap_or(1) as usize;
    crate::hal::primitives::reduce_max_last_dim(inputs[0], output, last_dim);
    Ok(())
}

fn dispatch_matmul(inputs: &[&[f32]], output: &mut [f32], meta: &OpShapeMeta) -> Result<(), String> {
    let a_shape = meta.input_shapes.get(0)
        .ok_or_else(|| "matmul: missing input shape 0".to_string())?;
    let b_shape = meta.input_shapes.get(1)
        .ok_or_else(|| "matmul: missing input shape 1".to_string())?;
    crate::hal::primitives::matmul_blas(inputs[0], inputs[1], output, a_shape, b_shape, true)
}

fn dispatch_softmax(inputs: &[&[f32]], output: &mut [f32], meta: &OpShapeMeta) -> Result<(), String> {
    let last_dim = meta.output_shape.last()
        .or_else(|| meta.input_shapes.get(0)?.last())
        .copied()
        .unwrap_or(1) as usize;
    crate::hal::primitives::fused_softmax(inputs[0], output, last_dim);
    Ok(())
}

fn dispatch_transpose(inputs: &[&[f32]], output: &mut [f32], meta: &OpShapeMeta) -> Result<(), String> {
    let input_shape = meta.input_shapes.get(0)
        .ok_or_else(|| "transpose: missing input shape".to_string())?;
    let output_shape = &meta.output_shape;
    let rank = input_shape.len();
    let perm: Vec<usize> = if let Some(kind) = &meta.kind {
        kind.split(',').filter_map(|s| s.trim().parse::<usize>().ok()).collect()
    } else {
        let mut p: Vec<usize> = (0..rank).collect();
        p.swap(rank - 2, rank - 1);
        p
    };
    crate::hal::primitives::transpose_nd(inputs[0], output, input_shape, output_shape, &perm)
}

fn dispatch_fill(output: &mut [f32], meta: &OpShapeMeta) -> Result<(), String> {
    match meta.kind.as_deref() {
        Some("arange") => {
            for (i, o) in output.iter_mut().enumerate() {
                *o = i as f32;
            }
        }
        _ => {
            let value = meta.value.unwrap_or(1.0) as f32;
            output.fill(value);
        }
    }
    Ok(())
}

fn dispatch_slice(inputs: &[&[f32]], output: &mut [f32], meta: &OpShapeMeta) -> Result<(), String> {
    let inp = inputs[0];
    let input_shape = meta.input_shapes.get(0)
        .ok_or_else(|| "slice: missing input shape".to_string())?;
    let output_shape = &meta.output_shape;
    if input_shape.is_empty() {
        return Err("slice: empty input shape".to_string());
    }
    let slice_dim = input_shape.iter().zip(output_shape.iter())
        .position(|(&i, &o)| i != o).unwrap_or(0);
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
            output[dst_off..dst_off + inner].copy_from_slice(&inp[src_off..src_off + inner]);
        }
    }
    Ok(())
}

fn dispatch_compare_le(inputs: &[&[f32]], output: &mut [f32]) -> Result<(), String> {
    for (o, (&a, &b)) in output.iter_mut().zip(inputs[0].iter().zip(inputs[1].iter())) {
        *o = if a <= b { 1.0 } else { 0.0 };
    }
    Ok(())
}

fn dispatch_compare_lt(inputs: &[&[f32]], output: &mut [f32]) -> Result<(), String> {
    for (o, (&a, &b)) in output.iter_mut().zip(inputs[0].iter().zip(inputs[1].iter())) {
        *o = if a < b { 1.0 } else { 0.0 };
    }
    Ok(())
}

fn dispatch_compare_gt(inputs: &[&[f32]], output: &mut [f32]) -> Result<(), String> {
    for (o, (&a, &b)) in output.iter_mut().zip(inputs[0].iter().zip(inputs[1].iter())) {
        *o = if a > b { 1.0 } else { 0.0 };
    }
    Ok(())
}

fn dispatch_compare_ge(inputs: &[&[f32]], output: &mut [f32]) -> Result<(), String> {
    for (o, (&a, &b)) in output.iter_mut().zip(inputs[0].iter().zip(inputs[1].iter())) {
        *o = if a >= b { 1.0 } else { 0.0 };
    }
    Ok(())
}

fn dispatch_compare_eq(inputs: &[&[f32]], output: &mut [f32]) -> Result<(), String> {
    for (o, (&a, &b)) in output.iter_mut().zip(inputs[0].iter().zip(inputs[1].iter())) {
        *o = if (a - b).abs() < 1e-6 { 1.0 } else { 0.0 };
    }
    Ok(())
}

fn dispatch_compare_ne(inputs: &[&[f32]], output: &mut [f32]) -> Result<(), String> {
    for (o, (&a, &b)) in output.iter_mut().zip(inputs[0].iter().zip(inputs[1].iter())) {
        *o = if (a - b).abs() >= 1e-6 { 1.0 } else { 0.0 };
    }
    Ok(())
}

fn dispatch_compare_logical_and(inputs: &[&[f32]], output: &mut [f32]) -> Result<(), String> {
    for (o, (&a, &b)) in output.iter_mut().zip(inputs[0].iter().zip(inputs[1].iter())) {
        *o = if a != 0.0 && b != 0.0 { 1.0 } else { 0.0 };
    }
    Ok(())
}
"""


# ── Header ─────────────────────────────────────────────────────────────

HEADER = """\
//! Auto-generated by EmitRust. Do not edit.
//! Source: hal_ir.json for {model_name}
//!
//! Dispatch layer — maps HAL IR op names to primitives in crate::hal::primitives.
//! Optimised implementations live in primitives/; this file is the glue.

#![allow(unused_variables, dead_code, unused_imports)]

use crate::hal::primitives;

/// Shape metadata for HAL CPU operations.
#[derive(Debug, Clone)]
pub struct OpShapeMeta {{
    pub input_shapes: Vec<Vec<i64>>,
    pub output_shape: Vec<i64>,
    pub kind: Option<String>,
    pub value: Option<f64>,
}}

impl OpShapeMeta {{
    pub fn new(input_shapes: Vec<Vec<i64>>, output_shape: Vec<i64>) -> Self {{
        Self {{ input_shapes, output_shape, kind: None, value: None }}
    }}
}}

"""


# ── Main generation function ────────────────────────────────────────────


def emit_rust(
    hal_ir_path: str | Path,
    output_path: str | Path | None = None,
) -> str:
    """Generate ``hal_ops_cpu.rs`` from ``hal_ir.json``.

    Args:
        hal_ir_path: Path to ``hal_ir.json``.
        output_path: Where to write the Rust file. If ``None``, writes to
            the same directory as ``hal_ir.json`` with name ``hal_ops_cpu.rs``.

    Returns:
        Absolute path to the written Rust file.
    """
    hal_ir_path = Path(hal_ir_path)
    if not hal_ir_path.exists():
        raise FileNotFoundError(f"hal_ir.json not found: {hal_ir_path}")

    data = json.loads(hal_ir_path.read_text())
    model_name: str = data.get("model_name", "unknown")
    functions: list[dict[str, Any]] = data.get("functions", [])

    # Collect unique op types in use
    op_types_in_use: set[str] = set()
    for fn in functions:
        for op in fn.get("ops", []):
            op_types_in_use.add(op["op"])

    _log.info(
        "EmitRust: model=%s functions=%d unique_ops=%s",
        model_name,
        len(functions),
        sorted(op_types_in_use),
    )

    # ── Assemble Rust source ──────────────────────────────────────────

    parts: list[str] = []

    # Header with OpShapeMeta struct
    parts.append(HEADER.format(model_name=model_name))

    # BLAS extern declaration (always needed for matmul)
    parts.append("""\
// BLAS FFI (linked via build.rs)
#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemm(
        order: i32, transA: i32, transB: i32,
        m: i32, n: i32, k: i32,
        alpha: f32, a: *const f32, lda: i32,
        b: *const f32, ldb: i32,
        beta: f32, c: *mut f32, ldc: i32,
    );
}

#[cfg(not(target_os = "macos"))]
extern "C" {
    fn cblas_sgemm(
        order: i32, transA: i32, transB: i32,
        m: i32, n: i32, k: i32,
        alpha: f32, a: *const f32, lda: i32,
        b: *const f32, ldb: i32,
        beta: f32, c: *mut f32, ldc: i32,
    );
}

""")

    # Dispatch helpers
    parts.append(DISPATCH_HELPERS)

    # Generate dispatch function
    dispatch_lines = ["pub fn dispatch("]
    dispatch_lines.append("    op_name: &str,")
    dispatch_lines.append("    inputs: &[&[f32]],")
    dispatch_lines.append("    output: &mut [f32],")
    dispatch_lines.append("    shape_meta: &OpShapeMeta,")
    dispatch_lines.append(") -> Result<(), String> {")
    dispatch_lines.append("    match op_name {")

    for op_type in sorted(op_types_in_use):
        if op_type in ("cache_read", "cache_write"):
            key = (op_type, None)
            if key in OP_DISPATCH:
                dispatch_lines.append(f'        "{op_type}" => {{ {OP_DISPATCH[key]} }},')
            continue

        # Get all kinds for this op type
        kinds: set[str | None] = set()
        for fn in functions:
            for op in fn.get("ops", []):
                if op["op"] == op_type:
                    kinds.add(op.get("kind"))

        for kind in sorted(kinds, key=lambda k: (k is None, k)):
            key = (op_type, kind)
            if key in OP_DISPATCH:
                expr = OP_DISPATCH[key]
                if kind is not None:
                    dispatch_lines.append(f'        "{op_type}:{kind}" => {{ {expr} }},')
                else:
                    dispatch_lines.append(f'        "{op_type}" => {{ {expr} }},')
            else:
                _log.warning("EmitRust: no dispatch for (%s, %s) — skipping", op_type, kind)

    # Always include cache ops (they may not be in hal_ir.json but are tested)
    if "cache_read" not in op_types_in_use:
        dispatch_lines.append('        "cache_read" => { Ok(()) },')
    if "cache_write" not in op_types_in_use:
        dispatch_lines.append('        "cache_write" => { Ok(()) },')

    dispatch_lines.append('        other => Err(format!("unknown op: {}", other)),')
    dispatch_lines.append("    }")
    dispatch_lines.append("}")
    parts.append("\n".join(dispatch_lines))

    rust_source = "\n".join(parts)

    # ── Write output ──────────────────────────────────────────────────

    if output_path is None:
        output_path = hal_ir_path.parent / "hal_ops_cpu.rs"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rust_source)

    _log.info("EmitRust: wrote %s (%d bytes)", output_path, len(rust_source))
    return str(output_path.resolve())


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python -m compiler.mlir_dialect.hal_ir.emit_rust <hal_ir.json> [output.rs]")
        sys.exit(1)

    hal_ir_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    path = emit_rust(hal_ir_path, output_path)
    print(f"Generated: {path}")
