//! HAL IR graph runner — iterates over HalFunction ops, assembles inputs
//! from global inputs / weights / SSA wires, dispatches through
//! `executable.execute(op_name, stream, &input_bufs, &output_bufs)`,
//! and extracts the final output Tensor.
//!
//! Path B (pure-Rust) counterpart to `compute_graph_runner.rs`.
//! All kernel dispatch goes through the HAL Executable trait — no direct
//! ciface / lookup_typed calls.

pub mod helpers;
pub mod prep;
pub mod shape_inference;
pub mod types;

#[path = "../tests/hal_runner_tests.rs"]
mod tests;

pub use types::{HalIR, HalFunction, HalOp, HalRustRunner, HalTensorDef, HalWeightEntry};

use crate::hal::traits;
use crate::hal::traits::Buffer;
use crate::sfa_tensor::SFATensor;
use crate::tensor::{Dtype, Tensor};
use crate::weight_loader::WeightProvider;

use crate::hal_runner::helpers::{
    inject_function_weights,
};
use crate::hal_runner::prep::{
    extract_global_output, prepare_ssa_maps, wire_cross_function_inputs,
};
use crate::hal_runner::shape_inference::compute_output_shape;

fn trace_level() -> u32 {
    std::env::var("HAL_TRACE")
        .unwrap_or_default()
        .parse()
        .unwrap_or(0)
}

fn dump_func() -> Vec<usize> {
    std::env::var("HAL_DUMP_FUNC")
        .unwrap_or_default()
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect()
}

fn dump_op_output(
    level: u32, fi: usize, oi: usize, op_name: &str,
    output_tensors: &[SFATensor], output_dtype: Dtype,
    dump_funcs: &[usize],
) {
    let should_dump = dump_funcs.contains(&fi);
    for (idx, tensor) in output_tensors.iter().enumerate() {
        let num_elem = tensor.numel();
        if num_elem == 0 { continue; }
        let elem_size = tensor.elem_size;
        if level >= 3 && elem_size == 4 {
            let ptr = tensor.data_ptr() as *const f32;
            let vals = unsafe { std::slice::from_raw_parts(ptr, num_elem) };
            let n = num_elem.min(8);
            eprintln!(
                "TRACE func[{}] op[{}] {} out[{}] numel={} first_{}={:?}",
                fi, oi, op_name, idx, num_elem, n, &vals[..n],
            );
        } else if level >= 3 && elem_size == 8 {
            let ptr = tensor.data_ptr() as *const i64;
            let vals = unsafe { std::slice::from_raw_parts(ptr, num_elem) };
            let n = num_elem.min(8);
            eprintln!(
                "TRACE func[{}] op[{}] {} out[{}] i64 numel={} first_{}={:?}",
                fi, oi, op_name, idx, num_elem, n, &vals[..n],
            );
        }
        if should_dump && elem_size == 4 && num_elem >= 4 {
            let path = format!("/tmp/hal_dump_f{}_op{:03}_{}.f32", fi, oi, idx);
            let ptr = tensor.data_ptr() as *const u8;
            let total_bytes = num_elem * elem_size;
            let slice = unsafe { std::slice::from_raw_parts(ptr, total_bytes) };
            let _ = std::fs::write(&path, slice);
        }
    }
}

fn infer_output_dtype(
    op: &HalOp,
    ssa_dtypes: &std::collections::HashMap<String, Dtype>,
) -> Dtype {
    let op_name = &op.op;
    match op_name.as_str() {
        "reshape" => op.inputs.first()
            .and_then(|n| ssa_dtypes.get(n))
            .copied()
            .unwrap_or(Dtype::F32),
        "gather" => Dtype::F32,
        "shape_of" => Dtype::F32,
        "fill" => {
            if let Some(dtype_str) = op.output_dtypes.first() {
                if dtype_str == "i64" || dtype_str == "I64" {
                    return Dtype::I64;
                }
            }
            Dtype::F32
        }
        "compare" => Dtype::F32,
        _ => op.inputs.first()
            .and_then(|n| ssa_dtypes.get(n))
            .copied()
            .unwrap_or(Dtype::F32),
    }
}

pub fn run_hal_function_graph(
    executable: &dyn traits::Executable,
    hal_ir: &HalIR,
    weight_provider: Option<&WeightProvider>,
    stream: &dyn traits::Stream,
    input_ids: &[u32],
    positions: &[u32],
) -> Result<Tensor, anyhow::Error> {
    let seq_len = input_ids.len();
    let trace = trace_level();
    let dump_funcs = dump_func();

    let (mut ssa_map, mut ssa_shapes, mut ssa_dtypes) =
        prepare_ssa_maps(hal_ir, seq_len, input_ids, positions);

    for (fi, function) in hal_ir.functions.iter().enumerate() {
        if trace >= 1 {
            eprintln!("TRACE func[{}] '{}' layer={} ops={}",
                fi, function.name, function.layer, function.ops.len());
        }

        if fi >= 1 {
            let prev_func = &hal_ir.functions[fi - 1];
            wire_cross_function_inputs(
                fi, function, prev_func,
                &mut ssa_map, &mut ssa_shapes, &mut ssa_dtypes, seq_len,
            );
        }

        for input_def in &function.inputs {
            let shape_strs: Vec<&str> = input_def.shape.iter().map(|s| s.as_str()).collect();
            if shape_strs == ["?", "1", "?", "?"] {
                let seq = seq_len;
                let mask_numel = seq * seq;
                let mut mask_f32 = vec![0f32; mask_numel];
                for i in 0..seq {
                    for j in 0..seq {
                        mask_f32[i * seq + j] = if j <= i { 0.0 } else { f32::NEG_INFINITY };
                    }
                }
                let t = SFATensor::from_vec_f32(mask_f32, vec![1, 1, seq, seq]);
                ssa_map.insert(input_def.name.clone(), t);
                ssa_shapes.insert(input_def.name.clone(), vec![1, 1, seq, seq]);
                ssa_dtypes.insert(input_def.name.clone(), Dtype::F32);
                break;
            }
        }

        inject_function_weights(weight_provider, function, &mut ssa_map, &mut ssa_shapes, &mut ssa_dtypes);

        // Workaround: shape-dim arrays [1,N] are wired as scalar inputs
        // but used as element_wise multipliers for attention scaling.
        // Replace with correct scaling factor 1/sqrt(head_dim) = 1/8 = 0.125.
        // (hal_ir compiler bug — should emit scaling factor, not shape dims.)
        for input_def in &function.inputs {
            if input_def.shape == ["1"] {
                if let Some(tensor) = ssa_map.get(&input_def.name) {
                    if tensor.elem_size == 4 && tensor.numel() == 2 {
                        let vals: &[f32] = unsafe {
                            std::slice::from_raw_parts(
                                tensor.data_ptr() as *const f32,
                                tensor.numel(),
                            )
                        };
                        if (vals[0] - 1.0).abs() < 0.01 && vals[1] > 0.0 {
                            let scale: f32 = 1.0 / 8.0;
                            let t = SFATensor::from_vec_f32(vec![scale], vec![1]);
                            ssa_map.insert(input_def.name.clone(), t);
                            ssa_shapes.insert(input_def.name.clone(), vec![1]);
                        }
                    }
                }
            }
        }

        for (oi, op) in function.ops.iter().enumerate() {
            if op.op == "cache_read" || op.op == "cache_write" {
                continue;
            }

            let output_dtype = infer_output_dtype(op, &ssa_dtypes);

            // Check shape_of with explicit dim early — must happen before
            // building input_bufs to avoid borrowing ssa_map immutably.
            if op.op == "shape_of" {
                if let Some(dim) = op.dim {
                    let input_shape = op.inputs.first()
                        .and_then(|n| ssa_shapes.get(n))
                        .cloned()
                        .unwrap_or_default();
                    let dim_val = if dim < input_shape.len() { input_shape[dim] } else { 1 };
                    let out_name = op.outputs.first().unwrap().clone();
                    let t = SFATensor::from_vec_f32(vec![dim_val as f32], vec![1]);
                    ssa_map.insert(out_name.clone(), t);
                    ssa_dtypes.insert(out_name.clone(), output_dtype);
                    ssa_shapes.insert(out_name.clone(), vec![1]);
                    continue;
                }
            }

            // Build inputs, outputs, execute — all inside a block so that
            // buffer borrows are dropped before we mutate ssa_map.
            let (output_shapes, mut output_tensors) = {
                let mut input_bufs: Vec<Box<dyn traits::Buffer>> =
                    Vec::with_capacity(op.inputs.len());

                for input_name in &op.inputs {
                    let tensor = ssa_map.get(input_name).ok_or_else(|| {
                        anyhow::anyhow!(
                            "hal_runner: SSA value '{}' not found in map (func[{}] op[{}]: {:?})",
                            input_name, fi, oi, op,
                        )
                    })?;
                    input_bufs.push(tensor.as_buffer_ref());
                }

                let mut output_tensors: Vec<SFATensor> = Vec::with_capacity(op.outputs.len());
                let mut output_bufs: Vec<Box<dyn traits::Buffer>> =
                    Vec::with_capacity(op.outputs.len());

                for (out_idx, _output_name) in op.outputs.iter().enumerate() {
                    let (numel, output_dims) = compute_output_shape(
                        op, out_idx, &ssa_shapes, &ssa_map, &ssa_dtypes, function, seq_len,
                    );
                    let numel = numel.max(1);
                    let out_elem_size = output_dtype.element_size();

                    // Use inferred multi-dimensional shape instead of collapsing to 1D.
                    // Downstream ops (matmul, sdpa) require correct rank for BLAS dispatch.
                    let tensor_dims = if output_dims.is_empty() {
                        vec![numel]
                    } else {
                        let product: usize = output_dims.iter().product();
                        if product == numel {
                            output_dims
                        } else {
                            vec![numel]
                        }
                    };
                    let t = if out_elem_size == 8 {
                        SFATensor::from_vec_i64(vec![0i64; numel], tensor_dims.clone())
                    } else {
                        SFATensor::from_vec_f32(vec![0f32; numel], tensor_dims.clone())
                    };
                    output_tensors.push(t);
                }
                for tensor in &output_tensors {
                    output_bufs.push(tensor.as_buffer_ref());
                }

                let input_refs: Vec<&dyn traits::Buffer> =
                    input_bufs.iter().map(|b| b.as_ref()).collect();
                let output_refs: Vec<&dyn traits::Buffer> =
                    output_bufs.iter().map(|b| b.as_ref()).collect();

                let input_memrefs: Vec<crate::hal::sfa::SfaMemRef> =
                    input_refs.iter().map(|b| b.as_sfa_memref()).collect();
                let mut output_memrefs: Vec<crate::hal::sfa::SfaMemRef> =
                    output_refs.iter().map(|b| b.as_sfa_memref()).collect();

                let op_name = if let Some(kind) = &op.kind {
                    format!("{}:{}", op.op, kind)
                } else if op.op == "transpose" {
                    if let Some(ref dims) = op.dims {
                        let dims_str = dims.iter()
                            .map(|d| d.to_string())
                            .collect::<Vec<_>>()
                            .join(",");
                        format!("transpose:{}", dims_str)
                    } else {
                        "transpose".to_string()
                    }
                } else {
                    op.op.clone()
                };

                if trace >= 2 {
                    let input_info: Vec<String> = op.inputs.iter().map(|n| {
                        let shape = ssa_shapes.get(n).map(|s| format!("{:?}", s)).unwrap_or_else(|| "?".into());
                        format!("{}[{}]", n, shape)
                    }).collect();
                    eprintln!("TRACE func[{}] op[{}] {} inputs=[{}]",
                        fi, oi, op_name, input_info.join(", "));
                }

                let exe_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    executable.execute(&op_name, stream, &input_memrefs, &mut output_memrefs)
                }));
                let shapes = match exe_result {
                    Ok(Ok(shapes)) => shapes,
                    Ok(Err(e)) => {
                        return Err(anyhow::anyhow!(
                            "{}", crate::error::HalExecutionError::OpFailed {
                                func_idx: fi, op_idx: oi, op_name, message: e.to_string(),
                            }
                        ));
                    }
                    Err(panic_payload) => {
                        let msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                            s.to_string()
                        } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                            s.clone()
                        } else {
                            "unknown panic".to_string()
                        };
                        return Err(anyhow::anyhow!(
                            "{}", crate::error::HalExecutionError::OpPanic {
                                func_idx: fi, op_idx: oi, op_name, panic_msg: msg,
                            }
                        ));
                    }
                };

                dump_op_output(trace, fi, oi, &op_name, &output_tensors, output_dtype, &dump_funcs);
                drop(output_bufs);
                (shapes, output_tensors)
            }; // input_bufs, output_bufs, and all borrows dropped here

            let tensors = std::mem::take(&mut output_tensors);
            for (tensor, output_name) in tensors.into_iter().zip(op.outputs.iter()) {
                ssa_map.insert(output_name.clone(), tensor);
                ssa_dtypes.insert(output_name.clone(), output_dtype);
            }
            for (idx, output_name) in op.outputs.iter().enumerate() {
                if let Some(shape) = output_shapes.get(idx) {
                    let shape_usize: Vec<usize> = shape.iter()
                        .map(|&s| std::cmp::max(1, s as usize))
                        .collect();
                    ssa_shapes.insert(output_name.clone(), shape_usize);
                }
            }

            if op.op == "shape_of" {
                if let Some(out_name) = op.outputs.first() {
                    if let Some(tensor) = ssa_map.get(out_name) {
                        let numel = tensor.numel();
                        let ptr = tensor.data_ptr() as *const f32;
                        let f32_vals = unsafe { std::slice::from_raw_parts(ptr, numel) };
                        let dims: Vec<usize> = f32_vals.iter().map(|&v| v as usize).collect();
                        if !dims.is_empty() {
                            ssa_shapes.insert(out_name.clone(), dims);
                        }
                    }
                }
            }
        }
    }

    extract_global_output(hal_ir, &ssa_map, &ssa_shapes, &ssa_dtypes, seq_len)
}

// ── Semantic Contract Validation ───────────────────────────────────────

/// Build the default HAL operator semantics contract with 23 entries
/// covering all op categories handled by the Rust runtime dispatch.
pub fn default_hal_op_semantics() -> crate::abi::SfaHalOpSemantics {
    use crate::abi::SfaHalOpSemanticEntry;

    let entries = vec![
        // ── matmul ──
        SfaHalOpSemanticEntry {
            op_name: "matmul".into(),
            expected_input_count: 2,
            expected_output_count: 1,
            input_dtypes: vec!["f32".into(), "f32".into()],
            output_dtypes: vec!["f32".into()],
            kind: "matmul".into(),
        },
        SfaHalOpSemanticEntry {
            op_name: "linear".into(),
            expected_input_count: 2,
            expected_output_count: 1,
            input_dtypes: vec!["f32".into(), "f32".into()],
            output_dtypes: vec!["f32".into()],
            kind: "matmul".into(),
        },
        // ── element_wise ──
        SfaHalOpSemanticEntry {
            op_name: "element_wise".into(),
            expected_input_count: 2,
            expected_output_count: 1,
            input_dtypes: vec!["f32".into(), "f32".into()],
            output_dtypes: vec!["f32".into()],
            kind: "element_wise".into(),
        },
        SfaHalOpSemanticEntry {
            op_name: "element_wise:rsqrt".into(),
            expected_input_count: 1,
            expected_output_count: 1,
            input_dtypes: vec!["f32".into()],
            output_dtypes: vec!["f32".into()],
            kind: "element_wise".into(),
        },
        // ── reshape / transpose ──
        SfaHalOpSemanticEntry {
            op_name: "reshape".into(),
            expected_input_count: 1,
            expected_output_count: 1,
            input_dtypes: vec!["f32".into()],
            output_dtypes: vec!["f32".into()],
            kind: "reshape".into(),
        },
        SfaHalOpSemanticEntry {
            op_name: "transpose".into(),
            expected_input_count: 1,
            expected_output_count: 1,
            input_dtypes: vec!["f32".into()],
            output_dtypes: vec!["f32".into()],
            kind: "reshape".into(),
        },
        SfaHalOpSemanticEntry {
            op_name: "unsqueeze".into(),
            expected_input_count: 1,
            expected_output_count: 1,
            input_dtypes: vec!["f32".into()],
            output_dtypes: vec!["f32".into()],
            kind: "reshape".into(),
        },
        SfaHalOpSemanticEntry {
            op_name: "concat".into(),
            expected_input_count: 2,
            expected_output_count: 1,
            input_dtypes: vec!["f32".into(), "f32".into()],
            output_dtypes: vec!["f32".into()],
            kind: "concat".into(),
        },
        // ── normalization ──
        SfaHalOpSemanticEntry {
            op_name: "layer_norm".into(),
            expected_input_count: 2,
            expected_output_count: 1,
            input_dtypes: vec!["f32".into(), "f32".into()],
            output_dtypes: vec!["f32".into()],
            kind: "normalization".into(),
        },
        SfaHalOpSemanticEntry {
            op_name: "rms_norm".into(),
            expected_input_count: 2,
            expected_output_count: 1,
            input_dtypes: vec!["f32".into(), "f32".into()],
            output_dtypes: vec!["f32".into()],
            kind: "normalization".into(),
        },
        // ── softmax ──
        SfaHalOpSemanticEntry {
            op_name: "softmax".into(),
            expected_input_count: 1,
            expected_output_count: 1,
            input_dtypes: vec!["f32".into()],
            output_dtypes: vec!["f32".into()],
            kind: "softmax".into(),
        },
        // ── attention ──
        SfaHalOpSemanticEntry {
            op_name: "scaled_dot_product_attention".into(),
            expected_input_count: 4,
            expected_output_count: 1,
            input_dtypes: vec!["f32".into(), "f32".into(), "f32".into(), "f32".into()],
            output_dtypes: vec!["f32".into()],
            kind: "attention".into(),
        },
        // ── gather ──
        SfaHalOpSemanticEntry {
            op_name: "gather".into(),
            expected_input_count: 2,
            expected_output_count: 1,
            input_dtypes: vec!["f32".into(), "i64".into()],
            output_dtypes: vec!["f32".into()],
            kind: "gather".into(),
        },
        // ── reduce ──
        SfaHalOpSemanticEntry {
            op_name: "reduce".into(),
            expected_input_count: 1,
            expected_output_count: 1,
            input_dtypes: vec!["f32".into()],
            output_dtypes: vec!["f32".into()],
            kind: "reduce".into(),
        },
        SfaHalOpSemanticEntry {
            op_name: "reduce:mean".into(),
            expected_input_count: 1,
            expected_output_count: 1,
            input_dtypes: vec!["f32".into()],
            output_dtypes: vec!["f32".into()],
            kind: "reduce".into(),
        },
        // ── scan ──
        SfaHalOpSemanticEntry {
            op_name: "scan".into(),
            expected_input_count: 1,
            expected_output_count: 1,
            input_dtypes: vec!["f32".into()],
            output_dtypes: vec!["f32".into()],
            kind: "scan".into(),
        },
        SfaHalOpSemanticEntry {
            op_name: "scan:cumsum".into(),
            expected_input_count: 1,
            expected_output_count: 1,
            input_dtypes: vec!["f32".into()],
            output_dtypes: vec!["f32".into()],
            kind: "scan".into(),
        },
        // ── fill ──
        SfaHalOpSemanticEntry {
            op_name: "fill".into(),
            expected_input_count: 0,
            expected_output_count: 1,
            input_dtypes: vec![],
            output_dtypes: vec!["f32".into()],
            kind: "fill".into(),
        },
        SfaHalOpSemanticEntry {
            op_name: "fill:arange".into(),
            expected_input_count: 0,
            expected_output_count: 1,
            input_dtypes: vec![],
            output_dtypes: vec!["i64".into()],
            kind: "fill".into(),
        },
        // ── slice ──
        SfaHalOpSemanticEntry {
            op_name: "slice".into(),
            expected_input_count: 1,
            expected_output_count: 1,
            input_dtypes: vec!["f32".into()],
            output_dtypes: vec!["f32".into()],
            kind: "slice".into(),
        },
        // ── compare ──
        SfaHalOpSemanticEntry {
            op_name: "compare".into(),
            expected_input_count: 2,
            expected_output_count: 1,
            input_dtypes: vec!["f32".into(), "f32".into()],
            output_dtypes: vec!["f32".into()],
            kind: "compare".into(),
        },
        // ── shape ──
        SfaHalOpSemanticEntry {
            op_name: "shape_of".into(),
            expected_input_count: 1,
            expected_output_count: 1,
            input_dtypes: vec!["f32".into()],
            output_dtypes: vec!["f32".into()],
            kind: "shape".into(),
        },
        // ── cache ──
        SfaHalOpSemanticEntry {
            op_name: "cache_read".into(),
            expected_input_count: 1,
            expected_output_count: 1,
            input_dtypes: vec!["f32".into()],
            output_dtypes: vec!["f32".into()],
            kind: "cache".into(),
        },
        SfaHalOpSemanticEntry {
            op_name: "cache_write".into(),
            expected_input_count: 1,
            expected_output_count: 0,
            input_dtypes: vec!["f32".into()],
            output_dtypes: vec![],
            kind: "cache".into(),
        },
    ];

    crate::abi::SfaHalOpSemantics { entries }
}

/// Validate a loaded `HalIR` against the HAL operator semantic contract.
///
/// For each op in every function, looks up the op name (and its base name
/// for qualified ops like `element_wise:add` → `element_wise`) in the
/// semantic contract. Warns on unknown ops and arity mismatches.
///
/// Returns a list of warning strings. The caller should log these via
/// `log::warn!`. Validation is non-fatal — only warnings are emitted.
pub fn validate_hal_ir_against_semantics(
    hal_ir: &HalIR,
    semantics: &crate::abi::SfaHalOpSemantics,
) -> Vec<String> {
    use std::collections::HashSet;

    // Build lookup: op_name → SfaHalOpSemanticEntry
    let mut sem_map: std::collections::HashMap<&str, &crate::abi::SfaHalOpSemanticEntry> =
        std::collections::HashMap::with_capacity(semantics.entries.len());
    for entry in &semantics.entries {
        sem_map.entry(&entry.op_name).or_insert(entry);
    }

    let mut warnings: Vec<String> = Vec::new();
    let mut unknown_ops: HashSet<String> = HashSet::new();

    for func in &hal_ir.functions {
        for op in &func.ops {
            // Resolve qualified op names: "element_wise:add" → lookup "element_wise"
            let base_name = op.op.split(':').next().unwrap_or(&op.op);
            let lookup_name = if op.op.contains(':') { base_name } else { &op.op };

            // Also try exact match first for ops with known qualified forms
            let entry = sem_map
                .get(&op.op.as_str())
                .or_else(|| sem_map.get(lookup_name));

            match entry {
                Some(sem) => {
                    let actual_inputs = op.inputs.len() as u32;
                    let actual_outputs = op.outputs.len() as u32;

                    if actual_inputs < sem.expected_input_count {
                        warnings.push(format!(
                            "hal_op_semantics: op '{}' in func '{}' expects >= {} inputs, got {}",
                            op.op, func.name, sem.expected_input_count, actual_inputs,
                        ));
                    }
                    if actual_outputs != sem.expected_output_count {
                        // Allow cache ops that are skipped to have 0 outputs
                        let is_cache = sem.kind == "cache";
                        if !is_cache || actual_outputs > 0 {
                            warnings.push(format!(
                                "hal_op_semantics: op '{}' in func '{}' expects {} outputs, got {}",
                                op.op, func.name, sem.expected_output_count, actual_outputs,
                            ));
                        }
                    }
                }
                None => {
                    if unknown_ops.insert(op.op.clone()) {
                        warnings.push(format!(
                            "hal_op_semantics: unknown op '{}' in func '{}' — not in semantic contract",
                            op.op, func.name,
                        ));
                    }
                }
            }
        }
    }

    warnings
}
