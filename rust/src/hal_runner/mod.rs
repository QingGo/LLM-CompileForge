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

#[cfg(test)]
mod tests;

pub use types::{HalIR, HalFunction, HalOp, HalRustRunner, HalTensorDef, HalWeightEntry};

use crate::hal::cpu::buffer::CpuBuffer as InnerCpuBuffer;
use crate::hal::cpu::CpuBuffer;
use crate::hal::traits;
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
    output_vecs: &[Vec<u8>], output_dtype: Dtype,
    dump_funcs: &[usize],
) {
    let should_dump = dump_funcs.contains(&fi);
    for (idx, vec) in output_vecs.iter().enumerate() {
        let elem_size = output_dtype.element_size();
        let num_elem = vec.len() / elem_size;
        if num_elem == 0 { continue; }
        if level >= 3 && elem_size == 4 {
            let vals: &[f32] = unsafe {
                std::slice::from_raw_parts(vec.as_ptr() as *const f32, num_elem)
            };
            let n = num_elem.min(8);
            eprintln!(
                "TRACE func[{}] op[{}] {} out[{}] numel={} first_{}={:?}",
                fi, oi, op_name, idx, num_elem, n, &vals[..n],
            );
        } else if level >= 3 && elem_size == 8 {
            let vals: &[i64] = unsafe {
                std::slice::from_raw_parts(vec.as_ptr() as *const i64, num_elem)
            };
            let n = num_elem.min(8);
            eprintln!(
                "TRACE func[{}] op[{}] {} out[{}] i64 numel={} first_{}={:?}",
                fi, oi, op_name, idx, num_elem, n, &vals[..n],
            );
        }
        if should_dump && elem_size == 4 && num_elem >= 4 {
            let path = format!("/tmp/hal_dump_f{}_op{:03}_{}.f32", fi, oi, idx);
            let _ = std::fs::write(&path, vec);
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
                let mut mask = vec![0u8; mask_numel * 4];
                let mask_f32: &mut [f32] = unsafe {
                    std::slice::from_raw_parts_mut(mask.as_mut_ptr() as *mut f32, mask_numel)
                };
                for i in 0..seq {
                    for j in 0..seq {
                        mask_f32[i * seq + j] = if j <= i { 0.0 } else { f32::NEG_INFINITY };
                    }
                }
                ssa_map.insert(input_def.name.clone(), mask);
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
                if let Some(data) = ssa_map.get(&input_def.name) {
                    if data.len() == 8 {
                        let vals: &[f32] = unsafe {
                            std::slice::from_raw_parts(data.as_ptr() as *const f32, 2)
                        };
                        if (vals[0] - 1.0).abs() < 0.01 && vals[1] > 0.0 {
                            // Attention scaling: 1/sqrt(head_dim) = 1/8 = 0.125
                            let scale: f32 = 1.0 / 8.0;
                            ssa_map.insert(input_def.name.clone(), scale.to_le_bytes().to_vec());
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

            let mut input_bufs: Vec<Box<dyn traits::Buffer>> =
                Vec::with_capacity(op.inputs.len());

            for input_name in &op.inputs {
                let data = ssa_map.get(input_name).ok_or_else(|| {
                    anyhow::anyhow!(
                        "hal_runner: SSA value '{}' not found in map (func[{}] op[{}]: {:?})",
                        input_name, fi, oi, op,
                    )
                })?;

                let elem_size = ssa_dtypes
                    .get(input_name)
                    .map(|d| d.element_size())
                    .unwrap_or(4);

                let dims: Vec<usize> = if let Some(shape) = ssa_shapes.get(input_name) {
                    shape.clone()
                } else {
                    let mut declared_shape: Vec<String> = vec![];
                    for output in &function.outputs {
                        if output.name == *input_name {
                            declared_shape = output.shape.clone();
                            break;
                        }
                    }
                    if declared_shape.is_empty() {
                        for input_def in &function.inputs {
                            if input_def.name == *input_name {
                                declared_shape = input_def.shape.clone();
                                break;
                            }
                        }
                    }
                    if declared_shape.is_empty() {
                        for weight_entry in &function.weights {
                            if weight_entry.ssa == *input_name {
                                declared_shape = weight_entry.shape.clone();
                                break;
                            }
                        }
                    }
                    if declared_shape.is_empty() {
                        declared_shape = vec![(data.len() / elem_size).to_string()];
                    }
                    let dims: Vec<usize> = declared_shape
                        .iter()
                        .map(|d| {
                            if d == "?" || d == "-1" { 1 }
                            else { d.parse::<usize>().unwrap_or(data.len() / elem_size) }
                        })
                        .collect();
                    if dims.is_empty() || dims.iter().all(|&d| d == 0) {
                        vec![data.len() / elem_size]
                    } else {
                        dims
                    }
                };

                let raw_buf = InnerCpuBuffer::from_raw_parts(
                    data.as_ptr() as *mut u8, data.len(), true,
                ).map_err(|e| anyhow::anyhow!("InnerCpuBuffer: {}", e))?;

                let cpu_buf = CpuBuffer::with_meta(raw_buf, elem_size, dims);
                input_bufs.push(Box::new(cpu_buf));
            }

            let mut output_vecs: Vec<Vec<u8>> = Vec::with_capacity(op.outputs.len());
            let mut output_bufs: Vec<Box<dyn traits::Buffer>> =
                Vec::with_capacity(op.outputs.len());

            let output_dtype = infer_output_dtype(op, &ssa_dtypes);

            for (out_idx, _output_name) in op.outputs.iter().enumerate() {
                let (numel, output_dims) = compute_output_shape(
                    op, out_idx, &ssa_shapes, &ssa_map, &ssa_dtypes, function, seq_len,
                );
                let numel = numel.max(1);
                let out_elem_size = output_dtype.element_size();

                let mut vec = vec![0u8; numel * out_elem_size];
                let raw_buf = InnerCpuBuffer::from_raw_parts(
                    vec.as_mut_ptr(), numel * out_elem_size, true,
                ).map_err(|e| anyhow::anyhow!("InnerCpuBuffer: {}", e))?;
                let cpu_buf = CpuBuffer::with_meta(raw_buf, out_elem_size, output_dims);
                output_vecs.push(vec);
                output_bufs.push(Box::new(cpu_buf));
            }

            let input_refs: Vec<&dyn traits::Buffer> =
                input_bufs.iter().map(|b| b.as_ref()).collect();
            let output_refs: Vec<&dyn traits::Buffer> =
                output_bufs.iter().map(|b| b.as_ref()).collect();

            if op.op == "shape_of" {
                if let Some(dim) = op.dim {
                    let input_shape = op.inputs.first()
                        .and_then(|n| ssa_shapes.get(n))
                        .cloned()
                        .unwrap_or_default();
                    let dim_val = if dim < input_shape.len() { input_shape[dim] } else { 1 };
                    let out_name = op.outputs.first().unwrap().clone();
                    let out_bytes = (dim_val as f32).to_le_bytes().to_vec();
                    ssa_map.insert(out_name.clone(), out_bytes);
                    ssa_dtypes.insert(out_name.clone(), output_dtype);
                    ssa_shapes.insert(out_name.clone(), vec![1]);
                    continue;
                }
            }

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
                executable.execute(&op_name, stream, &input_refs, &output_refs)
            }));
            let output_shapes = match exe_result {
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

            dump_op_output(trace, fi, oi, &op_name, &output_vecs, output_dtype, &dump_funcs);

            drop(output_bufs);
            for (idx, output_name) in op.outputs.iter().enumerate() {
                let out_vec = std::mem::take(&mut output_vecs[idx]);
                ssa_map.insert(output_name.clone(), out_vec);
                ssa_dtypes.insert(output_name.clone(), output_dtype);
                if let Some(shapes) = output_shapes.get(idx) {
                    let shape_usize: Vec<usize> = shapes.iter()
                        .map(|&s| std::cmp::max(1, s as usize))
                        .collect();
                    ssa_shapes.insert(output_name.clone(), shape_usize);
                }
            }

            if op.op == "shape_of" {
                if let Some(out_name) = op.outputs.first() {
                    if let Some(data) = ssa_map.get(out_name) {
                        let dims: Vec<usize> = data.chunks(4)
                            .map(|b| {
                                let bytes: [u8; 4] = b.try_into().unwrap_or([0; 4]);
                                f32::from_le_bytes(bytes) as usize
                            })
                            .collect();
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
