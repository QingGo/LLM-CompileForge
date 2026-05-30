//! SSA map preparation, cross-function wiring, and global output
//! extraction for the HAL IR graph runner.
//!
//! These functions are the "scaffolding" around the per-op execution loop
//! in `run_hal_function_graph`.  They handle initial state setup,
//! inter-function data flow, and final output assembly.

use std::collections::{HashMap, HashSet};

use crate::tensor::{Dtype, Tensor};
use crate::hal_runner::helpers::estimate_numel_from_shape;
use crate::hal_runner::types::{HalFunction, HalIR};

type SsaMaps = (HashMap<String, Vec<u8>>, HashMap<String, Vec<usize>>, HashMap<String, Dtype>);

// ── SSA map preparation ───────────────────────────────────────────────

/// Build the initial SSA value, shape, and dtype maps from global inputs,
/// function I/O metadata, and lazy zero-fill for invisible constants.
///
/// Returns `(ssa_map, ssa_shapes, ssa_dtypes)` ready for weight injection
/// and execution.
pub(super) fn prepare_ssa_maps(
    hal_ir: &HalIR,
    seq_len: usize,
    input_ids: &[u32],
    positions: &[u32],
) -> SsaMaps {
    let mut ssa_map: HashMap<String, Vec<u8>> = HashMap::new();
    let mut ssa_shapes: HashMap<String, Vec<usize>> = HashMap::new();
    let mut ssa_dtypes: HashMap<String, Dtype> = HashMap::new();

    // ── Inject global inputs (i64) for the entry function ────────────
    {
        // %arg0 = input_ids, stored as i64 bytes (8 bytes per element)
        let raw: Vec<u8> = input_ids
            .iter()
            .flat_map(|&id| (id as i64).to_le_bytes().to_vec())
            .collect();
        ssa_map.insert("%arg0".to_string(), raw);
        ssa_shapes.insert("%arg0".to_string(), vec![1, seq_len]);
        ssa_dtypes.insert("%arg0".to_string(), Dtype::I64);

        // %arg1 = position_ids, stored as i64 bytes (8 bytes per element)
        let raw: Vec<u8> = positions
            .iter()
            .flat_map(|&p| (p as i64).to_le_bytes().to_vec())
            .collect();
        ssa_map.insert("%arg1".to_string(), raw);
        ssa_shapes.insert("%arg1".to_string(), vec![1, seq_len]);
        ssa_dtypes.insert("%arg1".to_string(), Dtype::I64);
    }

    // ── Build set of SSA names referenced as op inputs vs produced ──
    // Only pre-populate tensors that ops reference but are NOT produced
    // by any op (weights, constants, function inputs).
    let mut referenced_ssa: HashSet<String> = HashSet::new();
    let mut produced_ssa: HashSet<String> = HashSet::new();
    for func in &hal_ir.functions {
        for op in &func.ops {
            for input_name in &op.inputs {
                referenced_ssa.insert(input_name.clone());
            }
            for output_name in &op.outputs {
                produced_ssa.insert(output_name.clone());
            }
        }
    }

    // ── Pre-populate all function outputs not yet in the map ────────
    for func in &hal_ir.functions {
        for output in &func.outputs {
            if !ssa_map.contains_key(&output.name) {
                let numel = estimate_numel_from_shape(&output.shape, seq_len);
                let dtype = Dtype::from_hal_str(&output.dtype);
                let raw_bytes = vec![0u8; numel * dtype.element_size()];
                ssa_map.insert(output.name.clone(), raw_bytes);
            }
            if !ssa_shapes.contains_key(&output.name) {
                let shape_dims: Vec<usize> = output.shape.iter().map(|d| {
                    if d == "?" || d == "-1" { 1 } else { d.parse::<usize>().unwrap_or(1) }
                }).collect();
                ssa_shapes.insert(output.name.clone(), shape_dims);
            }
            if !ssa_dtypes.contains_key(&output.name) {
                ssa_dtypes.insert(output.name.clone(), Dtype::from_hal_str(&output.dtype));
            }
        }
    }

    // ── Lazy zero-fill remaining referenced names ───────────────────
    for name in &referenced_ssa {
        if ssa_map.contains_key(name) || name.starts_with("%arg") {
            continue;
        }
        let mut found = false;
        for func in &hal_ir.functions {
            for output in &func.outputs {
                if output.name == *name {
                    let numel = estimate_numel_from_shape(&output.shape, seq_len);
                    let dtype = Dtype::from_hal_str(&output.dtype);
                    let raw_bytes = vec![0u8; numel * dtype.element_size()];
                    ssa_map.insert(name.clone(), raw_bytes);
                    let shape_dims: Vec<usize> = output.shape.iter().map(|d| {
                        if d == "?" || d == "-1" { 1 } else { d.parse::<usize>().unwrap_or(1) }
                    }).collect();
                    ssa_shapes.insert(name.clone(), shape_dims);
                    ssa_dtypes.insert(name.clone(), dtype);
                    log::trace!(
                        "hal_runner: zero-filled '{}' ({} elements, shape={:?}, dtype={:?})",
                        name, numel, output.shape, dtype,
                    );
                    found = true;
                    break;
                }
            }
            if found { break; }
        }
        if found { continue; }
        for func in &hal_ir.functions {
            for input_def in &func.inputs {
                if input_def.name == *name {
                    let numel = estimate_numel_from_shape(&input_def.shape, seq_len);
                    let dtype = Dtype::from_hal_str(&input_def.dtype);
                    let raw_bytes = vec![0u8; numel * dtype.element_size()];
                    ssa_map.insert(name.clone(), raw_bytes);
                    let shape_dims: Vec<usize> = input_def.shape.iter().map(|d| {
                        if d == "?" || d == "-1" { 1 } else { d.parse::<usize>().unwrap_or(1) }
                    }).collect();
                    ssa_shapes.insert(name.clone(), shape_dims);
                    ssa_dtypes.insert(name.clone(), dtype);
                    log::trace!(
                        "hal_runner: zero-filled input '{}' ({} elements, shape={:?}, dtype={:?})",
                        name, numel, input_def.shape, dtype,
                    );
                    found = true;
                    break;
                }
            }
            if found { break; }
        }
        if !found {
            const DEFAULT_CONSTANT_ELEMS: usize = 65536;
            let raw_bytes = vec![0u8; DEFAULT_CONSTANT_ELEMS * 4];
            ssa_map.insert(name.clone(), raw_bytes);
            ssa_dtypes.insert(name.clone(), Dtype::F32);
            log::trace!(
                "hal_runner: zero-filled invisible constant '{}' ({} elements)",
                name, DEFAULT_CONSTANT_ELEMS,
            );
        }
    }

    (ssa_map, ssa_shapes, ssa_dtypes)
}

// ── Cross-function wiring ──────────────────────────────────────────────

/// Wire dynamic-shape inputs from the previous function's outputs
/// to the current function's inputs, matching by shape pattern.
///
/// Called inside the main execution loop for functions >= 1.
pub(super) fn wire_cross_function_inputs(
    fi: usize,
    function: &HalFunction,
    prev_func: &HalFunction,
    ssa_map: &mut HashMap<String, Vec<u8>>,
    ssa_shapes: &mut HashMap<String, Vec<usize>>,
    ssa_dtypes: &mut HashMap<String, Dtype>,
    seq_len: usize,
) {
    // ── Cross-function wiring for functions >= 1 ────────────────
    //
    // Each function declares its own %arg0..%argN namespace.
    // We wire ALL dynamic-shape inputs from the previous function's
    // outputs to this function's inputs, matching by shape pattern.
    // Build a map of previous function's output shapes for matching.
    let mut prev_outputs_by_shape: HashMap<Vec<String>, (String, Vec<u8>)> =
        HashMap::new();
    for output in &prev_func.outputs {
        if let Some(data) = ssa_map.get(&output.name) {
            prev_outputs_by_shape
                .insert(output.shape.clone(), (output.name.clone(), data.clone()));
        }
    }

    // Wire each dynamic-shape input in current function
    // from the previous function's matching output.
    log::debug!(
        "hal_runner: wire_cross_function_inputs for function[{}], {} inputs, {} prev_outputs",
        fi, function.inputs.len(), prev_func.outputs.len(),
    );
    for input_def in &function.inputs {
        // Wire dynamic-shape inputs AND scalar constants (rank 1, non-dynamic).
        // Scalar inputs like shape [1] need wiring too — they carry scale factors,
        // dropout constants, etc. from earlier functions.
        let has_dyn = input_def.shape.iter().any(|d| d == "?" || d == "-1");
        let is_scalar_constant = input_def.shape.len() == 1
            && input_def.shape[0] != "?"
            && input_def.shape[0] != "-1";
        if !has_dyn && !is_scalar_constant {
            continue;
        }

        // Try direct name match first (check previous function's outputs).
        // We do NOT skip based on existing data because %arg names are
        // shared across functions with different semantics (e.g. %arg1
        // is position_ids in func[0] but hidden state in func[2]).
        if fi > 0 {
            if let Some(prev_output) = prev_func.outputs.iter().find(|o| o.name == input_def.name) {
                let prev_data_opt = ssa_map.get(&prev_output.name).cloned();
                if let Some(prev_data) = prev_data_opt {
                    if prev_data.iter().any(|&b| b != 0) {
                        ssa_map.insert(input_def.name.clone(), prev_data.clone());
                        if let Some(prev_shape) = ssa_shapes.get(&prev_output.name) {
                            ssa_shapes.insert(input_def.name.clone(), prev_shape.clone());
                        }
                        if let Some(prev_dtype) = ssa_dtypes.get(&prev_output.name) {
                            ssa_dtypes.insert(input_def.name.clone(), *prev_dtype);
                        }
                        log::debug!(
                            "hal_runner: wired function[{}] '{}' from same-name output ({} bytes)",
                            fi, input_def.name, prev_data.len(),
                        );
                        continue;
                    }
                }
            }
        }

        // Match by shape pattern directly
        log::debug!(
            "hal_runner: wire check function[{}] '{}' shape={:?}, prev_outputs_by_shape keys: {:?}",
            fi, input_def.name, input_def.shape,
            prev_outputs_by_shape.keys().collect::<Vec<_>>(),
        );
        if let Some((prev_name, prev_data)) =
            prev_outputs_by_shape.get(&input_def.shape)
        {
            ssa_map.insert(input_def.name.clone(), prev_data.clone());
            if let Some(prev_shape) = ssa_shapes.get(prev_name) {
                ssa_shapes.insert(input_def.name.clone(), prev_shape.clone());
            }
            if let Some(prev_dtype) = ssa_dtypes.get(prev_name) {
                ssa_dtypes.insert(input_def.name.clone(), *prev_dtype);
            }
            log::debug!(
                "hal_runner: wired function[{}] '{}' from function[{}] '{}' (shape={:?}, {} bytes)",
                fi, input_def.name, fi - 1, prev_name, input_def.shape, prev_data.len(),
            );
        } else {
            // Try fuzzy matching: find any output with same number of
            // dynamic dims and same static dims.
            let dyn_count = input_def
                .shape
                .iter()
                .filter(|d| *d == "?" || *d == "-1")
                .count();
            let input_static: Vec<&str> = input_def
                .shape
                .iter()
                .filter(|d| *d != "?" && *d != "-1")
                .map(|s| s.as_str())
                .collect();

            for output in &prev_func.outputs {
                let out_dyn_count = output
                    .shape
                    .iter()
                    .filter(|d| *d == "?" || *d == "-1")
                    .count();
                let out_static: Vec<&str> = output
                    .shape
                    .iter()
                    .filter(|d| *d != "?" && *d != "-1")
                    .map(|s| s.as_str())
                    .collect();

                if dyn_count == out_dyn_count && input_static == out_static {
                    if let Some(prev_data) = ssa_map.get(&output.name) {
                        ssa_map.insert(input_def.name.clone(), prev_data.clone());
                        if let Some(prev_shape) = ssa_shapes.get(&output.name) {
                            ssa_shapes.insert(input_def.name.clone(), prev_shape.clone());
                        }
                        if let Some(prev_dtype) = ssa_dtypes.get(&output.name) {
                            ssa_dtypes.insert(input_def.name.clone(), *prev_dtype);
                        }
                        log::debug!(
                            "hal_runner: fuzzy-wired function[{}] '{}' from function[{}] '{}' ({} dims match)",
                            fi, input_def.name, fi - 1, output.name, dyn_count,
                        );
                        break;
                    }
                }
            }
        }
    }

    // Pre-populate remaining %arg inputs with zeros.
    for input_def in &function.inputs {
        if ssa_map.contains_key(&input_def.name) {
            continue;
        }
        let numel = estimate_numel_from_shape(&input_def.shape, seq_len);
        let dtype = Dtype::from_hal_str(&input_def.dtype);
        let raw_bytes = vec![0u8; numel * dtype.element_size()];
        ssa_map.insert(input_def.name.clone(), raw_bytes);
        let shape_dims: Vec<usize> = input_def.shape.iter().map(|d| {
            if d == "?" || d == "-1" { 1 } else { d.parse::<usize>().unwrap_or(1) }
        }).collect();
        ssa_shapes.insert(input_def.name.clone(), shape_dims);
        ssa_dtypes.insert(input_def.name.clone(), dtype);
        log::trace!(
            "hal_runner: zero-filled function[{}] input '{}' ({} elements, shape={:?}, dtype={:?})",
            fi, input_def.name, numel, input_def.shape, dtype,
        );
    }
}

// ── Global output extraction ───────────────────────────────────────────

/// Extract the final output tensor from the last function's SSA map values.
///
/// The last function's first non-consumed output is identified as the global
/// output (typically logits).  Shape is validated against the declared shape
/// from function output metadata, falling back to flat shape on mismatch.
pub(super) fn extract_global_output(
    hal_ir: &HalIR,
    ssa_map: &HashMap<String, Vec<u8>>,
    ssa_shapes: &HashMap<String, Vec<usize>>,
    ssa_dtypes: &HashMap<String, Dtype>,
    seq_len: usize,
) -> Result<Tensor, anyhow::Error> {
    // The last function's first non-consumed output is the global
    // output (typically the logits tensor from main_15).
    let last_func = hal_ir
        .functions
        .last()
        .ok_or_else(|| anyhow::anyhow!("hal_runner: no functions in HAL IR"))?;

    let global_output_idx = last_func
        .outputs
        .iter()
        .position(|o| !o.name.is_empty())
        .unwrap_or(0);
    let global_output_def = &last_func.outputs[global_output_idx];

    let raw_bytes = ssa_map
        .get(&global_output_def.name)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "hal_runner: global output '{}' not found in SSA map",
                global_output_def.name
            )
        })?;

    let output_elem_size = ssa_dtypes
        .get(&global_output_def.name)
        .map(|d| d.element_size())
        .unwrap_or(4);
    let numel = raw_bytes.len() / output_elem_size;
    let mut result: Vec<f32> = Vec::with_capacity(numel);
    result.resize(numel, 0.0f32);
    for i in 0..numel {
        let bytes: [u8; 4] = raw_bytes[i * output_elem_size..i * output_elem_size + 4]
            .try_into()
            .map_err(|_| anyhow::anyhow!("hal_runner: invalid output byte slice"))?;
        result[i] = f32::from_le_bytes(bytes);
    }

    // Prefer runtime shape from ssa_shapes (updated during execution).
    // Fall back to declared shape from function output metadata.
    let output_shape: Vec<usize> = if let Some(runtime_shape) = ssa_shapes.get(&global_output_def.name) {
        runtime_shape.clone()
    } else {
        let mut first_dyn = true;
        global_output_def
            .shape
            .iter()
            .map(|d| {
                if d == "?" || d == "-1" {
                    if first_dyn {
                        first_dyn = false;
                        1
                    } else {
                        seq_len
                    }
                } else {
                    d.parse::<usize>().unwrap_or(1)
                }
            })
            .collect()
    };

    let shape_product: usize = output_shape.iter().product();
    let output_shape = if shape_product == numel || shape_product == 0 {
        output_shape
    } else {
        log::debug!(
            "hal_runner: global output shape mismatch: shape {:?} (product={}), actual numel={}. \
             Using flat shape.",
            output_shape,
            shape_product,
            numel,
        );
        vec![numel]
    };

    log::debug!(
        "hal_runner: global output '{}' shape={:?} numel={}",
        global_output_def.name,
        output_shape,
        numel,
    );

    Ok(Tensor::new_owned(output_shape, result, Dtype::F32))
}
