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

// ── SSA map preparation ───────────────────────────────────────────────

/// Build the initial SSA value and shape maps from global inputs, function
/// I/O metadata, and lazy zero-fill for invisible constants.
///
/// Returns `(ssa_map, ssa_shapes)` ready for weight injection and execution.
pub(super) fn prepare_ssa_maps(
    hal_ir: &HalIR,
    seq_len: usize,
    input_ids: &[u32],
    positions: &[u32],
) -> (HashMap<String, Vec<u8>>, HashMap<String, Vec<usize>>) {
    // ── Global SSA value map ─────────────────────────────────────────
    // Stores raw bytes keyed by SSA name (e.g. "%arg0", "%213", "%196").
    // All values are stored as f32 bytes (4 per element).  The gather op
    // internally casts f32 indices to i64.
    let mut ssa_map: HashMap<String, Vec<u8>> = HashMap::new();

    // ── Global SSA shape map ─────────────────────────────────────────
    // Tracks tensor shapes alongside ssa_map for cross-function shape
    // propagation.  Populated from global inputs, function I/O metadata,
    // weight entries, and actual op output shapes after execute().
    // Without this, tensors flowing between functions lose their shape
    // metadata and default to flat 1D (e.g. [65536] instead of [1,4,768]),
    // causing downstream matmul/transpose to fail with "expected rank >= 2".
    let mut ssa_shapes: HashMap<String, Vec<usize>> = HashMap::new();

    // ── Inject global inputs (f32) for the entry function ────────────
    {
        // %arg0 = input_ids, stored as f32 values (cast from u32)
        let raw: Vec<u8> = input_ids
            .iter()
            .flat_map(|&id| (id as f32).to_le_bytes().to_vec())
            .collect();
        ssa_map.insert("%arg0".to_string(), raw);
        ssa_shapes.insert("%arg0".to_string(), vec![1, seq_len]);

        // %arg1 = position_ids, stored as f32 values
        let raw: Vec<u8> = positions
            .iter()
            .flat_map(|&p| (p as f32).to_le_bytes().to_vec())
            .collect();
        ssa_map.insert("%arg1".to_string(), raw);
        ssa_shapes.insert("%arg1".to_string(), vec![1, seq_len]);
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
    // Ensures outputs like %203 (declared but never produced by any op)
    // are available for global output extraction.
    for func in &hal_ir.functions {
        for output in &func.outputs {
            if !ssa_map.contains_key(&output.name) {
                let numel = estimate_numel_from_shape(&output.shape, seq_len);
                let raw_bytes = vec![0u8; numel * 4];
                ssa_map.insert(output.name.clone(), raw_bytes);
            }
            // Always populate shape from function output metadata.
            if !ssa_shapes.contains_key(&output.name) {
                let shape_dims: Vec<usize> = output.shape.iter().map(|d| {
                    if d == "?" || d == "-1" { 1 } else { d.parse::<usize>().unwrap_or(1) }
                }).collect();
                ssa_shapes.insert(output.name.clone(), shape_dims);
            }
        }
    }

    // ── Lazy zero-fill remaining referenced names ───────────────────
    // Pre-populate any SSA name referenced by an op that is not
    // yet in the map (invisible constants, etc.).
    for name in &referenced_ssa {
        if ssa_map.contains_key(name) || name.starts_with("%arg") {
            continue;
        }
        // Search all functions' output lists for this name.
        let mut found = false;
        for func in &hal_ir.functions {
            for output in &func.outputs {
                if output.name == *name {
                    let numel = estimate_numel_from_shape(&output.shape, seq_len);
                    let raw_bytes = vec![0u8; numel * 4]; // zero-fill (f32 = 4 bytes)
                    ssa_map.insert(name.clone(), raw_bytes);
                    // Store shape from function output metadata.
                    let shape_dims: Vec<usize> = output.shape.iter().map(|d| {
                        if d == "?" || d == "-1" { 1 } else { d.parse::<usize>().unwrap_or(1) }
                    }).collect();
                    ssa_shapes.insert(name.clone(), shape_dims);
                    log::trace!(
                        "hal_runner: zero-filled '{}' ({} elements, shape={:?})",
                        name,
                        numel,
                        output.shape,
                    );
                    found = true;
                    break;
                }
            }
            if found {
                break;
            }
        }
        if found {
            continue;
        }
        // Search function input lists too.
        for func in &hal_ir.functions {
            for input_def in &func.inputs {
                if input_def.name == *name {
                    let numel = estimate_numel_from_shape(&input_def.shape, seq_len);
                    let raw_bytes = vec![0u8; numel * 4];
                    ssa_map.insert(name.clone(), raw_bytes);
                    // Store shape from function input metadata.
                    let shape_dims: Vec<usize> = input_def.shape.iter().map(|d| {
                        if d == "?" || d == "-1" { 1 } else { d.parse::<usize>().unwrap_or(1) }
                    }).collect();
                    ssa_shapes.insert(name.clone(), shape_dims);
                    log::trace!(
                        "hal_runner: zero-filled input '{}' ({} elements, shape={:?})",
                        name,
                        numel,
                        input_def.shape,
                    );
                    found = true;
                    break;
                }
            }
            if found {
                break;
            }
        }
        if !found {
            // Invisible constant — not in any I/O list.  Use 65536 elements
            // (256 KB), sized to accommodate 2D gather weight shapes
            // like [85, 768] for embed_dim=768.
            const DEFAULT_CONSTANT_ELEMS: usize = 65536;
            let raw_bytes = vec![0u8; DEFAULT_CONSTANT_ELEMS * 4];
            ssa_map.insert(name.clone(), raw_bytes);            log::trace!(
                "hal_runner: zero-filled invisible constant '{}' ({} elements)",
                name,
                DEFAULT_CONSTANT_ELEMS,
            );
        }
    }

    (ssa_map, ssa_shapes)
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
        let has_dyn = input_def.shape.iter().any(|d| d == "?" || d == "-1");
        if !has_dyn {
            continue;
        }

        // Skip if already populated (e.g., by weight injection).
        if let Some(data) = ssa_map.get(&input_def.name) {
            if data.iter().any(|&b| b != 0) {
                continue; // non-zero, already populated
            }
        }

        // Try direct name match first
        if let Some(prev_data) = ssa_map.get(&input_def.name) {
            let prev_data = prev_data.clone(); // release borrow before mutable insert
            if prev_data.iter().any(|&b| b != 0) {
                ssa_map.insert(input_def.name.clone(), prev_data.clone());
                if let Some(prev_shape) = ssa_shapes.get(&input_def.name) {
                    ssa_shapes.insert(input_def.name.clone(), prev_shape.clone());
                }
                log::debug!(
                    "hal_runner: wired function[{}] '{}' from same-name output ({} bytes)",
                    fi,
                    input_def.name,
                    prev_data.len(),
                );
                continue;
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
            log::debug!(
                "hal_runner: wired function[{}] '{}' from function[{}] '{}' (shape={:?}, {} bytes)",
                fi,
                input_def.name,
                fi - 1,
                prev_name,
                input_def.shape,
                prev_data.len(),
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
                        log::debug!(
                            "hal_runner: fuzzy-wired function[{}] '{}' from function[{}] '{}' ({} dims match)",
                            fi,
                            input_def.name,
                            fi - 1,
                            output.name,
                            dyn_count,
                        );
                        break;
                    }
                }
            }
        }
    }

    // Pre-populate remaining %arg inputs with zeros.
    // Weight injection from WeightProvider will be added in Task 5.
    for input_def in &function.inputs {
        if ssa_map.contains_key(&input_def.name) {
            continue;
        }
        let numel = estimate_numel_from_shape(&input_def.shape, seq_len);
        let raw_bytes = vec![0u8; numel * 4];
        ssa_map.insert(input_def.name.clone(), raw_bytes);
        // Store shape from function input metadata for cross-function
        // shape propagation.
        let shape_dims: Vec<usize> = input_def.shape.iter().map(|d| {
            if d == "?" || d == "-1" { 1 } else { d.parse::<usize>().unwrap_or(1) }
        }).collect();
        ssa_shapes.insert(input_def.name.clone(), shape_dims);
        log::trace!(
            "hal_runner: zero-filled function[{}] input '{}' ({} elements, shape={:?})",
            fi,
            input_def.name,
            numel,
            input_def.shape,
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
    seq_len: usize,
) -> Result<Tensor, anyhow::Error> {
    // The last function's first non-consumed output is the global
    // output (typically the logits tensor from main_15).
    let last_func = hal_ir
        .functions
        .last()
        .ok_or_else(|| anyhow::anyhow!("hal_runner: no functions in HAL IR"))?;

    // Find the first output that is not consumed internally.
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

    let numel = raw_bytes.len() / 4;
    let mut result: Vec<f32> = Vec::with_capacity(numel);
    // Initialize with zeros; immediately overwritten by from_le_bytes loop.
    result.resize(numel, 0.0f32);
    for i in 0..numel {
        let bytes: [u8; 4] = raw_bytes[i * 4..(i + 1) * 4]
            .try_into()
            .map_err(|_| anyhow::anyhow!("hal_runner: invalid output byte slice"))?;
        result[i] = f32::from_le_bytes(bytes);
    }

    // Build output shape from function output metadata.
    // First "?" = batch (always 1), subsequent "?" = sequence length.
    let output_shape: Vec<usize> = {
        let mut first_dyn = true;
        global_output_def
            .shape
            .iter()
            .map(|d| {
                if d == "?" || d == "-1" {
                    if first_dyn {
                        first_dyn = false;
                        1 // batch
                    } else {
                        seq_len
                    }
                } else {
                    d.parse::<usize>().unwrap_or(1)
                }
            })
            .collect()
    };

    // Validate shape product matches actual data.
    // The declared shape from function output metadata may not match the
    // runtime data (e.g. shape_of output declared as [1] but returning
    // rank elements).  Fall back to a flat shape when mismatched.
    let shape_product: usize = output_shape.iter().product();
    let output_shape = if shape_product == numel || shape_product == 0 {
        output_shape
    } else {
        log::debug!(
            "hal_runner: global output shape mismatch: declared {:?} (product={}), actual numel={}. \
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
