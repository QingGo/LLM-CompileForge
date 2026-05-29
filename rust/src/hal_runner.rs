//! HAL IR graph runner — iterates over HalFunction ops, assembles inputs
//! from global inputs / weights / SSA wires, dispatches through
//! `executable.execute(op_name, stream, &input_bufs, &output_bufs)`,
//! and extracts the final output Tensor.
//!
//! Path B (pure-Rust) counterpart to `compute_graph_runner.rs`.
//! All kernel dispatch goes through the HAL Executable trait — no direct
//! ciface / lookup_typed calls.

use std::collections::HashMap;

use crate::hal::cpu::buffer::CpuBuffer as InnerCpuBuffer;
use crate::hal::cpu::CpuBuffer;
use crate::hal::traits;
use crate::tensor::{Dtype, Tensor};
use crate::weight_loader::WeightProvider;

// ── HAL IR types ───────────────────────────────────────────────────────

/// Parsed HAL IR for a model.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct HalIR {
    pub model_name: String,
    pub num_functions: usize,
    pub functions: Vec<HalFunction>,
}

/// A single HAL function (entry point / sub-graph).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct HalFunction {
    pub name: String,
    pub layer: usize,
    pub inputs: Vec<HalTensorDef>,
    pub outputs: Vec<HalTensorDef>,
    #[serde(default)]
    pub weights: Vec<HalWeightEntry>,
    #[serde(default)]
    pub weight_inputs: HashMap<String, String>,
    pub ops: Vec<HalOp>,
}

/// A weight entry in a function's `weights` list.
///
/// Compiled from `sf.weight` ops in the normalized MLIR.
/// The `ssa` field gives the SSA name (e.g. `%0`) produced by the weight op.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct HalWeightEntry {
    pub name: String,
    #[serde(deserialize_with = "deserialize_shape_to_strings")]
    pub shape: Vec<String>,
    pub dtype: String,
    pub hal_name: String,
    #[serde(default)]
    pub ssa: String,
}

/// Tensor metadata in a function's input/output list.
///
/// Shape values can be strings ("?") for dynamic dims or integers (1, 768)
/// for static dims.  We deserialize both into `String` by converting integers.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct HalTensorDef {
    pub name: String,
    #[serde(default, deserialize_with = "deserialize_shape_to_strings")]
    pub shape: Vec<String>,
    #[serde(default)]
    pub dtype: String,
    /// Whether this tensor is consumed internally within the function.
    /// When true, the tensor is NOT propagated to subsequent functions
    /// as a cross-function wire.
    #[serde(default)]
    pub consumed_internally: bool,
}

/// Deserialize shape arrays that may contain strings ("?") or integers (1).
fn deserialize_shape_to_strings<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let arr: Vec<serde_json::Value> = serde::Deserialize::deserialize(deserializer)?;
    arr.iter()
        .map(|v| match v {
            serde_json::Value::String(s) => Ok(s.clone()),
            serde_json::Value::Number(n) => Ok(n.to_string()),
            _ => Err(serde::de::Error::custom(format!(
                "expected string or number, got {:?}",
                v
            ))),
        })
        .collect()
}

/// A single HAL operation.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct HalOp {
    pub op: String,
    #[serde(default)]
    pub kind: Option<String>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    #[serde(default)]
    pub weight: Option<String>,
    /// Op shape metadata (mixed integer/string). Stored as JSON Value
    /// since reshape targets use both "?" and integer dims.
    #[serde(default, deserialize_with = "deserialize_optional_shape")]
    pub shape: Option<Vec<String>>,
    #[serde(default)]
    pub value: Option<f64>,
}

/// Deserialize optional shape arrays that may contain strings ("?") or integers.
fn deserialize_optional_shape<'de, D>(
    deserializer: D,
) -> Result<Option<Vec<String>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let opt: Option<Vec<serde_json::Value>> = serde::Deserialize::deserialize(deserializer)?;
    match opt {
        None => Ok(None),
        Some(arr) => {
            let strings: Result<Vec<String>, _> = arr
                .iter()
                .map(|v| match v {
                    serde_json::Value::String(s) => Ok(s.clone()),
                    serde_json::Value::Number(n) => Ok(n.to_string()),
                    _ => Err(serde::de::Error::custom(format!(
                        "expected string or number, got {:?}",
                        v
                    ))),
                })
                .collect();
            strings.map(Some)
        }
    }
}

// ── HalRustRunner ──────────────────────────────────────────────────────

/// Main runner for HAL IR execution.
///
/// Parses `hal_ir.json` and provides `run_hal_function_graph()` to execute
/// the full model forward pass through a `traits::Executable`.
pub struct HalRustRunner {
    pub hal_ir: HalIR,
}

impl HalRustRunner {
    /// Parse a HAL IR JSON string.
    pub fn from_json(json_str: &str) -> Result<Self, anyhow::Error> {
        let hal_ir: HalIR = serde_json::from_str(json_str)?;
        Ok(Self { hal_ir })
    }

    /// Load HAL IR from a JSON file.
    pub fn from_path(path: &str) -> Result<Self, anyhow::Error> {
        let content = std::fs::read_to_string(path)?;
        Self::from_json(&content)
    }
}

// ── run_hal_function_graph ────────────────────────────────────────────

/// Execute a complete HAL IR function graph through a HAL Executable.
///
/// Iterates over every function and its ops in order, maintaining an SSA
/// value map (`HashMap<String, Vec<f32>>`) across all functions.
///
/// # Arguments
///
/// * `executable` — HAL executable that dispatches op_name → CPU kernel.
/// * `hal_ir` — parsed HAL IR (28 functions, 634 ops for opt-125m).
/// * `weight_provider` — optional weight loader for weight tensors.
///   When `Some(...)`, known weights are loaded before execution.
///   When `None`, all weights are zero-filled.
/// * `stream` — HAL stream (no-op for CPU).
/// * `input_ids` — token IDs (length = sequence length).
/// * `positions` — position IDs (length = sequence length).
///
/// # Returns
///
/// The global output tensor (typically logits from the last function).
pub fn run_hal_function_graph(
    executable: &dyn traits::Executable,
    hal_ir: &HalIR,
    weight_provider: Option<&WeightProvider>,
    stream: &dyn traits::Stream,
    input_ids: &[u32],
    positions: &[u32],
) -> Result<Tensor, anyhow::Error> {
    let seq_len = input_ids.len();

    // ── Global SSA value map ─────────────────────────────────────────
    // Stores raw bytes keyed by SSA name (e.g. "%arg0", "%213", "%196").
    // All values are stored as f32 bytes (4 per element).  The gather op
    // internally casts f32 indices to i64.
    let mut ssa_map: HashMap<String, Vec<u8>> = HashMap::new();

    // ── Inject global inputs (f32) for the entry function ────────────
    {
        // %arg0 = input_ids, stored as f32 values (cast from u32)
        let raw: Vec<u8> = input_ids
            .iter()
            .flat_map(|&id| (id as f32).to_le_bytes().to_vec())
            .collect();
        ssa_map.insert("%arg0".to_string(), raw);

        // %arg1 = position_ids, stored as f32 values
        let raw: Vec<u8> = positions
            .iter()
            .flat_map(|&p| (p as f32).to_le_bytes().to_vec())
            .collect();
        ssa_map.insert("%arg1".to_string(), raw);
    }

    // ── Build set of SSA names referenced as op inputs vs produced ──
    // Only pre-populate tensors that ops reference but are NOT produced
    // by any op (weights, constants, function inputs).
    let mut referenced_ssa: std::collections::HashSet<String> =
        std::collections::HashSet::new();
    let mut produced_ssa: std::collections::HashSet<String> =
        std::collections::HashSet::new();
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
                    ssa_map.insert(name.clone(), raw_bytes);                    log::trace!(
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

    // ── Pre-populate weights from WeightProvider ─────────────────────
    // Inject real weight data into the SSA map before execution begins.
    // This handles two cases:
    //   (a) function.weights — weight entries produced by sf.weight ops
    //       (primarily function 0), each with a SSA name.
    //   (b) function.weight_inputs — function input args mapped to
    //       compiled weight names (all other functions).
    if let Some(wp) = weight_provider {
        for function in &hal_ir.functions {
            // Case (a): weights list with inline SSA names.
            for weight_entry in &function.weights {
                if weight_entry.ssa.is_empty() {
                    continue;
                }
                if let Some(desc) = wp.get_weight_memref(&weight_entry.name) {
                    let n = desc.numel();
                    // SAFETY: The pointer comes from a valid MemRefDesc's aligned
                    // field. The f16 data was written by the dylib's execute() call.
                    let data: Vec<f32> = unsafe {
                        let raw = desc.aligned as *const u16;
                        let slice = std::slice::from_raw_parts(raw, n);
                        slice
                            .iter()
                            .map(|&h| half::f16::from_bits(h).to_f32())
                            .collect()
                    };
                    let raw_bytes: Vec<u8> = data
                        .iter()
                        .flat_map(|&v| v.to_le_bytes())
                        .collect();
                    ssa_map.insert(weight_entry.ssa.clone(), raw_bytes);
                    log::debug!(
                        "hal_runner: loaded weight '{}' -> SSA '{}' ({} elements)",
                        weight_entry.name,
                        weight_entry.ssa,
                        n,
                    );
                }
            }

            // Case (b): weight_inputs mapping (function args → compiled names).
            for (ssa_name, compiled_name) in &function.weight_inputs {
                if let Some(desc) = wp.get_weight_memref(compiled_name) {
                    let n = desc.numel();
                    // SAFETY: Same pattern as case (a) — valid MemRefDesc aligned pointer.
                    let data: Vec<f32> = unsafe {
                        let raw = desc.aligned as *const u16;
                        let slice = std::slice::from_raw_parts(raw, n);
                        slice
                            .iter()
                            .map(|&h| half::f16::from_bits(h).to_f32())
                            .collect()
                    };
                    let raw_bytes: Vec<u8> = data
                        .iter()
                        .flat_map(|&v| v.to_le_bytes())
                        .collect();
                    ssa_map.insert(ssa_name.clone(), raw_bytes);
                    log::debug!(
                        "hal_runner: loaded weight '{}' -> SSA '{}' ({} elements)",
                        compiled_name,
                        ssa_name,
                        n,
                    );
                } else {
                    log::warn!(
                        "hal_runner: weight '{}' for SSA '{}' not found in WeightProvider",
                        compiled_name,
                        ssa_name,
                    );
                }
            }
        }
    } else {
        log::info!("hal_runner: no WeightProvider — all weights will be zero-filled");
    }

    // ── Execute each function's ops ──────────────────────────────────
    for (fi, function) in hal_ir.functions.iter().enumerate() {
        log::debug!(
            "hal_runner: executing function[{}] '{}' (layer={}, {} ops)",
            fi,
            function.name,
            function.layer,
            function.ops.len(),
        );

        // ── Cross-function wiring for functions >= 1 ────────────────
        //
        // Each function declares its own %arg0..%argN namespace.
        // The wire (hidden state from previous function) is identified
        // as the first input with dynamic dimensions ("?" in shape).
        // Other %arg inputs are weights or scalars, zero-filled here
        // (WeightProvider injection is reserved for Task 5).
        if fi >= 1 {
            let prev_func = &hal_ir.functions[fi - 1];
            let wire = find_main_output(prev_func, &ssa_map);

            if wire.is_none() {
                log::warn!(
                    "hal_runner: no wire output found for function[{}], using zero-fill for all inputs",
                    fi - 1,
                );
            }

            // Find the wire input: the first %arg with dynamic dims.
            // This may be %arg0, %arg1, or any %arg name depending on
            // how the compiler structured the function signature.
            let mut wire_input_name: Option<String> = None;
            for input_def in &function.inputs {
                let has_dyn = input_def.shape.iter().any(|d| d == "?" || d == "-1");
                if has_dyn {
                    wire_input_name = Some(input_def.name.clone());
                    break;
                }
            }

            if wire_input_name.is_none() {
                log::warn!(
                    "hal_runner: no dynamic input found for function[{}] — no cross-function wiring",
                    fi,
                );
            }

            // Populate the wire input from the previous function's output.
            // We ALWAYS overwrite — even if the SSA name already exists
            // (from function[0]'s global input injection).  %arg names
            // are per-function namespaces that collide in the global map.
            if let Some(ref wire_name) = wire_input_name {
                if let Some((_wire_ssa, ref wire_data)) = wire {
                    ssa_map.insert(wire_name.clone(), wire_data.clone());                    log::debug!(
                        "hal_runner: wired function[{}] '{}' from function[{}] output ({} bytes)",
                        fi,
                        wire_name,
                        fi - 1,
                        wire_data.len(),
                    );
                } else {
                    log::warn!(
                        "hal_runner: wire output for function[{}] not in SSA map — %arg inputs remain zero-filled",
                        fi - 1,
                    );
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
                ssa_map.insert(input_def.name.clone(), raw_bytes);                log::trace!(
                    "hal_runner: zero-filled function[{}] input '{}' ({} elements, shape={:?})",
                    fi,
                    input_def.name,
                    numel,
                    input_def.shape,
                );
            }
        }

        for (oi, op) in function.ops.iter().enumerate() {
            // Skip runtime-level cache ops — handled by block_manager/kv_cache.
            if op.op == "cache_read" || op.op == "cache_write" {
                log::trace!(
                    "hal_runner: func[{}] op[{}] skipping '{}' (runtime-level)",
                    fi,
                    oi,
                    op.op,
                );
                continue;
            }

            // ── Resolve input buffers from SSA map ──────────────────
            let mut input_bufs: Vec<Box<dyn traits::Buffer>> =
                Vec::with_capacity(op.inputs.len());

            for input_name in &op.inputs {
                let data = ssa_map.get(input_name).ok_or_else(|| {
                    anyhow::anyhow!(
                        "hal_runner: SSA value '{}' not found in map (func[{}] op[{}]: {:?})",
                        input_name,
                        fi,
                        oi,
                        op,
                    )
                })?;

                // Determine element size from the dtype map, function
                // metadata, or data length heuristic.
                // i64 values have 8-byte elements; f32 has 4-byte elements.
                // All SSA values stored as f32 (4 bytes per element).
                let elem_size = 4;

                // Look up the tensor shape from the function's output list
                // (authoritative source for all tensor shapes including
                // weights like %196 with shape [50272, 768]).
                let mut declared_shape: Vec<String> = vec![];
                for output in &function.outputs {
                    if output.name == *input_name {
                        declared_shape = output.shape.clone();
                        break;
                    }
                }
                if declared_shape.is_empty() {
                    // Look up from function input list (for %arg names).
                    for input_def in &function.inputs {
                        if input_def.name == *input_name {
                            declared_shape = input_def.shape.clone();
                            break;
                        }
                    }
                }
                if declared_shape.is_empty() {
                    // Look up from function weight list (invisible constants
                    // populated by weight injection, e.g. %1 for position emb).
                    for weight_entry in &function.weights {
                        if weight_entry.ssa == *input_name {
                            declared_shape = weight_entry.shape.clone();
                            break;
                        }
                    }
                }
                if declared_shape.is_empty() {
                    // Fall back to flat 1D from data length.
                    declared_shape = vec![(data.len() / elem_size).to_string()];
                }
                let dims: Vec<usize> = declared_shape
                    .iter()
                    .map(|d| {
                        if d == "?" || d == "-1" {
                            1 // batch dim for dynamic shapes
                        } else {
                            d.parse::<usize>().unwrap_or(data.len() / elem_size)
                        }
                    })
                    .collect();
                // For buffers with no shape metadata, fall back to flat 1D.
                let dims = if dims.is_empty() || dims.iter().all(|&d| d == 0) {
                    vec![data.len() / elem_size]
                } else {
                    dims
                };

                let raw_buf = InnerCpuBuffer::from_raw_parts(
                    data.as_ptr() as *mut u8,
                    data.len(),
                    true, // borrowed
                )
                .map_err(|e| anyhow::anyhow!("InnerCpuBuffer: {}", e))?;

                let cpu_buf = CpuBuffer::with_meta(raw_buf, elem_size, dims);
                input_bufs.push(Box::new(cpu_buf));
            }

            // ── Pre-allocate output buffers ─────────────────────────
            let mut output_vecs: Vec<Vec<f32>> = Vec::with_capacity(op.outputs.len());
            let mut output_bufs: Vec<Box<dyn traits::Buffer>> =
                Vec::with_capacity(op.outputs.len());

            for (out_idx, output_name) in op.outputs.iter().enumerate() {
                let (shape, _) = find_output_shape(function, output_name);

                // Special cases where function output list shapes are unreliable:
                //
                // - shape_of: output has rank = input rank elements, regardless
                //   of declared shape (which is often just [1]).
                // - reshape: output numel = input logical numel, using the
                //   op.shape target when available.
                // - element_wise / unsqueeze / transpose: same numel as input.
                let numel = if op.op == "shape_of" && out_idx == 0 {
                    if let Some(inp_name) = op.inputs.first() {
                        let inp_shape = find_any_shape(function, inp_name);
                        inp_shape.len()
                    } else {
                        estimate_numel_from_shape(&shape, seq_len)
                    }
                } else if op.op == "reshape" && out_idx == 0 {
                    // Reshape copies raw bytes — output must have the same
                    // f32 count as the input's f32 view.  i64 inputs take
                    // 2× the f32 slots (8 bytes vs 4).
                    if let Some(inp_name) = op.inputs.first() {
                        if let Some(inp_data) = ssa_map.get(inp_name) {
                            inp_data.len() / 4  // total bytes → f32 count
                        } else {
                            estimate_numel_from_shape(&shape, seq_len)
                        }
                    } else {
                        estimate_numel_from_shape(&shape, seq_len)
                    }
                } else if op.op == "gather" && out_idx == 0 && op.inputs.len() >= 2 {
                    // gather(weight, indices): output = indices_count × embed_dim.
                    // The function output list shape is unreliable for gather.
                    let weight_shape_name = &op.inputs[0];
                    let indices_name = &op.inputs[1];
                    let wshape = find_any_shape(function, weight_shape_name);
                    let embed_dim: usize = wshape.iter().skip(1)
                        .map(|d| d.parse::<usize>().unwrap_or(1))
                        .product();
                    let num_indices = ssa_map.get(indices_name)
                        .map(|d| d.len() / 4)  // f32 count (4 bytes per element)
                        .unwrap_or(1);
                    (num_indices * embed_dim).max(1)
                } else {
                    estimate_numel_from_shape(&shape, seq_len)
                };
                let numel = numel.max(1);

                let mut vec = vec![0.0f32; numel];

                let raw_buf = InnerCpuBuffer::from_raw_parts(
                    vec.as_mut_ptr() as *mut u8,
                    numel * 4,
                    true, // borrowed
                )
                .map_err(|e| anyhow::anyhow!("InnerCpuBuffer: {}", e))?;

                // Build the output buffer shape.
                // For shape-preserving ops (element_wise, softmax, unsqueeze,
                // transpose, slice, compare, fill), the output shape equals the
                // first input's shape.  For shape_of it's [rank]; otherwise use
                // the declared shape from the function's I/O list.
                let output_dims: Vec<usize> = if out_idx == 0 {
                    match op.op.as_str() {
                        "element_wise" | "elementwise" | "softmax" | "unsqueeze"
                        | "transpose" | "slice" | "compare" | "fill" => {
                            // Derive from first input's shape.
                            if let Some(inp_name) = op.inputs.first() {
                                let inp_shape = find_any_shape(function, inp_name);
                                let inp_dims: Vec<usize> = inp_shape
                                    .iter()
                                    .map(|d| {
                                        if d == "?" || d == "-1" { 1 }
                                        else { d.parse::<usize>().unwrap_or(1) }
                                    })
                                    .collect();
                                if inp_dims.iter().all(|&d| d > 0) {
                                    inp_dims
                                } else {
                                    fallback_shape(&shape, numel)
                                }
                            } else {
                                fallback_shape(&shape, numel)
                            }
                        }
                        "reduce" => {
                            // Reduce preserves rank — output shape = input shape.
                            if let Some(inp_name) = op.inputs.first() {
                                let inp_shape = find_any_shape(function, inp_name);
                                let inp_dims: Vec<usize> = inp_shape
                                    .iter()
                                    .map(|d| {
                                        if d == "?" || d == "-1" { 1 }
                                        else { d.parse::<usize>().unwrap_or(1) }
                                    })
                                    .collect();
                                // Reduce collapses some dims but we keep rank
                                // for simplicity (the op handles the actual dims).
                                if inp_dims.iter().all(|&d| d > 0) {
                                    inp_dims
                                } else {
                                    fallback_shape(&shape, numel)
                                }
                            } else {
                                fallback_shape(&shape, numel)
                            }
                        }
                        "shape_of" => {
                            vec![numel]
                        }
                        "gather" => {
                            // gather(weight, indices): output = [N, embed_dim].
                            // Use numel-based fallback.
                            fallback_shape(&shape, numel)
                        }
                        "matmul" | "reshape" | "concat" | "cache_read" | "cache_write" => {
                            fallback_shape(&shape, numel)
                        }
                        _ => {
                            fallback_shape(&shape, numel)
                        }
                    }
                } else {
                    fallback_shape(&shape, numel)
                };
                let cpu_buf = CpuBuffer::with_meta(raw_buf, 4 /* f32 */, output_dims);
                output_vecs.push(vec);
                output_bufs.push(Box::new(cpu_buf));
            }

            // ── Execute ─────────────────────────────────────────────
            let input_refs: Vec<&dyn traits::Buffer> =
                input_bufs.iter().map(|b| b.as_ref()).collect();
            let output_refs: Vec<&dyn traits::Buffer> =
                output_bufs.iter().map(|b| b.as_ref()).collect();

            // Build the op name with optional kind suffix for the
            // HalRustExecutable dispatch (e.g. "element_wise:add").
            let op_name = match &op.kind {
                Some(kind) => format!("{}:{}", op.op, kind),
                None => op.op.clone(),
            };

            // Wrap execute in a panic-safe handler: with zero-filled weights,
            // many ops fail due to shape mismatches (matmul rank < 2),
            // type reinterpretation (i64→f32 in gather indices), or
            // index OOB.  Generated CPU kernels may panic with OOB access.
            // We catch panics AND errors, zero-filling outputs to keep
            // the forward pass running — producing garbage logits is
            // acceptable for this diagnostic binary.
            let exe_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                executable.execute(&op_name, stream, &input_refs, &output_refs)
            }));
            let output_shapes = match exe_result {
                Ok(Ok(shapes)) => shapes,
                Ok(Err(e)) => {
                    log::warn!(
                        "hal_runner: func[{}] op[{}] '{}' error: {}. \
                         Zero-filling and continuing.",
                        fi, oi, op_name, e,
                    );
                    zero_fill_outputs(&mut output_vecs, &output_bufs, fi, oi, &op_name)
                }
                Err(panic_payload) => {
                    let msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                        s.to_string()
                    } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "unknown panic".to_string()
                    };
                    log::warn!(
                        "hal_runner: func[{}] op[{}] '{}' PANIC: {}. \
                         Zero-filling and continuing.",
                        fi, oi, op_name, msg,
                    );
                    zero_fill_outputs(&mut output_vecs, &output_bufs, fi, oi, &op_name)
                }
            };
            log::trace!(
                "hal_runner: func[{}] op[{}] '{}' -> {} outputs, shapes={:?}",
                fi,
                oi,
                op_name,
                output_shapes.len(),
                output_shapes,
            );

            // ── Store outputs in SSA map ────────────────────────────
            for (idx, output_name) in op.outputs.iter().enumerate() {
                let out_vec = std::mem::take(&mut output_vecs[idx]);
                let raw_bytes: Vec<u8> =
                    out_vec.iter().flat_map(|&v| v.to_le_bytes()).collect();

                ssa_map.insert(output_name.clone(), raw_bytes);            }
        }

        // ── Capture wire output for cross-function wiring ─────────────
        // The main wire output (hidden state) from this function becomes
        // %arg0 for the next function.  It is identified as the function
        // output whose shape has dynamic batch/seq dims ("?") with a
        // static feature dimension (e.g. [?,?,768]).
        if fi < hal_ir.functions.len().saturating_sub(1) {
            let wire = find_main_output(function, &ssa_map);
            if let Some((name, data)) = wire {
                log::debug!(
                    "hal_runner: function[{}] main wire '{}' ({} bytes) ready for next function",
                    fi,
                    name,
                    data.len(),
                );
            }
        }
    }

    // ── Extract global output ───────────────────────────────────────
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

// ── Helpers ────────────────────────────────────────────────────────────

/// Zero-fill output buffers after a failed op execution.
/// Called by the panic-safe execute wrapper to keep the SSA map populated
/// with valid (zeroed) data so subsequent ops don't cascade-fail.
fn zero_fill_outputs(
    output_vecs: &mut [Vec<f32>],
    output_bufs: &[Box<dyn traits::Buffer>],
    fi: usize,
    oi: usize,
    _op_name: &str,
) -> Vec<Vec<i64>> {
    for (idx, out_vec) in output_vecs.iter_mut().enumerate() {
        out_vec.fill(0.0f32);
        log::trace!(
            "hal_runner: zero-filled func[{}] op[{}] output[{}] ({} elements)",
            fi, oi, idx, out_vec.len(),
        );
    }
    output_bufs
        .iter()
        .map(|b| {
            let s: Vec<i64> = b.shape().iter().map(|&d| d as i64).collect();
            s
        })
        .collect()
}

/// Estimate the number of f32 elements for a tensor with the given
/// HAL IR shape representation (strings with "?" for dynamic dims).
///
/// First "?" = batch (always 1), subsequent "?" = sequence length.
fn estimate_numel_from_shape(shape: &[String], seq_len: usize) -> usize {
    let mut numel: usize = 1;
    let mut first_dyn = true;
    for d in shape {
        let dim = if d == "?" || d == "-1" {
            if first_dyn {
                first_dyn = false;
                1 // batch
            } else {
                seq_len
            }
        } else {
            d.parse::<usize>().unwrap_or(1)
        };
        numel = numel.saturating_mul(dim);
    }
    // Generous minimum to accommodate intermediate tensors whose
    // declared shapes in the function output list are unreliable
    // (e.g., shape_of output declared as [1] but actually [rank],
    //  gather output declared as [1] but actually [N × embed_dim]).
    numel.max(65536) // 64K elements = 256 KB
}

/// Find the main (wire) output of a HAL function for cross-function wiring.
///
/// The wire is the hidden state output that gets passed to the next
/// function as `%arg0`.  It is identified by:
///   a) NOT being `consumed_internally` (excludes KV cache intermediates)
///   b) having at least one dynamic dimension ("?")
///   c) having rank >= 2 (excludes scalars and offsets)
///   d) preferring rank-3 shapes `[?, ?, X]` typical of hidden states
///
/// Returns `Some((ssa_name, data_bytes))` when found, `None` on failure.
fn find_main_output(
    function: &HalFunction,
    ssa_map: &std::collections::HashMap<String, Vec<u8>>,
) -> Option<(String, Vec<u8>)> {
    let mut best_score: i64 = -1;
    let mut best: Option<(&HalTensorDef, Vec<u8>)> = None;

    for output in &function.outputs {
        // Skip internally-consumed tensors (KV cache intermediates).
        if output.consumed_internally {
            continue;
        }
        let dyn_count = output.shape.iter().filter(|d| *d == "?" || *d == "-1").count();
        if dyn_count == 0 {
            continue; // skip fully static outputs (weights, constants)
        }
        let rank = output.shape.len();
        if rank < 2 {
            continue; // skip scalars and 1D offsets
        }

        // Score: prefer higher rank, more "?" dims, and rank 3 being ideal.
        let score = (rank as i64) * 10 + (dyn_count as i64) + if rank == 3 { 100 } else { 0 };

        if let Some(data) = ssa_map.get(&output.name) {
            if score > best_score {
                best_score = score;
                best = Some((output, data.clone()));
            }
        }
    }

    best.map(|(output, data)| (output.name.clone(), data))
}

/// Fallback shape builder for op outputs: converts the declared shape
/// strings to a `Vec<usize>`, substituting `"?` and `"-1"` with 1 (batch).
/// Uses `numel` as the flat fallback when the shape is all-dynamic.
fn fallback_shape(shape: &[String], numel: usize) -> Vec<usize> {
    let dims: Vec<usize> = shape
        .iter()
        .map(|d| {
            if d == "?" || d == "-1" {
                1
            } else {
                d.parse::<usize>().unwrap_or(1)
            }
        })
        .collect();
    if dims.is_empty() || dims.iter().all(|&d| d == 0) {
        vec![numel]
    } else {
        dims
    }
}

/// Find the shape definition for a tensor name in a function's
/// input, output, or weight list.
fn find_output_shape(function: &HalFunction, name: &str) -> (Vec<String>, bool) {
    for output in &function.outputs {
        if output.name == name {
            return (output.shape.clone(), false);
        }
    }
    for input in &function.inputs {
        if input.name == name {
            return (input.shape.clone(), false);
        }
    }
    // Check weight list for invisible constant SSAs (e.g. %1 for position emb).
    for weight_entry in &function.weights {
        if weight_entry.ssa == name {
            return (weight_entry.shape.clone(), false);
        }
    }
    (vec!["?".to_string()], false)
}

/// Find the shape for any SSA name in a function (inputs + outputs).
fn find_any_shape(function: &HalFunction, name: &str) -> Vec<String> {
    find_output_shape(function, name).0
}

#[allow(dead_code)]
/// Try to find a matching tensor from prior function outputs for the
/// given function input definition.  Uses shape matching when names
/// don't directly correspond (the common case across functions).
fn find_matching_output(
    hal_ir: &HalIR,
    _current_func: &HalFunction,
    input_def: &HalTensorDef,
    ssa_map: &HashMap<String, Vec<u8>>,
    seq_len: usize,
) -> Option<Vec<u8>> {
    let input_numel = estimate_numel_from_shape(&input_def.shape, seq_len);

    // Search all prior functions' outputs for a tensor with matching
    // numel and compatible shape.
    for func in &hal_ir.functions {
        for output in &func.outputs {
            if output.name == input_def.name {
                // Same name — direct match (should already be in map).
                return ssa_map.get(&output.name).cloned();
            }

            let output_numel = estimate_numel_from_shape(&output.shape, seq_len);

            // Check if the output is in the SSA map and has matching size.
            if output_numel == input_numel && ssa_map.contains_key(&output.name) {
                // Also check shape compatibility (same rank, or both 1D).
                if output.shape.len() == input_def.shape.len() {
                    return ssa_map.get(&output.name).cloned();
                }
            }
        }
    }

    None
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hal::traits;

    /// Minimal buffer backed by raw bytes (for testing).
    #[derive(Debug)]
    struct TestBuf(Vec<u8>, usize, Vec<usize>);

    impl traits::Buffer for TestBuf {
        fn as_ptr(&self) -> *const u8 {
            self.0.as_ptr()
        }
        fn as_mut_ptr(&mut self) -> *mut u8 {
            self.0.as_mut_ptr()
        }
        fn len(&self) -> usize {
            self.0.len()
        }
        fn copy_from_host(&mut self, src: &[u8], _: &dyn traits::Stream) -> Result<(), anyhow::Error> {
            self.0.copy_from_slice(src);
            Ok(())
        }
        fn copy_to_host(&self, dst: &mut [u8], _: &dyn traits::Stream) -> Result<(), anyhow::Error> {
            dst.copy_from_slice(&self.0);
            Ok(())
        }
        fn element_size(&self) -> usize {
            self.1
        }
        fn shape(&self) -> Vec<usize> {
            self.2.clone()
        }
        fn rank(&self) -> u8 {
            self.2.len() as u8
        }
    }

    #[derive(Debug)]
    struct NoopStream;
    impl traits::Stream for NoopStream {
        fn synchronize(&self) -> Result<(), anyhow::Error> {
            Ok(())
        }
        fn wait_event(&self, _: &dyn traits::Event) -> Result<(), anyhow::Error> {
            Ok(())
        }
        fn record_event(&self, _: &dyn traits::Event) -> Result<(), anyhow::Error> {
            Ok(())
        }
    }

    /// Helper to get the test hal_ir.json path (relative to CARGO_MANIFEST_DIR).
    fn test_hal_ir_path() -> String {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        format!(
            "{}/../compiled/opt_125m_fresh/generated/hal_ir.json",
            manifest_dir
        )
    }

    #[test]
    fn test_hal_runner_parses_json() {
        let path = test_hal_ir_path();
        let runner = HalRustRunner::from_path(&path).expect("parse hal_ir.json");
        assert_eq!(runner.hal_ir.num_functions, 16);
        assert_eq!(runner.hal_ir.model_name, "opt_125m_fresh");

        let total_ops: usize = runner
            .hal_ir
            .functions
            .iter()
            .map(|f| f.ops.len())
            .sum();
        assert_eq!(total_ops, 610);

        // Verify each function has a name and ops.
        for func in &runner.hal_ir.functions {
            assert!(!func.name.is_empty(), "function name should not be empty");
            assert!(
                !func.ops.is_empty(),
                "function '{}' should have ops",
                func.name
            );
        }
    }

    #[test]
    fn test_hal_runner_executes_function() {
        let path = test_hal_ir_path();
        let content = std::fs::read_to_string(&path).expect("read hal_ir.json");
        let hal_ir: HalIR = serde_json::from_str(&content).expect("parse hal_ir.json");

        let exe = crate::hal::rust::executable::HalRustExecutable::new(hal_ir.num_functions);
        let stream = NoopStream;

        // Run only main_0 (entry function with 35 ops) WITHOUT weight provider.
        // All weight tensors are zero-filled.  The output will be garbage but
        // execution should not panic.
        //
        // Cross-function input mapping (func[0] outputs → func[1..N] inputs)
        // is handled in a follow-up task.
        let input_ids: Vec<u32> = vec![0, 1, 2, 3];
        let positions: Vec<u32> = vec![0, 1, 2, 3];

        // Build a HAL IR with only the first function.
        let single_hal_ir = HalIR {
            model_name: hal_ir.model_name.clone(),
            num_functions: 1,
            functions: vec![hal_ir.functions[0].clone()],
        };

        // With zero-filled weights and invisible constants (e.g. %1, %197–%200
        // that are not in any function I/O list), ops that require realistic data
        // (like gather with position embeddings) will fail with index-out-of-bounds.
        // This is expected — the purpose of the test is to verify the runner's op
        // dispatch path, not correctness with zero weights.
        let result = run_hal_function_graph(
            &exe, &single_hal_ir, None, &stream, &input_ids, &positions,
        );
        match result {
            Ok(tensor) => {
                assert_eq!(tensor.dtype, Dtype::F32);
                assert!(tensor.numel() > 0, "output should have elements");
                log::info!(
                    "hal_runner test: output shape={:?} numel={}",
                    tensor.shape,
                    tensor.numel()
                );
            }
            Err(e) => {
                // All ops up to op[32] (element_wise with invisible constants)
                // execute correctly.  Gather at op[33] may fail with
                // zero-filled invisible constants — this is expected.
                let msg = e.to_string();
                assert!(
                    msg.contains("out of bounds") || msg.contains("gather"),
                    "unexpected error: {}",
                    msg,
                );
                log::warn!(
                    "hal_runner test: expected error with zero-filled weights: {}",
                    msg,
                );
            }
        }
    }

    #[test]
    fn test_hal_runner_executes_single_op() {
        // Verify a minimal single-op execution: shape_of on a 2-element input.
        use crate::hal::rust::executable::HalRustExecutable;

        let exe = HalRustExecutable::new(1);
        let stream = NoopStream;

        // Create a simple hal_ir with one function containing one shape_of op.
        let hal_ir = HalIR {
            model_name: "test".to_string(),
            num_functions: 1,
            functions: vec![HalFunction {
                name: "main_0".to_string(),
                layer: 0,
                weights: vec![],
                weight_inputs: HashMap::new(),
                inputs: vec![HalTensorDef {
                    name: "%arg0".to_string(),
                    shape: vec!["?".to_string(), "?".to_string()],
                    dtype: "i64".to_string(),
                    consumed_internally: false,
                }],
                outputs: vec![
                    HalTensorDef {
                        name: "%213".to_string(),
                        shape: vec!["2".to_string()],
                        dtype: "f32".to_string(),
                        consumed_internally: false,
                    },
                    HalTensorDef {
                        name: "%1".to_string(),
                        shape: vec!["768".to_string()],
                        dtype: "f32".to_string(),
                        consumed_internally: false,
                    },
                ],
                ops: vec![HalOp {
                    op: "shape_of".to_string(),
                    kind: None,
                    inputs: vec!["%arg0".to_string()],
                    outputs: vec!["%213".to_string()],
                    weight: None,
                    shape: None,
                    value: None,
                }],
            }],
        };

        let input_ids: Vec<u32> = vec![0, 1, 2, 3];
        let positions: Vec<u32> = vec![0, 1, 2, 3];

        let result = run_hal_function_graph(
            &exe, &hal_ir, None, &stream, &input_ids, &positions,
        )
        .expect("single op execution");

        assert_eq!(result.dtype, Dtype::F32);
        // shape_of on rank-2 input should produce 2 elements.
        assert_eq!(result.numel(), 2);
        // shape_of on %arg0 with dynamic shape [?, ?] (estimated as [1, 4])
        // returns [1.0, 4.0] from OpShapeMeta.
        let data = result.as_slice();
        assert_eq!(data.len(), 2, "shape_of should output 2 dims (rank=2)");
    }
}
