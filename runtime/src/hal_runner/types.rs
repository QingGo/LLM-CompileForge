//! HAL IR data structures — types used by the HAL IR graph runner.
//!
//! These are deserialized from `hal_ir.json` and represent the full
//! HAL function graph with weights, inputs, outputs, and ops.

use std::collections::HashMap;

use crate::model::sfa_tensor::SFATensor;
use crate::model::tensor::Tensor;

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
    /// Data type of this tensor (e.g. "f32", "i64", "f16").
    /// Defaults to "f32" if not specified in HAL IR.
    #[serde(default = "default_dtype")]
    pub dtype: String,
    /// Whether this tensor is consumed internally within the function.
    /// When true, the tensor is NOT propagated to subsequent functions
    /// as a cross-function wire.
    #[serde(default)]
    pub consumed_internally: bool,
}

fn default_dtype() -> String {
    "f32".to_string()
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
    #[allow(dead_code)]
    pub weight: Option<String>,
    /// Op shape metadata (mixed integer/string). Stored as JSON Value
    /// since reshape targets use both "?" and integer dims.
    #[serde(default, deserialize_with = "deserialize_optional_shape")]
    #[allow(dead_code)]
    pub shape: Option<Vec<String>>,
    #[serde(default)]
    #[allow(dead_code)]
    pub value: Option<f64>,
    /// Dtype annotations for each input tensor (e.g. ["i64", "f32"]).
    #[serde(default)]
    pub input_dtypes: Vec<String>,
    /// Dtype annotations for each output tensor (e.g. ["f32"]).
    #[serde(default)]
    pub output_dtypes: Vec<String>,
    /// Dimension permutation for transpose ops (e.g. [1, 2] means swap axes 1 and 2).
    #[serde(default)]
    pub dims: Option<Vec<usize>>,
    /// Dimension index for shape_of ops (extract a single dim instead of full shape).
    #[serde(default)]
    pub dim: Option<usize>,
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
    /// GraphRunner internal SSA store (trait impl state).
    #[doc(hidden)]
    pub(crate) gr_ssa_map: std::cell::RefCell<HashMap<String, SFATensor>>,
    /// GraphRunner internal SSA shapes (trait impl state).
    #[doc(hidden)]
    pub(crate) gr_ssa_shapes: std::cell::RefCell<HashMap<String, Vec<usize>>>,
    /// GraphRunner internal SSA dtypes (trait impl state).
    #[doc(hidden)]
    pub(crate) gr_ssa_dtypes: std::cell::RefCell<HashMap<String, crate::model::tensor::Dtype>>,
    /// GraphRunner weight cache (trait impl state).
    #[doc(hidden)]
    pub(crate) gr_weight_cache: std::cell::RefCell<HashMap<String, Tensor>>,
}

impl HalRustRunner {
    /// Parse a HAL IR JSON string.
    pub fn from_json(json_str: &str) -> Result<Self, anyhow::Error> {
        let hal_ir: HalIR = serde_json::from_str(json_str)?;
        let semantics = super::default_hal_op_semantics();
        let warnings = super::validate_hal_ir_against_semantics(
            &hal_ir, &semantics,
        );
        for w in &warnings {
            log::warn!("{}", w);
        }
        Ok(Self {
            hal_ir,
            gr_ssa_map: std::cell::RefCell::new(HashMap::new()),
            gr_ssa_shapes: std::cell::RefCell::new(HashMap::new()),
            gr_ssa_dtypes: std::cell::RefCell::new(HashMap::new()),
            gr_weight_cache: std::cell::RefCell::new(HashMap::new()),
        })
    }

    /// Load HAL IR from a JSON file.
    pub fn from_path(path: &str) -> Result<Self, anyhow::Error> {
        let content = std::fs::read_to_string(path)?;
        Self::from_json(&content)
    }
}
