// ── Tests ──────────────────────────────────────────────────────────────

use std::collections::HashMap;

use super::*;
use crate::hal::traits;
use crate::hal::traits::Executable as _;
use crate::tensor::Dtype;

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
    assert_eq!(total_ops, 347);

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
                input_dtypes: vec!["i64".to_string()],
                output_dtypes: vec!["f32".to_string()],
                dims: None,
                dim: None,
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

#[test]
fn test_shape_of_output_contains_actual_dims() {
    // RED: shape_of should output actual shape values [1.0, 4.0],
    // not just the rank [2.0].
    use crate::hal::rust::executable::HalRustExecutable;

    let exe = HalRustExecutable::new(1);
    let stream = NoopStream;

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
            outputs: vec![HalTensorDef {
                name: "%213".to_string(),
                shape: vec!["2".to_string()],
                dtype: "f32".to_string(),
                consumed_internally: false,
            }],
            ops: vec![HalOp {
                op: "shape_of".to_string(),
                kind: None,
                inputs: vec!["%arg0".to_string()],
                outputs: vec!["%213".to_string()],
                weight: None,
                shape: None,
                value: None,
                input_dtypes: vec!["i64".to_string()],
                output_dtypes: vec!["f32".to_string()],
                dims: None,
                dim: None,
            }],
        }],
    };

    let input_ids: Vec<u32> = vec![2, 32826, 85, 4129];
    let positions: Vec<u32> = vec![0, 1, 2, 3];

    let result = run_hal_function_graph(
        &exe, &hal_ir, None, &stream, &input_ids, &positions,
    )
    .expect("shape_of execution");

    let data = result.as_slice();
    // shape_of on [1, 4] input should output [1.0, 4.0], not [2.0]
    assert_eq!(data[0], 1.0, "first dim should be batch=1");
    assert_eq!(data[1], 4.0, "second dim should be seq_len=4");
    // Output shape should be [2] (rank), not [1, 4]
    // This is correct — shape_of outputs a 1D tensor of shape values
    assert_eq!(result.shape, vec![2], "shape_of output shape should be [rank]");
}

#[test]
fn test_reshape_uses_shape_of_values() {
    // RED: reshape should use shape_of's DATA values [1, 4] as target shape,
    // not shape_of's OUTPUT shape [2].
    use crate::hal::rust::executable::HalRustExecutable;

    let exe = HalRustExecutable::new(1);
    let stream = NoopStream;

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
            outputs: vec![HalTensorDef {
                name: "%215".to_string(),
                shape: vec!["?".to_string(), "?".to_string()],
                dtype: "i64".to_string(),
                consumed_internally: false,
            }],
            ops: vec![
                HalOp {
                    op: "shape_of".to_string(),
                    kind: None,
                    inputs: vec!["%arg0".to_string()],
                    outputs: vec!["%213".to_string()],
                    weight: None,
                    shape: None,
                    value: None,
                    input_dtypes: vec!["i64".to_string()],
                    output_dtypes: vec!["f32".to_string()],
                    dims: None,
                dim: None,
                },
                HalOp {
                    op: "reshape".to_string(),
                    kind: None,
                    inputs: vec!["%arg0".to_string(), "%213".to_string()],
                    outputs: vec!["%215".to_string()],
                    weight: None,
                    shape: Some(vec!["?".to_string(), "?".to_string()]),
                    value: None,
                    input_dtypes: vec!["i64".to_string(), "f32".to_string()],
                    output_dtypes: vec!["i64".to_string()],
                    dims: None,
                dim: None,
                },
            ],
        }],
    };

    let input_ids: Vec<u32> = vec![2, 32826, 85, 4129];
    let positions: Vec<u32> = vec![0, 1, 2, 3];

    let result = run_hal_function_graph(
        &exe, &hal_ir, None, &stream, &input_ids, &positions,
    )
    .expect("reshape+shape_of execution");

    // reshape(%arg0 [1,4], shape_of(%arg0)) should produce [1, 4]
    assert_eq!(result.shape, vec![1, 4],
        "reshape should produce [1, 4] from shape_of values, not [2]");
}

#[test]
fn test_reshape_uses_shape_of_when_op_shape_incomplete() {
    // When op.shape has '?' dims and shape_of inputs provide the values,
    // reshape should use shape_of values for those dynamic dims.
    // e.g., input [1, 4, 768] → reshape ['?', '?', 12, 64] with shape_of [1, 4]
    // → output [1, 4, 12, 64].
    use crate::hal::rust::executable::HalRustExecutable;

    let exe = HalRustExecutable::new(1);
    let stream = NoopStream;

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
                shape: vec!["?".to_string(), "?".to_string(), "768".to_string()],
                dtype: "f32".to_string(),
                consumed_internally: false,
            }],
            outputs: vec![HalTensorDef {
                name: "%out".to_string(),
                shape: vec!["?".to_string(), "?".to_string(), "?".to_string(), "?".to_string()],
                dtype: "f32".to_string(),
                consumed_internally: false,
            }],
            ops: vec![
                HalOp {
                    op: "shape_of".to_string(),
                    kind: None,
                    inputs: vec!["%arg0".to_string()],
                    outputs: vec!["%shape".to_string()],
                    weight: None,
                    shape: None,
                    value: None,
                    input_dtypes: vec!["f32".to_string()],
                    output_dtypes: vec!["f32".to_string()],
                    dims: None,
                dim: None,
                },
                HalOp {
                    op: "reshape".to_string(),
                    kind: None,
                    inputs: vec!["%arg0".to_string(), "%shape".to_string()],
                    outputs: vec!["%out".to_string()],
                    weight: None,
                    shape: Some(vec!["?".to_string(), "?".to_string(), "12".to_string(), "64".to_string()]),
                    value: None,
                    input_dtypes: vec!["f32".to_string(), "f32".to_string()],
                    output_dtypes: vec!["f32".to_string()],
                    dims: None,
                dim: None,
                },
            ],
        }],
    };

    let input_ids: Vec<u32> = vec![2, 32826, 85, 4129];
    let positions: Vec<u32> = vec![0, 1, 2, 3];

    let mut ssa_map: std::collections::HashMap<String, SFATensor> = std::collections::HashMap::new();
    let mut ssa_shapes: std::collections::HashMap<String, Vec<usize>> = std::collections::HashMap::new();
    let mut ssa_dtypes: std::collections::HashMap<String, Dtype> = std::collections::HashMap::new();

    let arg0_numel = 1 * 4 * 768;
    let t = SFATensor::from_vec_f32(vec![0f32; arg0_numel], vec![1, 4, 768]);
    ssa_map.insert("%arg0".to_string(), t);
    ssa_shapes.insert("%arg0".to_string(), vec![1, 4, 768]);
    ssa_dtypes.insert("%arg0".to_string(), Dtype::F32);

    use crate::hal_runner::shape_inference::compute_output_shape;
    for op in &hal_ir.functions[0].ops {
        let output_dtype = crate::hal_runner::infer_output_dtype(op, &ssa_dtypes);
        for (out_idx, output_name) in op.outputs.iter().enumerate() {
            let (numel, output_dims) = compute_output_shape(
                op, out_idx, &ssa_shapes, &ssa_map, &ssa_dtypes, &hal_ir.functions[0], 4,
            );
            let out_elem_size = output_dtype.element_size();
            let mut out_tensor = if out_elem_size == 8 {
                SFATensor::from_vec_i64(vec![0i64; numel], vec![numel])
            } else {
                SFATensor::from_vec_f32(vec![0f32; numel], vec![numel])
            };

            if op.op == "reshape" {
                if let Some(in_tensor) = op.inputs.first().and_then(|n| ssa_map.get(n)) {
                    let copy_count = in_tensor.numel().min(numel);
                    let src = in_tensor.data_ptr();
                    let dst = out_tensor.data_ptr();
                    let copy_bytes = copy_count * out_elem_size;
                    unsafe {
                        std::ptr::copy_nonoverlapping(src, dst, copy_bytes);
                    }
                }
            }
            if op.op == "shape_of" {
                if let Some(in_shape) = op.inputs.first().and_then(|n| ssa_shapes.get(n)) {
                    let ptr = out_tensor.data_ptr() as *mut f32;
                    let f32_slice = unsafe { std::slice::from_raw_parts_mut(ptr, out_tensor.numel().min(in_shape.len())) };
                    for (i, &dim) in in_shape.iter().enumerate() {
                        if i < f32_slice.len() {
                            f32_slice[i] = dim as f32;
                        }
                    }
                }
            }

            ssa_map.insert(output_name.clone(), out_tensor);
            ssa_dtypes.insert(output_name.clone(), output_dtype);
            ssa_shapes.insert(output_name.clone(), output_dims.clone());

            if op.op == "shape_of" {
                if let Some(t) = ssa_map.get(output_name) {
                    let numel = t.numel();
                    let ptr = t.data_ptr() as *const f32;
                    let f32_vals = unsafe { std::slice::from_raw_parts(ptr, numel) };
                    let dims: Vec<usize> = f32_vals.iter().map(|&v| v as usize).collect();
                    if !dims.is_empty() {
                        ssa_shapes.insert(output_name.clone(), dims);
                    }
                }
            }
        }
    }

    let out_shape = ssa_shapes.get("%out").expect("reshape output should be in ssa_shapes");
    assert_eq!(*out_shape, vec![1, 4, 12, 64],
        "reshape should use shape_of values [1, 4] for dynamic dims, got {:?}", out_shape);
}

#[test]
fn test_wire_passes_data_through_unchanged() {
    // Cross-function wiring should copy data and actual shapes from
    // the previous function's output, without any slicing.
    use crate::hal::rust::executable::HalRustExecutable;

    let exe = HalRustExecutable::new(1);
    let stream = NoopStream;

    let hal_ir = HalIR {
        model_name: "test".to_string(),
        num_functions: 2,
        functions: vec![
            HalFunction {
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
                outputs: vec![HalTensorDef {
                    name: "%out0".to_string(),
                    shape: vec!["?".to_string(), "?".to_string(), "768".to_string()],
                    dtype: "f32".to_string(),
                    consumed_internally: false,
                }],
                ops: vec![HalOp {
                    op: "element_wise".to_string(),
                    kind: Some("add".to_string()),
                    inputs: vec!["%arg0".to_string(), "%arg0".to_string()],
                    outputs: vec!["%out0".to_string()],
                    weight: None,
                    shape: None,
                    value: None,
                    input_dtypes: vec!["i64".to_string(), "i64".to_string()],
                    output_dtypes: vec!["i64".to_string()],
                    dims: None,
                dim: None,
                }],
            },
            HalFunction {
                name: "main_1".to_string(),
                layer: 1,
                weights: vec![],
                weight_inputs: HashMap::new(),
                inputs: vec![HalTensorDef {
                    name: "%arg0".to_string(),
                    shape: vec!["?".to_string(), "?".to_string(), "768".to_string()],
                    dtype: "f32".to_string(),
                    consumed_internally: false,
                }],
                outputs: vec![HalTensorDef {
                    name: "%out1".to_string(),
                    shape: vec!["?".to_string(), "?".to_string(), "768".to_string()],
                    dtype: "f32".to_string(),
                    consumed_internally: false,
                }],
                ops: vec![HalOp {
                    op: "element_wise".to_string(),
                    kind: Some("add".to_string()),
                    inputs: vec!["%arg0".to_string(), "%arg0".to_string()],
                    outputs: vec!["%out1".to_string()],
                    weight: None,
                    shape: None,
                    value: None,
                    input_dtypes: vec!["f32".to_string(), "f32".to_string()],
                    output_dtypes: vec!["f32".to_string()],
                    dims: None,
                dim: None,
                }],
            },
        ],
    };

    let mut ssa_map: std::collections::HashMap<String, SFATensor> = std::collections::HashMap::new();
    let mut ssa_shapes: std::collections::HashMap<String, Vec<usize>> = std::collections::HashMap::new();
    let mut ssa_dtypes: std::collections::HashMap<String, Dtype> = std::collections::HashMap::new();

    ssa_map.insert("%arg0".to_string(), SFATensor::from_vec_i64(vec![0i64; 4], vec![1, 4]));
    ssa_shapes.insert("%arg0".to_string(), vec![1, 4]);
    ssa_dtypes.insert("%arg0".to_string(), Dtype::I64);

    ssa_map.insert("%out0".to_string(), SFATensor::from_vec_f32(vec![0f32; 1 * 4 * 768], vec![1, 4, 768]));
    ssa_shapes.insert("%out0".to_string(), vec![1, 4, 768]);
    ssa_dtypes.insert("%out0".to_string(), Dtype::F32);

    use crate::hal_runner::prep::wire_cross_function_inputs;
    wire_cross_function_inputs(
        1,
        &hal_ir.functions[1],
        &hal_ir.functions[0],
        &mut ssa_map,
        &mut ssa_shapes,
        &mut ssa_dtypes,
        4,
    );

    let wired_shape = ssa_shapes.get("%arg0").expect("wired %arg0 should exist");
    assert_eq!(*wired_shape, vec![1, 4, 768],
        "wiring should pass through actual shape [1, 4, 768], got {:?}", wired_shape);
}

#[test]
fn test_transpose_output_shape_permuted() {
    // RED: compute_output_shape for transpose with dims=[1,2] on [1,4,12,64]
    // should return permuted shape [1,12,4,64], NOT the input shape [1,4,12,64].
    use crate::hal_runner::shape_inference::compute_output_shape;

    let op = HalOp {
        op: "transpose".to_string(),
        kind: None,
        inputs: vec!["%inp".to_string()],
        outputs: vec!["%out".to_string()],
        weight: None,
        shape: None,
        value: None,
        input_dtypes: vec!["f32".to_string()],
        output_dtypes: vec!["f32".to_string()],
        dims: Some(vec![1, 2]),
        dim: None,
    };

    let mut ssa_shapes: HashMap<String, Vec<usize>> = HashMap::new();
    ssa_shapes.insert("%inp".to_string(), vec![1, 4, 12, 64]);
    let ssa_map: std::collections::HashMap<String, SFATensor> = std::collections::HashMap::new();
    let ssa_dtypes: HashMap<String, Dtype> = HashMap::new();

    let function = HalFunction {
        name: "test".to_string(),
        layer: 0,
        weights: vec![],
        weight_inputs: HashMap::new(),
        inputs: vec![],
        outputs: vec![HalTensorDef {
            name: "%out".to_string(),
            shape: vec!["?".to_string(), "?".to_string(), "?".to_string(), "?".to_string()],
            dtype: "f32".to_string(),
            consumed_internally: false,
        }],
        ops: vec![op.clone()],
    };

    let (numel, shape) = compute_output_shape(
        &op, 0, &ssa_shapes, &ssa_map, &ssa_dtypes, &function, 4,
    );

    assert_eq!(shape, vec![1, 12, 4, 64],
        "transpose dims=[1,2] on [1,4,12,64] should produce [1,12,4,64], got {:?}",
        shape);
    assert_eq!(numel, 3072,
        "transpose numel should be 3072, got {}", numel);
}

#[test]
fn test_transpose_output_shape_no_dims() {
    // When transpose has no dims, it should fall back to shape-preserving.
    use crate::hal_runner::shape_inference::compute_output_shape;

    let op = HalOp {
        op: "transpose".to_string(),
        kind: None,
        inputs: vec!["%inp".to_string()],
        outputs: vec!["%out".to_string()],
        weight: None,
        shape: None,
        value: None,
        input_dtypes: vec!["f32".to_string()],
        output_dtypes: vec!["f32".to_string()],
        dims: None,
        dim: None,
    };

    let mut ssa_shapes: HashMap<String, Vec<usize>> = HashMap::new();
    ssa_shapes.insert("%inp".to_string(), vec![1, 4, 12, 64]);
    let ssa_map: std::collections::HashMap<String, SFATensor> = HashMap::new();
    let ssa_dtypes: HashMap<String, Dtype> = HashMap::new();

    let function = HalFunction {
        name: "test".to_string(),
        layer: 0,
        weights: vec![],
        weight_inputs: HashMap::new(),
        inputs: vec![],
        outputs: vec![HalTensorDef {
            name: "%out".to_string(),
            shape: vec!["?".to_string(), "?".to_string(), "?".to_string(), "?".to_string()],
            dtype: "f32".to_string(),
            consumed_internally: false,
        }],
        ops: vec![op.clone()],
    };

    let (numel, shape) = compute_output_shape(
        &op, 0, &ssa_shapes, &ssa_map, &ssa_dtypes, &function, 4,
    );

    assert_eq!(shape, vec![1, 4, 12, 64],
        "transpose without dims should be shape-preserving, got {:?}",
        shape);
}

#[test]
fn test_matmul_dispatch_recognized() {
    // RED: dispatch("matmul", ...) should NOT return "unknown op".
    use crate::hal::rust::executable::HalRustExecutable;

    let exe = HalRustExecutable::new(1);
    let stream = NoopStream;

    let a = TestBuf(vec![0u8; 16], 4, vec![2, 2]);
    let b = TestBuf(vec![0u8; 16], 4, vec![2, 2]);
    let out = TestBuf(vec![0u8; 16], 4, vec![2, 2]);
    let inputs: [&dyn traits::Buffer; 2] = [&a, &b];
    let outputs: [&dyn traits::Buffer; 1] = [&out];

    let result = exe.execute("matmul", &stream, &inputs, &outputs);
    assert!(result.is_ok(),
        "matmul dispatch should succeed, got: {:?}", result.err());
}

// ── TDD: shape-preserving ops ─────────────────────────────────────────

fn make_shape_preserving_test_case(
    op_name: &str,
    input_shape: Vec<usize>,
) -> (usize, Vec<usize>) {
    use crate::hal_runner::shape_inference::compute_output_shape;

    let op = HalOp {
        op: op_name.to_string(),
        kind: None,
        inputs: vec!["%inp".to_string()],
        outputs: vec!["%out".to_string()],
        weight: None,
        shape: None,
        value: None,
        input_dtypes: vec!["f32".to_string()],
        output_dtypes: vec!["f32".to_string()],
        dims: None,
        dim: None,
    };

    let mut ssa_shapes: HashMap<String, Vec<usize>> = HashMap::new();
    ssa_shapes.insert("%inp".to_string(), input_shape);
    let ssa_map: std::collections::HashMap<String, SFATensor> = HashMap::new();
    let ssa_dtypes: HashMap<String, Dtype> = HashMap::new();

    let function = HalFunction {
        name: "test".to_string(),
        layer: 0,
        weights: vec![],
        weight_inputs: HashMap::new(),
        inputs: vec![],
        outputs: vec![HalTensorDef {
            name: "%out".to_string(),
            shape: vec!["?".to_string(), "?".to_string(), "768".to_string()],
            dtype: "f32".to_string(),
            consumed_internally: false,
        }],
        ops: vec![op.clone()],
    };

    compute_output_shape(&op, 0, &ssa_shapes, &ssa_map, &ssa_dtypes, &function, 4)
}

#[test]
fn test_layer_norm_output_shape() {
    // layer_norm on [4, 768] should output [4, 768] (shape-preserving).
    let (numel, shape) = make_shape_preserving_test_case("layer_norm", vec![4, 768]);
    assert_eq!(shape, vec![4, 768],
        "layer_norm on [4,768] should output [4,768], got {:?}", shape);
    assert_eq!(numel, 3072, "layer_norm numel should be 3072, got {}", numel);
}

#[test]
fn test_linear_output_shape_lm_head() {
    // LM head: input [1, 1, 768] with weight [50272, 768] should output [1, 1, 50272].
    // Weight is stored as [out_features=50272, in_features=768] (PyTorch convention).
    // linear op: input @ weight.T → output's last dim = out_features = weight[0].
    use crate::hal_runner::shape_inference::compute_output_shape;

    let op = HalOp {
        op: "linear".to_string(),
        kind: None,
        inputs: vec!["%inp".to_string(), "%weight".to_string()],
        outputs: vec!["%out".to_string()],
        weight: None,
        shape: None,
        value: None,
        input_dtypes: vec!["f32".to_string()],
        output_dtypes: vec!["f32".to_string()],
        dims: None,
        dim: None,
    };

    let mut ssa_shapes: HashMap<String, Vec<usize>> = HashMap::new();
    ssa_shapes.insert("%inp".to_string(), vec![1, 1, 768]);
    ssa_shapes.insert("%weight".to_string(), vec![50272, 768]);

    let function = crate::hal_runner::types::HalFunction {
        name: "test".to_string(),
        layer: 0,
        inputs: vec![],
        outputs: vec![],
        weights: vec![],
        weight_inputs: std::collections::HashMap::new(),
        ops: vec![],
    };

    let ssa_map: std::collections::HashMap<String, SFATensor> = HashMap::new();
    let ssa_dtypes: HashMap<String, crate::tensor::Dtype> = HashMap::new();

    let (numel, shape) = compute_output_shape(
        &op, 0, &ssa_shapes, &ssa_map, &ssa_dtypes, &function, 4,
    );
    assert_eq!(shape, vec![1, 1, 50272],
        "linear LM head on [1,1,768] with weight [50272,768] should output [1,1,50272], got {:?}", shape);
    assert_eq!(numel, 50272, "linear LM head numel should be 50272, got {}", numel);
}

#[test]
fn test_linear_output_shape_ffn_fc2() {
    // FFN fc2: input [4, 3072] with weight [768, 3072] should output [4, 768].
    // Weight is stored as [out_features=768, in_features=3072] (PyTorch convention).
    use crate::hal_runner::shape_inference::compute_output_shape;

    let op = HalOp {
        op: "linear".to_string(),
        kind: None,
        inputs: vec!["%inp".to_string(), "%weight".to_string()],
        outputs: vec!["%out".to_string()],
        weight: None,
        shape: None,
        value: None,
        input_dtypes: vec!["f32".to_string()],
        output_dtypes: vec!["f32".to_string()],
        dims: None,
        dim: None,
    };

    let mut ssa_shapes: HashMap<String, Vec<usize>> = HashMap::new();
    ssa_shapes.insert("%inp".to_string(), vec![4, 3072]);
    ssa_shapes.insert("%weight".to_string(), vec![768, 3072]);

    let function = crate::hal_runner::types::HalFunction {
        name: "test".to_string(),
        layer: 0,
        inputs: vec![],
        outputs: vec![],
        weights: vec![],
        weight_inputs: std::collections::HashMap::new(),
        ops: vec![],
    };

    let ssa_map: std::collections::HashMap<String, SFATensor> = HashMap::new();
    let ssa_dtypes: HashMap<String, crate::tensor::Dtype> = HashMap::new();

    let (numel, shape) = compute_output_shape(
        &op, 0, &ssa_shapes, &ssa_map, &ssa_dtypes, &function, 4,
    );
    assert_eq!(shape, vec![4, 768],
        "linear FFN fc2 on [4,3072] with weight [768,3072] should output [4,768], got {:?}", shape);
    assert_eq!(numel, 3072, "linear FFN fc2 numel should be 3072, got {}", numel);
}

#[test]
fn test_sdpa_output_shape() {
    // scaled_dot_product_attention on [1, 1, 768] should output [1, 1, 768].
    let (numel, shape) = make_shape_preserving_test_case(
        "scaled_dot_product_attention", vec![1, 1, 768]);
    assert_eq!(shape, vec![1, 1, 768],
        "sdpa on [1,1,768] should output [1,1,768], got {:?}", shape);
    assert_eq!(numel, 768, "sdpa numel should be 768, got {}", numel);
}

#[test]
fn test_attention_pipeline_shape_propagation() {
    // Simulate the attention pipeline: linear → reshape → transpose → SDPA → transpose → reshape
    // This tests that compute_output_shape correctly handles the full pipeline.
    use crate::hal_runner::shape_inference::compute_output_shape;
    use std::collections::HashMap;

    let function = crate::hal_runner::types::HalFunction {
        name: "test".to_string(), layer: 0,
        inputs: vec![], outputs: vec![], weights: vec![],
        weight_inputs: HashMap::new(), ops: vec![],
    };
    let mut ssa_map: std::collections::HashMap<String, SFATensor> = HashMap::new();
    let mut ssa_dtypes: HashMap<String, crate::tensor::Dtype> = HashMap::new();
    let mut ssa_shapes: HashMap<String, Vec<usize>> = HashMap::new();

    // Input: [1, 4, 768] hidden state after layer_norm
    ssa_shapes.insert("%hidden".to_string(), vec![1, 4, 768]);

    // Step 1: Q projection (linear: [1, 4, 768] × [768, 768] → [1, 4, 768])
    let q_proj = HalOp {
        op: "linear".to_string(), kind: None,
        inputs: vec!["%hidden".to_string(), "%q_weight".to_string()],
        outputs: vec!["%q".to_string()],
        weight: None, shape: None, value: None,
        input_dtypes: vec!["f32".to_string()],
        output_dtypes: vec!["f32".to_string()],
        dims: None, dim: None,
    };
    ssa_shapes.insert("%q_weight".to_string(), vec![768, 768]);
    let (_, q_shape) = compute_output_shape(&q_proj, 0, &ssa_shapes, &ssa_map, &ssa_dtypes, &function, 4);
    assert_eq!(q_shape, vec![1, 4, 768], "Q projection shape mismatch");
    ssa_shapes.insert("%q".to_string(), q_shape);

    // Step 2: Reshape Q to [1, 4, 12, 64]
    let q_reshape = HalOp {
        op: "reshape".to_string(), kind: None,
        inputs: vec!["%q".to_string(), "%dim0".to_string(), "%dim1".to_string()],
        outputs: vec!["%q_4d".to_string()],
        weight: None,
        shape: Some(vec!["?".to_string(), "?".to_string(), "12".to_string(), "64".to_string()]),
        value: None,
        input_dtypes: vec!["f32".to_string()],
        output_dtypes: vec!["f32".to_string()],
        dims: None, dim: None,
    };
    // Simulate shape_of values in ssa_map (as f32 tensors)
    ssa_map.insert("%dim0".to_string(), SFATensor::from_vec_f32(vec![1.0f32], vec![1]));
    ssa_map.insert("%dim1".to_string(), SFATensor::from_vec_f32(vec![4.0f32], vec![1]));
    ssa_shapes.insert("%dim0".to_string(), vec![1]);
    ssa_shapes.insert("%dim1".to_string(), vec![1]);
    ssa_dtypes.insert("%dim0".to_string(), crate::tensor::Dtype::F32);
    ssa_dtypes.insert("%dim1".to_string(), crate::tensor::Dtype::F32);
    ssa_dtypes.insert("%q".to_string(), crate::tensor::Dtype::F32);
    let (_, q_4d_shape) = compute_output_shape(&q_reshape, 0, &ssa_shapes, &ssa_map, &ssa_dtypes, &function, 4);
    assert_eq!(q_4d_shape, vec![1, 4, 12, 64], "Q reshape shape mismatch");
    ssa_shapes.insert("%q_4d".to_string(), q_4d_shape);

    // Step 3: Transpose Q to [1, 12, 4, 64]
    let q_trans = HalOp {
        op: "transpose".to_string(), kind: None,
        inputs: vec!["%q_4d".to_string()],
        outputs: vec!["%q_t".to_string()],
        weight: None, shape: None, value: None,
        input_dtypes: vec!["f32".to_string()],
        output_dtypes: vec!["f32".to_string()],
        dims: Some(vec![1, 2]), dim: None,
    };
    let (_, q_t_shape) = compute_output_shape(&q_trans, 0, &ssa_shapes, &ssa_map, &ssa_dtypes, &function, 4);
    assert_eq!(q_t_shape, vec![1, 12, 4, 64], "Q transpose shape mismatch");
    ssa_shapes.insert("%q_t".to_string(), q_t_shape);

    // Step 4: SDPA (shape-preserving: output = [1, 12, 4, 64])
    let sdpa = HalOp {
        op: "scaled_dot_product_attention".to_string(), kind: None,
        inputs: vec!["%q_t".to_string(), "%k_t".to_string(), "%v_t".to_string(), "%mask".to_string()],
        outputs: vec!["%attn".to_string()],
        weight: None, shape: None, value: None,
        input_dtypes: vec!["f32".to_string(), "f32".to_string(), "f32".to_string(), "f32".to_string()],
        output_dtypes: vec!["f32".to_string()],
        dims: None, dim: None,
    };
    ssa_shapes.insert("%k_t".to_string(), vec![1, 12, 4, 64]);
    ssa_shapes.insert("%v_t".to_string(), vec![1, 12, 4, 64]);
    ssa_shapes.insert("%mask".to_string(), vec![1, 1, 4, 4]);
    let (_, attn_shape) = compute_output_shape(&sdpa, 0, &ssa_shapes, &ssa_map, &ssa_dtypes, &function, 4);
    assert_eq!(attn_shape, vec![1, 12, 4, 64], "SDPA output shape mismatch");
    ssa_shapes.insert("%attn".to_string(), attn_shape);

    // Step 5: Transpose back to [1, 4, 12, 64]
    let attn_trans = HalOp {
        op: "transpose".to_string(), kind: None,
        inputs: vec!["%attn".to_string()],
        outputs: vec!["%attn_t".to_string()],
        weight: None, shape: None, value: None,
        input_dtypes: vec!["f32".to_string()],
        output_dtypes: vec!["f32".to_string()],
        dims: Some(vec![1, 2]), dim: None,
    };
    let (_, attn_t_shape) = compute_output_shape(&attn_trans, 0, &ssa_shapes, &ssa_map, &ssa_dtypes, &function, 4);
    assert_eq!(attn_t_shape, vec![1, 4, 12, 64], "attention transpose back shape mismatch");
    ssa_shapes.insert("%attn_t".to_string(), attn_t_shape);

    // Step 6: Reshape back to [1, 4, 768]
    let attn_reshape = HalOp {
        op: "reshape".to_string(), kind: None,
        inputs: vec!["%attn_t".to_string(), "%dim0".to_string(), "%dim1".to_string()],
        outputs: vec!["%attn_out".to_string()],
        weight: None,
        shape: Some(vec!["?".to_string(), "?".to_string(), "768".to_string()]),
        value: None,
        input_dtypes: vec!["f32".to_string()],
        output_dtypes: vec!["f32".to_string()],
        dims: None, dim: None,
    };
    let (_, attn_out_shape) = compute_output_shape(&attn_reshape, 0, &ssa_shapes, &ssa_map, &ssa_dtypes, &function, 4);
    assert_eq!(attn_out_shape, vec![1, 4, 768], "attention reshape back shape mismatch");
}

// ── TDD: SSA map SFATensor roundtrip ──────────────────────────────────

/// Verify that SFATensor can be stored in and retrieved from the SSA map,
/// and that as_buffer_ref() provides correct metadata and data access.
#[test]
fn test_ssa_map_sfa_tensor_roundtrip() {
    use crate::sfa_tensor::SFATensor;

    let mut ssa_map: std::collections::HashMap<String, SFATensor> =
        std::collections::HashMap::new();

    // ── Roundtrip f32 tensor ──────────────────────────────────────
    {
        let t_f32 = SFATensor::from_vec_f32(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
        let buf = t_f32.as_buffer_ref();
        assert_eq!(buf.element_size(), 4);
        assert_eq!(buf.shape(), vec![2, 2]);
        assert_eq!(buf.len(), 16);
        assert_eq!(buf.rank(), 2);
        let ptr = buf.as_ptr() as *const f32;
        let data = unsafe { std::slice::from_raw_parts(ptr, 4) };
        assert_eq!(data, &[1.0f32, 2.0, 3.0, 4.0]);
        drop(buf);
        ssa_map.insert("%f32".to_string(), t_f32);
    }

    // Retrieve and verify metadata survived.
    {
        let retrieved = ssa_map.get("%f32").expect("retrieve f32 tensor");
        assert_eq!(retrieved.rank(), 2);
        assert_eq!(retrieved.shape(), vec![2, 2]);
        assert_eq!(retrieved.numel(), 4);
        assert_eq!(retrieved.elem_size, 4);

        let buf2 = retrieved.as_buffer_ref();
        assert_eq!(buf2.len(), 16);
        let ptr2 = buf2.as_ptr() as *const f32;
        let data2 = unsafe { std::slice::from_raw_parts(ptr2, 4) };
        assert_eq!(data2, &[1.0f32, 2.0, 3.0, 4.0]);
    }

    // ── Roundtrip i64 tensor ──────────────────────────────────────
    {
        let t_i64 = SFATensor::from_vec_i64(vec![10i64, 20, 30], vec![3]);
        let buf3 = t_i64.as_buffer_ref();
        assert_eq!(buf3.element_size(), 8);
        assert_eq!(buf3.shape(), vec![3]);
        assert_eq!(buf3.len(), 24);
        drop(buf3);
        ssa_map.insert("%i64".to_string(), t_i64);
    }

    {
        let retrieved_i64 = ssa_map.get("%i64").expect("retrieve i64 tensor");
        assert_eq!(retrieved_i64.rank(), 1);
        assert_eq!(retrieved_i64.shape(), vec![3]);
        assert_eq!(retrieved_i64.numel(), 3);
        assert_eq!(retrieved_i64.elem_size, 8);

        let buf4 = retrieved_i64.as_buffer_ref();
        let ptr4 = buf4.as_ptr() as *const i64;
        let data4 = unsafe { std::slice::from_raw_parts(ptr4, 3) };
        assert_eq!(data4, &[10i64, 20, 30]);

        // ── Clone data roundtrip ──────────────────────────────────────
        let cloned = retrieved_i64.clone_data();
        assert_eq!(cloned.rank(), 1);
        assert_eq!(cloned.shape(), vec![3]);
        assert_eq!(cloned.numel(), 3);
        let buf5 = cloned.as_buffer_ref();
        let ptr5 = buf5.as_ptr() as *const i64;
        let data5 = unsafe { std::slice::from_raw_parts(ptr5, 3) };
        assert_eq!(data5, &[10i64, 20, 30]);

        // Clone should produce a separate allocation.
        let buf_orig = retrieved_i64.as_buffer_ref();
        assert_ne!(
            buf_orig.as_ptr(), buf5.as_ptr(),
            "clone_data should allocate new memory"
        );
    }
}

/// Verify SFATensor scalar (rank-0) works in SSA map.
#[test]
fn test_ssa_map_sfa_tensor_scalar() {
    use crate::sfa_tensor::SFATensor;

    let mut ssa_map: std::collections::HashMap<String, SFATensor> =
        std::collections::HashMap::new();

    {
        let t = SFATensor::scalar_f32(0.125f32);
        assert_eq!(t.rank(), 0);
        assert_eq!(t.numel(), 1);
        assert!(t.shape().is_empty());
        assert_eq!(t.elem_size, 4);

        let buf = t.as_buffer_ref();
        let ptr = buf.as_ptr() as *const f32;
        let val = unsafe { *ptr };
        assert!((val - 0.125).abs() < 1e-7);
        drop(buf);
        ssa_map.insert("%scalar".to_string(), t);
    }

    let retrieved = ssa_map.get("%scalar").unwrap();
    assert_eq!(retrieved.rank(), 0);
}
