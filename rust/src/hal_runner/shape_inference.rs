//! Output shape inference for HAL IR ops.
//!
//! Determines the output tensor shape (numel + dimensions) for each HAL op
//! before execution, so output buffers can be pre-allocated.

use std::collections::HashMap;

use crate::sfa_tensor::SFATensor;
use crate::hal_runner::types::HalFunction;
use crate::hal_runner::types::HalOp;

/// Compute the output shape for a HAL op.
///
/// Returns `(numel, shape)` where `numel` is the total number of elements
/// and `shape` is a `Vec<usize>` of dimension sizes.
pub(crate) fn compute_output_shape(
    op: &HalOp,
    out_idx: usize,
    ssa_shapes: &HashMap<String, Vec<usize>>,
    ssa_map: &HashMap<String, SFATensor>,
    ssa_dtypes: &HashMap<String, crate::tensor::Dtype>,
    function: &HalFunction,
    _seq_len: usize,
) -> (usize, Vec<usize>) {
    let result = match op.op.as_str() {
        "shape_of" => {
            if op.dim.is_some() {
                (1, vec![1])
            } else {
                let rank = op
                    .inputs
                    .first()
                    .and_then(|n| ssa_shapes.get(n))
                    .map(|s| s.len())
                    .unwrap_or(2);
                (rank, vec![rank])
            }
        }
        "reshape" => shape_of_reshape(op, out_idx, ssa_shapes, ssa_map, ssa_dtypes, function),
        "gather" => shape_of_gather(op, ssa_shapes),
        "matmul" => shape_of_matmul(op, ssa_shapes),
        "fill" => shape_of_fill(op, ssa_shapes),
        "transpose" => shape_of_transpose(op, ssa_shapes),
        "linear" => shape_of_linear(op, ssa_shapes),
        "element_wise" | "elementwise" | "softmax" | "unsqueeze"
        | "slice" | "compare" | "layer_norm"
        | "scaled_dot_product_attention" | "scan" => shape_preserving(op, ssa_shapes),
        "reduce" => shape_of_reduce(op, ssa_shapes),
        "concat" => shape_of_concat(op, ssa_shapes, ssa_map),
        _ => {
            let shape = function
                .outputs
                .get(out_idx)
                .map(|o| {
                    o.shape
                        .iter()
                        .map(|d| {
                            if d == "?" || d == "-1" {
                                1
                            } else {
                                d.parse::<usize>().unwrap_or(1)
                            }
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_else(|| vec![1]);
            let numel = shape.iter().product::<usize>().max(1);
            (numel, shape)
        }
    };
    result
}

fn shape_of_reshape(
    op: &HalOp,
    _out_idx: usize,
    ssa_shapes: &HashMap<String, Vec<usize>>,
    ssa_map: &HashMap<String, SFATensor>,
    ssa_dtypes: &HashMap<String, crate::tensor::Dtype>,
    _function: &HalFunction,
) -> (usize, Vec<usize>) {
    let input_numel = op
        .inputs
        .first()
        .and_then(|n| ssa_map.get(n))
        .map(|t| t.numel())
        .unwrap_or(1);
    let input_shape = op.inputs.first()
        .and_then(|n| ssa_shapes.get(n))
        .cloned()
        .unwrap_or_else(|| vec![input_numel]);

    let shape_of_vals: Vec<usize> = op.inputs[1..]
        .iter()
        .filter_map(|n| {
            ssa_map.get(n).map(|t| {
                if t.numel() > 0 && t.elem_size == 4 {
                    t.read_f32(0) as usize
                } else {
                    1
                }
            })
        })
        .collect();

    let target_shape = if let Some(shape_strs) = &op.shape {
        let mut so_iter = shape_of_vals.iter();
        let mut result: Vec<usize> = shape_strs.iter().map(|d| {
            if d == "?" || d == "-1" {
                so_iter.next().copied().unwrap_or(0)
            } else {
                d.parse::<usize>().unwrap_or(1)
            }
        }).collect();
        let known: usize = result.iter().filter(|&&d| d > 0).product();
        let zeros = result.iter().filter(|&&d| d == 0).count();
        if zeros == 1 && known > 0 {
            for d in result.iter_mut() {
                if *d == 0 { *d = input_numel / known; }
            }
        } else if zeros > 1 {
            for (i, d) in result.iter_mut().enumerate() {
                if *d == 0 {
                    *d = input_shape.get(i).copied().unwrap_or(1);
                }
            }
        }
        result
    } else {
        input_shape.clone()
    };
    (input_numel, target_shape)
}

fn shape_of_gather(
    op: &HalOp,
    ssa_shapes: &HashMap<String, Vec<usize>>,
) -> (usize, Vec<usize>) {
    if op.inputs.len() >= 3 {
        let data_shape = op.inputs.first()
            .and_then(|n| ssa_shapes.get(n))
            .cloned()
            .unwrap_or_else(|| vec![1]);
        let rank = data_shape.len();
        let mid_shape: Vec<usize> = if rank >= 2 {
            data_shape[1..rank-1].to_vec()
        } else {
            vec![1]
        };
        let numel = mid_shape.iter().product::<usize>().max(1);
        (numel, mid_shape)
    } else {
        let embed_dim = op.inputs.get(0)
            .and_then(|n| ssa_shapes.get(n))
            .map(|s| s.iter().skip(1).product::<usize>())
            .unwrap_or(768);
        let input_shape = op.inputs.get(1)
            .and_then(|n| ssa_shapes.get(n))
            .cloned()
            .unwrap_or_else(|| vec![1]);
        let mut output_shape = input_shape;
        output_shape.push(embed_dim);
        let numel = output_shape.iter().product::<usize>().max(1);
        (numel, output_shape)
    }
}

fn shape_of_matmul(
    op: &HalOp,
    ssa_shapes: &HashMap<String, Vec<usize>>,
) -> (usize, Vec<usize>) {
    let a_shape = op
        .inputs
        .get(0)
        .and_then(|n| ssa_shapes.get(n))
        .cloned()
        .unwrap_or_else(|| vec![1, 1]);
    let b_shape = op
        .inputs
        .get(1)
        .and_then(|n| ssa_shapes.get(n))
        .cloned()
        .unwrap_or_else(|| vec![1, 1]);
    // matmul_blas with transpose_b=true uses b_shape[-2] as output N
    // when the K dimensions match (k_b == k). Otherwise b_shape[-1] is used.
    // Match the logic in matmul_blas: if b_shape[-1] != a_shape[-1],
    // then n = b_shape[-1]; otherwise n = b_shape[-2].
    let k = a_shape.last().copied().unwrap_or(1);
    let k_b = b_shape.last().copied().unwrap_or(1);
    let n = if k_b != k {
        b_shape.last().copied().unwrap_or(1)
    } else {
        b_shape.get(b_shape.len().saturating_sub(2)).copied().unwrap_or(1)
    };
    let mut output_shape: Vec<usize> = a_shape[..a_shape.len().saturating_sub(1)].to_vec();
    output_shape.push(n);
    let numel = output_shape.iter().product::<usize>().max(1);
    (numel, output_shape)
}

fn shape_of_fill(
    op: &HalOp,
    ssa_shapes: &HashMap<String, Vec<usize>>,
) -> (usize, Vec<usize>) {
    let shape = op.inputs.first()
        .and_then(|n| ssa_shapes.get(n))
        .cloned()
        .unwrap_or_else(|| vec![1]);
    let numel = shape.iter().product::<usize>().max(1);
    (numel, shape)
}

fn shape_of_transpose(
    op: &HalOp,
    ssa_shapes: &HashMap<String, Vec<usize>>,
) -> (usize, Vec<usize>) {
    let input_shape = op
        .inputs
        .first()
        .and_then(|n| ssa_shapes.get(n))
        .cloned()
        .unwrap_or_else(|| vec![1]);
    let output_shape = if let Some(ref dims) = op.dims {
        let rank = input_shape.len();
        let mut perm: Vec<usize> = (0..rank).collect();
        for pair in dims.chunks(2) {
            if pair.len() == 2 && pair[0] < rank && pair[1] < rank {
                perm.swap(pair[0], pair[1]);
            }
        }
        perm.iter().map(|&d| input_shape[d]).collect()
    } else {
        input_shape.clone()
    };
    let numel = output_shape.iter().product::<usize>().max(1);
    (numel, output_shape)
}

fn shape_of_linear(
    op: &HalOp,
    ssa_shapes: &HashMap<String, Vec<usize>>,
) -> (usize, Vec<usize>) {
    let weight_shape = op.inputs.get(1)
        .and_then(|n| ssa_shapes.get(n))
        .cloned()
        .unwrap_or_else(|| vec![1, 1]);
    let out_features = weight_shape.first().copied().unwrap_or(768);
    let input_shape = op.inputs.first()
        .and_then(|n| ssa_shapes.get(n))
        .cloned()
        .unwrap_or_else(|| vec![1]);
    let mut output_shape: Vec<usize> = input_shape[..input_shape.len().saturating_sub(1)].to_vec();
    output_shape.push(out_features);
    let numel = output_shape.iter().product::<usize>().max(1);
    (numel, output_shape)
}

fn shape_preserving(
    op: &HalOp,
    ssa_shapes: &HashMap<String, Vec<usize>>,
) -> (usize, Vec<usize>) {
    let shape = op
        .inputs
        .first()
        .and_then(|n| ssa_shapes.get(n))
        .cloned()
        .unwrap_or_else(|| vec![1]);
    let numel = shape.iter().product::<usize>().max(1);
    (numel, shape)
}

fn shape_of_reduce(
    op: &HalOp,
    ssa_shapes: &HashMap<String, Vec<usize>>,
) -> (usize, Vec<usize>) {
    let shape = op
        .inputs
        .first()
        .and_then(|n| ssa_shapes.get(n))
        .cloned()
        .unwrap_or_else(|| vec![1]);
    let reduced: Vec<usize> = if shape.len() > 1 {
        shape[..shape.len() - 1].to_vec()
    } else {
        vec![1]
    };
    let numel = reduced.iter().product::<usize>().max(1);
    (numel, reduced)
}

fn shape_of_concat(
    op: &HalOp,
    ssa_shapes: &HashMap<String, Vec<usize>>,
    ssa_map: &HashMap<String, SFATensor>,
) -> (usize, Vec<usize>) {
    let total_numel: usize = op
        .inputs
        .iter()
        .filter_map(|n| ssa_map.get(n))
        .map(|t| t.numel())
        .sum();
    let shape = ssa_shapes
        .get(&op.inputs[0])
        .cloned()
        .unwrap_or_else(|| vec![total_numel]);
    (total_numel.max(1), shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hal_runner::types::HalFunction;

    fn make_shape_preserving_test_case(op_name: &str, input_shape: Vec<usize>) -> (usize, Vec<usize>) {
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
        let function = HalFunction {
            name: "test".to_string(),
            layer: 0,
            inputs: vec![],
            outputs: vec![],
            weights: vec![],
            weight_inputs: std::collections::HashMap::new(),
            ops: vec![],
        };
        let ssa_map: HashMap<String, SFATensor> = HashMap::new();
        let ssa_dtypes: HashMap<String, crate::tensor::Dtype> = HashMap::new();
        compute_output_shape(&op, 0, &ssa_shapes, &ssa_map, &ssa_dtypes, &function, 4)
    }

    #[test]
    fn test_layer_norm_output_shape() {
        let (numel, shape) = make_shape_preserving_test_case("layer_norm", vec![4, 768]);
        assert_eq!(shape, vec![4, 768]);
        assert_eq!(numel, 3072);
    }

    #[test]
    fn test_sdpa_output_shape() {
        let (numel, shape) = make_shape_preserving_test_case("scaled_dot_product_attention", vec![1, 1, 768]);
        assert_eq!(shape, vec![1, 1, 768]);
        assert_eq!(numel, 768);
    }

    #[test]
    fn test_linear_output_shape_lm_head() {
        let op = HalOp {
            op: "linear".to_string(),
            kind: None,
            inputs: vec!["%inp".to_string(), "%weight".to_string()],
            outputs: vec!["%out".to_string()],
            weight: None, shape: None, value: None,
            input_dtypes: vec!["f32".to_string()],
            output_dtypes: vec!["f32".to_string()],
            dims: None, dim: None,
        };
        let mut ssa_shapes: HashMap<String, Vec<usize>> = HashMap::new();
        ssa_shapes.insert("%inp".to_string(), vec![1, 1, 768]);
        ssa_shapes.insert("%weight".to_string(), vec![50272, 768]);
        let function = HalFunction {
            name: "test".to_string(), layer: 0, inputs: vec![], outputs: vec![],
            weights: vec![], weight_inputs: std::collections::HashMap::new(), ops: vec![],
        };
        let ssa_map: HashMap<String, SFATensor> = HashMap::new();
        let ssa_dtypes: HashMap<String, crate::tensor::Dtype> = HashMap::new();
        let (numel, shape) = compute_output_shape(&op, 0, &ssa_shapes, &ssa_map, &ssa_dtypes, &function, 4);
        assert_eq!(shape, vec![1, 1, 50272]);
        assert_eq!(numel, 50272);
    }

    #[test]
    fn test_linear_output_shape_ffn_fc2() {
        let op = HalOp {
            op: "linear".to_string(), kind: None,
            inputs: vec!["%inp".to_string(), "%weight".to_string()],
            outputs: vec!["%out".to_string()],
            weight: None, shape: None, value: None,
            input_dtypes: vec!["f32".to_string()],
            output_dtypes: vec!["f32".to_string()],
            dims: None, dim: None,
        };
        let mut ssa_shapes: HashMap<String, Vec<usize>> = HashMap::new();
        ssa_shapes.insert("%inp".to_string(), vec![4, 3072]);
        ssa_shapes.insert("%weight".to_string(), vec![768, 3072]);
        let function = HalFunction {
            name: "test".to_string(), layer: 0, inputs: vec![], outputs: vec![],
            weights: vec![], weight_inputs: std::collections::HashMap::new(), ops: vec![],
        };
        let ssa_map: HashMap<String, SFATensor> = HashMap::new();
        let ssa_dtypes: HashMap<String, crate::tensor::Dtype> = HashMap::new();
        let (numel, shape) = compute_output_shape(&op, 0, &ssa_shapes, &ssa_map, &ssa_dtypes, &function, 4);
        assert_eq!(shape, vec![4, 768]);
        assert_eq!(numel, 3072);
    }
}
