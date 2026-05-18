//! Compute graph — the execution plan for a compiled model.
//!
//! Describes which ``_mlir_ciface_*`` functions to call in which order,
//! how to wire tensor values between them, and which inputs are weights
//! from the embedded SFCF registry versus SSA values produced by prior
//! functions.

use crate::sfcf;
use crate::tensor::Dtype;

// ── Input binding ──────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum InputBinding {
    Weight(String),
    Ssa { producer_func: usize, output_idx: usize },
    GlobalInput,
}

// ── I/O tensor descriptor ──────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct IOTensorDef {
    pub rank: u8,
    pub shape: Vec<u64>,
}

impl IOTensorDef {
    pub fn numel(&self) -> usize {
        self.shape.iter().filter(|&&d| d > 0).product::<u64>() as usize
    }
}

// ── Function definition ────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct FuncDef {
    pub index: usize,
    pub symbol: String,
    pub num_inputs: usize,
    pub num_outputs: usize,
    pub inputs: Vec<(InputBinding, IOTensorDef)>,
    pub outputs: Vec<IOTensorDef>,
}

impl FuncDef {
    /// Number of C ABI arguments for the ciface wrapper:
    /// 1 sret pointer + num_inputs descriptor pointers.
    pub fn total_args(&self) -> usize {
        1 + self.num_inputs
    }
}

// ── Compute graph ──────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ComputeGraph {
    pub functions: Vec<FuncDef>,
    pub global_input: (usize, usize),
    pub global_output: (usize, usize),
}

impl ComputeGraph {
    pub fn parse(data: &[u8], pos: &mut usize) -> Result<Self, anyhow::Error> {
        let num_funcs = sfcf::read_u32(data, pos)? as usize;
        let mut functions = Vec::with_capacity(num_funcs);

        for fi in 0..num_funcs {
            let symbol = sfcf::read_string(data, pos)?;
            let num_inputs = sfcf::read_u32(data, pos)? as usize;
            let num_outputs = sfcf::read_u32(data, pos)? as usize;

            let mut inputs: Vec<(InputBinding, IOTensorDef)> =
                Vec::with_capacity(num_inputs);
            for _ in 0..num_inputs {
                let binding_type = sfcf::read_u8(data, pos)?;
                let binding = match binding_type {
                    0 => {
                        let key = sfcf::read_string(data, pos)?;
                        InputBinding::Weight(key)
                    }
                    1 => {
                        let pf = sfcf::read_u32(data, pos)? as usize;
                        let oi = sfcf::read_u32(data, pos)? as usize;
                        InputBinding::Ssa {
                            producer_func: pf,
                            output_idx: oi,
                        }
                    }
                    2 => InputBinding::GlobalInput,
                    _ => anyhow::bail!(
                        "unknown binding type {} at func {}",
                        binding_type,
                        fi
                    ),
                };
                let rank = sfcf::read_u8(data, pos)?;
                let num_dims = sfcf::read_u32(data, pos)? as usize;
                let mut shape = Vec::with_capacity(num_dims);
                for _ in 0..num_dims {
                    shape.push(sfcf::read_u64(data, pos)?);
                }
                inputs.push((binding, IOTensorDef { rank, shape }));
            }

            let mut outputs = Vec::with_capacity(num_outputs);
            for _ in 0..num_outputs {
                let rank = sfcf::read_u8(data, pos)?;
                let num_dims = sfcf::read_u32(data, pos)? as usize;
                let mut shape = Vec::with_capacity(num_dims);
                for _ in 0..num_dims {
                    shape.push(sfcf::read_u64(data, pos)?);
                }
                outputs.push(IOTensorDef { rank, shape });
            }

            functions.push(FuncDef {
                index: fi,
                symbol,
                num_inputs,
                num_outputs,
                inputs,
                outputs,
            });
        }

        let global_input = (
            sfcf::read_u32(data, pos)? as usize,
            sfcf::read_u32(data, pos)? as usize,
        );
        let global_output = (
            sfcf::read_u32(data, pos)? as usize,
            sfcf::read_u32(data, pos)? as usize,
        );
        Ok(Self {
            functions,
            global_input,
            global_output,
        })
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> Vec<u8> {
        let mut buf = Vec::new();
        // Header
        buf.extend_from_slice(b"SFCF");
        buf.extend_from_slice(&2u32.to_le_bytes()); // version
        buf.extend_from_slice(&0u32.to_le_bytes()); // 0 name mappings
        buf.extend_from_slice(&0u32.to_le_bytes()); // 0 constants
        // Compute graph section
        buf.extend_from_slice(&2u32.to_le_bytes()); // 2 functions

        // Function 0: _mlir_ciface_main_0, 2 inputs, 1 output
        {
            let s = b"_mlir_ciface_main_0";
            buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
            buf.extend_from_slice(s);
            buf.extend_from_slice(&2u32.to_le_bytes()); // num_inputs
            buf.extend_from_slice(&1u32.to_le_bytes()); // num_outputs

            // Input 0: global_input, rank 2, shape [1, 0]
            buf.push(2u8); // global_input
            buf.push(2u8); // rank
            buf.extend_from_slice(&2u32.to_le_bytes()); // 2 dims
            buf.extend_from_slice(&1u64.to_le_bytes());
            buf.extend_from_slice(&0u64.to_le_bytes());

            // Input 1: weight "w", rank 2, shape [4096, 4096]
            buf.push(0u8); // weight
            let key = b"w";
            buf.extend_from_slice(&(key.len() as u32).to_le_bytes());
            buf.extend_from_slice(key);
            buf.push(2u8); // rank
            buf.extend_from_slice(&2u32.to_le_bytes());
            buf.extend_from_slice(&4096u64.to_le_bytes());
            buf.extend_from_slice(&4096u64.to_le_bytes());

            // Output 0: rank 2, shape [1, 0]
            buf.push(2u8); // rank
            buf.extend_from_slice(&2u32.to_le_bytes());
            buf.extend_from_slice(&1u64.to_le_bytes());
            buf.extend_from_slice(&0u64.to_le_bytes());
        }

        // Function 1: _mlir_ciface_main_1, 1 input, 1 output
        {
            let s = b"_mlir_ciface_main_1";
            buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
            buf.extend_from_slice(s);
            buf.extend_from_slice(&1u32.to_le_bytes()); // num_inputs
            buf.extend_from_slice(&1u32.to_le_bytes()); // num_outputs

            // Input 0: ssa from func 0 output 0, rank 2, shape [1, 0]
            buf.push(1u8); // ssa
            buf.extend_from_slice(&0u32.to_le_bytes()); // producer func 0
            buf.extend_from_slice(&0u32.to_le_bytes()); // output idx 0
            buf.push(2u8); // rank
            buf.extend_from_slice(&2u32.to_le_bytes());
            buf.extend_from_slice(&1u64.to_le_bytes());
            buf.extend_from_slice(&0u64.to_le_bytes());

            // Output 0: rank 3, shape [1, 0, 152064]
            buf.push(3u8); // rank
            buf.extend_from_slice(&3u32.to_le_bytes());
            buf.extend_from_slice(&1u64.to_le_bytes());
            buf.extend_from_slice(&0u64.to_le_bytes());
            buf.extend_from_slice(&152064u64.to_le_bytes());
        }

        // Global I/O
        buf.extend_from_slice(&0u32.to_le_bytes()); // global_input_func
        buf.extend_from_slice(&0u32.to_le_bytes()); // global_input_arg
        buf.extend_from_slice(&1u32.to_le_bytes()); // global_output_func
        buf.extend_from_slice(&0u32.to_le_bytes()); // global_output_idx

        buf
    }

    #[test]
    fn test_parse_compute_graph() {
        let data = make_test_data();
        // Skip SFCF header + name_mapping (0 entries) + constants (0 entries)
        let mut pos = 8 + 4 + 4; // magic(4) + version(4) + nm_count(4) + const_count(4)

        let graph = ComputeGraph::parse(&data, &mut pos).unwrap();
        assert_eq!(graph.functions.len(), 2);

        assert_eq!(graph.functions[0].symbol, "_mlir_ciface_main_0");
        assert_eq!(graph.functions[0].num_inputs, 2);
        assert_eq!(graph.functions[0].num_outputs, 1);
        assert!(matches!(
            graph.functions[0].inputs[0].0,
            InputBinding::GlobalInput
        ));
        assert!(matches!(
            graph.functions[0].inputs[1].0,
            InputBinding::Weight(_)
        ));

        assert_eq!(graph.functions[1].symbol, "_mlir_ciface_main_1");
        assert!(matches!(
            graph.functions[1].inputs[0].0,
            InputBinding::Ssa {
                producer_func: 0,
                output_idx: 0
            }
        ));

        assert_eq!(graph.global_input, (0, 0));
        assert_eq!(graph.global_output, (1, 0));
    }
}
