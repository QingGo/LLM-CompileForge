//! Compute graph — the execution plan for a compiled model.
//!
//! Describes which ``_mlir_ciface_*`` functions to call in which order,
//! how to wire tensor values between them, and which inputs are weights
//! from the embedded SFCF registry versus SSA values produced by prior
//! functions.

use crate::sfcf;

// ── Input binding ──────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum InputBinding {
    #[allow(dead_code)]
    Weight(String),
    Ssa { producer_func: usize, output_idx: usize },
    GlobalInput,
}

// ── I/O tensor descriptor ──────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct IOTensorDef {
    pub rank: u8,
    pub shape: Vec<u64>,
    pub consumed_internally: bool,
}

impl IOTensorDef {
    #[allow(dead_code)]
    pub fn numel(&self) -> usize {
        self.shape.iter().filter(|&&d| d > 0).product::<u64>() as usize
    }

    /// Create an IOTensorDef with consumed_internally (used for outputs).
    #[allow(dead_code)]
    pub fn new(rank: u8, shape: Vec<u64>, consumed_internally: bool) -> Self {
        Self { rank, shape, consumed_internally }
    }
}

// ── Function definition ────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct FuncDef {
    #[allow(dead_code)]
    pub index: usize,
    pub symbol: String,
    pub num_inputs: usize,
    #[allow(dead_code)]
    pub num_outputs: usize,
    pub inputs: Vec<(InputBinding, IOTensorDef)>,
    pub outputs: Vec<IOTensorDef>,
}

impl FuncDef {
    /// Number of C ABI arguments for the ciface wrapper:
    /// 1 sret pointer + num_inputs descriptor pointers.
    #[allow(dead_code)]
    pub fn total_args(&self) -> usize {
        1 + self.num_inputs
    }
}

// ── Compute graph ──────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ComputeGraph {
    pub functions: Vec<FuncDef>,
    #[allow(dead_code)]
    pub global_input: (usize, usize),
    pub global_output: (usize, usize),
}

impl ComputeGraph {
    pub fn parse(data: &[u8], pos: &mut usize, sfcf_version: u32) -> Result<Self, anyhow::Error> {
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
                inputs.push((binding, IOTensorDef { rank, shape, consumed_internally: false }));
            }

            let mut outputs = Vec::with_capacity(num_outputs);
            for _ in 0..num_outputs {
                let consumed_internally = if sfcf_version >= 3 {
                    sfcf::read_u8(data, pos)? != 0
                } else {
                    false
                };
                let rank = sfcf::read_u8(data, pos)?;
                let num_dims = sfcf::read_u32(data, pos)? as usize;
                let mut shape = Vec::with_capacity(num_dims);
                for _ in 0..num_dims {
                    shape.push(sfcf::read_u64(data, pos)?);
                }
                outputs.push(IOTensorDef { rank, shape, consumed_internally });
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

            // Output 0: consumed_internally=0, rank 2, shape [1, 0]
            buf.push(0u8); // consumed_internally
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

            // Output 0: consumed_internally=0, rank 3, shape [1, 0, 152064]
            buf.push(0u8); // consumed_internally
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

    fn make_test_data_v2() -> Vec<u8> {
        // Build test binary in SFCF v2 format (no consumed_internally bytes per output).
        let mut buf = Vec::new();
        // SFCF header
        buf.extend_from_slice(b"SFCF");
        buf.extend_from_slice(&2u32.to_le_bytes()); // version 2
        // Name mapping (0 entries)
        buf.extend_from_slice(&0u32.to_le_bytes());
        // Constants (0 entries)
        buf.extend_from_slice(&0u32.to_le_bytes());
        // Compute graph section: 2 functions
        buf.extend_from_slice(&2u32.to_le_bytes());

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
            buf.extend_from_slice(&2u32.to_le_bytes());
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

            // Output 0: rank 2, shape [1, 0]  (v2 — no consumed_internally byte)
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
            buf.extend_from_slice(&0u32.to_le_bytes());
            buf.extend_from_slice(&0u32.to_le_bytes());
            buf.push(2u8); // rank
            buf.extend_from_slice(&2u32.to_le_bytes());
            buf.extend_from_slice(&1u64.to_le_bytes());
            buf.extend_from_slice(&0u64.to_le_bytes());

            // Output 0: rank 3, shape [1, 0, 152064] (v2 — no consumed_internally byte)
            buf.push(3u8); // rank
            buf.extend_from_slice(&3u32.to_le_bytes());
            buf.extend_from_slice(&1u64.to_le_bytes());
            buf.extend_from_slice(&0u64.to_le_bytes());
            buf.extend_from_slice(&152064u64.to_le_bytes());
        }

        // Global I/O
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());

        buf
    }

    fn make_test_data_v3_with_kv() -> Vec<u8> {
        // Build test binary in SFCF v3 format where some outputs have
        // consumed_internally=true.  Simulates a split KV-head function:
        //   output[0]=logits(False), output[1]=K(True), output[2]=V(True).
        let mut buf = Vec::new();
        // SFCF header
        buf.extend_from_slice(b"SFCF");
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&0u32.to_le_bytes()); // 0 name mappings
        buf.extend_from_slice(&0u32.to_le_bytes()); // 0 constants
        // Compute graph section: 1 function with 3 outputs
        buf.extend_from_slice(&1u32.to_le_bytes()); // 1 function

        // Function 0: _mlir_ciface_main_0, 2 inputs, 3 outputs
        {
            let s = b"_mlir_ciface_main_0";
            buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
            buf.extend_from_slice(s);
            buf.extend_from_slice(&2u32.to_le_bytes()); // num_inputs
            buf.extend_from_slice(&3u32.to_le_bytes()); // num_outputs

            // Input 0: global_input, rank 2, shape [1, 0]
            buf.push(2u8); // global_input
            buf.push(2u8); // rank
            buf.extend_from_slice(&2u32.to_le_bytes());
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

            // Output 0: logits — consumed_internally=false, rank 2, shape [1, 0]
            buf.push(0u8); // consumed_internally = false
            buf.push(2u8); // rank
            buf.extend_from_slice(&2u32.to_le_bytes());
            buf.extend_from_slice(&1u64.to_le_bytes());
            buf.extend_from_slice(&0u64.to_le_bytes());

            // Output 1: K cache — consumed_internally=true, rank 3, shape [1, 32, 128]
            buf.push(1u8); // consumed_internally = true
            buf.push(3u8); // rank
            buf.extend_from_slice(&3u32.to_le_bytes());
            buf.extend_from_slice(&1u64.to_le_bytes());
            buf.extend_from_slice(&32u64.to_le_bytes());
            buf.extend_from_slice(&128u64.to_le_bytes());

            // Output 2: V cache — consumed_internally=true, rank 3, shape [1, 32, 128]
            buf.push(1u8); // consumed_internally = true
            buf.push(3u8); // rank
            buf.extend_from_slice(&3u32.to_le_bytes());
            buf.extend_from_slice(&1u64.to_le_bytes());
            buf.extend_from_slice(&32u64.to_le_bytes());
            buf.extend_from_slice(&128u64.to_le_bytes());
        }

        // Global I/O
        buf.extend_from_slice(&0u32.to_le_bytes()); // global_input_func
        buf.extend_from_slice(&0u32.to_le_bytes()); // global_input_arg
        buf.extend_from_slice(&0u32.to_le_bytes()); // global_output_func
        buf.extend_from_slice(&0u32.to_le_bytes()); // global_output_idx

        buf
    }

    #[test]
    fn test_parse_compute_graph_v2_backward_compat() {
        let data = make_test_data_v2();
        let mut pos = 8 + 4 + 4;

        let graph = ComputeGraph::parse(&data, &mut pos, 2).unwrap();
        assert_eq!(graph.functions.len(), 2);

        // All outputs in v2 should default to consumed_internally = false
        for fi in 0..graph.functions.len() {
            for oi in 0..graph.functions[fi].num_outputs {
                let output = &graph.functions[fi].outputs[oi];
                assert!(
                    !output.consumed_internally,
                    "func[{}] output[{}]: expected consumed_internally=false for v2",
                    fi, oi
                );
            }
        }

        // Verify shapes still parse correctly
        assert_eq!(graph.functions[0].outputs[0].rank, 2);
        assert_eq!(graph.functions[0].outputs[0].shape, vec![1, 0]);
        assert_eq!(graph.functions[1].outputs[0].rank, 3);
        assert_eq!(graph.functions[1].outputs[0].shape, vec![1, 0, 152064]);

        assert_eq!(graph.global_input, (0, 0));
        assert_eq!(graph.global_output, (1, 0));
    }

    #[test]
    fn test_parse_consumed_internally() {
        let data = make_test_data_v3_with_kv();
        // Skip SFCF header + name_mapping (0 entries) + constants (0 entries)
        let mut pos = 8 + 4 + 4; // magic(4) + version(4) + nm_count(4) + const_count(4)

        let graph = ComputeGraph::parse(&data, &mut pos, 3).unwrap();
        assert_eq!(graph.functions.len(), 1);

        // Output 0: consumed_internally=false (logits)
        assert!(
            !graph.functions[0].outputs[0].consumed_internally,
            "output[0] (logits): expected consumed_internally=false"
        );
        assert_eq!(graph.functions[0].outputs[0].rank, 2);
        assert_eq!(graph.functions[0].outputs[0].shape, vec![1, 0]);

        // Output 1: consumed_internally=true (K cache)
        assert!(
            graph.functions[0].outputs[1].consumed_internally,
            "output[1] (K): expected consumed_internally=true"
        );
        assert_eq!(graph.functions[0].outputs[1].rank, 3);
        assert_eq!(graph.functions[0].outputs[1].shape, vec![1, 32, 128]);

        // Output 2: consumed_internally=true (V cache)
        assert!(
            graph.functions[0].outputs[2].consumed_internally,
            "output[2] (V): expected consumed_internally=true"
        );
        assert_eq!(graph.functions[0].outputs[2].rank, 3);
        assert_eq!(graph.functions[0].outputs[2].shape, vec![1, 32, 128]);

        assert_eq!(graph.global_input, (0, 0));
        assert_eq!(graph.global_output, (0, 0));
    }

    #[test]
    fn test_parse_compute_graph() {
        let data = make_test_data();
        // Skip SFCF header + name_mapping (0 entries) + constants (0 entries)
        let mut pos = 8 + 4 + 4; // magic(4) + version(4) + nm_count(4) + const_count(4)

        let graph = ComputeGraph::parse(&data, &mut pos, 3).unwrap();
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
