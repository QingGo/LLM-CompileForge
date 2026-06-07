//! GraphRunner trait — common execution interface shared by
//! ComputeGraphRunner (Path A, dylib) and HalRustRunner (Path B, HAL IR).
//!
//! Both runners iterate over a graph of functions/ops, build inputs from
//! global inputs / weights / SSA wires, execute via
//! `executable.execute(op_name, stream, &inputs, &outputs)`,
//! extract outputs, and wire SSA data flow.
//!
//! The trait formalizes the shared interface while each runner keeps its
//! own execution loop and internal data representation.

use std::collections::HashMap;

use crate::hal::sfa::SfaMemRef;
use crate::hal::traits::{Buffer, Executable, Stream};
use crate::model::tensor::{Dtype, Tensor};

// ── GraphRunner trait ──────────────────────────────────────────────────

/// Common interface for graph-based model execution.
///
/// Implementations provide weight loading, input building, output
/// allocation, and SSA wiring — the building blocks used by both
/// the dylib Path A runner and the HAL IR Path B runner.
pub trait GraphRunner {
    /// Specification for building a global input tensor.
    type InputSpec;

    /// Specification for allocating an output buffer.
    type OutputSpec;

    /// Load a weight tensor from the weight provider, converting f16→f32.
    fn load_weight_tensor(&self, name: &str, dtype: Dtype) -> Result<Tensor, anyhow::Error>;

    /// Allocate an output buffer with the given shape and element type.
    fn allocate_output_buffer(
        &self,
        shape: &[usize],
        dtype: Dtype,
    ) -> Result<Box<dyn Buffer>, anyhow::Error>;

    /// Build a global input tensor from input_ids / positions via the
    /// provided spec.
    fn build_global_input(&self, spec: &Self::InputSpec) -> Result<Tensor, anyhow::Error>;

    /// Store a tensor as an SSA output for later retrieval by downstream
    /// graph nodes.
    fn wire_ssa_output(&mut self, name: &str, tensor: Tensor);

    /// Retrieve a previously wired SSA tensor by name.
    fn get_ssa_input(&self, name: &str) -> Option<Tensor>;
}

// ── Shared free functions ──────────────────────────────────────────────

/// Wrap raw buffer data (pointer + shape) into an [`SfaMemRef`] descriptor.
///
/// Both runners use this to construct SFA-compatible descriptors before
/// calling `executable.execute()`. The returned descriptor borrows the
/// caller's data — the caller must ensure the buffer outlives the descriptor.
pub fn wrap_tensor_buffer(
    data: *mut u8,
    shape: &[usize],
    elem_size: usize,
) -> SfaMemRef {
    let ptr = data as *mut std::ffi::c_void;
    SfaMemRef::from_shape(ptr, shape, elem_size).unwrap_or_else(|_| {
        let n = shape.iter().product::<usize>().max(1);
        SfaMemRef::r1(ptr, [n as i64], [1], elem_size)
    })
}

/// Convert a slice of HAL Buffers into SfaMemRef descriptors, execute the
/// named operation, and return the output shapes reported by the executable.
///
/// This is the core dispatch shared by both runners — all kernel calls go
/// through `executable.execute()`.
pub fn build_sfa_and_execute(
    op_name: &str,
    executable: &dyn Executable,
    stream: &dyn Stream,
    input_bufs: &[Box<dyn Buffer>],
    output_bufs: &[Box<dyn Buffer>],
) -> Result<Vec<Vec<i64>>, anyhow::Error> {
    let input_refs: Vec<&dyn Buffer> = input_bufs.iter().map(|b| b.as_ref()).collect();
    let output_refs: Vec<&dyn Buffer> = output_bufs.iter().map(|b| b.as_ref()).collect();

    let input_memrefs: Vec<SfaMemRef> =
        input_refs.iter().map(|b| b.as_sfa_memref()).collect();
    let mut output_memrefs: Vec<SfaMemRef> =
        output_refs.iter().map(|b| b.as_sfa_memref()).collect();

    executable.execute(op_name, stream, &input_memrefs, &mut output_memrefs)
}

/// SSA tensor registry backed by a `HashMap<String, Tensor>`.
///
/// Used by both runner implementations to store and retrieve intermediate
/// tensors that flow between functions / ops.
#[derive(Debug, Default)]
pub struct SsaRegistry {
    store: HashMap<String, Tensor>,
}

impl SsaRegistry {
    pub fn new() -> Self {
        Self {
            store: HashMap::new(),
        }
    }

    /// Store a tensor with the given SSA name.
    pub fn insert(&mut self, name: &str, tensor: Tensor) {
        self.store.insert(name.to_string(), tensor);
    }

    /// Retrieve a tensor by SSA name.
    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.store.get(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wrap_tensor_buffer_rank1() {
        let data = vec![1.0f32, 2.0, 3.0];
        let sfa = wrap_tensor_buffer(
            data.as_ptr() as *mut u8,
            &[3],
            4, /* f32 */
        );
        assert_eq!(sfa.rank(), 1);
        assert_eq!(sfa.sizes(), vec![3]);
        assert_eq!(sfa.element_size(), 4);
    }

    #[test]
    fn test_wrap_tensor_buffer_rank2() {
        let data = vec![0.0f32; 6];
        let sfa = wrap_tensor_buffer(
            data.as_ptr() as *mut u8,
            &[2, 3],
            4, /* f32 */
        );
        assert_eq!(sfa.rank(), 2);
        assert_eq!(sfa.sizes(), vec![2, 3]);
        assert_eq!(sfa.element_size(), 4);
    }

    #[test]
    fn test_ssa_registry_insert_and_get() {
        let mut reg = SsaRegistry::new();
        let t = Tensor::new_owned(vec![4], vec![1.0, 2.0, 3.0, 4.0], Dtype::F32);
        reg.insert("%0", t);
        let retrieved = reg.get("%0").expect("should find %0");
        assert_eq!(retrieved.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
        assert!(reg.get("%1").is_none());
    }

    #[test]
    fn test_build_sfa_and_execute_mock() {
        use crate::hal::cpu::buffer::RawBuffer as InnerCpuBuffer;
        use crate::hal::cpu::CpuBuffer;
        use crate::hal::cpu::CpuStream;
        use std::sync::Mutex;

        #[derive(Debug)]
        struct MockExec {
            calls: Mutex<Vec<String>>,
        }
        impl Executable for MockExec {
            fn execute(
                &self,
                op_name: &str,
                _stream: &dyn Stream,
                _inputs: &[SfaMemRef],
                outputs: &mut [SfaMemRef],
            ) -> Result<Vec<Vec<i64>>, anyhow::Error> {
                self.calls.lock().unwrap().push(op_name.to_string());
                Ok(vec![vec![0i64; 0]; outputs.len()])
            }
            fn function_count(&self) -> usize {
                1
            }
        }

        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let raw = InnerCpuBuffer::from_raw_parts(
            data.as_ptr() as *mut u8,
            data.len() * 4,
            true, // borrowed
        )
        .unwrap();
        let buf: Box<dyn Buffer> =
            Box::new(CpuBuffer::with_meta(raw, 4, vec![4]));

        let mut out_data: Vec<f32> = vec![0.0; 4];
        let raw_out = InnerCpuBuffer::from_raw_parts(
            out_data.as_mut_ptr() as *mut u8,
            out_data.len() * 4,
            true,
        )
        .unwrap();
        let out_buf: Box<dyn Buffer> =
            Box::new(CpuBuffer::with_meta(raw_out, 4, vec![4]));

        let mock = MockExec {
            calls: Mutex::new(Vec::new()),
        };
        let result = build_sfa_and_execute(
            "test_op",
            &mock,
            &CpuStream,
            &[buf],
            &[out_buf],
        );
        assert!(result.is_ok());
        assert_eq!(*mock.calls.lock().unwrap(), vec!["test_op"]);
    }
}
