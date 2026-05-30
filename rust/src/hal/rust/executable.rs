//! HalRustExecutable — pure-Rust HAL backend dispatch.
//!
//! Implements ``traits::Executable`` by dispatching to the generated
//! ``*_cpu`` functions from ``hal_ops_cpu.rs`` (emitted by EmitRust).
//!
//! Dispatch logic lives in the sibling ``dispatch`` module — each
//! ``dispatch_*`` free function handles one HAL operation type.
//!
//! Each input/output Buffer is converted to a ``&[f32]``/``&mut [f32]``
//! slice, and an ``OpShapeMeta`` is constructed from the buffer shapes.
//! The generated ``*_cpu`` function is then called inline.

use crate::hal::traits;

// ── HalRustExecutable ─────────────────────────────────────────────────

/// A pure-Rust HAL executable that dispatches to generated CPU kernels.
///
/// # Type parameters
///
/// ``N`` — number of functions (entry points) in the model forward pass.
///   Set to the function count from the compute graph.
#[derive(Debug)]
pub struct HalRustExecutable {
    function_count: usize,
    /// SFCF blob (serveforge_constants_data) containing weight registry,
    /// compute graph, and contract metadata. Read from ``constants.bin``
    /// at model-load time; returned by ``module_data()`` so the caller can
    /// parse the compute graph and weight mappings without needing a dylib.
    blob: Vec<u8>,
}

impl HalRustExecutable {
    /// Create a new ``HalRustExecutable`` with no embedded blob.
    ///
    /// ``function_count`` is the number of functions in the model's
    /// compute graph (typically 28 for a KV-cache model).
    ///
    /// ``module_data()`` returns an empty slice — only suitable for tests
    /// or backends that load weights via an alternative mechanism.
    pub fn new(function_count: usize) -> Self {
        Self { function_count, blob: Vec::new() }
    }

    /// Create a ``HalRustExecutable`` with the SFCF constants blob.
    ///
    /// Use this constructor when ``module_data()`` must return real
    /// weight-registry / compute-graph data (e.g. for the hal-rust
    /// integration path in ``ModelExecutor::load_with_device``).
    pub fn with_blob(function_count: usize, blob: Vec<u8>) -> Self {
        Self { function_count, blob }
    }

    /// Convert a trait Buffer to a ``&[f32]`` slice.
    ///
    /// # Safety
    ///
    /// The buffer must contain f32 data (element_size == 4).
    #[doc(hidden)]
    pub(crate) unsafe fn buf_as_f32_slice(buf: &dyn traits::Buffer) -> &[f32] {
        let ptr = buf.as_ptr() as *const f32;
        let len = buf.len() / 4;
        std::slice::from_raw_parts(ptr, len)
    }

    /// Convert an output trait Buffer to a ``&mut [f32]`` slice.
    ///
    /// # Safety
    ///
    /// The buffer must contain f32 data (element_size == 4) and be writable.
    #[doc(hidden)]
    #[allow(clippy::mut_from_ref)]
    pub(crate) unsafe fn buf_as_f32_mut(buf: &dyn traits::Buffer) -> &mut [f32] {
        let ptr = buf.as_ptr() as *mut f32;
        let len = buf.len() / 4;
        std::slice::from_raw_parts_mut(ptr, len)
    }

    /// Convert a trait Buffer to raw bytes.
    #[doc(hidden)]
    pub(crate) unsafe fn buf_as_bytes(buf: &dyn traits::Buffer) -> &[u8] {
        let ptr = buf.as_ptr();
        let len = buf.len();
        std::slice::from_raw_parts(ptr, len)
    }

    /// Convert a trait Buffer to mutable raw bytes.
    #[doc(hidden)]
    #[allow(clippy::mut_from_ref)]
    pub(crate) unsafe fn buf_as_mut_bytes(buf: &dyn traits::Buffer) -> &mut [u8] {
        let ptr = buf.as_ptr() as *mut u8;
        let len = buf.len();
        std::slice::from_raw_parts_mut(ptr, len)
    }

    /// Build an ``OpShapeMeta`` from input and output buffers.
    #[doc(hidden)]
    pub(crate) fn build_shape_meta(
        inputs: &[&dyn traits::Buffer],
        outputs: &[&dyn traits::Buffer],
    ) -> crate::hal::hal_ops_cpu::OpShapeMeta {
        let input_shapes: Vec<Vec<i64>> = inputs
            .iter()
            .map(|b| b.shape().into_iter().map(|s| s as i64).collect())
            .collect();
        let output_shape: Vec<i64> = outputs
            .first()
            .map(|b| b.shape().into_iter().map(|s| s as i64).collect())
            .unwrap_or_default();
        crate::hal::hal_ops_cpu::OpShapeMeta::new(input_shapes, output_shape)
    }
}

impl traits::Executable for HalRustExecutable {
    fn execute(
        &self,
        op_name: &str,
        _stream: &dyn traits::Stream,
        inputs: &[&dyn traits::Buffer],
        outputs: &[&dyn traits::Buffer],
    ) -> Result<Vec<Vec<i64>>, anyhow::Error> {
        let output_shapes: Vec<Vec<i64>> = outputs
            .iter()
            .map(|b| b.shape().into_iter().map(|s| s as i64).collect())
            .collect();
        let meta = Self::build_shape_meta(inputs, outputs);

        // Special handling for gather op
        if op_name == "gather" && inputs.len() >= 2 {
            let out_buf = outputs.first().ok_or_else(|| anyhow::anyhow!("gather: no output buffer"))?;

            if inputs.len() >= 3 {
                // sf.index: 3-input gather (data, batch_idx, position_idx)
                // For scalar indices: data[idx0, :, :, idx1] → middle dims
                let data_buf = inputs[0];
                let idx0_buf = inputs[1];
                let idx1_buf = inputs[2];

                // SAFETY: Data buffer contains f32 tensor from SSA map.
                let data_slice = unsafe { Self::buf_as_f32_slice(data_buf) };
                // SAFETY: Output buffer pre-allocated by runner.
                let out_slice = unsafe { Self::buf_as_f32_mut(*out_buf) };

                let data_shape: Vec<usize> = data_buf.shape();
                let rank = data_shape.len();

                let idx0: usize = if idx0_buf.element_size() == 8 {
                    // SAFETY: Index buffer is i64 (8 bytes/elem).
                    let bytes = unsafe { Self::buf_as_bytes(idx0_buf) };
                    if bytes.len() >= 8 { i64::from_le_bytes(bytes[..8].try_into().unwrap_or([0;8])) as usize } else { 0 }
                } else {
                    // SAFETY: Index buffer is f32 (4 bytes/elem).
                    let slice = unsafe { Self::buf_as_f32_slice(idx0_buf) };
                    if !slice.is_empty() { slice[0] as usize } else { 0 }
                };

                let idx1: usize = if idx1_buf.element_size() == 8 {
                    // SAFETY: Index buffer is i64 (8 bytes/elem).
                    let bytes = unsafe { Self::buf_as_bytes(idx1_buf) };
                    if bytes.len() >= 8 { i64::from_le_bytes(bytes[..8].try_into().unwrap_or([0;8])) as usize } else { 0 }
                } else {
                    // SAFETY: Index buffer is f32 (4 bytes/elem).
                    let slice = unsafe { Self::buf_as_f32_slice(idx1_buf) };
                    if !slice.is_empty() { slice[0] as usize } else { 0 }
                };

                if rank >= 2 {
                    let inner: usize = data_shape[1..rank-1].iter().product();
                    let last = data_shape[rank - 1];
                    let src_base = idx0 * inner * last + idx1;
                    for i in 0..inner {
                        let src_off = src_base + i * last;
                        let dst_off = i * last;
                        if src_off + last <= data_slice.len() && dst_off + last <= out_slice.len() {
                            out_slice[dst_off..dst_off + last]
                                .copy_from_slice(&data_slice[src_off..src_off + last]);
                        }
                    }
                }
            } else {
                // Standard 2-input gather (embedding lookup)
                let weight_buf = inputs[0];
                let indices_buf = inputs[1];

                let embed_dim = meta.input_shapes.first()
                    .map(|s| s.iter().skip(1).map(|&d| d as usize).product())
                    .unwrap_or(768);

                // SAFETY: Weight buffer is f32 embedding table.
                let weight_slice = unsafe { Self::buf_as_f32_slice(weight_buf) };
                // SAFETY: Indices buffer may be i64 or f32; accessed as raw bytes.
                let indices_bytes = unsafe { Self::buf_as_bytes(indices_buf) };
                // SAFETY: Output buffer pre-allocated by runner.
                let out_slice = unsafe { Self::buf_as_f32_mut(*out_buf) };

                let index_dtype = if indices_buf.element_size() == 8 {
                    crate::tensor::Dtype::I64
                } else {
                    crate::tensor::Dtype::F32
                };

                eprintln!(
                    "[gather_debug] indices: element_size={}, numel={}, bytes={}, dtype={:?}, shape={:?}",
                    indices_buf.element_size(),
                    indices_buf.len() / indices_buf.element_size() as usize,
                    indices_buf.len(),
                    index_dtype,
                    indices_buf.shape(),
                );

                crate::hal::primitives::gather_from_bytes(
                    weight_slice, indices_bytes, out_slice, embed_dim, index_dtype,
                ).map_err(|e| anyhow::anyhow!("{}", e))?;
            }

            return Ok(output_shapes);
        }

        // Special handling for reshape op to preserve raw bytes
        if op_name == "reshape" && inputs.len() >= 1 {
            let in_buf = inputs[0];
            let out_buf = outputs.first().ok_or_else(|| anyhow::anyhow!("reshape: no output buffer"))?;

            // SAFETY: Input buffer accessed as raw bytes for byte-level copy.
            let in_bytes = unsafe { Self::buf_as_bytes(in_buf) };
            // SAFETY: Output buffer accessed as mutable bytes for byte-level copy.
            let out_bytes = unsafe { Self::buf_as_mut_bytes(*out_buf) };

            if in_bytes.len() != out_bytes.len() {
                return Err(anyhow::anyhow!(
                    "reshape: numel mismatch: input {} bytes != output {} bytes (in_shape={:?}, out_shape={:?})",
                    in_bytes.len(), out_bytes.len(),
                    in_buf.shape(), out_buf.shape(),
                ));
            }
            out_bytes.copy_from_slice(in_bytes);

            return Ok(output_shapes);
        }

        // Special handling for layer_norm op
        if op_name == "layer_norm" && inputs.len() >= 2 {
            let in_buf = inputs[0];
            let weight_buf = inputs[1];
            let out_buf = outputs.first().ok_or_else(|| anyhow::anyhow!("layer_norm: no output buffer"))?;

            // SAFETY: Input buffer is f32 tensor from SSA map.
            let in_slice = unsafe { Self::buf_as_f32_slice(in_buf) };
            // SAFETY: Weight buffer is f32 layer norm parameters.
            let weight_slice = unsafe { Self::buf_as_f32_slice(weight_buf) };
            // SAFETY: Output buffer pre-allocated by runner.
            let out_slice = unsafe { Self::buf_as_f32_mut(*out_buf) };

            let cols = meta.input_shapes.first()
                .map(|s| s.last().copied().unwrap_or(768) as usize)
                .unwrap_or(768);

            crate::hal::primitives::fused_rms_norm(in_slice, out_slice, weight_slice, cols, 1e-5);
            return Ok(output_shapes);
        }

        // Special handling for linear/matmul op
        if (op_name == "linear" || op_name == "matmul") && inputs.len() >= 2 {
            let in_buf = inputs[0];
            let weight_buf = inputs[1];
            let out_buf = outputs.first().ok_or_else(|| anyhow::anyhow!("linear: no output buffer"))?;

            // SAFETY: Input buffer is f32 activation tensor.
            let in_slice = unsafe { Self::buf_as_f32_slice(in_buf) };
            // SAFETY: Weight buffer is f32 linear weight matrix.
            let weight_slice = unsafe { Self::buf_as_f32_slice(weight_buf) };
            // SAFETY: Output buffer pre-allocated by runner.
            let out_slice = unsafe { Self::buf_as_f32_mut(*out_buf) };

            let a_shape = meta.input_shapes.first().cloned().unwrap_or_default();
            let b_shape = meta.input_shapes.get(1).cloned().unwrap_or_default();

            crate::hal::primitives::matmul_blas(in_slice, weight_slice, out_slice, &a_shape, &b_shape, true)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            return Ok(output_shapes);
        }

        // Special handling for scaled_dot_product_attention op
        if op_name == "scaled_dot_product_attention" && inputs.len() >= 4 {
            let q_buf = inputs[0];
            let k_buf = inputs[1];
            let v_buf = inputs[2];
            let mask_buf = inputs[3];
            let out_buf = outputs.first().ok_or_else(|| anyhow::anyhow!("sdpa: no output buffer"))?;

            // SAFETY: Input buffers are f32 tensors from SSA map.
            let q_slice = unsafe { Self::buf_as_f32_slice(q_buf) };
            let k_slice = unsafe { Self::buf_as_f32_slice(k_buf) };
            let v_slice = unsafe { Self::buf_as_f32_slice(v_buf) };
            let mask_slice = unsafe { Self::buf_as_f32_slice(mask_buf) };
            // SAFETY: Output buffer pre-allocated by runner.
            let out_slice = unsafe { Self::buf_as_f32_mut(*out_buf) };

            let q_shape = meta.input_shapes.first().cloned().unwrap_or_default();
            let k_shape = meta.input_shapes.get(1).cloned().unwrap_or_default();
            let v_shape = meta.input_shapes.get(2).cloned().unwrap_or_default();
            let mask_shape = meta.input_shapes.get(3).cloned().unwrap_or_default();

            crate::hal::primitives::fused_sdpa(
                q_slice, k_slice, v_slice, mask_slice, out_slice,
                &q_shape, &k_shape, &v_shape, &mask_shape,
            )
            .map_err(|e| anyhow::anyhow!("{}", e))?;
            return Ok(output_shapes);
        }

        // Special handling for element_wise ops with i64 or f32 inputs
        if op_name.starts_with("element_wise:") && inputs.len() >= 2 {
            let a_buf = inputs[0];
            let b_buf = inputs[1];
            let out_buf = outputs.first().ok_or_else(|| anyhow::anyhow!("element_wise: no output buffer"))?;
            let kind = op_name.strip_prefix("element_wise:").unwrap_or("add");

            // Check if inputs are i64 or f32
            let is_i64 = a_buf.element_size() == 8 && b_buf.element_size() == 8;
            let is_f32 = a_buf.element_size() == 4 && b_buf.element_size() == 4;
            if is_i64 {
                // SAFETY: Buffer is valid for the lifetime of inputs/outputs refs.
                // Element-wise ops guarantee inputs have the same byte length.
                let a_bytes = unsafe { Self::buf_as_bytes(a_buf) };
                let b_bytes = unsafe { Self::buf_as_bytes(b_buf) };
                // SAFETY: Output buffer is pre-allocated by the runner with
                // the correct size. Only written within bounds.
                let out_bytes = unsafe { Self::buf_as_mut_bytes(*out_buf) };

                let num_elems = a_bytes.len().max(b_bytes.len()) / 8;
                let a_scalar = a_bytes.len() / 8 == 1;
                let b_scalar = b_bytes.len() / 8 == 1;
                let kind = op_name.strip_prefix("element_wise:").unwrap_or("add");

                for i in 0..num_elems {
                    let a_idx = if a_scalar { 0 } else { i };
                    let b_idx = if b_scalar { 0 } else { i };
                    let a_val = i64::from_le_bytes(a_bytes[a_idx*8..(a_idx+1)*8].try_into().unwrap_or([0; 8]));
                    let b_val = i64::from_le_bytes(b_bytes[b_idx*8..(b_idx+1)*8].try_into().unwrap_or([0; 8]));
                    let result = match kind {
                        "add" => a_val.wrapping_add(b_val),
                        "sub" => a_val.wrapping_sub(b_val),
                        "mul" => a_val.wrapping_mul(b_val),
                        _ => a_val,
                    };
                    let result_bytes = result.to_le_bytes();
                    out_bytes[i*8..(i+1)*8].copy_from_slice(&result_bytes);
                }
                return Ok(output_shapes);
            }

            // Handle f32 element_wise ops with broadcasting
            if is_f32 {
                let a_slice = unsafe { Self::buf_as_f32_slice(a_buf) };
                let b_slice = unsafe { Self::buf_as_f32_slice(b_buf) };
                let out_slice = unsafe { Self::buf_as_f32_mut(*out_buf) };
                let a_scalar = a_slice.len() == 1;
                let b_scalar = b_slice.len() == 1;
                // Use output numel for iteration — shape inference ensures
                // output size matches.  For non-scalar inputs, wrap-around
                // indexing supports numpy-style broadcasting.
                let num_elems = out_slice.len();
                for i in 0..num_elems {
                    let a_idx = if a_scalar { 0 } else { i % a_slice.len() };
                    let b_idx = if b_scalar { 0 } else { i % b_slice.len() };
                    let a_val = a_slice[a_idx];
                    let b_val = b_slice[b_idx];
                    let result = match kind {
                        "add" => a_val + b_val,
                        "sub" => a_val - b_val,
                        "mul" => a_val * b_val,
                        "div" => a_val / b_val,
                        _ => a_val,
                    };
                    out_slice[i] = result;
                }
                return Ok(output_shapes);
            }
        }

        // Special handling for transpose op
        if op_name.starts_with("transpose") {
            let in_buf = inputs.first().ok_or_else(|| anyhow::anyhow!("transpose: no input"))?;
            let out_buf = outputs.first().ok_or_else(|| anyhow::anyhow!("transpose: no output"))?;

            // SAFETY: Input buffer contains f32 tensor from SSA map.
            let input_slice = unsafe { Self::buf_as_f32_slice(*in_buf) };
            // SAFETY: Output buffer is pre-allocated as f32 by the runner.
            let out_slice = unsafe { Self::buf_as_f32_mut(*out_buf) };

            let input_shape: Vec<i64> = in_buf.shape().into_iter().map(|s| s as i64).collect();
            let output_shape: Vec<i64> = out_buf.shape().into_iter().map(|s| s as i64).collect();

            let rank = input_shape.len();
            let perm: Vec<usize> = if let Some(kind_str) = op_name.strip_prefix("transpose:") {
                // Parse axis pairs from "transpose:1,2" and build full permutation.
                let dims: Vec<usize> = kind_str.split(',')
                    .filter_map(|s| s.trim().parse::<usize>().ok())
                    .collect();
                let mut p: Vec<usize> = (0..rank).collect();
                for pair in dims.chunks(2) {
                    if pair.len() == 2 && pair[0] < rank && pair[1] < rank {
                        p.swap(pair[0], pair[1]);
                    }
                }
                p
            } else {
                let mut p: Vec<usize> = (0..rank).collect();
                p.swap(rank - 2, rank - 1);
                p
            };

            crate::hal::primitives::transpose_nd(input_slice, out_slice, &input_shape, &output_shape, &perm)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            return Ok(output_shapes);
        }

        // Special handling for scan:cumsum op (prefix sum)
        if op_name == "scan:cumsum" {
            let in_buf = inputs.first().ok_or_else(|| anyhow::anyhow!("scan: no input"))?;
            let out_buf = outputs.first().ok_or_else(|| anyhow::anyhow!("scan: no output"))?;
            let input_slice = unsafe { Self::buf_as_f32_slice(*in_buf) };
            let out_slice = unsafe { Self::buf_as_f32_mut(*out_buf) };
            let mut running = 0.0f32;
            for i in 0..input_slice.len().min(out_slice.len()) {
                running += input_slice[i];
                out_slice[i] = running;
            }
            return Ok(output_shapes);
        }

        // Special handling for softmax op
        if op_name == "softmax" {
            let in_buf = inputs.first().ok_or_else(|| anyhow::anyhow!("softmax: no input"))?;
            let out_buf = outputs.first().ok_or_else(|| anyhow::anyhow!("softmax: no output"))?;
            let input_slice = unsafe { Self::buf_as_f32_slice(*in_buf) };
            let out_slice = unsafe { Self::buf_as_f32_mut(*out_buf) };
            let shape: Vec<usize> = in_buf.shape();
            let last_dim = *shape.last().unwrap_or(&1);
            crate::hal::primitives::fused_softmax(input_slice, out_slice, last_dim);
            return Ok(output_shapes);
        }

        // Special handling for element_wise:rsqrt op (1/sqrt(x))
        if op_name == "element_wise:rsqrt" {
            let in_buf = inputs.first().ok_or_else(|| anyhow::anyhow!("rsqrt: no input"))?;
            let out_buf = outputs.first().ok_or_else(|| anyhow::anyhow!("rsqrt: no output"))?;
            let input_slice = unsafe { Self::buf_as_f32_slice(*in_buf) };
            let out_slice = unsafe { Self::buf_as_f32_mut(*out_buf) };
            for i in 0..input_slice.len().min(out_slice.len()) {
                out_slice[i] = 1.0 / input_slice[i].sqrt();
            }
            return Ok(output_shapes);
        }

        // Special handling for reduce:mean op (mean reduction)
        if op_name == "reduce:mean" {
            let in_buf = inputs.first().ok_or_else(|| anyhow::anyhow!("reduce: no input"))?;
            let out_buf = outputs.first().ok_or_else(|| anyhow::anyhow!("reduce: no output"))?;
            let input_slice = unsafe { Self::buf_as_f32_slice(*in_buf) };
            let out_slice = unsafe { Self::buf_as_f32_mut(*out_buf) };
            let mean = if !input_slice.is_empty() {
                input_slice.iter().sum::<f32>() / input_slice.len() as f32
            } else { 0.0 };
            for i in 0..out_slice.len() {
                out_slice[i] = mean;
            }
            return Ok(output_shapes);
        }

        // Default path: convert all inputs to f32 slices
        let input_slices: Vec<&[f32]> = inputs
            .iter()
            // SAFETY: All inputs are f32 buffers (element_size == 4).
            .map(|b| unsafe { Self::buf_as_f32_slice(*b) })
            .collect();

        if let Some(out_buf) = outputs.first() {
            // SAFETY: Output buffer is pre-allocated as f32 by the runner.
            let out_slice = unsafe { Self::buf_as_f32_mut(*out_buf) };
            crate::hal::hal_ops_cpu::dispatch(op_name, &input_slices, out_slice, &meta)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
        } else {
            crate::hal::hal_ops_cpu::dispatch(op_name, &input_slices, &mut [], &meta)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
        }
        Ok(output_shapes)
    }

    fn function_count(&self) -> usize {
        self.function_count
    }

    fn module_data(&self) -> &[u8] {
        &self.blob
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hal::traits;
    use crate::hal::traits::Executable as _;

    /// A minimal buffer backed by a Vec<f32> for testing.
    #[derive(Debug)]
    struct TestBuf(Vec<u8>, usize, Vec<usize>);

    impl traits::Buffer for TestBuf {
        fn as_ptr(&self) -> *const u8 { self.0.as_ptr() }
        fn as_mut_ptr(&mut self) -> *mut u8 { self.0.as_mut_ptr() }
        fn len(&self) -> usize { self.0.len() }
        fn copy_from_host(&mut self, src: &[u8], _: &dyn traits::Stream) -> Result<(), anyhow::Error> {
            self.0.copy_from_slice(src);
            Ok(())
        }
        fn copy_to_host(&self, dst: &mut [u8], _: &dyn traits::Stream) -> Result<(), anyhow::Error> {
            dst.copy_from_slice(&self.0);
            Ok(())
        }
        fn element_size(&self) -> usize { self.1 }
        fn shape(&self) -> Vec<usize> { self.2.clone() }
        fn rank(&self) -> u8 { self.2.len() as u8 }
    }

    #[derive(Debug)]
    struct NoopStream;
    impl traits::Stream for NoopStream {
        fn synchronize(&self) -> Result<(), anyhow::Error> { Ok(()) }
        fn wait_event(&self, _: &dyn traits::Event) -> Result<(), anyhow::Error> { Ok(()) }
        fn record_event(&self, _: &dyn traits::Event) -> Result<(), anyhow::Error> { Ok(()) }
    }

    #[test]
    fn test_hal_rust_executable_new() {
        let exe = HalRustExecutable::new(28);
        assert_eq!(exe.function_count, 28);
    }

    #[test]
    fn test_hal_rust_executable_function_count() {
        let exe = HalRustExecutable::new(16);
        assert_eq!(exe.function_count(), 16);

        let exe2 = HalRustExecutable::new(28);
        assert_eq!(exe2.function_count(), 28);
    }

    #[test]
    fn test_hal_rust_executable_module_data_empty() {
        let exe = HalRustExecutable::new(1);
        assert!(exe.module_data().is_empty());
    }

    #[test]
    fn test_hal_rust_executable_execute_unknown_op() {
        let exe = HalRustExecutable::new(1);
        let stream = NoopStream;
        let result = exe.execute("nonexistent_op", &stream, &[], &[]);
        assert!(result.is_err(), "unknown op should return error");
        assert!(
            result.unwrap_err().to_string().contains("unknown op"),
            "error message should mention 'unknown op'"
        );
    }

    #[test]
    fn test_hal_rust_executable_cache_ops_noop() {
        // cache_read and cache_write are no-op stubs that return output shapes.
        let exe = HalRustExecutable::new(1);
        let stream = NoopStream;
        let input = TestBuf(vec![0u8; 16], 4, vec![4]);
        let output = TestBuf(vec![0u8; 16], 4, vec![4]);
        let inputs: [&dyn traits::Buffer; 1] = [&input];
        let outputs: [&dyn traits::Buffer; 1] = [&output];

        let result = exe.execute("cache_read", &stream, &inputs, &outputs);
        assert!(result.is_ok(), "cache_read should be a no-op");
        let shapes = result.unwrap();
        assert_eq!(shapes, vec![vec![4i64]]);

        let result2 = exe.execute("cache_write", &stream, &inputs, &outputs);
        assert!(result2.is_ok(), "cache_write should be a no-op");
    }

    #[test]
    fn test_hal_rust_executable_register_expert_kernel() {
        // Verify that register_expert_kernel works (default impl errors).
        // Use the trait method directly via the Executable import.
        let mut exe = HalRustExecutable::new(1);
        let result = traits::Executable::register_expert_kernel(
            &mut exe, "test_op", Box::new(NoopExpertKernel),
        );
        assert!(result.is_err(), "default register_expert_kernel should error");
    }

    #[derive(Debug)]
    struct NoopExpertKernel;
    impl traits::ExpertKernel for NoopExpertKernel {
        fn execute(
            &self,
            _stream: &dyn traits::Stream,
            _inputs: &[&dyn traits::Buffer],
            _outputs: &[&dyn traits::Buffer],
        ) -> Result<(), anyhow::Error> {
            Ok(())
        }
    }
}
