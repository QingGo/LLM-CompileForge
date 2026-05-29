//! HalRustExecutable — pure-Rust HAL backend dispatch.
//!
//! Implements ``traits::Executable`` by dispatching to the generated
//! ``*_cpu`` functions from ``hal_ops_cpu.rs`` (emitted by EmitRust).
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
/// - ``N`` — number of functions (entry points) in the model forward pass.
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
    unsafe fn buf_as_f32_slice(buf: &dyn traits::Buffer) -> &[f32] {
        let ptr = buf.as_ptr() as *const f32;
        let len = buf.len() / 4; // f32 = 4 bytes
        std::slice::from_raw_parts(ptr, len)
    }

    /// Convert an output trait Buffer to a ``&mut [f32]`` slice.
    ///
    /// Uses the raw pointer from ``as_ptr()`` cast to mutable — the caller
    /// guarantees the buffer is writable (same pattern as CpuExecutable).
    ///
    /// # Safety
    ///
    /// The buffer must contain f32 data (element_size == 4) and be writable.
    #[allow(clippy::mut_from_ref)]
    unsafe fn buf_as_f32_mut(buf: &dyn traits::Buffer) -> &mut [f32] {
        let ptr = buf.as_ptr() as *mut f32;
        let len = buf.len() / 4;
        std::slice::from_raw_parts_mut(ptr, len)
    }

    /// Build an ``OpShapeMeta`` from input and output buffers.
    fn build_shape_meta(
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

    /// Dispatch a matmul operation.
    fn dispatch_matmul(
        &self,
        inputs: &[&dyn traits::Buffer],
        outputs: &[&dyn traits::Buffer],
    ) -> Result<Vec<Vec<i64>>, anyhow::Error> {
        let input_slices: Vec<&[f32]> = inputs
            .iter()
            .map(|b| {
                // SAFETY: buf_as_f32_slice requires f32 data (element_size == 4).
                // Matmul inputs are f32 tensors from weights or SSA wires, guaranteed
                // by the compute graph runner.
                unsafe { Self::buf_as_f32_slice(*b) }
            })
            .collect();
        let output_shapes: Vec<Vec<i64>> = outputs
            .iter()
            .map(|b| b.shape().into_iter().map(|s| s as i64).collect())
            .collect();
        let meta = Self::build_shape_meta(inputs, outputs);

        if let Some(out_buf) = outputs.first() {
            // SAFETY: buf_as_f32_mut requires f32 data and write access. The output
            // buffer is pre-allocated as f32 by the compute graph runner and valid
            // for the matmul result size.
            let out_slice = unsafe { Self::buf_as_f32_mut(*out_buf) };
            crate::hal::hal_ops_cpu::matmul_cpu(&input_slices, out_slice, &meta)
                .map_err(|e| anyhow::anyhow!("matmul_cpu: {}", e))?;
        }
        Ok(output_shapes)
    }

    /// Dispatch an element_wise operation.
    fn dispatch_element_wise(
        &self,
        inputs: &[&dyn traits::Buffer],
        outputs: &[&dyn traits::Buffer],
        kind: &str,
    ) -> Result<Vec<Vec<i64>>, anyhow::Error> {
        let input_slices: Vec<&[f32]> = inputs
            .iter()
            .map(|b| {
                // SAFETY: buf_as_f32_slice requires f32 data. Element-wise
                // inputs are f32 tensors guaranteed by the compute graph runner.
                unsafe { Self::buf_as_f32_slice(*b) }
            })
            .collect();
        let output_shapes: Vec<Vec<i64>> = outputs
            .iter()
            .map(|b| b.shape().into_iter().map(|s| s as i64).collect())
            .collect();
        let mut meta = Self::build_shape_meta(inputs, outputs);
        meta.kind = Some(kind.to_string());

        if let Some(out_buf) = outputs.first() {
            // SAFETY: buf_as_f32_mut requires f32 data and write access.
            // The output buffer is pre-allocated as f32 by the compute graph runner.
            let out_slice = unsafe { Self::buf_as_f32_mut(*out_buf) };
            crate::hal::hal_ops_cpu::element_wise_cpu(&input_slices, out_slice, &meta)
                .map_err(|e| anyhow::anyhow!("element_wise_cpu: {}", e))?;
        }
        Ok(output_shapes)
    }

    /// Dispatch a softmax operation.
    fn dispatch_softmax(
        &self,
        inputs: &[&dyn traits::Buffer],
        outputs: &[&dyn traits::Buffer],
    ) -> Result<Vec<Vec<i64>>, anyhow::Error> {
        let input_slices: Vec<&[f32]> = inputs
            .iter()
            .map(|b| {
                // SAFETY: buf_as_f32_slice requires f32 data. Softmax inputs
                // are f32 tensors guaranteed by the compute graph runner.
                unsafe { Self::buf_as_f32_slice(*b) }
            })
            .collect();
        let output_shapes: Vec<Vec<i64>> = outputs
            .iter()
            .map(|b| b.shape().into_iter().map(|s| s as i64).collect())
            .collect();
        let meta = Self::build_shape_meta(inputs, outputs);

        if let Some(out_buf) = outputs.first() {
            // SAFETY: buf_as_f32_mut requires f32 data and write access.
            // The output buffer is pre-allocated as f32 by the compute graph runner.
            let out_slice = unsafe { Self::buf_as_f32_mut(*out_buf) };
            crate::hal::hal_ops_cpu::softmax_cpu(&input_slices, out_slice, &meta)
                .map_err(|e| anyhow::anyhow!("softmax_cpu: {}", e))?;
        }
        Ok(output_shapes)
    }

    /// Dispatch a reshape operation.
    fn dispatch_reshape(
        &self,
        inputs: &[&dyn traits::Buffer],
        outputs: &[&dyn traits::Buffer],
    ) -> Result<Vec<Vec<i64>>, anyhow::Error> {
        let input_slices: Vec<&[f32]> = inputs
            .iter()
            .map(|b| {
                // SAFETY: buf_as_f32_slice requires f32 data. Reshape inputs
                // are f32 tensors guaranteed by the compute graph runner.
                unsafe { Self::buf_as_f32_slice(*b) }
            })
            .collect();
        let output_shapes: Vec<Vec<i64>> = outputs
            .iter()
            .map(|b| b.shape().into_iter().map(|s| s as i64).collect())
            .collect();
        let meta = Self::build_shape_meta(inputs, outputs);

        if let Some(out_buf) = outputs.first() {
            // SAFETY: buf_as_f32_mut requires f32 data and write access.
            // The output buffer is pre-allocated as f32 by the compute graph runner.
            let out_slice = unsafe { Self::buf_as_f32_mut(*out_buf) };
            crate::hal::hal_ops_cpu::reshape_cpu(&input_slices, out_slice, &meta)
                .map_err(|e| anyhow::anyhow!("reshape_cpu: {}", e))?;
        }
        Ok(output_shapes)
    }

    /// Dispatch a transpose operation.
    fn dispatch_transpose(
        &self,
        inputs: &[&dyn traits::Buffer],
        outputs: &[&dyn traits::Buffer],
        perm: Option<&str>,
    ) -> Result<Vec<Vec<i64>>, anyhow::Error> {
        let input_slices: Vec<&[f32]> = inputs
            .iter()
            .map(|b| {
                // SAFETY: buf_as_f32_slice requires f32 data. Transpose inputs
                // are f32 tensors guaranteed by the compute graph runner.
                unsafe { Self::buf_as_f32_slice(*b) }
            })
            .collect();
        let output_shapes: Vec<Vec<i64>> = outputs
            .iter()
            .map(|b| b.shape().into_iter().map(|s| s as i64).collect())
            .collect();
        let mut meta = Self::build_shape_meta(inputs, outputs);
        meta.kind = perm.map(|s| s.to_string());

        if let Some(out_buf) = outputs.first() {
            // SAFETY: buf_as_f32_mut requires f32 data and write access.
            // The output buffer is pre-allocated as f32 by the compute graph runner.
            let out_slice = unsafe { Self::buf_as_f32_mut(*out_buf) };
            crate::hal::hal_ops_cpu::transpose_cpu(&input_slices, out_slice, &meta)
                .map_err(|e| anyhow::anyhow!("transpose_cpu: {}", e))?;
        }
        Ok(output_shapes)
    }

    /// Dispatch a reduce operation.
    fn dispatch_reduce(
        &self,
        inputs: &[&dyn traits::Buffer],
        outputs: &[&dyn traits::Buffer],
        kind: &str,
    ) -> Result<Vec<Vec<i64>>, anyhow::Error> {
        let input_slices: Vec<&[f32]> = inputs
            .iter()
            .map(|b| {
                // SAFETY: buf_as_f32_slice requires f32 data. Reduce inputs
                // are f32 tensors guaranteed by the compute graph runner.
                unsafe { Self::buf_as_f32_slice(*b) }
            })
            .collect();
        let output_shapes: Vec<Vec<i64>> = outputs
            .iter()
            .map(|b| b.shape().into_iter().map(|s| s as i64).collect())
            .collect();
        let mut meta = Self::build_shape_meta(inputs, outputs);
        meta.kind = Some(kind.to_string());

        if let Some(out_buf) = outputs.first() {
            // SAFETY: buf_as_f32_mut requires f32 data and write access.
            // The output buffer is pre-allocated as f32 by the compute graph runner.
            let out_slice = unsafe { Self::buf_as_f32_mut(*out_buf) };
            crate::hal::hal_ops_cpu::reduce_cpu(&input_slices, out_slice, &meta)
                .map_err(|e| anyhow::anyhow!("reduce_cpu: {}", e))?;
        }
        Ok(output_shapes)
    }

    /// Dispatch a gather operation.
    fn dispatch_gather(
        &self,
        inputs: &[&dyn traits::Buffer],
        outputs: &[&dyn traits::Buffer],
    ) -> Result<Vec<Vec<i64>>, anyhow::Error> {
        let input_slices: Vec<&[f32]> = inputs
            .iter()
            .map(|b| {
                // SAFETY: buf_as_f32_slice requires f32 data. Gather inputs
                // are f32 tensors guaranteed by the compute graph runner.
                unsafe { Self::buf_as_f32_slice(*b) }
            })
            .collect();
        let output_shapes: Vec<Vec<i64>> = outputs
            .iter()
            .map(|b| b.shape().into_iter().map(|s| s as i64).collect())
            .collect();
        let meta = Self::build_shape_meta(inputs, outputs);

        if let Some(out_buf) = outputs.first() {
            // SAFETY: buf_as_f32_mut requires f32 data and write access.
            // The output buffer is pre-allocated as f32 by the compute graph runner.
            let out_slice = unsafe { Self::buf_as_f32_mut(*out_buf) };
            crate::hal::hal_ops_cpu::gather_cpu(&input_slices, out_slice, &meta)
                .map_err(|e| anyhow::anyhow!("gather_cpu: {}", e))?;
        }
        Ok(output_shapes)
    }

    /// Dispatch a fill operation.
    fn dispatch_fill(
        &self,
        inputs: &[&dyn traits::Buffer],
        outputs: &[&dyn traits::Buffer],
        kind: Option<&str>,
        value: Option<f64>,
    ) -> Result<Vec<Vec<i64>>, anyhow::Error> {
        let input_slices: Vec<&[f32]> = inputs
            .iter()
            .map(|b| {
                // SAFETY: buf_as_f32_slice requires f32 data. Fill inputs
                // are f32 tensors guaranteed by the compute graph runner.
                unsafe { Self::buf_as_f32_slice(*b) }
            })
            .collect();
        let output_shapes: Vec<Vec<i64>> = outputs
            .iter()
            .map(|b| b.shape().into_iter().map(|s| s as i64).collect())
            .collect();
        let mut meta = Self::build_shape_meta(inputs, outputs);
        meta.kind = kind.map(|s| s.to_string());
        meta.value = value;

        if let Some(out_buf) = outputs.first() {
            // SAFETY: buf_as_f32_mut requires f32 data and write access.
            // The output buffer is pre-allocated as f32 by the compute graph runner.
            let out_slice = unsafe { Self::buf_as_f32_mut(*out_buf) };
            crate::hal::hal_ops_cpu::fill_cpu(&input_slices, out_slice, &meta)
                .map_err(|e| anyhow::anyhow!("fill_cpu: {}", e))?;
        }
        Ok(output_shapes)
    }

    /// Dispatch a shape_of operation.
    fn dispatch_shape_of(
        &self,
        inputs: &[&dyn traits::Buffer],
        outputs: &[&dyn traits::Buffer],
    ) -> Result<Vec<Vec<i64>>, anyhow::Error> {
        let input_slices: Vec<&[f32]> = inputs
            .iter()
            .map(|b| {
                // SAFETY: buf_as_f32_slice requires f32 data. Shape-of inputs
                // are f32 tensors guaranteed by the compute graph runner.
                unsafe { Self::buf_as_f32_slice(*b) }
            })
            .collect();
        let output_shapes: Vec<Vec<i64>> = outputs
            .iter()
            .map(|b| b.shape().into_iter().map(|s| s as i64).collect())
            .collect();
        let meta = Self::build_shape_meta(inputs, outputs);

        if let Some(out_buf) = outputs.first() {
            // SAFETY: buf_as_f32_mut requires f32 data and write access.
            // The output buffer is pre-allocated as f32 by the compute graph runner.
            let out_slice = unsafe { Self::buf_as_f32_mut(*out_buf) };
            crate::hal::hal_ops_cpu::shape_of_cpu(&input_slices, out_slice, &meta)
                .map_err(|e| anyhow::anyhow!("shape_of_cpu: {}", e))?;
        }
        Ok(output_shapes)
    }

    /// Dispatch a slice operation.
    fn dispatch_slice(
        &self,
        inputs: &[&dyn traits::Buffer],
        outputs: &[&dyn traits::Buffer],
    ) -> Result<Vec<Vec<i64>>, anyhow::Error> {
        let input_slices: Vec<&[f32]> = inputs
            .iter()
            .map(|b| {
                // SAFETY: buf_as_f32_slice requires f32 data. Slice inputs
                // are f32 tensors guaranteed by the compute graph runner.
                unsafe { Self::buf_as_f32_slice(*b) }
            })
            .collect();
        let output_shapes: Vec<Vec<i64>> = outputs
            .iter()
            .map(|b| b.shape().into_iter().map(|s| s as i64).collect())
            .collect();
        let meta = Self::build_shape_meta(inputs, outputs);

        if let Some(out_buf) = outputs.first() {
            // SAFETY: buf_as_f32_mut requires f32 data and write access.
            // The output buffer is pre-allocated as f32 by the compute graph runner.
            let out_slice = unsafe { Self::buf_as_f32_mut(*out_buf) };
            crate::hal::hal_ops_cpu::slice_cpu(&input_slices, out_slice, &meta)
                .map_err(|e| anyhow::anyhow!("slice_cpu: {}", e))?;
        }
        Ok(output_shapes)
    }

    /// Dispatch an unsqueeze operation.
    fn dispatch_unsqueeze(
        &self,
        inputs: &[&dyn traits::Buffer],
        outputs: &[&dyn traits::Buffer],
    ) -> Result<Vec<Vec<i64>>, anyhow::Error> {
        let input_slices: Vec<&[f32]> = inputs
            .iter()
            .map(|b| {
                // SAFETY: buf_as_f32_slice requires f32 data. Unsqueeze inputs
                // are f32 tensors guaranteed by the compute graph runner.
                unsafe { Self::buf_as_f32_slice(*b) }
            })
            .collect();
        let output_shapes: Vec<Vec<i64>> = outputs
            .iter()
            .map(|b| b.shape().into_iter().map(|s| s as i64).collect())
            .collect();
        let meta = Self::build_shape_meta(inputs, outputs);

        if let Some(out_buf) = outputs.first() {
            // SAFETY: buf_as_f32_mut requires f32 data and write access.
            // The output buffer is pre-allocated as f32 by the compute graph runner.
            let out_slice = unsafe { Self::buf_as_f32_mut(*out_buf) };
            crate::hal::hal_ops_cpu::unsqueeze_cpu(&input_slices, out_slice, &meta)
                .map_err(|e| anyhow::anyhow!("unsqueeze_cpu: {}", e))?;
        }
        Ok(output_shapes)
    }

    /// Dispatch a compare operation.
    fn dispatch_compare(
        &self,
        inputs: &[&dyn traits::Buffer],
        outputs: &[&dyn traits::Buffer],
        kind: &str,
    ) -> Result<Vec<Vec<i64>>, anyhow::Error> {
        let input_slices: Vec<&[f32]> = inputs
            .iter()
            .map(|b| {
                // SAFETY: buf_as_f32_slice requires f32 data. Compare inputs
                // are f32 tensors guaranteed by the compute graph runner.
                unsafe { Self::buf_as_f32_slice(*b) }
            })
            .collect();
        let output_shapes: Vec<Vec<i64>> = outputs
            .iter()
            .map(|b| b.shape().into_iter().map(|s| s as i64).collect())
            .collect();
        let mut meta = Self::build_shape_meta(inputs, outputs);
        meta.kind = Some(kind.to_string());

        if let Some(out_buf) = outputs.first() {
            // SAFETY: buf_as_f32_mut requires f32 data and write access.
            // The output buffer is pre-allocated as f32 by the compute graph runner.
            let out_slice = unsafe { Self::buf_as_f32_mut(*out_buf) };
            crate::hal::hal_ops_cpu::compare_cpu(&input_slices, out_slice, &meta)
                .map_err(|e| anyhow::anyhow!("compare_cpu: {}", e))?;
        }
        Ok(output_shapes)
    }

    /// Dispatch a concat operation (not yet implemented in Rust backend).
    fn dispatch_concat(
        &self,
        inputs: &[&dyn traits::Buffer],
        outputs: &[&dyn traits::Buffer],
    ) -> Result<Vec<Vec<i64>>, anyhow::Error> {
        let output_shapes: Vec<Vec<i64>> = outputs
            .iter()
            .map(|b| b.shape().into_iter().map(|s| s as i64).collect())
            .collect();
        // Simple flat copy (works if no actual concat needed — metadata-only)
        if let (Some(inp), Some(out)) = (inputs.first(), outputs.first()) {
            // SAFETY: buf_as_f32_slice requires f32 data. Concat inputs
            // are f32 tensors guaranteed by the compute graph runner.
            let in_slice = unsafe { Self::buf_as_f32_slice(*inp) };
            // SAFETY: buf_as_f32_mut requires f32 data and write access.
            // The output buffer is pre-allocated as f32 by the compute graph runner.
            let out_slice = unsafe { Self::buf_as_f32_mut(*out) };
            let n = in_slice.len().min(out_slice.len());
            out_slice[..n].copy_from_slice(&in_slice[..n]);
        }
        Ok(output_shapes)
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
        // Parse optional kind suffix from op_name: "element_wise:add" → kind="add".
        // This allows the caller to specify the operation variant while keeping
        // simple names ("element_wise") backward-compatible with defaults.
        let (base_op, kind) = if let Some(pos) = op_name.find(':') {
            (&op_name[..pos], Some(&op_name[pos + 1..]))
        } else {
            (op_name, None)
        };
        match base_op {
            "matmul" => self.dispatch_matmul(inputs, outputs),
            "element_wise" | "elementwise" => {
                self.dispatch_element_wise(inputs, outputs, kind.unwrap_or("add"))
            }
            "softmax" => self.dispatch_softmax(inputs, outputs),
            "reshape" => self.dispatch_reshape(inputs, outputs),
            "transpose" => self.dispatch_transpose(inputs, outputs, kind),
            "reduce" => self.dispatch_reduce(inputs, outputs, kind.unwrap_or("sum")),
            "gather" => self.dispatch_gather(inputs, outputs),
            "fill" => self.dispatch_fill(inputs, outputs, kind, None),
            "shape_of" => self.dispatch_shape_of(inputs, outputs),
            "slice" => self.dispatch_slice(inputs, outputs),
            "unsqueeze" => self.dispatch_unsqueeze(inputs, outputs),
            "compare" => self.dispatch_compare(inputs, outputs, kind.unwrap_or("eq")),
            "concat" => self.dispatch_concat(inputs, outputs),
            "cache_read" | "cache_write" => {
                // Cache ops are handled by the runtime (block_manager/kv_cache).
                // The HAL CPU kernel is a no-op stub.
                let output_shapes: Vec<Vec<i64>> = outputs
                    .iter()
                    .map(|b| b.shape().into_iter().map(|s| s as i64).collect())
                    .collect();
                Ok(output_shapes)
            }
            other => {
                anyhow::bail!("HalRustExecutable: unknown op '{}'", other)
            }
        }
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

