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
}

impl HalRustExecutable {
    /// Create a new ``HalRustExecutable``.
    ///
    /// ``function_count`` is the number of functions in the model's
    /// compute graph (typically 28 for a KV-cache model).
    pub fn new(function_count: usize) -> Self {
        Self { function_count }
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
            .map(|b| unsafe { Self::buf_as_f32_slice(*b) })
            .collect();
        let output_shapes: Vec<Vec<i64>> = outputs
            .iter()
            .map(|b| b.shape().into_iter().map(|s| s as i64).collect())
            .collect();
        let meta = Self::build_shape_meta(inputs, outputs);

        if let Some(out_buf) = outputs.first() {
            let mut out_slice = unsafe { Self::buf_as_f32_mut(*out_buf) };
            crate::hal::hal_ops_cpu::matmul_cpu(&input_slices, &mut out_slice, &meta)
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
            .map(|b| unsafe { Self::buf_as_f32_slice(*b) })
            .collect();
        let output_shapes: Vec<Vec<i64>> = outputs
            .iter()
            .map(|b| b.shape().into_iter().map(|s| s as i64).collect())
            .collect();
        let mut meta = Self::build_shape_meta(inputs, outputs);
        meta.kind = Some(kind.to_string());

        if let Some(out_buf) = outputs.first() {
            let mut out_slice = unsafe { Self::buf_as_f32_mut(*out_buf) };
            crate::hal::hal_ops_cpu::element_wise_cpu(&input_slices, &mut out_slice, &meta)
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
            .map(|b| unsafe { Self::buf_as_f32_slice(*b) })
            .collect();
        let output_shapes: Vec<Vec<i64>> = outputs
            .iter()
            .map(|b| b.shape().into_iter().map(|s| s as i64).collect())
            .collect();
        let meta = Self::build_shape_meta(inputs, outputs);

        if let Some(out_buf) = outputs.first() {
            let mut out_slice = unsafe { Self::buf_as_f32_mut(*out_buf) };
            crate::hal::hal_ops_cpu::softmax_cpu(&input_slices, &mut out_slice, &meta)
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
            .map(|b| unsafe { Self::buf_as_f32_slice(*b) })
            .collect();
        let output_shapes: Vec<Vec<i64>> = outputs
            .iter()
            .map(|b| b.shape().into_iter().map(|s| s as i64).collect())
            .collect();
        let meta = Self::build_shape_meta(inputs, outputs);

        if let Some(out_buf) = outputs.first() {
            let mut out_slice = unsafe { Self::buf_as_f32_mut(*out_buf) };
            crate::hal::hal_ops_cpu::reshape_cpu(&input_slices, &mut out_slice, &meta)
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
            .map(|b| unsafe { Self::buf_as_f32_slice(*b) })
            .collect();
        let output_shapes: Vec<Vec<i64>> = outputs
            .iter()
            .map(|b| b.shape().into_iter().map(|s| s as i64).collect())
            .collect();
        let mut meta = Self::build_shape_meta(inputs, outputs);
        meta.kind = perm.map(|s| s.to_string());

        if let Some(out_buf) = outputs.first() {
            let mut out_slice = unsafe { Self::buf_as_f32_mut(*out_buf) };
            crate::hal::hal_ops_cpu::transpose_cpu(&input_slices, &mut out_slice, &meta)
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
            .map(|b| unsafe { Self::buf_as_f32_slice(*b) })
            .collect();
        let output_shapes: Vec<Vec<i64>> = outputs
            .iter()
            .map(|b| b.shape().into_iter().map(|s| s as i64).collect())
            .collect();
        let mut meta = Self::build_shape_meta(inputs, outputs);
        meta.kind = Some(kind.to_string());

        if let Some(out_buf) = outputs.first() {
            let mut out_slice = unsafe { Self::buf_as_f32_mut(*out_buf) };
            crate::hal::hal_ops_cpu::reduce_cpu(&input_slices, &mut out_slice, &meta)
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
            .map(|b| unsafe { Self::buf_as_f32_slice(*b) })
            .collect();
        let output_shapes: Vec<Vec<i64>> = outputs
            .iter()
            .map(|b| b.shape().into_iter().map(|s| s as i64).collect())
            .collect();
        let meta = Self::build_shape_meta(inputs, outputs);

        if let Some(out_buf) = outputs.first() {
            let mut out_slice = unsafe { Self::buf_as_f32_mut(*out_buf) };
            crate::hal::hal_ops_cpu::gather_cpu(&input_slices, &mut out_slice, &meta)
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
            .map(|b| unsafe { Self::buf_as_f32_slice(*b) })
            .collect();
        let output_shapes: Vec<Vec<i64>> = outputs
            .iter()
            .map(|b| b.shape().into_iter().map(|s| s as i64).collect())
            .collect();
        let mut meta = Self::build_shape_meta(inputs, outputs);
        meta.kind = kind.map(|s| s.to_string());
        meta.value = value;

        if let Some(out_buf) = outputs.first() {
            let mut out_slice = unsafe { Self::buf_as_f32_mut(*out_buf) };
            crate::hal::hal_ops_cpu::fill_cpu(&input_slices, &mut out_slice, &meta)
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
            .map(|b| unsafe { Self::buf_as_f32_slice(*b) })
            .collect();
        let output_shapes: Vec<Vec<i64>> = outputs
            .iter()
            .map(|b| b.shape().into_iter().map(|s| s as i64).collect())
            .collect();
        let meta = Self::build_shape_meta(inputs, outputs);

        if let Some(out_buf) = outputs.first() {
            let mut out_slice = unsafe { Self::buf_as_f32_mut(*out_buf) };
            crate::hal::hal_ops_cpu::shape_of_cpu(&input_slices, &mut out_slice, &meta)
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
            .map(|b| unsafe { Self::buf_as_f32_slice(*b) })
            .collect();
        let output_shapes: Vec<Vec<i64>> = outputs
            .iter()
            .map(|b| b.shape().into_iter().map(|s| s as i64).collect())
            .collect();
        let meta = Self::build_shape_meta(inputs, outputs);

        if let Some(out_buf) = outputs.first() {
            let mut out_slice = unsafe { Self::buf_as_f32_mut(*out_buf) };
            crate::hal::hal_ops_cpu::slice_cpu(&input_slices, &mut out_slice, &meta)
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
            .map(|b| unsafe { Self::buf_as_f32_slice(*b) })
            .collect();
        let output_shapes: Vec<Vec<i64>> = outputs
            .iter()
            .map(|b| b.shape().into_iter().map(|s| s as i64).collect())
            .collect();
        let meta = Self::build_shape_meta(inputs, outputs);

        if let Some(out_buf) = outputs.first() {
            let mut out_slice = unsafe { Self::buf_as_f32_mut(*out_buf) };
            crate::hal::hal_ops_cpu::unsqueeze_cpu(&input_slices, &mut out_slice, &meta)
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
            .map(|b| unsafe { Self::buf_as_f32_slice(*b) })
            .collect();
        let output_shapes: Vec<Vec<i64>> = outputs
            .iter()
            .map(|b| b.shape().into_iter().map(|s| s as i64).collect())
            .collect();
        let mut meta = Self::build_shape_meta(inputs, outputs);
        meta.kind = Some(kind.to_string());

        if let Some(out_buf) = outputs.first() {
            let mut out_slice = unsafe { Self::buf_as_f32_mut(*out_buf) };
            crate::hal::hal_ops_cpu::compare_cpu(&input_slices, &mut out_slice, &meta)
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
            let in_slice = unsafe { Self::buf_as_f32_slice(*inp) };
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
        match op_name {
            "matmul" => self.dispatch_matmul(inputs, outputs),
            "element_wise" | "elementwise" => self.dispatch_element_wise(inputs, outputs, "add"),
            "softmax" => self.dispatch_softmax(inputs, outputs),
            "reshape" => self.dispatch_reshape(inputs, outputs),
            "transpose" => self.dispatch_transpose(inputs, outputs, None),
            "reduce" => self.dispatch_reduce(inputs, outputs, "sum"),
            "gather" => self.dispatch_gather(inputs, outputs),
            "fill" => self.dispatch_fill(inputs, outputs, None, None),
            "shape_of" => self.dispatch_shape_of(inputs, outputs),
            "slice" => self.dispatch_slice(inputs, outputs),
            "unsqueeze" => self.dispatch_unsqueeze(inputs, outputs),
            "compare" => self.dispatch_compare(inputs, outputs, "eq"),
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
        // Weights are loaded from safetensors, not embedded.
        &[]
    }
}
