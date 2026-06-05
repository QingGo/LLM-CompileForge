//! KernelOp trait — formal contract for all HAL CPU kernel types.
//!
//! This trait defines the interface that every compiler-generated kernel
//! (from ``hal_ops_cpu.rs``) must implement. The registry provides
//! trait-object dispatch as a replacement for string-based ``dispatch()``.
//!
//! Each kernel type is a zero-sized wrapper struct around a generated
//! ``*_cpu`` function. The trait enforces at compile time that every
//! registered kernel provides:
//! - A name for look-up
//! - A list of supported dtypes
//! - A typed ``execute_typed`` entry point

use std::collections::HashMap;

use crate::model::error::HalExecutionError;
use crate::model::tensor::Dtype;

// ── Re-export OpShapeMeta for convenience ──────────────────────────────
pub use crate::hal::hal_ops_cpu::OpShapeMeta;

// ═══════════════════════════════════════════════════════════════════════
//  KernelOp trait
// ═══════════════════════════════════════════════════════════════════════

/// Formal contract for a HAL CPU kernel.
///
/// Every compiler-generated ``*_cpu`` function is wrapped by a kernel type
/// that implements this trait. The trait object is looked up by name at
/// dispatch time via ``kernel_registry()``.
///
/// # Safety
///
/// Implementations must guarantee that ``execute_typed`` only accesses
/// the provided input/output slices within their bounds.
pub trait KernelOp: Send + Sync {
    /// Unique operation name used for dispatch look-up.
    fn op_name(&self) -> &'static str;

    /// Data types this kernel supports.
    fn supported_dtypes(&self) -> &[Dtype];

    /// Execute the kernel on f32 input/output slices.
    ///
    /// # Arguments
    /// * `inputs` — read-only f32 slices, one per input buffer.
    /// * `output` — mutable f32 slice for the (first) output buffer.
    /// * `meta` — shape metadata (ranks, dims, op-specific kind/value).
    ///
    /// # Errors
    /// Returns ``HalExecutionError`` on shape mismatch, unsupported dtype,
    /// or out-of-bounds access.
    fn execute_typed(
        &self,
        inputs: &[&[f32]],
        output: &mut [f32],
        meta: &OpShapeMeta,
    ) -> Result<(), HalExecutionError>;
}

// ═══════════════════════════════════════════════════════════════════════
//  Kernel type implementations (wrapping generated *_cpu functions)
// ═══════════════════════════════════════════════════════════════════════

// ── MatmulKernel ─────────────────────────────────────────────────────

/// Wraps ``matmul_cpu`` from the generated ``hal_ops_cpu`` module.
#[derive(Debug, Clone, Default)]
pub struct MatmulKernel;

impl KernelOp for MatmulKernel {
    fn op_name(&self) -> &'static str {
        "matmul"
    }

    fn supported_dtypes(&self) -> &[Dtype] {
        &[Dtype::F32]
    }

    fn execute_typed(
        &self,
        inputs: &[&[f32]],
        output: &mut [f32],
        meta: &OpShapeMeta,
    ) -> Result<(), HalExecutionError> {
        crate::hal::hal_ops_cpu::matmul_cpu(inputs, output, meta)
            .map_err(|msg| HalExecutionError::Runner(msg))
    }
}

// ── ElementWiseKernel ────────────────────────────────────────────────

/// Wraps ``element_wise_cpu`` from the generated ``hal_ops_cpu`` module.
///
/// Delegates to ``meta.kind`` to select the concrete element-wise variant
/// (add / sub / mul / div / relu / silu / gelu / ...).
#[derive(Debug, Clone, Default)]
pub struct ElementWiseKernel;

impl KernelOp for ElementWiseKernel {
    fn op_name(&self) -> &'static str {
        "element_wise"
    }

    fn supported_dtypes(&self) -> &[Dtype] {
        &[Dtype::F32]
    }

    fn execute_typed(
        &self,
        inputs: &[&[f32]],
        output: &mut [f32],
        meta: &OpShapeMeta,
    ) -> Result<(), HalExecutionError> {
        crate::hal::hal_ops_cpu::element_wise_cpu(inputs, output, meta)
            .map_err(|msg| HalExecutionError::Runner(msg))
    }
}

// ── ReduceKernel ─────────────────────────────────────────────────────

/// Wraps ``reduce_cpu`` from the generated ``hal_ops_cpu`` module.
///
/// Supports reduction kinds via ``meta.kind`` (sum / mean / max).
#[derive(Debug, Clone, Default)]
pub struct ReduceKernel;

impl KernelOp for ReduceKernel {
    fn op_name(&self) -> &'static str {
        "reduce"
    }

    fn supported_dtypes(&self) -> &[Dtype] {
        &[Dtype::F32]
    }

    fn execute_typed(
        &self,
        inputs: &[&[f32]],
        output: &mut [f32],
        meta: &OpShapeMeta,
    ) -> Result<(), HalExecutionError> {
        crate::hal::hal_ops_cpu::reduce_cpu(inputs, output, meta)
            .map_err(|msg| HalExecutionError::Runner(msg))
    }
}

// ── SoftmaxKernel ────────────────────────────────────────────────────

/// Wraps ``softmax_cpu`` from the generated ``hal_ops_cpu`` module.
#[derive(Debug, Clone, Default)]
pub struct SoftmaxKernel;

impl KernelOp for SoftmaxKernel {
    fn op_name(&self) -> &'static str {
        "softmax"
    }

    fn supported_dtypes(&self) -> &[Dtype] {
        &[Dtype::F32]
    }

    fn execute_typed(
        &self,
        inputs: &[&[f32]],
        output: &mut [f32],
        meta: &OpShapeMeta,
    ) -> Result<(), HalExecutionError> {
        crate::hal::hal_ops_cpu::softmax_cpu(inputs, output, meta)
            .map_err(|msg| HalExecutionError::Runner(msg))
    }
}

// ── ReshapeKernel ────────────────────────────────────────────────────

/// Wraps ``reshape_cpu`` from the generated ``hal_ops_cpu`` module.
///
/// Performs a flat byte-copy (metadata-only shape change).
#[derive(Debug, Clone, Default)]
pub struct ReshapeKernel;

impl KernelOp for ReshapeKernel {
    fn op_name(&self) -> &'static str {
        "reshape"
    }

    fn supported_dtypes(&self) -> &[Dtype] {
        &[Dtype::F32]
    }

    fn execute_typed(
        &self,
        inputs: &[&[f32]],
        output: &mut [f32],
        meta: &OpShapeMeta,
    ) -> Result<(), HalExecutionError> {
        crate::hal::hal_ops_cpu::reshape_cpu(inputs, output, meta)
            .map_err(|msg| HalExecutionError::Runner(msg))
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Kernel registry
// ═══════════════════════════════════════════════════════════════════════

/// Build the canonical kernel registry.
///
/// Returns a ``HashMap`` keyed by op name, mapping each name to a
/// trait-object reference. Callers use ``registry.get(op_name)``
/// to resolve an op at dispatch time.
///
/// New kernel types are registered by adding an entry here.
pub fn kernel_registry() -> HashMap<&'static str, Box<dyn KernelOp>> {
    let mut m: HashMap<&'static str, Box<dyn KernelOp>> = HashMap::new();
    m.insert("matmul", Box::new(MatmulKernel));
    m.insert("element_wise", Box::new(ElementWiseKernel));
    m.insert("reduce", Box::new(ReduceKernel));
    m.insert("softmax", Box::new(SoftmaxKernel));
    m.insert("reshape", Box::new(ReshapeKernel));
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_has_five_kernels() {
        let reg = kernel_registry();
        assert!(reg.len() >= 5, "registry should have at least 5 kernels");
        assert!(reg.contains_key("matmul"));
        assert!(reg.contains_key("element_wise"));
        assert!(reg.contains_key("reduce"));
        assert!(reg.contains_key("softmax"));
        assert!(reg.contains_key("reshape"));
    }

    #[test]
    fn test_kernel_op_names_consistent() {
        let reg = kernel_registry();
        for (key, kernel) in reg.iter() {
            assert_eq!(
                *key, kernel.op_name(),
                "registry key '{}' must match kernel.op_name() '{}'",
                key, kernel.op_name()
            );
        }
    }

    #[test]
    fn test_kernel_supported_dtypes() {
        let reg = kernel_registry();
        for kernel in reg.values() {
            let dtypes = kernel.supported_dtypes();
            assert!(!dtypes.is_empty(), "{} must support at least one dtype", kernel.op_name());
            assert!(dtypes.contains(&Dtype::F32), "{} must support F32", kernel.op_name());
        }
    }

    #[test]
    fn test_matmul_kernel_name() {
        let k = MatmulKernel;
        assert_eq!(k.op_name(), "matmul");
    }

    #[test]
    fn test_kernel_trait_object_send_sync() {
        // Compile-time assertion: Box<dyn KernelOp> must be Send + Sync.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Box<dyn KernelOp>>();
    }
}
