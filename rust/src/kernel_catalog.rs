use std::fmt::Debug;

use crate::compute_graph::ComputeGraph;
#[allow(unused_imports)]
use crate::executor::ModelExecutor;

/// Result of kernel selection — which kernel to use for this batch config.
#[derive(Debug, Clone)]
pub struct KernelSelection {
    /// Symbol name in the compiled .dylib
    pub symbol: String,
    /// Expected input shape for this kernel (rank-matched to ComputeGraph)
    pub expected_shape: Vec<Option<usize>>,
}

/// Catalog of available compiled kernels.
///
/// In Phase 0, this is always a single dynamic dylib that handles all shapes.
/// Future: multi-shape AOT kernels selected by batch/seq pattern.
pub trait KernelCatalog: Debug + Send + Sync {
    /// Select the best kernel for a given input configuration.
    fn select_kernel(
        &self,
        input_shape: &[usize],
        compute_graph: &ComputeGraph,
    ) -> Result<KernelSelection, anyhow::Error>;

    /// Number of available kernel variants.
    fn num_variants(&self) -> usize;
}

/// Default implementation: single dynamic dylib that handles all shapes.
#[derive(Debug)]
pub struct DynamicDylib;

impl KernelCatalog for DynamicDylib {
    fn select_kernel(
        &self,
        _input_shape: &[usize],
        _compute_graph: &ComputeGraph,
    ) -> Result<KernelSelection, anyhow::Error> {
        // Always returns the compiled dynamic kernel (symbol from ComputeGraph)
        Ok(KernelSelection {
            symbol: "_mlir_ciface_main_0".to_string(),
            expected_shape: vec![None, None], // fully dynamic
        })
    }

    fn num_variants(&self) -> usize {
        1
    }
}
