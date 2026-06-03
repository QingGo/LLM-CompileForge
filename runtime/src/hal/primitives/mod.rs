//! Primitive operations for HAL CPU kernels.
//!
//! These are the building blocks for both direct dispatch and fused kernels.
//! Each primitive is a standalone, testable function.
//!
//! Submodules:
//! - ``traits``  — ``KernelOp`` trait, kernel type wrappers, and the registry
//!   that replaces string-based dispatch.

pub mod traits;
pub mod vec_ops;
pub mod reduce_ops;
pub mod matmul;
pub mod gather;
pub mod transpose;
pub mod fused_softmax;
pub mod fused_layer_norm;
pub mod fused_sdpa;

pub use vec_ops::*;
pub use reduce_ops::*;
pub use matmul::*;
pub use gather::*;
pub use transpose::*;
pub use fused_softmax::*;
pub use fused_layer_norm::*;
pub use fused_sdpa::*;
