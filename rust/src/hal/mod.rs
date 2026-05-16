//! Hardware Abstraction Layer (HAL) module.
//!
//! Provides backend-independent interfaces and CPU backend implementation.
//! Future backends: ``pub mod cuda; pub mod metal; pub mod npu;``

pub mod traits;
pub mod cpu;
