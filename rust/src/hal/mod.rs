//! Hardware Abstraction Layer (HAL) module.
//!
//! Provides backend-independent interfaces and CPU backend implementation.
//! Future backends: ``pub mod cuda; pub mod metal; pub mod npu;``

pub mod traits;
pub mod cpu;

#[cfg(feature = "hal-rust")]
pub mod rust;

#[cfg(feature = "hal-rust")]
#[path = "../../../compiled/opt_125m_kv/generated/hal_ops_cpu.rs"]
pub mod hal_ops_cpu;
