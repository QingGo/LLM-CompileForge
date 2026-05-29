//! Hardware Abstraction Layer (HAL) module.
//!
//! Provides backend-independent interfaces and CPU backend implementation.
//! Future backends: ``pub mod cuda; pub mod metal; pub mod npu;``

pub mod traits;
pub mod cpu;

#[cfg(feature = "hal-rust")]
pub mod rust;

#[cfg(feature = "hal-rust")]
#[path = "../../../compiled/opt_125m_fresh/generated/hal_ops_cpu.rs"]
#[allow(clippy::excessive_precision, clippy::get_first, clippy::len_zero, clippy::manual_memcpy, clippy::needless_range_loop, clippy::manual_is_multiple_of)]
pub mod hal_ops_cpu;
