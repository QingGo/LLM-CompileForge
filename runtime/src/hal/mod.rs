//! Hardware Abstraction Layer (HAL) module.
//!
//! Provides backend-independent interfaces and CPU backend implementation.
//! Future backends: ``pub mod cuda; pub mod metal; pub mod npu;``

pub mod traits;
pub mod sfa;
pub mod cpu;

#[cfg(feature = "hal-rust")]
pub mod rust;

#[cfg(feature = "hal-rust")]
pub mod primitives;

#[cfg(feature = "hal-rust")]
// To switch models: set HAL_OPS_CPU_PATH to the target model's
// generated/ directory, then recompile with `cargo build --features hal-rust`.
// Example: HAL_OPS_CPU_PATH=<model_output_dir>/generated cargo build --features hal-rust
pub mod hal_ops_cpu {
    #![allow(clippy::excessive_precision, clippy::get_first, clippy::len_zero, clippy::manual_memcpy, clippy::needless_range_loop, clippy::manual_is_multiple_of)]
    #![allow(unused_variables, dead_code)]
    include!(concat!(env!("HAL_OPS_CPU_PATH"), "/hal_ops_cpu.rs"));
}
