//! CPU backend — top-level module.
//!
//! Sub-modules:
//! - ``device`` — ``CpuDevice`` (HAL Device impl), ``CpuStream``, ``CpuEvent``
//! - ``buffer`` — ``RawBuffer`` (raw allocator-backed buffer) + ``CpuBuffer`` (HAL Buffer wrapper)
//! - ``executable`` — ``RawCpuExecutable`` (dylib loader) + ``CpuExecutable`` (HAL Executable wrapper)
//! - ``kernel``, ``memref``, ``sret`` — MemRef descriptor helpers
//!
//! Re-exports ``CpuDevice``, ``CpuStream``, ``CpuEvent``, ``CpuExecutable``,
//! ``CpuBuffer``, and MemRef descriptor types for use by the rest of the crate.

pub mod buffer;
pub mod device;
pub mod executable;
pub mod kernel;
pub mod memref;
pub mod sret;

// ── Re-exports (used by executor.rs, weight_loader.rs, device.rs) ─

#[allow(unused_imports)]
pub(crate) use buffer::CpuBuffer;
#[allow(unused_imports)]
pub use device::{CpuDevice, CpuEvent, CpuStream};
#[allow(unused_imports)]
pub use executable::CpuExecutable;
#[allow(unused_imports)]
pub use executable::CpuExecutable as Executable;
pub use memref::MemRefDesc2;
pub(crate) use memref::MemRefDescAny;
pub use sret::read_sret_descriptor;

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "../../tests/cpu_tests.rs"]
mod tests;
