//! HAL trait definitions — hardware abstraction layer for LLM-ServeForge.
//!
//! All backends (CPU/GPU/NPU) must implement these traits.
//! The runtime and compiler dispatch through trait objects, never calling
//! platform-specific APIs directly.
//!
//! Four core traits:
//! - ``Device`` — compute device lifecycle: alloc, compile, create_stream
//! - ``Buffer`` — device memory view: as_ptr, copy_from/to_host
//! - ``Executable`` — compiled compute module: execute, entry_count
//! - ``Stream`` — async execution ordering: synchronize

use std::fmt::Debug;

/// Compute device abstraction.
///
/// # Safety
/// Implementors must guarantee:
/// - `alloc()` returns memory valid on this device until the Buffer is dropped
/// - `compile()` returns an Executable that only requires the same device type
/// - `create_stream()` returns a Stream usable for all operations on this device
pub trait Device: Debug + Send + Sync {
    /// Allocate `size` bytes on the device.
    /// Returns uninitialized memory.
    #[allow(dead_code)]
    fn alloc(&self, size: usize) -> Result<Box<dyn Buffer>, anyhow::Error>;

    /// Create an asynchronous execution stream.
    /// Operations on the same Stream execute in FIFO order.
    /// Operations on different Streams may execute in parallel.
    #[allow(dead_code)]
    fn create_stream(&self) -> Result<Box<dyn Stream>, anyhow::Error>;

    /// Compile a serialized compute module into an executable.
    /// `module_data` contains the compiler-generated IR (SFCF binary
    /// with compute graph + constants; for CPU the .dylib path is
    /// derived from the module metadata).
    fn compile(&self, module_data: &[u8]) -> Result<Box<dyn Executable>, anyhow::Error>;

    /// Human-readable device name (for logging and debugging).
    #[allow(dead_code)]
    fn name(&self) -> &str;
}

/// Device memory buffer.
#[allow(dead_code)]
pub trait Buffer: Debug + Send + Sync {
    /// Read-only pointer to buffer data.
    /// For GPU buffers this returns a device pointer (not directly host-accessible).
    fn as_ptr(&self) -> *const u8;

    /// Mutable pointer to buffer data.
    fn as_mut_ptr(&mut self) -> *mut u8;

    /// Buffer size in bytes.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Copy data from host memory into this buffer.
    fn copy_from_host(&mut self, src: &[u8], stream: &dyn Stream) -> Result<(), anyhow::Error>;

    /// Copy data from this buffer into host memory.
    fn copy_to_host(&self, dst: &mut [u8], stream: &dyn Stream) -> Result<(), anyhow::Error>;
}

/// Compiled compute module.
///
/// An Executable wraps the complete model forward pass.
/// `execute()` is synchronous — on return, all computation is complete.
pub trait Executable: Debug + Send + Sync {
    /// Execute the compiled module with given inputs and outputs.
    /// `stream` provides execution ordering (no-op for CPU).
    /// `inputs` and `outputs` are Buffer reference arrays.
    #[allow(dead_code)]
    fn execute(
        &self,
        stream: &dyn Stream,
        inputs: &[&dyn Buffer],
        outputs: &[&dyn Buffer],
    ) -> Result<(), anyhow::Error>;

    /// Number of functions / entry points in this module.
    #[allow(dead_code)]
    fn entry_count(&self) -> usize;
}

/// Asynchronous execution stream.
#[allow(dead_code)]
pub trait Stream: Debug + Send + Sync {
    /// Wait for all operations on this stream to complete.
    /// CPU implementation is a no-op.
    fn synchronize(&self) -> Result<(), anyhow::Error>;
}
