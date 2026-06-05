//! HAL trait definitions — hardware abstraction layer for LLM-ServeForge.
//!
//! All backends (CPU/GPU/NPU) must implement these traits.
//! The runtime and compiler dispatch through trait objects, never calling
//! platform-specific APIs directly.
//!
//! Five core traits:
//! - ``Device`` — compute device lifecycle: alloc, compile, create_stream, create_event
//! - ``Buffer`` — device memory view: as_ptr, copy_from/to_host
//! - ``Executable`` — compiled compute module: execute(op_name, …), function_count
//! - ``Stream`` — async execution ordering: synchronize, wait_event, record_event
//! - ``Event`` — synchronization primitive: is_complete, synchronize

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

    /// Create a synchronization event.
    #[allow(dead_code)]
    fn create_event(&self) -> Result<Box<dyn Event>, anyhow::Error>;

    /// Compile a serialized compute module into an executable.
    /// For CPU backends, `module_data` is the .dylib file path as UTF-8.
    fn compile(&self, module_data: &[u8]) -> Result<Box<dyn Executable>, anyhow::Error>;

    /// Human-readable device name (for logging and debugging).
    #[allow(dead_code)]
    fn name(&self) -> &str;
}

/// Device memory buffer.
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

    /// Element size in bytes. Default is 4 (f32).
    /// Override for non-f32 inputs (e.g. 8 for i64 GlobalInputs).
    fn element_size(&self) -> usize {
        4
    }

    /// Logical shape of the buffer contents.
    /// Default is rank-1 with element_count elements.
    fn shape(&self) -> Vec<usize> {
        let n = self.len() / self.element_size();
        vec![n]
    }

    /// Tensor rank for MemRef descriptor construction.
    /// Default is 1 (rank-1 descriptor).
    fn rank(&self) -> u8 {
        1
    }

    /// Convert this buffer into a unified ``SfaMemRef`` descriptor.
    fn as_sfa_memref(&self) -> super::sfa::SfaMemRef {
        let ptr = self.as_ptr() as *mut std::ffi::c_void;
        let shape = self.shape();
        let elem_size = self.element_size();
        super::sfa::SfaMemRef::from_shape(ptr, &shape, elem_size)
            .unwrap_or_else(|_| {
                let n = self.len() / elem_size.max(1);
                super::sfa::SfaMemRef::r1(ptr, [n as i64], [1], elem_size)
            })
    }
}

/// Compiled compute module.
///
/// An Executable wraps the complete model forward pass.
/// `execute()` dispatches to a named operation — the runtime selects
/// which kernel/function to run via `op_name`.
pub trait Executable: Debug + Send + Sync {
    /// Execute a named operation with given inputs and outputs.
    /// `op_name` identifies which kernel/function to run.
    /// `stream` provides execution ordering (no-op for CPU).
    /// `inputs` and `outputs` are SfaMemRef descriptor arrays.
    /// Returns the actual output shapes (sizes) for each output.
    fn execute(
        &self,
        op_name: &str,
        stream: &dyn Stream,
        inputs: &[super::sfa::SfaMemRef],
        outputs: &mut [super::sfa::SfaMemRef],
    ) -> Result<Vec<Vec<i64>>, anyhow::Error>;

    /// Number of functions / entry points in this module.
    fn function_count(&self) -> usize;

    /// List of operation names this executable supports.
    /// Default returns empty slice (unknown/unconstrained).
    fn supported_ops(&self) -> &[&str] {
        &[]
    }

}

/// Synchronization event.
///
/// An Event tracks operation completion on a Stream.
/// CPU implementation is a no-op (CPU is synchronous).
#[allow(dead_code)]
pub trait Event: Debug + Send + Sync {
    /// Check whether the event has completed.
    fn is_complete(&self) -> bool;

    /// Block until the event completes.
    fn synchronize(&self) -> Result<(), anyhow::Error>;
}

/// Asynchronous execution stream.
///
/// Operations on the same Stream execute in FIFO order.
/// Operations on different Streams may execute in parallel.
/// CPU implementation is a no-op (CPU is synchronous).
#[allow(dead_code)]
pub trait Stream: Debug + Send + Sync {
    /// Wait for all operations on this stream to complete.
    fn synchronize(&self) -> Result<(), anyhow::Error>;

    /// Make future operations on this stream wait until the event completes.
    fn wait_event(&self, event: &dyn Event) -> Result<(), anyhow::Error>;

    /// Record that all prior operations on this stream have completed.
    /// The event is signalled when all preceding work is done.
    fn record_event(&self, event: &dyn Event) -> Result<(), anyhow::Error>;
}
