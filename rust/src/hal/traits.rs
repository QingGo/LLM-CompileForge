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
    /// `inputs` and `outputs` are Buffer reference arrays.
    /// Returns the actual output shapes (sizes) for each output.
    #[allow(dead_code)]
    fn execute(
        &self,
        op_name: &str,
        stream: &dyn Stream,
        inputs: &[&dyn Buffer],
        outputs: &[&dyn Buffer],
    ) -> Result<Vec<Vec<i64>>, anyhow::Error>;

    /// Number of functions / entry points in this module.
    #[allow(dead_code)]
    fn function_count(&self) -> usize;

    /// Return the embedded constants data (serveforge_constants_data/size)
    /// as a byte slice. This is the raw binary blob containing weight
    /// registry, compute graph, and contract metadata.
    #[allow(dead_code)]
    fn module_data(&self) -> &[u8];

    /// List of operation names this executable supports.
    /// Default returns empty slice (unknown/unconstrained).
    fn supported_ops(&self) -> &[&str] {
        &[]
    }

    /// Register an expert kernel for a named operation.
    /// Expert kernels override the default execution path for specific ops.
    /// Default implementation returns an error — backends with expert
    /// kernel support must override.
    fn register_expert_kernel(
        &mut self,
        op_name: &str,
        kernel: Box<dyn ExpertKernel>,
    ) -> Result<(), anyhow::Error> {
        anyhow::bail!("register_expert_kernel not supported by this backend")
    }
}

/// An expert kernel — a specialized implementation for a single operation.
///
/// Expert kernels allow backends to register optimized or hardware-specific
/// implementations for individual ops, overriding the default compiled
/// kernel path.
pub trait ExpertKernel: Debug + Send + Sync {
    /// Execute this expert kernel with the given inputs and outputs.
    fn execute(
        &self,
        stream: &dyn Stream,
        inputs: &[&dyn Buffer],
        outputs: &[&dyn Buffer],
    ) -> Result<(), anyhow::Error>;
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
