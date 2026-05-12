//! Model executor — loads compiled .dylib and orchestrates inference.
//!
//! The executor is responsible for:
//! 1. Loading the compiled per-function .dylib.
//! 2. Reading the embedded weight registry (name mapping + constants).
//! 3. Managing KV cache blocks via the Rust ``BlockManager``.
//! 4. Assembling the inference loop: embed → per-function forward → sample.

use crate::hal_cpu::{Buffer, Device, Executable, KernelFn};
use crate::weight_loader::WeightProvider;

/// One compiled model function loaded from the .dylib.
#[allow(dead_code)]
pub struct CompiledFunction {
    pub name: String,
    pub func: KernelFn,
    pub num_inputs: usize,
    pub num_outputs: usize,
}

/// The model executor.
pub struct ModelExecutor {
    device: Device,
    #[allow(dead_code)]
    executable: Executable,
    pub functions: Vec<CompiledFunction>,
    pub weight_provider: Option<WeightProvider>,
}

impl ModelExecutor {
    /// Load a compiled model from a .dylib path.
    ///
    /// The .dylib contains the compiled compute functions and embedded
    /// weight registry (name mapping + constants).  Optionally, an
    /// external HF safetensors file provides the original model weights.
    pub fn load(
        dylib_path: &str,
        safetensors_path: Option<&str>,
    ) -> Result<Self, anyhow::Error> {
        let executable = Executable::load(dylib_path)?;
        let lib = unsafe { libloading::Library::new(dylib_path)? };
        let registry = crate::weight_loader::load_registry_from_dylib(&lib)?;

        let weight_provider = {
            let st_path = safetensors_path.map(std::path::Path::new);
            Some(WeightProvider::new(registry, st_path)?)
        };

        Ok(Self {
            device: Device::new(),
            executable,
            functions: Vec::new(),
            weight_provider,
        })
    }

    /// Register a compiled function from the .dylib by name.
    #[allow(dead_code)]
    pub fn register_function(
        &mut self,
        name: &str,
        num_inputs: usize,
        num_outputs: usize,
    ) -> Result<(), anyhow::Error> {
        let func = self.executable.lookup(name)?;
        self.functions.push(CompiledFunction {
            name: name.to_string(),
            func,
            num_inputs,
            num_outputs,
        });
        Ok(())
    }

    /// Allocate a host-side buffer of the given byte size.
    #[allow(dead_code)]
    pub fn allocate(&mut self, size: usize) -> Buffer {
        self.device.allocate(size)
    }

    /// Return total bytes allocated.
    #[allow(dead_code)]
    pub fn allocated_bytes(&self) -> usize {
        self.device.total_allocated()
    }
}
