//! CPU Executable — dynamic library loading and typed symbol lookup.

use crate::error::ExecutorError;

use super::kernel::{CifaceFn1, CifaceFn2, CifaceFn3, CifaceFn4, CifaceFn5, CifaceFn6, CifaceFn7, CifaceFn8, KernelFn};

#[derive(Debug)]
pub struct CpuExecutable {
    lib: libloading::Library,
}

impl CpuExecutable {
    /// Load a compiled .dylib.
    ///
    /// # Safety
    /// The caller must ensure `path` refers to a valid dynamic library.
    /// `libloading::Library::new` is unsafe because it may execute
    /// initializers in the loaded library.
    pub fn load(path: &str) -> Result<Self, anyhow::Error> {
        // SAFETY: The caller guarantees the path points to a trusted
        // compiled model .dylib. We control the compilation pipeline,
        // so the dylib has no malicious initializers.
        let lib = unsafe { libloading::Library::new(path)? };
        Ok(Self { lib })
    }

    pub fn lib(&self) -> &libloading::Library {
        &self.lib
    }

    /// Look up a raw symbol by name (for sret-based ciface calls).
    #[allow(dead_code)]
    pub fn lookup_symbol(
        &self,
        name: &str,
    ) -> Result<unsafe extern "C" fn(), anyhow::Error> {
        let sym: libloading::Symbol<unsafe extern "C" fn()> = unsafe {
            self.lib.get(name.as_bytes())
        }?;
        Ok(*sym)
    }

    /// Read the embedded constants data (serveforge_constants_data/size)
    /// from the loaded dynamic library. This is the raw binary blob
    /// containing the weight registry, compute graph, and contract.
    pub fn load_constants(&self) -> Result<Vec<u8>, anyhow::Error> {
        let data_ptr: *const u8 = {
            // SAFETY: Symbol lookup from a loaded dylib returns a valid pointer
            // if the dylib exports the named symbol.
            let sym: libloading::Symbol<*const std::ffi::c_void> = unsafe {
                self.lib.get(b"serveforge_constants_data")
                    .map_err(|e| anyhow::anyhow!("missing serveforge_constants_data: {}", e))?
            };
            *sym as *const u8
        };
        let size_val: u64 = {
            // SAFETY: Same as above — symbol lookup from a loaded dylib.
            let sym = unsafe {
                self.lib.get::<*const u64>(b"serveforge_constants_size")
                    .map_err(|e| anyhow::anyhow!("missing serveforge_constants_size: {}", e))?
            };
            // SAFETY: Dereferencing a symbol pointer that was just successfully
            // looked up from the dylib.
            unsafe { *(*sym) }
        };
        // SAFETY: data_ptr and size_val come from the dylib's exported constants.
        // The dylib guarantees the data region is at least size_val bytes.
        let data: &[u8] =
            unsafe { std::slice::from_raw_parts(data_ptr, size_val as usize) };
        Ok(data.to_vec())
    }

    /// Look up a kernel function by symbol name with a specific arity.
    pub fn lookup_typed(
        &self,
        name: &str,
        arity: usize,
    ) -> Result<KernelFn, anyhow::Error> {
        if arity > 300 {
            return Err(ExecutorError::KernelArityMismatch {
                expected: 300, actual: arity,
            }.into());
        }
        match arity {
            // SAFETY: libloading::get() looks up typed symbols valid for self.lib's lifetime.
            1 => { let sym: libloading::Symbol<CifaceFn1> = unsafe { self.lib.get(name.as_bytes()) }?; Ok(KernelFn::Arity1(*sym)) }
            // SAFETY: Same as above, arity 2.
            2 => { let sym: libloading::Symbol<CifaceFn2> = unsafe { self.lib.get(name.as_bytes()) }?; Ok(KernelFn::Arity2(*sym)) }
            // SAFETY: Same as above, arity 3.
            3 => { let sym: libloading::Symbol<CifaceFn3> = unsafe { self.lib.get(name.as_bytes()) }?; Ok(KernelFn::Arity3(*sym)) }
            // SAFETY: Same as above, arity 4.
            4 => { let sym: libloading::Symbol<CifaceFn4> = unsafe { self.lib.get(name.as_bytes()) }?; Ok(KernelFn::Arity4(*sym)) }
            // SAFETY: Same as above, arity 5.
            5 => { let sym: libloading::Symbol<CifaceFn5> = unsafe { self.lib.get(name.as_bytes()) }?; Ok(KernelFn::Arity5(*sym)) }
            // SAFETY: Same as above, arity 6.
            6 => { let sym: libloading::Symbol<CifaceFn6> = unsafe { self.lib.get(name.as_bytes()) }?; Ok(KernelFn::Arity6(*sym)) }
            // SAFETY: Same as above, arity 7.
            7 => { let sym: libloading::Symbol<CifaceFn7> = unsafe { self.lib.get(name.as_bytes()) }?; Ok(KernelFn::Arity7(*sym)) }
            // SAFETY: Same as above, arity 8.
            8 => { let sym: libloading::Symbol<CifaceFn8> = unsafe { self.lib.get(name.as_bytes()) }?; Ok(KernelFn::Arity8(*sym)) }
              _ => {
                // SAFETY: libloading::get() looks up a raw symbol; transmute
                // is safe because the symbol was just successfully resolved.
                let sym: libloading::Symbol<*const ()> = unsafe { self.lib.get(name.as_bytes()) }?;
                Ok(KernelFn::HighArity(crate::ciface_high::FnPtr(
                    // SAFETY: transmute from *const () to fn ptr is safe here
                    // because *sym was resolved from a valid dylib symbol.
                    unsafe { std::mem::transmute::<*const (), unsafe extern "C" fn()>(*sym) }
                )))
            }
        }
    }
}
