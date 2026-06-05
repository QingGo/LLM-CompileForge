//! CPU Executable — dynamic library loading and typed symbol lookup.

use std::ffi::c_void;

use crate::model::error::ExecutorError;

use super::kernel::{CifaceFn1, CifaceFn2, CifaceFn3, CifaceFn4, CifaceFn5, CifaceFn6, CifaceFn7, CifaceFn8, KernelFn};
use super::super::traits;
use super::super::sfa::SfaMemRef;
use super::memref::MemRefDescAny;
use super::sret;

// ── RawCpuExecutable (low-level dylib loader) ───────────────────────────

#[derive(Debug)]
pub struct RawCpuExecutable {
    lib: libloading::Library,
}

impl RawCpuExecutable {
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
                Ok(KernelFn::HighArity(crate::model::ciface_high::FnPtr(
                    // SAFETY: transmute from *const () to fn ptr is safe here
                    // because *sym was resolved from a valid dylib symbol.
                    unsafe { std::mem::transmute::<*const (), unsafe extern "C" fn()>(*sym) }
                )))
            }
        }
    }
}

// ── CpuExecutable (high-level HAL Executable wrapper) ───────────────────

#[derive(Debug)]
pub struct CpuExecutable {
    inner: RawCpuExecutable,
    /// Cached function pointer for `serveforge_free` exported by the dylib.
    /// Eliminates per-call libloading symbol lookup and memory leak.
    free_fn: unsafe extern "C" fn(*mut c_void),
}

impl CpuExecutable {
    #[allow(dead_code)]
    pub fn inner(&self) -> &RawCpuExecutable { &self.inner }

    pub(crate) fn new(
        inner: RawCpuExecutable,
        free_fn: unsafe extern "C" fn(*mut c_void),
    ) -> Self {
        Self { inner, free_fn }
    }
}

impl traits::Executable for CpuExecutable {
    fn execute(
        &self,
        op_name: &str,
        _stream: &dyn traits::Stream,
        inputs: &[SfaMemRef],
        outputs: &mut [SfaMemRef],
    ) -> Result<Vec<Vec<i64>>, anyhow::Error> {
        // Step 1: Look up kernel by symbol name (arity = 1 sret + inputs)
        let arity = 1 + inputs.len();
        let kernel = self.inner.lookup_typed(op_name, arity)?;

        // Step 2: Construct MemRef descriptors from SfaMemRef inputs.
        // Build descriptors in the Vec FIRST, then collect stable pointers
        // from the Vec's heap allocation.  Stack-local desc pointers are
        // invalid after desc is moved into the Vec.
        let mut input_descs: Vec<MemRefDescAny> = Vec::with_capacity(inputs.len());
        let input_data_ptrs: Vec<*const u8> = inputs.iter()
            .map(|sfa| sfa.data_ptr() as *const u8)
            .collect();
        for sfa in inputs {
            input_descs.push(sfa.to_memref_desc_any());
        }
        let input_ptrs: Vec<*const c_void> = input_descs.iter()
            .map(|desc| desc.as_input_ptr())
            .collect();

        // Step 3: Allocate sret buffer for output descriptors
        const SRET_BUF_SIZE: usize = 131072;
        let mut sret: Vec<u8> = vec![0u8; SRET_BUF_SIZE];
        let sret_ptr = sret.as_mut_ptr() as *mut c_void;

        // Step 4: Build argument list and call the ciface kernel
        let mut all_args: Vec<*const c_void> = Vec::with_capacity(1 + input_ptrs.len());
        all_args.push(sret_ptr);
        all_args.extend(input_ptrs);
        let raw_ptr = kernel.as_raw_ptr();
        // SAFETY: kernel was loaded from a valid compiled .dylib.
        // sret_ptr and input_ptrs point to writable/readable buffers.
        // The kernel is a _mlir_ciface_* function that reads input descriptors
        // and writes output descriptors to the sret buffer.
        unsafe {
            crate::model::ciface_high::call_high_arity(raw_ptr, &all_args);
        }
        // Step 5: Parse sret output descriptors, copy data to output
        // buffers, collect shapes and dylib-allocated pointers to free.
        //
        // Two-pass strategy prevents use-after-free and double-free:
        //   Pass 1: parse all descriptors, copy data from dylib buffers
        //           to pre-allocated Rust output buffers.
        //   Pass 2: deduplicate allocated pointers and free them via
        //           the dylib's own serveforge_free.
        let mut output_shapes: Vec<Vec<i64>> = Vec::with_capacity(outputs.len());
        if !outputs.is_empty() {
            let mut sret_offset: usize = 0;
            let mut to_free: Vec<*mut std::ffi::c_void> =
                Vec::with_capacity(outputs.len());

            // ── Pass 1: parse + copy ──────────────────────────────
            for (oi, output_sfa) in outputs.iter().enumerate() {
                if sret_offset >= SRET_BUF_SIZE - 24 {
                    anyhow::bail!(
                        "sret overflow at output {} (offset {} >= {})",
                        oi, sret_offset, SRET_BUF_SIZE,
                    );
                }

                let out_rank = output_sfa.rank() as usize;
                if !(1..=4).contains(&out_rank) {
                    anyhow::bail!(
                        "output {}: unsupported rank {} for sret parsing",
                        oi, out_rank,
                    );
                }
                let desc_size = 24 + 16 * out_rank;
                if sret_offset + desc_size > sret.len() {
                    anyhow::bail!(
                        "sret overflow at output {} (offset {} + {} > {})",
                        oi, sret_offset, desc_size, sret.len(),
                    );
                }
                let slice = &sret[sret_offset..sret_offset + desc_size];
                // SAFETY: `read_sret_descriptor` validates the slice length
                // and non-null pointers internally; we already verified the
                // slice is within bounds above.
                let (allocated, aligned, sizes) = unsafe {
                    sret::read_sret_descriptor(slice, out_rank)?
                };
                sret_offset += desc_size;

                let n: usize = sret::checked_product_from_i64(&sizes)
                    .ok_or_else(|| anyhow::anyhow!(
                        "output {} sret sizes overflow: {:?}", oi, sizes
                    ))?;
                // When the dylib returns unresolved dynamic dimension
                // markers (negative values like -2, -3 in the sizes
                // array), checked_product clamps to 0 → n_bytes=0 →
                // no data would be copied.
                //
                // For these outputs we MUST NOT copy dylib data
                // (the dylib's malloc buffer for sentinel outputs may
                // be mis-sized, causing non-deterministic reads).
                // Instead, leave the Rust pre-allocated buffer as zeros
                // and push the RESOLVED shape from output buffer metadata
                // (which comes from the compute graph's io_def, not the
                // dylib's unreliable sret).
                //
                // This preserves the output_shapes count (so downstream
                // SSA wiring indices remain correct) while avoiding
                // non-deterministic data from the dylib's internal
                // malloc calls that used sentinel values in arithmetic.
                let has_negative = sizes.iter().any(|&s| s < 0);
                if has_negative {
                    let resolved_sizes: Vec<i64> =
                        output_sfa.sizes_i64();
                    let resolved_n: usize =
                        resolved_sizes.iter().map(|&s| s as usize).product();
                    let n_bytes = resolved_n
                        * output_sfa.element_size();
                    if n_bytes > 0 && n_bytes <= output_sfa.byte_len() {
                        let dst = output_sfa.data_ptr();
                        log::trace!(
                            "execute: output[{}] sret neg sizes {:?}, \
                             resolved={:?} n_bytes={}",
                            oi, sizes, resolved_sizes, n_bytes,
                        );
                        // SAFETY: aligned from dylib sret, dst is
                        // pre-allocated Rust buffer.
                        unsafe {
                            std::ptr::copy_nonoverlapping(aligned, dst, n_bytes);
                        }
                    }
                    output_shapes.push(resolved_sizes);
                    continue;
                }
                let n_bytes = n * 4;

                log::trace!(
                    "execute: output[{}] sret rank={} sizes={:?} n={} n_bytes={} buf_cap={} \
                      allocated={:p} aligned={:p}",
                    oi, out_rank, sizes, n, n_bytes, output_sfa.byte_len(),
                    allocated, aligned,
                );

                if n_bytes == 0 {
                    output_shapes.push(sizes);
                    continue;
                }

                if n_bytes > output_sfa.byte_len() {
                    anyhow::bail!(
                        "output {}: dylib output {} bytes exceeds buffer capacity {} bytes",
                        oi, n_bytes, output_sfa.byte_len(),
                    );
                }

                debug_assert!(
                    n_bytes <= output_sfa.byte_len(),
                    "output {}: panic on n_bytes={} > buf.len()={}",
                    oi, n_bytes, output_sfa.byte_len(),
                );

                let dst = output_sfa.data_ptr();
                // SAFETY: `aligned` points to valid dylib output data.
                // `dst` is a pre-allocated Rust buffer. Regions are disjoint.
                unsafe {
                    std::ptr::copy_nonoverlapping(aligned, dst, n_bytes);
                }

                // Track for deferred free (Pass 2). Only enqueue heap
                // pointers; skip nullptr and MLIR sentinel 0xdeadbeef
                // (global constants from memref.get_global).
                let allocated_addr = allocated as usize;
                if allocated_addr != 0 && allocated_addr != 0xdeadbeef {
                    to_free.push(allocated as *mut std::ffi::c_void);
                }

                output_shapes.push(sizes);
            }

            // ── Pass 2: dedup + free ──────────────────────────────
            //
            // Only free pointers that are verifiably heap-allocated
            // AND not owned by Rust.
            //
            // Some MemRef descriptors in the sret have `allocated`
            // pointing to non-malloc memory (stack, globals, or
            // garbage from misaligned sret parsing). Use the system
            // allocator's own validation (malloc_zone_from_ptr) to
            // distinguish safe-to-free pointers from the rest.
            //
            // CRITICAL: The dylib's ciface calls pass through weight
            // and SSA wire tensors as outputs.  Their MemRef descriptors
            // have `allocated` pointing to Rust-owned memory (weight
            // cache, func_outputs Vecs).  Freeing these causes a
            // double-free crash (SIGABRT in Arc::drop_slow) because
            // Rust's allocator detects the double-free.
            // Use input_data_ptrs to skip these pass-through pointers.
            to_free.sort();
            to_free.dedup();
            for ptr in &to_free {
                let addr = *ptr as *const std::ffi::c_void;
                if addr.is_null() {
                    continue;
                }
                // Skip if this pointer matches an input buffer — it's
                // a pass-through weight or SSA wire owned by Rust.
                let addr_u8 = addr as *const u8;
                if input_data_ptrs.contains(&addr_u8) {
                    log::trace!(
                        "execute: skip free of pass-through input ptr {:p}",
                        addr,
                    );
                    continue;
                }
                // SAFETY: malloc_zone_from_ptr is thread-safe and
                // read-only. Returns null if addr is not in any
                // malloc zone (stack, global, or invalid pointer).
                #[link(name = "System")]
                extern "C" {
                    fn malloc_zone_from_ptr(ptr: *const std::ffi::c_void) -> *const std::ffi::c_void;
                }
                // SAFETY: `malloc_zone_from_ptr` is an Apple system call that
                // is thread-safe and accepts any pointer (returns null for
                // non-zone allocations).
                let zone = unsafe { malloc_zone_from_ptr(addr) };
                if !zone.is_null() {
                    // SAFETY: `self.free_fn` is the dylib's `serveforge_free`
                    // function pointer, valid for the lifetime of the executable.
                    // `*ptr` was allocated by the dylib and is safe to free.
                    unsafe { (self.free_fn)(*ptr); }
                }
            }
        }
        Ok(output_shapes)
    }

    /// Number of dylib entry points. Returns 0 because the actual function
    /// count is stored in the compute graph (parsed via proto ABI during
    /// model loading). Callers should use `compute_graph.functions.len()`.
    fn function_count(&self) -> usize { 0 }
}
