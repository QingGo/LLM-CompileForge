//! High-arity support for MLIR ciface functions using exact-arity C trampolines.

use std::ffi::c_void;

include!(concat!(env!("OUT_DIR"), "/ciface_gen.rs"));

/// A raw MLIR ciface function pointer for arbitrary arity.
#[derive(Clone, Copy)]
pub struct FnPtr(pub unsafe extern "C" fn());

/// Call an MLIR ciface function with a dynamic number of pointer arguments.
///
/// # Safety
///
/// - `fn_ptr` must be a valid function pointer to an MLIR ciface entry point.
/// - `all_args` must contain the correct number and types of arguments for
///   the function. The first argument must be the sret (struct return) buffer,
///   and the remaining arguments are input descriptors.
/// - All pointers in `all_args` must be valid for the duration of the call.
///
/// Flat arg list: first arg is sret buffer, remaining are input descriptors.
pub unsafe fn call_high_arity(fn_ptr: *const (), all_args: &[*const c_void]) {
    let n = all_args.len();
    if n == 0 || n > 512 {
        panic!(
            "call_high_arity: unsupported arity {} (function pointer: {:p}, expected 1-512)",
            n, fn_ptr,
        );
    }

    // call_n dispatches to a generated exact-arity C trampoline for
    // 1..=512 pointer arguments. The first element of `all_args` is the
    // writable sret buffer and the rest are valid input descriptor pointers
    // for the duration of the call.
    //
    // SAFETY: all caller preconditions documented on this function are met.
    unsafe { call_n(fn_ptr, all_args[0] as *mut c_void, &all_args[1..]) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::sync::atomic::{AtomicU64, Ordering};

    static PROBE_COUNTER: AtomicU64 = AtomicU64::new(0);

    struct ProbeArtifacts {
        dylib: PathBuf,
        source: PathBuf,
    }

    impl Drop for ProbeArtifacts {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.dylib);
            let _ = std::fs::remove_file(&self.source);
        }
    }

    fn probe_c_source() -> String {
        let mut params = String::from("    void *sret");
        let mut checks = String::new();
        for i in 1..=152usize {
            params.push_str(&format!(",\n    void *a{i:03}"));
            checks.push_str(&format!(
                "    if (a{i:03} != (void *)((uintptr_t)sret + {i})) bad = {i};\n"
            ));
        }
        format!(
            r#"/* Synthetic arity-153 MLIR-style ciface probe.
 * Total arguments: sret + 152 pointer args = 153.
 * The probe never dereferences the pointer args; it verifies that each one
 * arrived at the expected position and writes the result into sret.
 */
#include <stdint.h>

void _mlir_ciface_arity153_probe(
{params}
) {{
    uint64_t *out = (uint64_t *)sret;
    int bad = 0;
{checks}
    out[0] = (uint64_t)bad;
    out[1] = (uint64_t)0x513153;
}}
"#
        )
    }

    fn build_probe_dylib() -> (ProbeArtifacts, libloading::Library) {
        let source = std::env::temp_dir().join(format!(
            "llm_serveforge_arity153_probe_{}_{}.c",
            std::process::id(),
            PROBE_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&source, probe_c_source()).expect("write arity153 probe source");

        let ext = if cfg!(target_os = "macos") {
            "dylib"
        } else {
            "so"
        };
        let dylib = source.with_extension(ext);
        let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
        let status = Command::new(cc)
            .arg("-O0")
            .arg("-fPIC")
            .arg("-shared")
            .arg("-o")
            .arg(&dylib)
            .arg(&source)
            .status()
            .expect("run C compiler for arity153 probe");
        assert!(status.success(), "failed to compile arity153 probe dylib");

        // SAFETY: the dylib was just compiled from the source above and only
        // exports a harmless probe function.
        let lib = unsafe { libloading::Library::new(&dylib) }.expect("load arity153 probe");
        (ProbeArtifacts { dylib, source }, lib)
    }

    fn call_probe_three_times(fn_ptr: *const ()) {
        let mut sret: Vec<u64> = vec![0, 0, 0, 0];
        for round in 0..3 {
            let base = sret.as_mut_ptr() as usize;
            let mut args: Vec<*const c_void> = Vec::with_capacity(153);
            args.push(sret.as_mut_ptr() as *const c_void);
            for i in 1..153usize {
                args.push((base + i) as *const c_void);
            }

            // SAFETY: `fn_ptr` is the arity-153 probe exported by the freshly
            // built dylib; args contains the writable sret followed by 152
            // valid probe pointers. The probe never dereferences them.
            unsafe { call_high_arity(fn_ptr, &args) };

            assert_eq!(
                sret[0], 0,
                "arity153 probe reported wrong argument placement at {round}: {}",
                sret[0]
            );
            assert_eq!(
                sret[1], 0x513153,
                "arity153 probe magic mismatch at {round}"
            );

            // Churn the Rust allocator between calls. This is the same pattern
            // that exposed the old libffi TypeArray/Cif drop corruption.
            let mut churn: Vec<Vec<u8>> = Vec::with_capacity(128);
            for i in 0..128 {
                let mut block: Vec<u8> = Vec::with_capacity(64 + i * 97);
                block.resize(64 + i * 97, i as u8);
                churn.push(block);
            }
            drop(churn);
        }
    }

    /// Synthetic arity-153 ciface regression: three consecutive high-arity
    /// calls must be ABI-correct and leave the heap intact for later
    /// allocations/frees.
    #[test]
    fn test_arity153_three_calls_no_corruption() {
        let (_guard, lib) = build_probe_dylib();
        // SAFETY: the golden probe source exports this exact symbol.
        let sym: libloading::Symbol<*const ()> =
            unsafe { lib.get(b"_mlir_ciface_arity153_probe") }.expect("probe symbol");
        let fn_ptr = *sym;
        call_probe_three_times(fn_ptr);
        drop(lib);
    }

    #[test]
    fn test_call_high_arity_rejects_out_of_range() {
        for n in [0usize, 513] {
            let args = vec![std::ptr::null::<c_void>(); n];
            let result = std::panic::catch_unwind(|| {
                // SAFETY: call_high_arity rejects this arity before touching
                // fn_ptr or the argument pointers.
                unsafe { call_high_arity(std::ptr::null(), &args) };
            });
            assert!(result.is_err(), "arity {n} should be rejected");
        }
    }
}
