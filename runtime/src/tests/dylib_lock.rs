//! Serializes tests that load compiled `.dylib` files.
//!
//! On macOS, dylibs are loaded into a flat namespace: two model dylibs
//! (e.g. `opt_125m_fresh` and `opt_125m_kv`) export overlapping symbols
//! (`_mlir_ciface_main_*`, `sfa_abi`, ...). Loading or using them
//! concurrently in one test process can interpose symbols across
//! handles and cause SIGSEGV or nondeterministic results. Tests that
//! touch a compiled dylib must hold this lock for the whole duration
//! of their use:
//!
//! ```ignore
//! let _dylib_guard = crate::tests::dylib_lock::lock();
//! let executor = compiled_executor();
//! ```
//!
//! Non-dylib tests continue to run in parallel.

use std::sync::{Mutex, MutexGuard};

static DYLIB_TEST_LOCK: Mutex<()> = Mutex::new(());

pub fn lock() -> MutexGuard<'static, ()> {
    DYLIB_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}
