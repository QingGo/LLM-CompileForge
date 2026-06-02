//! High-arity support for MLIR ciface functions using libffi.

use std::ffi::c_void;

use libffi::middle::{Cif, CodePtr, Type};

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
/// Uses libffi to push the correct number of args on the stack/registers.
/// Flat arg list: first arg is sret buffer, remaining are input descriptors.
pub unsafe fn call_high_arity(
    fn_ptr: *const (),
    all_args: &[*const c_void],
) {
    let n = all_args.len();
    if n == 0 || n > 300 {
        panic!(
            "call_high_arity: unsupported arity {} (function pointer: {:p}, expected 1-300)",
            n, fn_ptr,
        );
    }

    let cif = Cif::new(
        std::iter::repeat_n(Type::pointer(), n),
        Type::void(),
    );

    let args: Vec<libffi::middle::Arg> = all_args.iter().map(|p| {
        libffi::middle::Arg::new(p)
    }).collect();

    cif.call_return_into(CodePtr(fn_ptr as *mut c_void), &args, libffi::middle::Ret::void());
}
