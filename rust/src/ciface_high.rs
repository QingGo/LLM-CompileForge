/// High-arity support for MLIR ciface functions using libffi.

use std::ffi::c_void;

use libffi::middle::{Cif, CodePtr, Type};

/// A raw MLIR ciface function pointer for arbitrary arity.
#[derive(Clone, Copy)]
pub struct FnPtr(pub unsafe extern "C" fn());

/// Call an MLIR ciface function with a dynamic number of pointer arguments.
///
/// Uses libffi to push the correct number of args on the stack/registers.
/// Flat arg list: first arg is sret buffer, remaining are input descriptors.
pub unsafe fn call_high_arity(
    fn_ptr: *const (),
    all_args: &[*const c_void],
) {
    let n = all_args.len();
    if n == 0 || n > 300 {
        panic!("unsupported arity: {n}");
    }

    // Build CIF for n pointer args and void return
    let cif = Cif::new(
        std::iter::repeat(Type::pointer()).take(n),
        Type::void(),
    );

    let args: Vec<libffi::middle::Arg> = all_args.iter().map(|p| {
        libffi::middle::Arg::new(p)
    }).collect();

    // Call using libffi (void return)
    cif.call_return_into(CodePtr(fn_ptr as *mut c_void), &args, libffi::middle::Ret::void());
}
