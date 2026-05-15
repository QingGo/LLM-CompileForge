/// High-arity support for MLIR ciface functions using libffi.

use std::ffi::c_void;

use libffi::middle::{Cif, CodePtr, Type};

/// A raw MLIR ciface function pointer for arbitrary arity.
#[derive(Clone, Copy)]
pub struct FnPtr(pub unsafe extern "C" fn());

/// Call an MLIR ciface function with a dynamic number of pointer arguments.
///
/// Uses libffi to push the correct number of args on the stack/registers.
pub unsafe fn call_high_arity(fn_ptr: *const (), out: *mut c_void, inputs: &[*const c_void]) {
    let n = inputs.len();
    if n == 0 || n > 300 {
        panic!("unsupported arity: {n}");
    }

    // Build CIF for (n+1) pointer args and void return
    let cif = Cif::new(
        std::iter::repeat(Type::pointer()).take(n + 1),
        Type::void(),
    );

    // Build args on the stack (Vec backed by heap) — all are *const c_void sized
    let out_ptr: *const c_void = out as *const c_void;
    let mut args: Vec<libffi::middle::Arg> = Vec::with_capacity(n + 1);
    args.push(libffi::middle::Arg::new(&out_ptr));
    for inp in inputs.iter() {
        args.push(libffi::middle::Arg::new(inp));
    }

    // Call using libffi (void return)
    cif.call_return_into(CodePtr(fn_ptr as *mut c_void), &args, libffi::middle::Ret::void());
}
