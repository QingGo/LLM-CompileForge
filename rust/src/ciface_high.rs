/// High-arity support for MLIR ciface functions using libffi.

use std::ffi::c_void;

use libffi::middle::{Cif, CodePtr, Type};

/// A raw MLIR ciface function pointer for arbitrary arity.
#[derive(Clone, Copy)]
pub struct FnPtr(pub unsafe extern "C" fn());

/// Call an MLIR ciface function with a dynamic number of pointer arguments.
///
/// Uses libffi to push the correct number of args on the stack/registers.
/// The ciface convention is: (out0, out1, ..., outN, in0, in1, ..., inM).
pub unsafe fn call_high_arity(
    fn_ptr: *const (),
    outputs: &[*mut c_void],
    inputs: &[*const c_void],
) {
    let total_args = outputs.len() + inputs.len();
    if total_args == 0 || total_args > 300 {
        panic!("unsupported arity: {total_args} (outputs={}, inputs={})",
               outputs.len(), inputs.len());
    }

    // Build CIF for total_args pointer args and void return
    let cif = Cif::new(
        std::iter::repeat(Type::pointer()).take(total_args),
        Type::void(),
    );

    // Build args on the stack: all outputs first, then all inputs
    let mut ptrs: Vec<*const c_void> = Vec::with_capacity(total_args);
    for o in outputs.iter() {
        ptrs.push(*o as *const c_void);
    }
    for inp in inputs.iter() {
        ptrs.push(*inp);
    }
    let args: Vec<libffi::middle::Arg> = ptrs.iter().map(|p| {
        libffi::middle::Arg::new(p)
    }).collect();

    // Call using libffi (void return)
    cif.call_return_into(CodePtr(fn_ptr as *mut c_void), &args, libffi::middle::Ret::void());
}
