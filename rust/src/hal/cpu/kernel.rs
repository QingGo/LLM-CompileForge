//! Kernel dispatch — type-erased function pointers and arity dispatch.

use std::ffi::c_void;

// ── Typed ciface function pointers ─────────────────────────────────

pub type CifaceFn1 = unsafe extern "C" fn(*mut c_void);
pub type CifaceFn2 = unsafe extern "C" fn(*mut c_void, *const c_void);
pub type CifaceFn3 = unsafe extern "C" fn(*mut c_void, *const c_void, *const c_void);
pub type CifaceFn4 =
    unsafe extern "C" fn(*mut c_void, *const c_void, *const c_void, *const c_void);
pub type CifaceFn5 = unsafe extern "C" fn(
    *mut c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *const c_void,
);
pub type CifaceFn6 = unsafe extern "C" fn(
    *mut c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *const c_void,
);
pub type CifaceFn7 = unsafe extern "C" fn(
    *mut c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *const c_void,
);
pub type CifaceFn8 = unsafe extern "C" fn(
    *mut c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *const c_void,
);

// ── Kernel function enum ──────────────────────────────────────────

pub enum KernelFn {
    Arity1(CifaceFn1),
    Arity2(CifaceFn2),
    Arity3(CifaceFn3),
    Arity4(CifaceFn4),
    Arity5(CifaceFn5),
    Arity6(CifaceFn6),
    Arity7(CifaceFn7),
    Arity8(CifaceFn8),
    HighArity(crate::ciface_high::FnPtr),
}

impl KernelFn {
    pub fn as_raw_ptr(&self) -> *const () {
        match self {
            KernelFn::Arity1(f) => *f as *const (),
            KernelFn::Arity2(f) => *f as *const (),
            KernelFn::Arity3(f) => *f as *const (),
            KernelFn::Arity4(f) => *f as *const (),
            KernelFn::Arity5(f) => *f as *const (),
            KernelFn::Arity6(f) => *f as *const (),
            KernelFn::Arity7(f) => *f as *const (),
            KernelFn::Arity8(f) => *f as *const (),
            KernelFn::HighArity(f) => f.0 as *const (),
        }
    }

    #[allow(dead_code)]
    pub fn arity(&self) -> usize {
        match self {
            KernelFn::Arity1(_) => 1,
            KernelFn::Arity2(_) => 2,
            KernelFn::Arity3(_) => 3,
            KernelFn::Arity4(_) => 4,
            KernelFn::Arity5(_) => 5,
            KernelFn::Arity6(_) => 6,
            KernelFn::Arity7(_) => 7,
            KernelFn::Arity8(_) => 8,
            KernelFn::HighArity(_) => 0,
        }
    }

    /// Call with sret convention: first arg is sret buffer, rest are inputs.
    ///
    /// # Safety
    ///
    /// - `fn_ptr` must be a valid function pointer to an MLIR ciface entry point.
    /// - `args` must contain the correct number of arguments matching the callee's arity.
    /// - All pointer arguments must be valid for the duration of the call.
    #[allow(dead_code)]
    pub unsafe fn call_high_arity_raw(
        &self,
        fn_ptr: unsafe extern "C" fn(),
        args: &[*const c_void],
    ) {
        let ptr = fn_ptr as *const ();
        crate::ciface_high::call_high_arity(ptr, args);
    }

    /// Call a kernel function with output and input pointers.
    ///
    /// # Safety
    ///
    /// - The kernel variant must match the number of arguments.
    /// - All pointers in `outputs` and `inputs` must be valid for the duration
    ///   of the call.
    #[allow(dead_code)]
    pub unsafe fn call(
        &self,
        outputs: &[*mut c_void],
        inputs: &[*const c_void],
    ) -> Result<(), anyhow::Error> {
        let total = outputs.len() + inputs.len();
        match (self, outputs.len(), inputs.len()) {
            (KernelFn::Arity1(f), 0, 0) => { f(outputs[0]); Ok(()) }
            (KernelFn::Arity2(f), 0, 1) => { f(outputs[0], inputs[0]); Ok(()) }
            (KernelFn::Arity3(f), 0, 2) => { f(outputs[0], inputs[0], inputs[1]); Ok(()) }
            (KernelFn::Arity4(f), 0, 3) => { f(outputs[0], inputs[0], inputs[1], inputs[2]); Ok(()) }
            (KernelFn::Arity5(f), 0, 4) => { f(outputs[0], inputs[0], inputs[1], inputs[2], inputs[3]); Ok(()) }
            (KernelFn::Arity6(f), 0, 5) => { f(outputs[0], inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]); Ok(()) }
            (KernelFn::Arity7(f), 0, 6) => { f(outputs[0], inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5]); Ok(()) }
            (KernelFn::Arity8(f), 0, 7) => { f(outputs[0], inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6]); Ok(()) }
            (KernelFn::HighArity(f), _, _) if (1..=300).contains(&total) => {
                let ptr = f.0 as *const ();
                let mut all_args: Vec<*const c_void> = Vec::with_capacity(total);
                for o in outputs { all_args.push(*o as *const c_void); }
                for inp in inputs { all_args.push(*inp); }
                crate::ciface_high::call_high_arity(ptr, &all_args);
                Ok(())
            }
            _ => anyhow::bail!(
                "KernelFn::call: kernel arity mismatch: outputs={}, inputs={}, total={}",
                outputs.len(), inputs.len(), total,
            ),
        }
    }
}

// SAFETY: KernelFn contains only function pointers which are Send+Sync.
unsafe impl Send for KernelFn {}
unsafe impl Sync for KernelFn {}
