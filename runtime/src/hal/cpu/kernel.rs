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
    HighArity(crate::model::ciface_high::FnPtr),
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
}

// SAFETY: KernelFn contains only function pointers which are Send+Sync.
unsafe impl Send for KernelFn {}
unsafe impl Sync for KernelFn {}
