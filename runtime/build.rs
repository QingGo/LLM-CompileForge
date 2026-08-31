use std::io::Write;

fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();

    // Generate C trampolines for arities 1..512
    let c_path = std::path::Path::new(&out_dir).join("call_gen.c");
    let mut f = std::fs::File::create(&c_path).unwrap();

    writeln!(f, "// Auto-generated trampolines for MLIR ciface calls").unwrap();
    writeln!(f, "#include <stdint.h>").unwrap();
    writeln!(f).unwrap();

    for n in 1..=512 {
        writeln!(f, "void call_{n}(void (*fn)(), void *out, void **inputs) {{").unwrap();
        write!(f, "    ((void (*)(void*").unwrap();
        for _ in 0..n { write!(f, ", void*").unwrap(); }
        write!(f, "))fn)(out").unwrap();
        for i in 0..n { write!(f, ", inputs[{i}]").unwrap(); }
        writeln!(f, ");").unwrap();
        writeln!(f, "}}").unwrap();
    }

    writeln!(f, "\nvoid call_mlir(void (*fn)(), void *out, void **inputs, int n) {{").unwrap();
    writeln!(f, "    switch (n) {{").unwrap();
    for n in 1..=512 {
        writeln!(f, "        case {n}: call_{n}(fn, out, inputs); break;").unwrap();
    }
    writeln!(f, "    }}").unwrap();
    writeln!(f, "}}").unwrap();

    // Compile the C file
    cc::Build::new()
        .file(&c_path)
        .compile("call_gen");

    // Generate Rust wrapper using the compiled C library
    let rs_path = std::path::Path::new(&out_dir).join("ciface_gen.rs");
    let mut r = std::fs::File::create(&rs_path).unwrap();
    writeln!(r, "extern \"C\" {{").unwrap();
    writeln!(r, "    fn call_mlir(fn_ptr: *const (), out: *mut std::ffi::c_void, inputs: *const *const std::ffi::c_void, n: i32);").unwrap();
    writeln!(r, "}}").unwrap();
    writeln!(r, "pub unsafe fn call_n(fn_ptr: *const (), out: *mut std::ffi::c_void, inputs: &[*const std::ffi::c_void]) {{").unwrap();
    writeln!(r, "    let n = inputs.len() as i32;").unwrap();
    writeln!(r, "    if n >= 1 && n <= 512 {{").unwrap();
    writeln!(r, "        call_mlir(fn_ptr, out, inputs.as_ptr(), n);").unwrap();
    writeln!(r, "    }} else {{").unwrap();
    writeln!(r, "        panic!(\"unsupported arity: {{}}\", n);").unwrap();
    writeln!(r, "    }}").unwrap();
    writeln!(r, "}}").unwrap();

    // Force linking call_gen for ALL targets (lib + bins)
    println!("cargo:rustc-link-lib=static=call_gen");
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rerun-if-changed=build.rs");

    // BLAS linking for HAL matmul (cblas_sgemm)
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap();
    match target_os.as_str() {
        "macos" => println!("cargo:rustc-link-lib=framework=Accelerate"),
        "linux" => println!("cargo:rustc-link-lib=openblas"),
        other => panic!("unsupported target OS for BLAS linking: {other}"),
    }
}
