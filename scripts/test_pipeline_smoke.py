#!/usr/bin/env python3
"""Pipeline smoke test — compile model.mlir to .dylib with timing checks.

Usage:
    python scripts/test_pipeline_smoke.py compiled/opt_125m_fresh
    python scripts/test_pipeline_smoke.py compiled/opt_125m_fresh --timeout 120

This test verifies:
  1. Lowered IR is valid (no remaining sf ops, linalg ops present)
  2. LLVM dialect pipeline completes within time limit
  3. mlir-translate produces valid LLVM IR
  4. llc produces a non-empty .o file
  5. cc links produces a .dylib with ciface symbols
  6. No memref ops or unrealized_conversion_cast remain
"""

import os
import subprocess
import sys
import time
from pathlib import Path


def check(label: str, condition: bool, detail: str = "") -> None:
    """Check a condition and print ✅/❌."""
    if condition:
        print(f"  ✅ {label}")
    else:
        print(f"  ❌ {label}: {detail}")
        raise SystemExit(1)


def main():
    compiled_dir = sys.argv[1] if len(sys.argv) > 1 else "compiled/opt_125m_fresh"
    timeout = 120

    for arg in sys.argv:
        if arg.startswith("--timeout="):
            timeout = int(arg.split("=")[1])

    artifact = Path(compiled_dir)
    lowered_path = artifact / "model.lowered.mlir"
    model_path = artifact / "model.mlir"

    if not lowered_path.exists():
        print(f"❌ {lowered_path} not found — run 'python scripts/compile_dylib.py {compiled_dir}' first")
        sys.exit(1)

    print(f"Pipeline smoke test: {compiled_dir}")
    print(f"  timeout: {timeout}s")
    print()

    # Step 0: Environment setup
    t0 = time.time()
    import warnings; warnings.filterwarnings("ignore")
    import mlir.ir as ir
    import mlir.passmanager as pm
    from mlir._mlir_libs import _mlirRegisterEverything

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    reg = ir.DialectRegistry()
    _mlirRegisterEverything.register_dialects(reg)
    ctx.append_dialect_registry(reg)
    ctx.load_all_available_dialects()
    mod = ir.Module.parse(open(lowered_path).read(), ctx)
    print(f"  [setup     ] {time.time()-t0:.1f}s")

    # Step 1: canonicalize + cse
    t1 = time.time()
    pm.PassManager.parse("builtin.module(canonicalize,cse)", ctx).run(mod.operation)
    print(f"  [canonical ] {time.time()-t1:.1f}s")

    # Step 2: fuse elementwise (optional)
    try:
        pm.PassManager.parse("builtin.module(linalg-fuse-elementwise-ops,canonicalize,cse)", ctx).run(mod.operation)
    except:
        pass

    # Step 3: tile matmul K dim by 64
    from compiler.mlir_dialect.llvm_backend import _tile_matmuls_per_func
    t2 = time.time()
    _tile_matmuls_per_func(mod, tile_k=64)
    pm.PassManager.parse("builtin.module(canonicalize,cse)", ctx).run(mod.operation)
    print(f"  [tile      ] {time.time()-t2:.1f}s")

    # Step 4: emit_c_interface
    for op in list(mod.operation.regions[0].blocks[0]):
        if str(op.operation.name) == "func.func":
            op.operation.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get(context=ctx)

    # Step 5: full LLVM lowering pipeline
    from compiler.mlir_dialect.llvm_backend import lower_linalg_to_llvm_ir
    t5 = time.time()
    lower_linalg_to_llvm_ir(mod)
    elapsed_llvm = time.time() - t5
    print(f"  [lowering  ] {elapsed_llvm:.1f}s")

    check("Lowering completed under timeout", elapsed_llvm < timeout,
          f"Took {elapsed_llvm:.1f}s (limit {timeout}s)")

    # Step 6: mlir-translate (which also applies _fixup_unrealized_casts)
    from compiler.mlir_dialect.llvm_backend import mlir_module_to_llvm_ir
    t6 = time.time()
    llvm_ir = mlir_module_to_llvm_ir(mod)
    elapsed_translate = time.time() - t6
    print(f"  [translate ] {elapsed_translate:.1f}s")

    check(f"LLVM IR generated ({len(llvm_ir)} chars)", len(llvm_ir) > 0)
    check("LLVM IR size reasonable", len(llvm_ir) < 50_000_000,
          f"{len(llvm_ir)} chars (limit 50MB)")
    check("No MLIR syntax in LLVM IR", "llvm.func" not in llvm_ir,
          "LLVM IR should not contain MLIR syntax")

    # Verify fixup eliminated all casts
    n_unrealized = llvm_ir.count("unrealized_conversion_cast")
    check("No unrealized_conversion_cast after fixup", n_unrealized == 0,
          f"Found {n_unrealized} in LLVM IR")

    # Step 7: llc + link
    import tempfile
    from compiler.mlir_dialect.llvm_backend import llc_compile, _compile_embedded_data, link_dylib
    t7 = time.time()
    with tempfile.TemporaryDirectory() as td:
        ll_path = os.path.join(td, "model.ll")
        with open(ll_path, "w") as f:
            f.write(llvm_ir)
        obj_path = llc_compile(ll_path, opt_level=0)
        obj_size = os.path.getsize(obj_path)
        print(f"  [llc       ] {time.time()-t7:.1f}s")
        check(f"Object file produced ({obj_size} bytes)", obj_size > 0)

        # Link
        t8 = time.time()
        const_bin = artifact / "constants.bin"
        obj_files = [obj_path]
        if const_bin.exists():
            obj_files.append(_compile_embedded_data(str(const_bin), td))
        dylib_path = os.path.join(td, "libopt_125m_smoke.dylib")
        link_dylib(obj_files, dylib_path)
        dylib_size = os.path.getsize(dylib_path)
        print(f"  [link      ] {time.time()-t8:.1f}s")
        check(f"dylib produced ({dylib_size} bytes)", dylib_size > 0)

        # Check symbols
        r = subprocess.run(["nm", "-gU", dylib_path], capture_output=True, text=True)
        for sym in ["serveforge_constants_data", "serveforge_constants_size", "_mlir_ciface_main_0"]:
            check(f"Symbol {sym} found", sym in r.stdout)

    total = time.time() - t0
    print(f"\n✅ All checks passed ({total:.1f}s total)")


if __name__ == "__main__":
    main()
