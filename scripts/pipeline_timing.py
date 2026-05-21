"""Pipeline timing diagnostics: step-by-step profiling of the full compile.

Usage:
    python scripts/pipeline_timing.py compiled/opt_125m_fresh

This script runs each pipeline step independently with timing and
timeout detection.  If a step hangs or crashes, the script reports
which step failed, making it easy to locate the root cause.

Exit codes:
    0 — all steps completed within time limits
    1 — step timed out (exceeded per-step or per-function limit)
    2 — step crashed with non-zero exit
"""

import os
import subprocess
import sys
import time
from pathlib import Path

# ── Per-step time limits (seconds) ──────────────────────────────
STEP_TIMEOUT: dict[str, int] = {
    "sf-promote-weights": 30,
    "sf-lower-to-linalg": 120,
    "canonicalize": 30,
    "bufferize": 120,
    "convert-to-llvm": 120,
    "finalize-memref": 60,
    "func-to-llvm": 30,
    "mlir-translate": 120,
    "llc": 120,
    "link": 30,
}

LLVM_BIN = os.environ.get("SERVE_FORGE_LLVM_BIN",
    str(Path(__file__).resolve().parent.parent / "llvm-project/build/bin"))


def _step(
    label: str,
    cmd: list[str],
    timeout: int,
    cwd: str | None = None,
) -> str:
    """Run a single pipeline step with timeout.

    Returns stdout if successful.
    Raises TimeoutExpired on timeout, CalledProcessError on non-zero exit.
    """
    t0 = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=cwd,
    )
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else "FAIL"
    sys.stdout.write(f"  [{label:20s}] {elapsed:6.1f}s  {status}\n")
    sys.stdout.flush()
    if result.returncode != 0:
        # Print last 5 lines of stderr for context
        for line in result.stderr.strip().split("\n")[-5:]:
            sys.stdout.write(f"    stderr: {line}\n")
        result.check_returncode()
    return result.stdout


def time_pipeline(artifact_dir: str) -> None:
    """Run the full compile pipeline step by step, timing each step."""
    ad = Path(artifact_dir)
    if not ad.is_dir():
        sys.stderr.write(f"Directory not found: {artifact_dir}\n")
        sys.exit(1)

    mlir_path = ad / "model.mlir"
    lowered_path = ad / "model.lowered.mlir"
    ll_path = ad / "model.ll"
    o_path = ad / "model.o"
    dylib_path = ad / "libopt_125m.dylib"

    if not mlir_path.exists():
        sys.stderr.write(f"model.mlir not found at {mlir_path}\n")
        sys.exit(1)

    _project_root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env.setdefault("DYLD_LIBRARY_PATH",
        str(_project_root / "llvm-project/build/tools/mlir/python_packages/mlir_core/mlir/_mlir_libs"))
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    if "CONDA_PREFIX" in env:
        del env["CONDA_PREFIX"]
    if LLVM_BIN:
        env.setdefault("SERVE_FORGE_LLVM_BIN", LLVM_BIN)

    # ── Step 1: C++ lowering ──────────────────────────────────────
    sys.stdout.write("Step 1: C++ lowering (sf→linalg)\n")
    try:
        _step(
            "sf-lower-to-linalg",
            [sys.executable, "-c", f"""
import sys; sys.path.insert(0, '.')
from compiler.mlir_artifact import _parse_mlir_text, mlir_module_to_ir_module
import mlir.ir as ir, mlir.passmanager as pm
from mlir_sf._mlir_libs._sfDialectsNanobind import sf

module = _parse_mlir_text(open(r'{mlir_path}').read())
ctx = ir.Context()
sf.register_dialects(ctx._CAPIPtr, load=True)
ir_mod = mlir_module_to_ir_module(module, ctx=ctx)
pman = pm.PassManager.parse('builtin.module(sf-promote-weights,canonicalize,cse,sf-lower-to-linalg)', ctx)
pman.enable_verifier(False)
pman.run(ir_mod.operation)
asm = ir_mod.operation.get_asm(print_generic_op_form=True)
open(r'{lowered_path}', 'w').write(asm)
print(f'Lowered: {{len(asm)}} chars')
import re
sf_remaining = len(re.findall(r'"sf\\.\\w+"', asm))
print(f'Sf ops remaining: {{sf_remaining}}')
"""],
            STEP_TIMEOUT["sf-lower-to-linalg"],
        )
    except subprocess.TimeoutExpired:
        sys.stdout.write(f"  ** sf-lower-to-linalg TIMED OUT ({STEP_TIMEOUT['sf-lower-to-linalg']}s)\n")
        sys.exit(1)

    # ── Step 2: LLVM dialect lowering ─────────────────────────────
    sys.stdout.write("Step 2: LLVM dialect lowering\n")
    lowered_text = lowered_path.read_text()

    # 2a: bufferize
    try:
        _step("bufferize", [
            sys.executable, "-c", f"""
import sys; sys.path.insert(0, '.')
import mlir.ir as ir, mlir.passmanager as pm
ctx = ir.Context()
with ctx:
    mod = ir.Module.parse(open(r'{lowered_path}').read())
    pman = pm.PassManager.parse(
        'builtin.module(canonicalize,cse,one-shot-bufferize{{bufferize-function-boundaries}})',
        ctx)
    pman.run(mod.operation)
"""], STEP_TIMEOUT["bufferize"])
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        sys.stdout.write(f"  ** bufferize {'TIMED OUT' if isinstance(e, subprocess.TimeoutExpired) else 'FAILED'}\n")
        sys.exit(1)

    # 2b: rename SDPA init tensor ops → full LLVM pipeline
    try:
        _step("convert-to-llvm", [
            sys.executable, "-c", f"""
import sys; sys.path.insert(0, '.')
import mlir.ir as ir, mlir.passmanager as pm
ctx = ir.Context()
with ctx:
    mod = ir.Module.parse(open(r'{lowered_path}').read())
    pipeline = (
        'builtin.module('
        'canonicalize,cse,'
        'one-shot-bufferize{{bufferize-function-boundaries}},'
        'canonicalize,cse,'
        'convert-bufferization-to-memref,'
        'convert-linalg-to-loops,'
        'lower-affine,'
        'convert-scf-to-cf,'
        'expand-strided-metadata,'
        'lower-affine,'
        'func.func(lower-vector-mask),'
        'func.func(convert-vector-to-scf),'
        'canonicalize,cse,'
        'convert-scf-to-cf,'
        'lower-affine,'
        'convert-to-llvm,'
        'finalize-memref-to-llvm,'
        'reconcile-unrealized-casts'
        ')'
    )
    pm.PassManager.parse(pipeline, ctx).run(mod.operation)
    text = str(mod)
    import re
    n_vec = len(re.findall(r'vector\\.', text))
    n_mem = len(re.findall(r'memref\\.', text))
    n_ciface = text.count('_mlir_ciface')
    print(f'vector={{n_vec}} memref={{n_mem}} ciface={{n_ciface}}')
"""], STEP_TIMEOUT["convert-to-llvm"])
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        status = 'TIMED OUT' if isinstance(e, subprocess.TimeoutExpired) else 'FAILED'
        sys.stdout.write(f"  ** convert-to-llvm {status}\n")
        sys.exit(1)

    # 2c: func-to-llvm + reconcile
    try:
        _step("func-to-llvm", [
            sys.executable, "-c", f"""
import sys; sys.path.insert(0, '.')
import mlir.ir as ir, mlir.passmanager as pm
ctx = ir.Context()
with ctx:
    mod = ir.Module.parse(open(r'{lowered_path}').read())
    # Same pipeline as 2b but including func-to-llvm
    pipeline = (
        'builtin.module('
        'canonicalize,cse,'
        'one-shot-bufferize{{bufferize-function-boundaries}},'
        'canonicalize,cse,'
        'convert-bufferization-to-memref,'
        'convert-linalg-to-loops,'
        'lower-affine,'
        'convert-scf-to-cf,'
        'expand-strided-metadata,'
        'lower-affine,'
        'func.func(lower-vector-mask),'
        'func.func(convert-vector-to-scf),'
        'canonicalize,cse,'
        'convert-scf-to-cf,'
        'lower-affine,'
        'finalize-memref-to-llvm{{use-generic-functions=false}},'
        'convert-cf-to-llvm,'
        'convert-math-to-llvm,'
        'convert-arith-to-llvm,'
        'convert-ub-to-llvm,'
        'convert-func-to-llvm,'
        'reconcile-unrealized-casts'
        ')'
    )
    pm.PassManager.parse(pipeline, ctx).run(mod.operation)
    print(f'LLVM dialect: {{len(str(mod))}} chars')
"""], STEP_TIMEOUT["func-to-llvm"])
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        sys.stdout.write(f"  ** func-to-llvm {'TIMED OUT' if isinstance(e, subprocess.TimeoutExpired) else 'FAILED'}\n")
        sys.exit(1)

    # ── Step 3: mlir-translate ────────────────────────────────────
    sys.stdout.write("Step 3: mlir-translate (LLVM dialect → LLVM IR)\n")
    try:
        import mlir.ir as ir

        from compiler.mlir_dialect.llvm_backend import mlir_module_to_llvm_ir
        ctx = ir.Context()
        with ctx:
            mod = ir.Module.parse(lowered_text)
            pipeline = (
                "builtin.module("
                "canonicalize,cse,"
                "one-shot-bufferize{bufferize-function-boundaries},"
                "canonicalize,cse,"
                "convert-bufferization-to-memref,"
                "convert-linalg-to-loops,"
                "lower-affine,"
                "convert-scf-to-cf,"
                "expand-strided-metadata,"
                "lower-affine,"
                "func.func(lower-vector-mask),"
                "func.func(convert-vector-to-scf),"
                "canonicalize,cse,"
                "convert-scf-to-cf,"
                "lower-affine,"
                "convert-to-llvm,"
                "finalize-memref-to-llvm,"
                "reconcile-unrealized-casts"
                ")"
            )
            import mlir.passmanager as pm
            pm.PassManager.parse(pipeline, ctx).run(mod.operation)
            t0 = time.time()
            llvm_ir = mlir_module_to_llvm_ir(mod)
            elapsed = time.time() - t0
            open(r'{ll_path}', 'w').write(llvm_ir)
            n_ciface = llvm_ir.count('_mlir_ciface')
            print(f'  [mlir-translate       ] {elapsed:6.1f}s  OK  ({len(llvm_ir)} chars, ciface={n_ciface})')
    except ImportError as e:
        sys.stdout.write(f"  ** mlir-translate: Python binding error: {e}\n")
        sys.exit(1)

    # ── Step 4: llc ───────────────────────────────────────────────
    sys.stdout.write("Step 4: llc (.ll → .o)\n")
    try:
        _step("llc", [
            "/usr/local/opt/llvm/bin/llc",
            "-O0", "-filetype=obj",
            "-o", str(o_path),
            str(ll_path),
        ], STEP_TIMEOUT["llc"])
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        sys.stdout.write(f"  ** llc {'TIMED OUT' if isinstance(e, subprocess.TimeoutExpired) else 'FAILED'}\n")
        sys.exit(1)

    # ── Step 5: link ──────────────────────────────────────────────
    sys.stdout.write("Step 5: link (.o → .dylib)\n")
    try:
        _step("link", [
            "cc", "-shared",
            "-o", str(dylib_path),
            str(o_path),
        ], STEP_TIMEOUT["link"])
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        sys.stdout.write(f"  ** link {'TIMED OUT' if isinstance(e, subprocess.TimeoutExpired) else 'FAILED'}\n")
        sys.exit(1)

    # ── Check symbols ─────────────────────────────────────────────
    nm_result = subprocess.run(
        ["nm", "-g", str(dylib_path)],
        capture_output=True, text=True,
    )
    symbols = [line.strip() for line in nm_result.stdout.split("\n") if " T " in line]
    sys.stdout.write(f"\nSymbols in .dylib: {len(symbols)} T (text) symbols\n")
    for s in symbols:
        name = s.split()[-1]
        if name.startswith("main"):
            sys.stdout.write(f"  ✓ {name}\n")

    sys.stdout.write(f"\n✅ Pipeline complete: {dylib_path}\n")
    sys.stdout.write(f"   ({dylib_path.stat().st_size} bytes)\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    time_pipeline(sys.argv[1])
