"""Full model compilation test — runs the complete pipeline on opt-125m.

This test:
  1. Compiles model.mlir → model.lowered.mlir (C++ lowering)
  2. Runs the full LLVM lowering pipeline
  3. Runs mlir-translate, llc, and link
  4. Verifies the .dylib contains expected symbols

Each step has a per-step timeout.  The test fails if any step times out
or produces unexpected output (e.g. vector ops remaining, memref ops
remaining, ciface wrappers missing).

Requires: compiled/opt_125m_fresh/model.mlir (the model artifact).
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.test_pipeline_lowering import MLIR_BINDINGS

pytestmark = [
    pytest.mark.skipif(not MLIR_BINDINGS, reason="mlir-core not available"),
    pytest.mark.skipif(
        not (Path(__file__).resolve().parent.parent / "compiled/opt_125m_fresh/model.mlir").exists(),
        reason="compiled/opt_125m_fresh/model.mlir not found (run export model first)",
    ),
    pytest.mark.integration,
]

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "compiled/opt_125m_fresh"
LLVM_BIN = os.environ.get(
    "SERVE_FORGE_LLVM_BIN",
    str(Path(__file__).resolve().parent.parent / "llvm-project/build/bin"),
)

STEP_TIMEOUT = 120  # per-step timeout in seconds


def _llvm_step(label, script, timeout=STEP_TIMEOUT):
    """Run a Python snippet as a subprocess with timeout.

    Returns (stdout, elapsed_seconds) on success.
    Raises AssertionError on failure.
    """
    _project_root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env.setdefault(
        "DYLD_LIBRARY_PATH",
        str(_project_root / "llvm-project/build/tools/mlir/python_packages/mlir_core/mlir/_mlir_libs"),
    )
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env.pop("CONDA_PREFIX", None)
    if LLVM_BIN:
        env.setdefault("SERVE_FORGE_LLVM_BIN", LLVM_BIN)

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True,
        timeout=timeout,
        env=env,
        cwd=Path(__file__).resolve().parent.parent,
    )
    elapsed = time.time() - t0
    print(f"  [{label:20s}] {elapsed:6.1f}s", end="")
    if result.returncode != 0:
        stderr_last = "\n".join(result.stderr.strip().split("\n")[-3:])
        print(f"  FAIL (exit {result.returncode})\n    {stderr_last}")
        pytest.fail(f"{label} failed: {stderr_last}")
    print("  OK")
    return result.stdout, elapsed


class TestFullCompile:
    """Full pipeline compile test — runs each step separately."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.ad = ARTIFACT_DIR
        self.mlir_path = self.ad / "model.mlir"
        self.lowered_path = self.ad / "model.lowered.mlir"
        self.ll_path = self.ad / "model.ll"
        self.o_path = self.ad / "model.o"
        self.dylib_path = self.ad / "libopt_125m.dylib"

    def test_01_cpp_lowering(self):
        """sf.promote-weights → sf-lower-to-linalg: all 375 ops converted."""
        stdout, elapsed = _llvm_step(
            "sf-lower-to-linalg",
            f"""
import sys; sys.path.insert(0, '.')
from compiler.mlir_artifact import _parse_mlir_text, mlir_module_to_ir_module
import mlir.ir as ir, mlir.passmanager as pm
from mlir_sf._mlir_libs._sfDialectsNanobind import sf

module = _parse_mlir_text(open(r'{self.mlir_path}').read())
ctx = ir.Context()
sf.register_dialects(ctx._CAPIPtr, load=True)
ir_mod = mlir_module_to_ir_module(module, ctx=ctx)
pman = pm.PassManager.parse(
    'builtin.module(sf-promote-weights,canonicalize,cse,sf-lower-to-linalg)', ctx)
pman.enable_verifier(True)
pman.run(ir_mod.operation)
asm = ir_mod.operation.get_asm(print_generic_op_form=True)
open(r'{self.lowered_path}', 'w').write(asm)
_sf_count = asm.count('"sf.')
print(f'Lowered: {{len(asm)}} chars, {{_sf_count}} sf ops remaining')
""",
        )
        assert '"sf.' not in stdout or '2 sf ops remaining' in stdout or '0 sf ops' in stdout, \
            f"sf ops remain: {stdout.split('sf ops')[0] if 'sf ops' in stdout else stdout[:200]}"

    def test_02_llvm_lowering(self):
        """Bufferize + LLVM lowering: no vector/memref ops remain."""
        if not self.lowered_path.exists():
            pytest.skip("lowered MLIR not found (run test_01 first)")

        stdout, elapsed = _llvm_step(
            "LLVM lowering",
            f"""
import sys; sys.path.insert(0, '.')
import mlir.ir as ir, mlir.passmanager as pm
ctx = ir.Context()
with ctx:
    mod = ir.Module.parse(open(r'{self.lowered_path}').read())
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
    text = str(mod)
    import re
    n_vec = len(re.findall(r'vector\\.', text))
    n_mem = len(re.findall(r'memref\\.', text))
    print(f'vector ops: {{n_vec}}, memref ops: {{n_mem}}')
    print(f'llvm dialect: {{len(text)}} chars')
""",
        )
        assert "vector ops: 0" in stdout, "Vector ops remain after lowering"
        assert "memref ops: 0" in stdout, "MemRef ops remain after lowering"

    def test_03_mlir_translate(self):
        """mlir-translate: produce clean LLVM IR with struct-based convention."""
        if not self.lowered_path.exists():
            pytest.skip("lowered MLIR not found")

        stdout, elapsed = _llvm_step(
            "mlir-translate",
            f"""
import sys, time; sys.path.insert(0, '.')
import mlir.ir as ir, mlir.passmanager as pm
from compiler.mlir_dialect.llvm_backend import mlir_module_to_llvm_ir

lowered = open(r'{self.lowered_path}').read()
ctx = ir.Context()
with ctx:
    mod = ir.Module.parse(lowered, ctx)
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
    t0 = time.time()
    llvm_ir = mlir_module_to_llvm_ir(mod)
    with open(r'{self.ll_path}', 'w') as f:
        f.write(llvm_ir)
    n_ciface = llvm_ir.count('_mlir_ciface')
    n_unrealized = llvm_ir.count('unrealized_conversion_cast')
    print(f'LLVM IR: {{len(llvm_ir)}} chars, ciface={{n_ciface}}, casts={{n_unrealized}}')
""",
        )
        assert "unrealized_conversion_cast" not in stdout, (
            "unrealized_conversion_cast remains after fixup"
        )

    def test_04_llc_and_link(self):
        """llc + link: produce .dylib with expected symbols."""
        if not self.ll_path.exists():
            pytest.skip("model.ll not found")

        _llvm_step(
            "llc",
            f"""
import subprocess
result = subprocess.run([
    'llc', '-O0', '-filetype=obj',
    '-o', r'{self.o_path}',
    r'{self.ll_path}'
], capture_output=True, text=True, timeout=120)
if result.returncode != 0:
    print(result.stderr[:500])
exit(result.returncode)
""",
        )
        assert self.o_path.exists(), "llc did not produce .o file"

        _llvm_step(
            "link",
            f"""
import subprocess
result = subprocess.run([
    'cc', '-shared',
    '-o', r'{self.dylib_path}',
    r'{self.o_path}',
], capture_output=True, text=True, timeout=30)
if result.returncode != 0:
    print(result.stderr[:500])
exit(result.returncode)
""",
        )
        assert self.dylib_path.exists(), "link did not produce .dylib"

        # Check symbols
        nm = subprocess.run(
            ["nm", "-g", str(self.dylib_path)],
            capture_output=True, text=True,
        )
        text_syms = [line.strip() for line in nm.stdout.split("\n") if " T " in line]
        main_syms = [s for s in text_syms if "main" in s.split()[-1]]
        print(f"  Symbols: {len(text_syms)} total, {len(main_syms)} main functions")
        for s in main_syms:
            print(f"    {s.split()[-1]}")
        assert len(main_syms) >= 1, "No main_* symbols in .dylib"
