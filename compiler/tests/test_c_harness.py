"""Contract tests for compiler output — validates compiled dylibs via C harness.

These tests do NOT depend on ctypes.  Each test:
1. Compiles a minimal MLIR function → .dylib through the full pipeline
2. Generates a C verifier program with expected inputs/outputs baked in
3. Compiles + runs the C program
4. Asserts exit code 0

This replaces the ctypes-based compiler output validation path entirely.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

import mlir.ir as ir
import mlir.passmanager as pm
from mlir_sf._mlir_libs._sfDialectsNanobind import sf

from compiler.pipeline.lowering import SF_LOWERING_PIPELINE
from compiler.backend.llvm_backend import lower_linalg_to_llvm_ir
from compiler.backend.fixups import _fixup_unrealized_casts_pass
from compiler.backend.compile_utils import _compile_serveforge_free

# ── Local helpers (copied from other test files) ──────────────────────

def _find_tool(name: str) -> str:
    """Locate a build tool in the LLVM build directory or PATH."""
    build_bin = os.path.join(_project_root, "llvm-project", "build", "bin")
    tool_path = os.path.join(build_bin, name)
    if os.path.isfile(tool_path):
        return tool_path
    import shutil
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(f"Tool '{name}' not found in {build_bin} or PATH")


def _compile_mlir_to_dylib(mlir_text: str, output_dir: str) -> str:
    """Compile MLIR text → .dylib using the full pipeline. Returns dylib path."""
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    sf.register_dialects(ctx._CAPIPtr, load=True)

    mod = ir.Module.parse(mlir_text, ctx)

    # SF lowering
    pman = pm.PassManager.parse(f"builtin.module({SF_LOWERING_PIPELINE})", ctx)
    pman.run(mod.operation)

    # Linalg → LLVM
    lower_linalg_to_llvm_ir(mod)
    _fixup_unrealized_casts_pass(mod)

    # Write MLIR
    mlir_path = os.path.join(output_dir, "model.mlir")
    with open(mlir_path, "w") as f:
        f.write(str(mod))

    # mlir-translate → .ll
    ll_path = os.path.join(output_dir, "model.ll")
    subprocess.run(
        [_find_tool("mlir-translate"), "--mlir-to-llvmir", mlir_path, "-o", ll_path],
        capture_output=True, check=True, timeout=60,
    )

    # llc → .o (use LLVM's own compiler — avoids Apple Clang version mismatch)
    obj_path = os.path.join(output_dir, "model.o")
    subprocess.run(
        [_find_tool("llc"), "-filetype=obj", ll_path, "-o", obj_path],
        capture_output=True, check=True, timeout=60,
    )

    # Link → .dylib (system cc is fine for linking .o files)
    free_o = _compile_serveforge_free(output_dir)
    dylib_path = os.path.join(output_dir, "libtest.dylib")
    subprocess.run(
        [_find_tool("cc"), "-shared", "-o", dylib_path, obj_path, free_o],
        capture_output=True, check=True, timeout=60,
    )

    # Link → .dylib
    free_o = _compile_serveforge_free(output_dir)
    dylib_path = os.path.join(output_dir, "libtest.dylib")
    subprocess.run(
        [_find_tool("cc"), "-shared", "-o", dylib_path, obj_path, free_o],
        capture_output=True, check=True, timeout=60,
    )
    return dylib_path


def _generate_c_verifier(
    symbol: str,
    inputs: list[tuple[np.ndarray, str]],  # (data, var_name) pairs
    expected_output: np.ndarray,
) -> str:
    """Generate a C program that loads a dylib, calls `symbol`, and checks output.

    `inputs`: list of (numpy_array, c_var_name) — each becomes a memref descriptor.
    `expected_output`: the expected output tensor values.
    """
    total_numel = int(np.prod(expected_output.shape))
    expected_str = ", ".join(f"{v:.8f}f" for v in expected_output.ravel()[:total_numel])

    # Generate input data arrays and descriptors
    input_data_blocks = []
    input_desc_blocks = []
    input_arg_names = []
    for data, name in inputs:
        rank = data.ndim
        numel = int(np.prod(data.shape))
        vals = ", ".join(f"{v:.8f}f" for v in data.ravel()[:numel])

        # Sizes padded to 4
        sizes = list(data.shape) + [1] * (4 - rank)
        sizes_str = ", ".join(str(s) for s in sizes)

        # Row-major strides
        stride = 1
        strides = [0] * 4
        for i in range(rank - 1, -1, -1):
            strides[i] = stride
            stride *= data.shape[i]
        strides_str = ", ".join(str(s) for s in strides)

        input_data_blocks.append(f"    float {name}_data[{numel}] = {{ {vals} }};")
        input_desc_blocks.append(
            f"    memref_t {name}_desc = {{ {name}_data, {name}_data, 0, "
            f"{{ {sizes_str} }}, {{ {strides_str} }} }};"
        )
        input_arg_names.append(f"&{name}_desc")

    n_inputs = len(inputs)
    # CifaceFn signature: void (*)(void* sret, memref_t* arg0, memref_t* arg1, ...)
    fn_params = "memref_t*" + ", memref_t*" * (n_inputs - 1) if n_inputs > 0 else "void"
    func_cast = f"typedef void (*fn_t)(void*, {fn_params});"
    input_args_str = ", ".join(input_arg_names)

    return f'''/* Auto-generated verifier for {symbol} */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {{
    float* allocated; float* aligned;
    int64_t offset; int64_t sizes[4]; int64_t strides[4];
}} memref_t;

int main(int argc, char** argv) {{
    if (argc < 2) {{ fprintf(stderr, "Usage: %s <dylib>\\n", argv[0]); return 2; }}

    void* h = dlopen(argv[1], RTLD_NOW);
    if (!h) {{ fprintf(stderr, "dlopen: %s\\n", dlerror()); return 1; }}
    {func_cast}
    fn_t func = (fn_t)dlsym(h, "{symbol}");
    if (!func) {{ fprintf(stderr, "dlsym: %s\\n", dlerror()); dlclose(h); return 1; }}

{chr(10).join(input_data_blocks)}

{chr(10).join(input_desc_blocks)}

    float expected[{total_numel}] = {{ {expected_str} }};

    size_t sret_sz = sizeof(memref_t) + {total_numel} * sizeof(float);
    void* sret = calloc(1, sret_sz);
    func(sret, {input_args_str});

    memref_t* out = (memref_t*)sret;
    float* out_data = out->aligned ? out->aligned : (float*)((char*)sret + sizeof(memref_t));

    int failed = 0;
    for (int i = 0; i < {total_numel}; i++) {{
        if (fabsf(out_data[i] - expected[i]) > 1e-4) {{
            fprintf(stderr, "MISMATCH[%d]: got %.6f expected %.6f\\n", i, out_data[i], expected[i]);
            failed = 1;
        }}
    }}
    if (!failed) printf("OK\\n");
    free(sret);
    dlclose(h);
    return failed ? 1 : 0;
}}
'''


class TestCompilerViaCHarness:
    """Validate compiler-produced dylibs via C harness (no ctypes dependency)."""

    def test_layer_norm_identity(self) -> None:
        """LayerNorm with w=[1,1], b=[0,0] — output should match normalized input."""
        rng = np.random.RandomState(42)
        x = rng.randn(2, 4, 64).astype(np.float32)
        nw = np.ones(64, dtype=np.float32)
        nb = np.zeros(64, dtype=np.float32)

        # Expected: manual layer_norm
        eps = 1e-5
        m = x.mean(axis=-1, keepdims=True)
        v = ((x - m) ** 2).mean(axis=-1, keepdims=True)
        expected = (x - m) / np.sqrt(v + eps) * nw + nb

        mlir = (
            'module {\n'
            '  func.func @main_0(%x: tensor<2x4x64xf32>, %nw: tensor<64xf32>, %nb: tensor<64xf32>)\n'
            '      -> tensor<2x4x64xf32> {\n'
            '    %0 = "sf.layer_norm"(%x, %nw, %nb) {axis = 2 : i64, eps = 1.0e-5 : f64}\n'
            '      : (tensor<2x4x64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<2x4x64xf32>\n'
            '    return %0 : tensor<2x4x64xf32>\n'
            '  }\n'
            '}\n'
        )

        with tempfile.TemporaryDirectory(prefix="c_harness_") as tmpdir:
            dylib_path = _compile_mlir_to_dylib(mlir, tmpdir)

            c_code = _generate_c_verifier(
                symbol="_mlir_ciface_main_0",
                inputs=[
                    (x, "x"),
                    (nw, "nw"),
                    (nb, "nb"),
                ],
                expected_output=expected,
            )

            c_path = os.path.join(tmpdir, "verify.c")
            bin_path = os.path.join(tmpdir, "verify")
            with open(c_path, "w") as f:
                f.write(c_code)

            subprocess.run(
                [_find_tool("cc"), "-O0", "-o", bin_path, c_path],
                capture_output=True, check=True, timeout=30,
            )

            result = subprocess.run(
                [bin_path, dylib_path],
                capture_output=True, text=True, timeout=30,
            )
            assert result.returncode == 0, f"Harness failed:\n{result.stderr}"
