"""Phase 2: Multi-function compiler precision test.

Verifies that a multi-function sf dialect module compiles through the full
pipeline (sf→linalg→LLVM→cc→dylib) and that each ciface function produces
numerically correct output matching torch reference.

Three independent functions form a logical chain:
  func_0: rms_norm  [2,8] * [8]     → [2,8]
  func_1: matmul    [2,8] @ [8,4]   → [2,4]
  func_2: add       [2,4] + [4]     → [2,4]

Each function is called independently via ctypes and compared against
torch reference (eps=1e-5, matching the C++ rms_norm lowering).
"""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np
import pytest
import torch

from compiler.tests.test_precision_contract import (
    _cosine_similarity,
    _find_tool,
    _get_sret_output,
    _make_memref_struct,
)

# C++ sf.rms_norm lowering uses eps=1e-5 (SfLowerNormalization.cpp:233)
_RMS_NORM_EPS = 1e-5


def _build_multifunc_mlir() -> str:
    """Build multi-function sf dialect MLIR.

    Each function uses fully static shapes (no ? dims) to avoid
    dynamic-shape issues in the lowering pipeline.
    """
    return """\
module {
  func.func @func_0(%input: tensor<2x8xf32>, %weight: tensor<8xf32>) -> tensor<2x8xf32> {
    %0 = "sf.rms_norm"(%input, %weight) : (tensor<2x8xf32>, tensor<8xf32>) -> tensor<2x8xf32>
    return %0 : tensor<2x8xf32>
  }
  func.func @func_1(%input: tensor<2x8xf32>, %weight: tensor<8x4xf32>) -> tensor<2x4xf32> {
    %0 = "sf.matmul"(%input, %weight) : (tensor<2x8xf32>, tensor<8x4xf32>) -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }
  func.func @func_2(%input: tensor<2x4xf32>, %bias: tensor<4xf32>) -> tensor<2x4xf32> {
    %0 = "sf.add"(%input, %bias) : (tensor<2x4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }
}"""


def _compile_sf_to_dylib_fixed(
    sf_mlir_text: str,
    tmp_dir: str,
    dylib_name: str = "libtest",
) -> str:
    """Compile sf dialect MLIR text to a .dylib, with LLVM IR fixup for Apple clang.

    Uses the same pipeline as ``_compile_sf_to_dylib`` but strips
    ``nocreateundeforpoison`` from the generated .ll file so that
    Apple clang 17 can compile it.
    """
    import mlir.ir as ir
    from mlir_sf._mlir_libs._sfDialectsNanobind import sf  # type: ignore[import-untyped]

    from compiler.backend.fixups import _fixup_unrealized_casts_pass
    from compiler.backend.llvm_backend import lower_linalg_to_llvm_ir
    from compiler.pipeline import _apply_sf_to_linalg

    # Step 1: sf → linalg
    lowered = _apply_sf_to_linalg(sf_mlir_text)
    assert "linalg" in lowered, f"sf→linalg lowering failed:\n{lowered[:500]}"

    # Step 2: linalg → LLVM
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    sf.register_dialects(ctx._CAPIPtr, load=True)
    with ir.Location.unknown(ctx):
        module = ir.Module.parse(lowered, ctx)
        lower_linalg_to_llvm_ir(module)

    # Step 3: Fix residual casts
    _fixup_unrealized_casts_pass(module)

    # Step 4: mlir-translate → .ll
    mlir_path = os.path.join(tmp_dir, "model.mlir")
    ll_path = os.path.join(tmp_dir, "model.ll")
    with open(mlir_path, "w") as f:
        f.write(str(module))
    subprocess.run(
        [_find_tool("mlir-translate"), "--mlir-to-llvmir", mlir_path, "-o", ll_path],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )

    # Step 4b: Fix LLVM IR — strip ``nocreateundeforpoison`` which
    # Apple clang 17 rejects with "unterminated attribute group".
    with open(ll_path) as f:
        ll_text = f.read()
    ll_text = ll_text.replace("nocreateundeforpoison ", "")
    with open(ll_path, "w") as f:
        f.write(ll_text)

    # Step 5: cc -c → .o
    o_path = os.path.join(tmp_dir, "model.o")
    subprocess.run(
        [_find_tool("cc"), "-c", ll_path, "-o", o_path, "-O0"],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )

    # Step 6: cc -shared → .dylib
    dylib_path = os.path.join(tmp_dir, f"{dylib_name}.dylib")
    subprocess.run(
        [_find_tool("cc"), "-shared", "-o", dylib_path, o_path],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return dylib_path


def _call_dylib_func(
    dylib_path: str,
    func_name: str,
    *input_arrays: np.ndarray,
    output_ndim: int,
) -> np.ndarray:
    """Load dylib via ctypes, call a named ciface function, return output.

    Args:
        dylib_path: Path to .dylib file.
        func_name: CIFace symbol name (e.g. ``_mlir_ciface_func_0``).
        input_arrays: One numpy array per function argument (f32).
        output_ndim: Expected output rank for sret descriptor parsing.

    Returns:
        Output numpy array (f32).
    """
    lib = ctypes.CDLL(dylib_path)
    arrays = [np.asarray(a, dtype=np.float32) for a in input_arrays]
    memrefs = [_make_memref_struct(a.ctypes.data, a.ndim, a.shape) for a in arrays]

    sret_buf = (ctypes.c_uint8 * 1024)()
    ciface_func = getattr(lib, func_name)

    args = [ctypes.byref(sret_buf)] + [ctypes.byref(m) for m in memrefs]
    ciface_func(*args)

    return _get_sret_output(sret_buf, output_ndim)


def _compute_torch_rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = _RMS_NORM_EPS) -> np.ndarray:
    """Torch reference for sf.rms_norm: x * rsqrt(mean(x²) + eps) * weight."""
    t_x = torch.from_numpy(x)
    t_w = torch.from_numpy(weight)
    return (t_x * torch.rsqrt(t_x.pow(2).mean(dim=-1, keepdim=True) + eps) * t_w).numpy()


def _compute_torch_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Torch reference for sf.matmul."""
    return (torch.from_numpy(a) @ torch.from_numpy(b)).numpy()


def _compute_torch_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Torch reference for sf.add with broadcasting."""
    return (torch.from_numpy(a) + torch.from_numpy(b)).numpy()


@pytest.mark.unit
@pytest.mark.timeout(120)
class TestMultiFuncPrecision:
    """Verify multi-function compiled dylib matches torch reference."""

    @pytest.fixture(scope="class")
    def multifunc_data(self) -> dict[str, np.ndarray]:
        """Generate reproducible random data for all 3 functions."""
        np.random.seed(42)
        return {
            "input_0": np.random.randn(2, 8).astype(np.float32),
            "weight_0": np.random.randn(8).astype(np.float32),
            "W_1": np.random.randn(8, 4).astype(np.float32),
            "bias_2": np.random.randn(4).astype(np.float32),
        }

    @pytest.fixture(scope="class")
    def compiled_dylib(self, tmp_path_factory: pytest.TempPathFactory) -> str:
        """Compile MLIR to dylib once, share across all test methods."""
        sf_mlir = _build_multifunc_mlir()
        td = tmp_path_factory.mktemp("multifunc_dylib")
        return _compile_sf_to_dylib_fixed(sf_mlir, str(td), "test_multifunc")

    # ── func_0: rms_norm ──────────────────────────────────────────

    def test_func_0_rms_norm(self, compiled_dylib: str, multifunc_data: dict[str, np.ndarray]) -> None:
        """func_0 (rms_norm) output matches torch reference."""
        actual = _call_dylib_func(
            compiled_dylib,
            "_mlir_ciface_func_0",
            multifunc_data["input_0"],
            multifunc_data["weight_0"],
            output_ndim=2,
        )
        expected = _compute_torch_rms_norm(multifunc_data["input_0"], multifunc_data["weight_0"])
        cos = _cosine_similarity(actual, expected)
        assert cos >= 0.9999, (
            f"func_0 rms_norm: cosine={cos:.6f} < 0.9999\n"
            f"  actual[:3]={actual.ravel()[:6].tolist()}\n"
            f"  expected[:3]={expected.ravel()[:6].tolist()}"
        )

    # ── func_1: matmul ────────────────────────────────────────────

    def test_func_1_matmul(self, compiled_dylib: str, multifunc_data: dict[str, np.ndarray]) -> None:
        """func_1 (matmul) output matches torch reference.

        Uses torch-computed func_0 output as input to avoid cascading
        errors — only the matmul operation itself is compared.
        """
        rms_out = _compute_torch_rms_norm(multifunc_data["input_0"], multifunc_data["weight_0"])
        actual = _call_dylib_func(
            compiled_dylib,
            "_mlir_ciface_func_1",
            rms_out,
            multifunc_data["W_1"],
            output_ndim=2,
        )
        expected = _compute_torch_matmul(rms_out, multifunc_data["W_1"])
        cos = _cosine_similarity(actual, expected)
        assert cos >= 0.9999, (
            f"func_1 matmul: cosine={cos:.6f} < 0.9999\n"
            f"  actual[:3]={actual.ravel()[:6].tolist()}\n"
            f"  expected[:3]={expected.ravel()[:6].tolist()}"
        )

    # ── func_2: add ───────────────────────────────────────────────

    def test_func_2_add(self, compiled_dylib: str, multifunc_data: dict[str, np.ndarray]) -> None:
        """func_2 (add) output matches torch reference.

        Uses torch-computed func_0+func_1 output as input to isolate
        the add operation.
        """
        rms_out = _compute_torch_rms_norm(multifunc_data["input_0"], multifunc_data["weight_0"])
        matmul_out = _compute_torch_matmul(rms_out, multifunc_data["W_1"])
        actual = _call_dylib_func(
            compiled_dylib,
            "_mlir_ciface_func_2",
            matmul_out,
            multifunc_data["bias_2"],
            output_ndim=2,
        )
        expected = _compute_torch_add(matmul_out, multifunc_data["bias_2"])
        cos = _cosine_similarity(actual, expected)
        assert cos >= 0.9999, (
            f"func_2 add: cosine={cos:.6f} < 0.9999\n"
            f"  actual[:3]={actual.ravel()[:6].tolist()}\n"
            f"  expected[:3]={expected.ravel()[:6].tolist()}"
        )

    # ── Chain: func_0 → func_1 → func_2 argmax ────────────────────

    def test_chain_argmax_matches_torch(self, compiled_dylib: str, multifunc_data: dict[str, np.ndarray]) -> None:
        """End-to-end chain through dylib: argmax of final output matches torch.

        Feeds compiled output through all 3 functions (not torch intermediates)
        to catch cascading numerical deviations that per-op tests may miss.
        """
        # Run the full chain through compiled dylib
        dylib_out_0 = _call_dylib_func(
            compiled_dylib,
            "_mlir_ciface_func_0",
            multifunc_data["input_0"],
            multifunc_data["weight_0"],
            output_ndim=2,
        )
        dylib_out_1 = _call_dylib_func(
            compiled_dylib,
            "_mlir_ciface_func_1",
            dylib_out_0,
            multifunc_data["W_1"],
            output_ndim=2,
        )
        dylib_out_2 = _call_dylib_func(
            compiled_dylib,
            "_mlir_ciface_func_2",
            dylib_out_1,
            multifunc_data["bias_2"],
            output_ndim=2,
        )

        # Run the same chain through torch
        torch_out_0 = _compute_torch_rms_norm(multifunc_data["input_0"], multifunc_data["weight_0"])
        torch_out_1 = _compute_torch_matmul(torch_out_0, multifunc_data["W_1"])
        torch_out_2 = _compute_torch_add(torch_out_1, multifunc_data["bias_2"])

        # Argmax must agree — this catches deviations that cos_sim may miss
        dylib_argmax = int(np.argmax(dylib_out_2.ravel()))
        torch_argmax = int(np.argmax(torch_out_2.ravel()))
        assert dylib_argmax == torch_argmax, (
            f"Chain argmax mismatch: dylib={dylib_argmax}, torch={torch_argmax}\n"
            f"  dylib_out_2[:4]={dylib_out_2.ravel()[:4].tolist()}\n"
            f"  torch_out_2[:4]={torch_out_2.ravel()[:4].tolist()}\n"
            f"  cosine={_cosine_similarity(dylib_out_2, torch_out_2):.6f}"
        )
