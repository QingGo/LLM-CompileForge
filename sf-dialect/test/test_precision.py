"""sf-dialect numerical precision tests.

Validates that sf→linalg lowering produces numerically correct results
for all cases in precision_cases.pb.  Each case is compiled through the
full pipeline (sf→linalg→LLVM→dylib) and compared against golden values
using cosine similarity.

Independent of torch.export / FX Graph — starts from sf MLIR text.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from compiler.dylib_ffi import DEFAULT_SRET_SIZE  # noqa: E402
from gen.proto.python.sfa_precision_pb2 import PrecisionContract  # noqa: E402

FIXTURES_DIR = ROOT / "tests" / "contract" / "fixtures"


# ── helpers ──────────────────────────────────────────────────────────


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.ravel().astype(np.float64)
    b_f = b.ravel().astype(np.float64)
    denom = np.linalg.norm(a_f) * np.linalg.norm(b_f) + 1e-12
    return float(np.dot(a_f, b_f) / denom)


def _find_tool(name: str) -> str:
    candidates = [name]
    if name in ("cc", "clang"):
        candidates.insert(0, "/usr/local/opt/llvm/bin/clang")
    candidates.append(str(ROOT / "llvm-project" / "build" / "bin" / name))
    for c in candidates:
        if Path(c).is_file():
            return str(c)
        try:
            r = subprocess.run([c, "--version"], capture_output=True, timeout=5)
            if r.returncode == 0:
                return c
        except FileNotFoundError:
            continue
    raise RuntimeError(f"{name} not found")


# ── compilation pipeline ─────────────────────────────────────────────


def _compile_sf_to_dylib(
    sf_mlir_text: str,
    tmp_dir: str,
    dylib_name: str = "libtest",
) -> str:
    """Compile sf dialect MLIR text to a .dylib via the full pipeline.

    Pipeline: sf→linalg → linalg→LLVM → fix casts → mlir-translate → cc → dylib.
    Returns path to .dylib.
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


# ── ctypes / FFI ─────────────────────────────────────────────────────

# Re-use the well-tested memref descriptor construction from compiler/dylib_ffi.py
from compiler.dylib_ffi import make_memref_descriptor, parse_sret_outputs  # noqa: E402


def _call_dylib(
    dylib_path: str,
    input_arrays: list[np.ndarray],
    output_ndim: int = 2,
    output_shape: tuple[int, ...] | None = None,
    symbol: str = "_mlir_ciface_main_0",
) -> np.ndarray:
    """Load a dylib, call its ciface kernel, and return the output tensor."""
    lib = ctypes.CDLL(dylib_path)
    kernel = getattr(lib, symbol)

    # Ensure input arrays are contiguous float32 copies so the data pointer
    # stays valid for the duration of the ctypes call.
    arrays: list[np.ndarray] = []
    memrefs: list[ctypes.Structure] = []
    for arr in input_arrays:
        a = np.asarray(arr, dtype=np.float32, order="C")
        arrays.append(a)  # keep alive
        memrefs.append(make_memref_descriptor(a))

    sret_buf = (ctypes.c_uint8 * DEFAULT_SRET_SIZE)()
    args = [ctypes.byref(sret_buf)] + [ctypes.byref(m) for m in memrefs]

    kernel.argtypes = [ctypes.c_void_p] * len(args)
    kernel.restype = None
    kernel(*args)

    # Parse the output from sret using the proven parser
    od_rank = output_ndim
    od_shape = list(output_shape) if output_shape else [0] * od_rank
    output_defs = [{"rank": od_rank, "shape": od_shape}]
    outputs = parse_sret_outputs(bytes(sret_buf), output_defs)
    return outputs[0]


# ── MLIR generation ──────────────────────────────────────────────────


def _make_mlir_for_case(case) -> str:
    """Generate sf dialect MLIR text matching the op described by the case."""
    name = case.name
    in_shape = tuple(int(d) for d in case.input_shape)

    if "matmul" in name:
        w_shape = tuple(int(d) for d in case.weight_shape)
        in_type = f"tensor<{'x'.join(str(d) for d in in_shape)}xf32>"
        w_type = f"tensor<{'x'.join(str(d) for d in w_shape)}xf32>"
        return f"""module {{
  func.func @main_0(%input: {in_type}, %weight: {w_type}) -> {in_type} {{
    %0 = "sf.matmul"(%input, %weight) : ({in_type}, {w_type}) -> {in_type}
    return %0 : {in_type}
  }}
}}"""

    elif "rms_norm" in name:
        w_shape = tuple(int(d) for d in case.weight_shape)
        in_type = f"tensor<{'x'.join(str(d) for d in in_shape)}xf32>"
        w_type = f"tensor<{'x'.join(str(d) for d in w_shape)}xf32>"
        return f"""module {{
  func.func @main_0(%input: {in_type}, %weight: {w_type}) -> {in_type} {{
    %0 = "sf.rms_norm"(%input, %weight) : ({in_type}, {w_type}) -> {in_type}
    return %0 : {in_type}
  }}
}}"""

    elif "ln" in name or "layer_norm" in name:
        # layer_norm: 3 inputs (input, gamma, bias) → 1 output
        # weight_shape=[2,64] → gamma(64) + beta(64)
        # expected_shape=[2,2,4,64] → 2 outputs of (2,4,64)
        last_dim = in_shape[-1]
        gamma_shape = (last_dim,)
        beta_shape = (last_dim,)
        out_shape = in_shape
        in_type = f"tensor<{'x'.join(str(d) for d in in_shape)}xf32>"
        g_type = f"tensor<{'x'.join(str(d) for d in gamma_shape)}xf32>"
        b_type = g_type
        out_type = f"tensor<{'x'.join(str(d) for d in out_shape)}xf32>"
        return f"""module {{
  func.func @main_0(%input: {in_type}, %gamma: {g_type}, %beta: {b_type}) -> {out_type} {{
    %0 = "sf.layer_norm"(%input, %gamma, %beta) : ({in_type}, {g_type}, {b_type}) -> {out_type}
    return %0 : {out_type}
  }}
}}"""

    else:
        raise ValueError(f"Unknown op type for case: {name}")


# ── test class ───────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.timeout(120)
class TestSfDialectPrecision:
    """Verify sf→linalg lowering numerical output matches precision contract."""

    @pytest.fixture(scope="class")
    def contract(self) -> PrecisionContract:
        contract = PrecisionContract()
        contract.ParseFromString((FIXTURES_DIR / "precision_cases.pb").read_bytes())
        return contract

    # ── matmul_2x2_f32 ──────────────────────────────────────────────

    def test_matmul_2x2_f32(self, contract: PrecisionContract) -> None:
        case = next((c for c in contract.cases if c.name == "matmul_2x2_f32"), None)
        assert case is not None, "matmul_2x2_f32 not found in contract"

        input_data = np.array(case.input_data, dtype=np.float32).reshape(tuple(case.input_shape))
        weight_data = np.array(case.weight_data, dtype=np.float32).reshape(tuple(case.weight_shape))
        expected = np.array(case.expected_output, dtype=np.float32).reshape(tuple(case.expected_shape))

        sf_mlir = _make_mlir_for_case(case)
        with tempfile.TemporaryDirectory() as td:
            dylib_path = _compile_sf_to_dylib(sf_mlir, td, "test_matmul")
            actual = _call_dylib(
                dylib_path,
                [input_data, weight_data],
                output_ndim=len(case.expected_shape),
            )

        cos = _cosine_similarity(actual, expected)
        assert cos >= case.min_cosine, (
            f"{case.name}: cos={cos:.6f} < {case.min_cosine}. "
            f"expected[:6]={expected.ravel()[:6].tolist()}, "
            f"actual[:6]={actual.ravel()[:6].tolist()}"
        )

    # ── rms_norm_2x4_f32 ────────────────────────────────────────────

    def test_rms_norm_2x4_f32(self, contract: PrecisionContract) -> None:
        case = next((c for c in contract.cases if c.name == "rms_norm_2x4_f32"), None)
        assert case is not None, "rms_norm_2x4_f32 not found in contract"

        input_data = np.array(case.input_data, dtype=np.float32).reshape(tuple(case.input_shape))
        weight_data = np.array(case.weight_data, dtype=np.float32).reshape(tuple(case.weight_shape))
        expected = np.array(case.expected_output, dtype=np.float32).reshape(tuple(case.expected_shape))

        sf_mlir = _make_mlir_for_case(case)
        with tempfile.TemporaryDirectory() as td:
            dylib_path = _compile_sf_to_dylib(sf_mlir, td, "test_rms_norm")
            actual = _call_dylib(
                dylib_path,
                [input_data, weight_data],
                output_ndim=len(case.expected_shape),
            )

        cos = _cosine_similarity(actual, expected)
        assert cos >= case.min_cosine, (
            f"{case.name}: cos={cos:.6f} < {case.min_cosine}. "
            f"expected[:6]={expected.ravel()[:6].tolist()}, "
            f"actual[:6]={actual.ravel()[:6].tolist()}"
        )

    # ── multi_out_ln_f32 ────────────────────────────────────────────

    def test_multi_out_ln_f32(self, contract: PrecisionContract) -> None:
        """LayerNorm with two outputs encoded in expected_shape=[2,2,4,64].

        weight_data=[gamma(64)..., beta(64)...] → gamma, beta
        expected_output=[out1_flat..., out2_flat...] → two (2,4,64) outputs
        """
        case = next((c for c in contract.cases if c.name == "multi_out_ln_f32"), None)
        assert case is not None, "multi_out_ln_f32 not found in contract"

        # Parse input
        input_shape = tuple(int(d) for d in case.input_shape)
        input_data = np.array(case.input_data, dtype=np.float32).reshape(input_shape)

        # Parse weight: [gamma(64)..., beta(64)...] from weight_shape=[2, 64]
        weight_shape = tuple(int(d) for d in case.weight_shape)
        assert weight_shape[0] == 2, f"Expected 2 weight params, got shape {weight_shape}"
        last_dim = weight_shape[1]
        all_weights = np.array(case.weight_data, dtype=np.float32)
        gamma = all_weights[:last_dim].copy()
        beta = all_weights[last_dim:].copy()

        # Parse expected: [out1_flat..., out2_flat...] from expected_shape=[2,2,4,64]
        expected_shape = tuple(int(d) for d in case.expected_shape)
        num_outputs = expected_shape[0]
        out_shape = expected_shape[1:]
        out_size = int(np.prod(out_shape))
        all_expected = np.array(case.expected_output, dtype=np.float32)
        expected_out1 = all_expected[:out_size].reshape(out_shape)
        expected_out2 = all_expected[out_size:].reshape(out_shape)

        # Generate MLIR for sf.layer_norm(input, gamma, beta) → output
        sf_mlir = _make_mlir_for_case(case)

        with tempfile.TemporaryDirectory() as td:
            dylib_path = _compile_sf_to_dylib(sf_mlir, td, "test_multi_ln")

            # Output 1: layer_norm(x, gamma, beta)
            actual1 = _call_dylib(
                dylib_path,
                [input_data, gamma, beta],
                output_ndim=len(out_shape),
            )

            # Output 2: layer_norm(x+1, gamma, beta)
            input_plus1 = input_data + 1.0
            actual2 = _call_dylib(
                dylib_path,
                [input_plus1, gamma, beta],
                output_ndim=len(out_shape),
            )

        cos1 = _cosine_similarity(actual1, expected_out1)
        cos2 = _cosine_similarity(actual2, expected_out2)
        min_cos = min(cos1, cos2)

        assert min_cos >= case.min_cosine, (
            f"{case.name}: cos1={cos1:.6f}, cos2={cos2:.6f} < {case.min_cosine}. "
            f"expected_out1[:6]={expected_out1.ravel()[:6].tolist()}, "
            f"actual1[:6]={actual1.ravel()[:6].tolist()}"
        )
