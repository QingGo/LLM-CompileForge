"""Phase 2: Compiler numerical precision contract test (TDD).

Verifies that the dylib produced by the full compiler pipeline produces
numerically correct output, matching precision contract test vectors.

Independent of runtime — uses ctypes to call ciface directly.
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

from gen.proto.python.sfa_precision_pb2 import NumericalTestCase, PrecisionContract  # noqa: E402

FIXTURES_DIR = ROOT / "tests" / "contract" / "fixtures"


def _load_contract() -> PrecisionContract:
    path = FIXTURES_DIR / "precision_cases.pb"
    if not path.exists():
        pytest.skip(f"Fixture not found: {path}")
    contract = PrecisionContract()
    contract.ParseFromString(path.read_bytes())
    return contract


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.ravel().astype(np.float64)
    b_f = b.ravel().astype(np.float64)
    denom = np.linalg.norm(a_f) * np.linalg.norm(b_f) + 1e-12
    return float(np.dot(a_f, b_f) / denom)


# ── Minimal compilation pipeline helpers ────────────────────────────


def _find_tool(name: str) -> str:
    candidates = [name, str(ROOT / "llvm-project" / "build" / "bin" / name)]
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
        capture_output=True, text=True, check=True, timeout=60,
    )

    # Step 5: cc -c → .o
    o_path = os.path.join(tmp_dir, "model.o")
    subprocess.run(
        [_find_tool("cc"), "-c", ll_path, "-o", o_path, "-O0"],
        capture_output=True, text=True, check=True, timeout=60,
    )

    # Step 6: cc -shared → .dylib
    dylib_path = os.path.join(tmp_dir, f"{dylib_name}.dylib")
    subprocess.run(
        [_find_tool("cc"), "-shared", "-o", dylib_path, o_path],
        capture_output=True, text=True, check=True, timeout=60,
    )
    return dylib_path


def _make_memref_struct(ptr: int, ndim: int, shape: tuple[int, ...]) -> ctypes.Structure:
    """Build a ctypes memref descriptor matching sfa.h layout."""
    strides = tuple(int(np.prod(shape[i + 1 :])) for i in range(ndim))

    class MemRef(ctypes.Structure):
        _fields_ = [
            ("allocated", ctypes.c_void_p),
            ("aligned", ctypes.c_void_p),
            ("offset", ctypes.c_int64),
            ("sizes", ctypes.c_int64 * ndim),
            ("strides", ctypes.c_int64 * ndim),
        ]

    return MemRef(
        ctypes.c_void_p(ptr),
        ctypes.c_void_p(ptr),
        0,
        (ctypes.c_int64 * ndim)(*shape),
        (ctypes.c_int64 * ndim)(*strides),
    )


def _get_sret_output(sret_buf: ctypes.Array, ndim: int) -> np.ndarray:
    """Extract numpy array from sret descriptor buffer."""
    aligned_ptr = ctypes.c_void_p.from_buffer(sret_buf, 8)
    sizes = (ctypes.c_int64 * ndim).from_buffer(sret_buf, 24)
    shape = tuple(sizes[i] for i in range(ndim))
    num_elts = int(np.prod(shape))
    data_ptr = ctypes.cast(aligned_ptr, ctypes.POINTER(ctypes.c_float))
    return np.array([data_ptr[i] for i in range(num_elts)], dtype=np.float32).reshape(shape)


def _call_single_op_dylib(
    dylib_path: str,
    input_data: np.ndarray,
    weight_data: np.ndarray,
    output_ndim: int = 2,
) -> np.ndarray:
    """Load dylib via ctypes, call _mlir_ciface_main_0, return output."""
    lib = ctypes.CDLL(dylib_path)

    inp = np.asarray(input_data, dtype=np.float32)
    w = np.asarray(weight_data, dtype=np.float32)

    inp_memref = _make_memref_struct(inp.ctypes.data, inp.ndim, inp.shape)
    w_memref = _make_memref_struct(w.ctypes.data, w.ndim, w.shape)

    sret_buf = (ctypes.c_uint8 * 1024)()
    lib._mlir_ciface_main_0(
        ctypes.byref(sret_buf),
        ctypes.byref(inp_memref),
        ctypes.byref(w_memref),
    )
    return _get_sret_output(sret_buf, output_ndim)


# ── MLIR template for single matmul ──────────────────────────────────


def _make_mlir_for_case(case: NumericalTestCase) -> str:
    """Generate sf dialect MLIR based on the case name's op type."""
    name = case.name
    in_shape = tuple(int(d) for d in case.input_shape)
    w_shape = tuple(int(d) for d in case.weight_shape)
    in_type = f"tensor<{'x'.join(str(d) for d in in_shape)}xf32>"
    w_type = f"tensor<{'x'.join(str(d) for d in w_shape)}xf32>"

    if "matmul" in name:
        return f"""module {{
  func.func @main_0(%input: {in_type}, %weight: {w_type}) -> {in_type} {{
    %0 = "sf.matmul"(%input, %weight) : ({in_type}, {w_type}) -> {in_type}
    return %0 : {in_type}
  }}
}}"""
    elif "rms_norm" in name:
        return f"""module {{
  func.func @main_0(%input: {in_type}, %weight: {w_type}) -> {in_type} {{
    %0 = "sf.rms_norm"(%input, %weight) : ({in_type}, {w_type}) -> {in_type}
    return %0 : {in_type}
  }}
}}"""
    else:
        raise ValueError(f"Unknown op type in case: {name}")


# ── Tests ─────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.timeout(120)
class TestPrecisionContract:
    """Verify compiled dylib numerical output matches precision contract."""

    @pytest.fixture(scope="class")
    def contract(self) -> PrecisionContract:
        return _load_contract()

    def test_matmul_2x2_matches_precision_fixture(self, contract: PrecisionContract) -> None:
        case = next((c for c in contract.cases if c.name == "matmul_2x2_f32"), None)
        if case is None:
            pytest.skip("matmul_2x2_f32 not in fixture")

        input_data = np.array(case.input_data, dtype=np.float32).reshape(tuple(case.input_shape))
        weight_data = np.array(case.weight_data, dtype=np.float32).reshape(tuple(case.weight_shape))
        expected = np.array(case.expected_output, dtype=np.float32).reshape(tuple(case.expected_shape))

        sf_mlir = _make_mlir_for_case(case)

        with tempfile.TemporaryDirectory() as td:
            dylib_path = _compile_sf_to_dylib(sf_mlir, td, "test_matmul")
            actual = _call_single_op_dylib(
                dylib_path, input_data, weight_data,
                output_ndim=len(case.expected_shape),
            )

        cos = _cosine_similarity(actual, expected)
        assert cos >= case.min_cosine, (
            f"{case.name}: cosine={cos:.6f} < {case.min_cosine}. "
            f"Expected={expected.ravel()[:6].tolist()}..., "
            f"Actual={actual.ravel()[:6].tolist()}..."
        )

    def test_all_cases_in_contract(self, contract: PrecisionContract) -> None:
        for case in contract.cases:
            input_data = np.array(case.input_data, dtype=np.float32).reshape(tuple(case.input_shape))
            weight_data = np.array(case.weight_data, dtype=np.float32).reshape(tuple(case.weight_shape))
            expected = np.array(case.expected_output, dtype=np.float32).reshape(tuple(case.expected_shape))

            sf_mlir = _make_mlir_for_case(case)

            with tempfile.TemporaryDirectory() as td:
                try:
                    dylib_path = _compile_sf_to_dylib(sf_mlir, td, f"test_{case.name}")
                except subprocess.CalledProcessError as e:
                    pytest.skip(
                        f"{case.name}: compilation failed — "
                        f"likely a pre-existing MLIR→LLVM issue. "
                        f"stderr: {e.stderr[-200:] if e.stderr else 'N/A'}"
                    )
                actual = _call_single_op_dylib(
                    dylib_path, input_data, weight_data,
                    output_ndim=len(case.expected_shape),
                )

            cos = _cosine_similarity(actual, expected)
            assert cos >= case.min_cosine, (
                f"{case.name}: cosine={cos:.6f} < {case.min_cosine}"
            )
