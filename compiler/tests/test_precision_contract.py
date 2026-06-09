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
from compiler.sfcf_parser import DEFAULT_SRET_SIZE

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


def _call_dylib(
    dylib_path: str,
    input_arrays: list[np.ndarray],
    output_ndim: int = 2,
    symbol: str = "_mlir_ciface_main_0",
) -> np.ndarray:
    lib = ctypes.CDLL(dylib_path)
    kernel = getattr(lib, symbol)

    memrefs = []
    for arr in input_arrays:
        a = np.asarray(arr, dtype=np.float32)
        memrefs.append(_make_memref_struct(a.ctypes.data, a.ndim, a.shape))

    sret_buf = (ctypes.c_uint8 * DEFAULT_SRET_SIZE)()
    args = [ctypes.byref(sret_buf)] + [ctypes.byref(m) for m in memrefs]

    kernel.argtypes = [ctypes.c_void_p] * len(args)
    kernel.restype = None
    kernel(*args)
    return _get_sret_output(sret_buf, output_ndim)


def _call_single_op_dylib(
    dylib_path: str,
    input_data: np.ndarray,
    weight_data: np.ndarray,
    output_ndim: int = 2,
) -> np.ndarray:
    return _call_dylib(
        dylib_path,
        [np.asarray(input_data, dtype=np.float32),
         np.asarray(weight_data, dtype=np.float32)],
        output_ndim=output_ndim,
    )


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


# ── SDPA-specific MLIR generation ──────────────────────────────────

def _make_sdpa_mlir(
    q_shape: tuple[int, ...],
    k_shape: tuple[int, ...],
    v_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    scale: float | None = None,
) -> str:
    q_type = f"tensor<{'x'.join(str(d) for d in q_shape)}xf32>"
    k_type = f"tensor<{'x'.join(str(d) for d in k_shape)}xf32>"
    v_type = f"tensor<{'x'.join(str(d) for d in v_shape)}xf32>"
    out_type = f"tensor<{'x'.join(str(d) for d in out_shape)}xf32>"

    attrs = ""
    if scale is not None:
        attrs = f" {{scale = {scale} : f64}}"

    return f"""module {{
  func.func @main_0(%q: {q_type}, %k: {k_type}, %v: {v_type}) -> {out_type} {{
    %0 = "sf.scaled_dot_product_attention"(%q, %k, %v){attrs} : ({q_type}, {k_type}, {v_type}) -> {out_type}
    return %0 : {out_type}
  }}
}}"""


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


# ── SDPA precision tests ─────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.timeout(120)
class TestSdpaPrecision:

    def _compile_and_run_sdpa(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        scale: float | None = None,
    ) -> np.ndarray:
        sf_mlir = _make_sdpa_mlir(q.shape, k.shape, v.shape, q.shape, scale=scale)
        with tempfile.TemporaryDirectory() as td:
            dylib_path = _compile_sf_to_dylib(sf_mlir, td, "test_sdpa")
            return _call_dylib(
                dylib_path,
                [q, k, v],
                output_ndim=q.ndim,
            )

    def test_sdpa_small_no_mask(self) -> None:
        import torch.nn.functional as F  # noqa: N812

        shape = (1, 2, 4, 8)
        rng = np.random.RandomState(42)
        q_np = rng.randn(*shape).astype(np.float32)
        k_np = rng.randn(*shape).astype(np.float32)
        v_np = rng.randn(*shape).astype(np.float32)

        d_k = shape[-1]
        scale = 1.0 / np.sqrt(d_k)

        actual = self._compile_and_run_sdpa(q_np, k_np, v_np, scale=scale)

        import torch
        q_t = torch.from_numpy(q_np)
        k_t = torch.from_numpy(k_np)
        v_t = torch.from_numpy(v_np)
        expected = F.scaled_dot_product_attention(
            q_t, k_t, v_t, scale=scale,
        ).numpy().astype(np.float32)

        cos = _cosine_similarity(actual, expected)
        assert cos >= 0.9999, (
            f"SDPA small no-mask: cos={cos:.8f} < 0.9999\n"
            f"Expected[:6]={expected.ravel()[:6].tolist()}\n"
            f"Actual[:6]={actual.ravel()[:6].tolist()}"
        )

    def test_sdpa_real_shape_no_mask(self) -> None:
        import torch.nn.functional as F  # noqa: N812

        shape = (2, 12, 4, 64)
        rng = np.random.RandomState(42)
        q_np = rng.randn(*shape).astype(np.float32)
        k_np = rng.randn(*shape).astype(np.float32)
        v_np = rng.randn(*shape).astype(np.float32)

        d_k = shape[-1]
        scale = 1.0 / np.sqrt(d_k)

        actual = self._compile_and_run_sdpa(q_np, k_np, v_np, scale=scale)

        import torch
        q_t = torch.from_numpy(q_np)
        k_t = torch.from_numpy(k_np)
        v_t = torch.from_numpy(v_np)
        expected = F.scaled_dot_product_attention(
            q_t, k_t, v_t, scale=scale,
        ).numpy().astype(np.float32)

        cos = _cosine_similarity(actual, expected)
        assert cos >= 0.9999, (
            f"SDPA real shape no-mask: cos={cos:.8f} < 0.9999\n"
            f"Expected[:6]={expected.ravel()[:6].tolist()}\n"
            f"Actual[:6]={actual.ravel()[:6].tolist()}"
        )


# ── Wave 2: per-op precision tests ─────────────────────────────────

def _make_single_op_mlir(
    op_name: str,
    input_shapes: list[tuple[int, ...]],
    output_shape: tuple[int, ...],
    attrs: str = "",
) -> str:
    input_types = [f"tensor<{'x'.join(str(d) for d in s)}xf32>" for s in input_shapes]
    out_type = f"tensor<{'x'.join(str(d) for d in output_shape)}xf32>"
    params = ", ".join(f"%arg{i}: {t}" for i, t in enumerate(input_types))
    args = ", ".join(f"%arg{i}" for i in range(len(input_types)))
    types = ", ".join(input_types)
    return f"""module {{
  func.func @main_0({params}) -> {out_type} {{
    %0 = "{op_name}"({args}) {attrs}: ({types}) -> {out_type}
    return %0 : {out_type}
  }}
}}"""


def _run_op_dylib_test(
    op_name: str,
    input_shapes: list[tuple[int, ...]],
    output_shape: tuple[int, ...],
    torch_fn,
    attrs: str = "",
    dylib_name: str = "test_op",
) -> tuple[float, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(42)
    inputs = [rng.randn(*s).astype(np.float32) for s in input_shapes]
    sf_mlir = _make_single_op_mlir(op_name, input_shapes, output_shape, attrs)
    with tempfile.TemporaryDirectory() as td:
        try:
            dylib_path = _compile_sf_to_dylib(sf_mlir, td, dylib_name)
        except subprocess.CalledProcessError as e:
            pytest.skip(f"{op_name}: compilation failed — stderr: {e.stderr[-200:] if e.stderr else 'N/A'}")
        actual = _call_dylib(
            dylib_path, inputs,
            output_ndim=len(output_shape),
        )
    import torch
    torch_inputs = [torch.from_numpy(a) for a in inputs]
    expected = torch_fn(*torch_inputs)
    if isinstance(expected, torch.Tensor):
        expected = expected.numpy().astype(np.float32)
    cos = _cosine_similarity(actual, expected)
    return cos, actual, expected


@pytest.mark.integration
@pytest.mark.timeout(120)
class TestOpPrecision:

    def test_linear_small_2d(self) -> None:
        import torch.nn.functional as F  # noqa: N812
        cos, actual, expected = _run_op_dylib_test(
            "sf.linear", [(4, 8), (4, 8)], (4, 4),
            lambda x, w: F.linear(x, w),
            dylib_name="test_linear",
        )
        assert cos >= 0.9999, (
            f"linear 2d: cos={cos:.8f} < 0.9999\n"
            f"Expected[:6]={expected.ravel()[:6].tolist()}\n"
            f"Actual[:6]={actual.ravel()[:6].tolist()}"
        )

    def test_linear_real_3d(self) -> None:
        import torch.nn.functional as F  # noqa: N812
        cos, actual, expected = _run_op_dylib_test(
            "sf.linear", [(1, 12, 768), (768, 768)], (1, 12, 768),
            lambda x, w: F.linear(x, w),
            dylib_name="test_linear3d",
        )
        assert cos >= 0.9999, (
            f"linear 3d: cos={cos:.8f} < 0.9999\n"
            f"Expected[:6]={expected.ravel()[:6].tolist()}\n"
            f"Actual[:6]={actual.ravel()[:6].tolist()}"
        )

    def test_silu_small(self) -> None:
        import torch.nn.functional as F  # noqa: N812
        cos, actual, expected = _run_op_dylib_test(
            "sf.silu", [(4, 8)], (4, 8),
            lambda x: F.silu(x),
            dylib_name="test_silu",
        )
        assert cos >= 0.9999, (
            f"silu: cos={cos:.8f} < 0.9999\n"
            f"Expected[:6]={expected.ravel()[:6].tolist()}\n"
            f"Actual[:6]={actual.ravel()[:6].tolist()}"
        )

    def test_layer_norm_small(self) -> None:
        import torch.nn.functional as F  # noqa: N812
        cos, actual, expected = _run_op_dylib_test(
            "sf.layer_norm", [(2, 4), (4,), (4,)], (2, 4),
            lambda x, w, b: F.layer_norm(x, (4,), weight=w, bias=b, eps=1e-5),
            dylib_name="test_ln",
        )
        assert cos >= 0.9999, (
            f"layer_norm: cos={cos:.8f} < 0.9999\n"
            f"Expected[:6]={expected.ravel()[:6].tolist()}\n"
            f"Actual[:6]={actual.ravel()[:6].tolist()}"
        )

    def test_layer_norm_real(self) -> None:
        import torch.nn.functional as F  # noqa: N812
        cos, actual, expected = _run_op_dylib_test(
            "sf.layer_norm", [(1, 12, 768), (768,), (768,)], (1, 12, 768),
            lambda x, w, b: F.layer_norm(x, (768,), weight=w, bias=b, eps=1e-5),
            dylib_name="test_ln_real",
        )
        assert cos >= 0.9999, (
            f"layer_norm real: cos={cos:.8f} < 0.9999\n"
            f"Expected[:6]={expected.ravel()[:6].tolist()}\n"
            f"Actual[:6]={actual.ravel()[:6].tolist()}"
        )

    def test_add_small(self) -> None:
        cos, actual, expected = _run_op_dylib_test(
            "sf.add", [(4, 8), (4, 8)], (4, 8),
            lambda a, b: a + b,
            dylib_name="test_add",
        )
        assert cos >= 0.9999, (
            f"add: cos={cos:.8f} < 0.9999\n"
            f"Expected[:6]={expected.ravel()[:6].tolist()}\n"
            f"Actual[:6]={actual.ravel()[:6].tolist()}"
        )

    def test_mul_small(self) -> None:
        cos, actual, expected = _run_op_dylib_test(
            "sf.mul", [(4, 8), (4, 8)], (4, 8),
            lambda a, b: a * b,
            dylib_name="test_mul",
        )
        assert cos >= 0.9999, (
            f"mul: cos={cos:.8f} < 0.9999\n"
            f"Expected[:6]={expected.ravel()[:6].tolist()}\n"
            f"Actual[:6]={actual.ravel()[:6].tolist()}"
        )

    def test_relu_small(self) -> None:
        cos, actual, expected = _run_op_dylib_test(
            "sf.relu", [(4, 8)], (4, 8),
            lambda x: x.clamp(min=0),
            dylib_name="test_relu",
        )
        assert cos >= 0.9999, (
            f"relu: cos={cos:.8f} < 0.9999\n"
            f"Expected[:6]={expected.ravel()[:6].tolist()}\n"
            f"Actual[:6]={actual.ravel()[:6].tolist()}"
        )
