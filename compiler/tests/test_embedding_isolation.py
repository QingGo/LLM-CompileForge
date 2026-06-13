"""TDD isolation: compile sf.embedding standalone → dylib → test output.

This determines whether the bug is in the embedding lowering itself or
in the interaction of ops within the full model's main_0.
"""

from __future__ import annotations

import ctypes
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from compiler.dylib_ffi import DEFAULT_SRET_SIZE


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    af = a.ravel().astype(np.float64)
    bf = b.ravel().astype(np.float64)
    return float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))


def _find_tool(name: str) -> str:
    c = [name]
    if name in ("cc", "clang"):
        c.insert(0, "/usr/local/opt/llvm/bin/clang")
    c.append(str(ROOT / "llvm-project" / "build" / "bin" / name))
    for p in c:
        if Path(p).is_file():
            return str(p)
        try:
            if subprocess.run([p, "--version"], capture_output=True, timeout=5).returncode == 0:
                return p
        except FileNotFoundError:
            continue
    raise RuntimeError(f"{name} not found")


def _memref(ptr, ndim, shape):
    strides = tuple(int(np.prod(shape[i + 1 :])) for i in range(ndim))

    class M(ctypes.Structure):
        _fields_ = [
            ("allocated", ctypes.c_void_p),
            ("aligned", ctypes.c_void_p),
            ("offset", ctypes.c_int64),
            ("sizes", ctypes.c_int64 * ndim),
            ("strides", ctypes.c_int64 * ndim),
        ]

    return M(
        ctypes.c_void_p(ptr),
        ctypes.c_void_p(ptr),
        0,
        (ctypes.c_int64 * ndim)(*shape),
        (ctypes.c_int64 * ndim)(*strides),
    )


def _compile_sf_to_dylib(sf_mlir: str, tmp_dir: str, name: str) -> str:
    import mlir.ir as ir
    from mlir_sf._mlir_libs._sfDialectsNanobind import sf

    from compiler.backend.compile_utils import _compile_serveforge_free, link_dylib
    from compiler.backend.fixups import _fixup_unrealized_casts_pass
    from compiler.backend.llvm_backend import lower_linalg_to_llvm_ir
    from compiler.pipeline import _apply_sf_to_linalg

    lowered = _apply_sf_to_linalg(sf_mlir)
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    sf.register_dialects(ctx._CAPIPtr, load=True)
    with ir.Location.unknown(ctx):
        mod = ir.Module.parse(lowered, ctx)
        lower_linalg_to_llvm_ir(mod)
        _fixup_unrealized_casts_pass(mod)
        m = os.path.join(tmp_dir, "m.mlir")
        l = os.path.join(tmp_dir, "m.ll")
        o = os.path.join(tmp_dir, "m.o")
        d = os.path.join(tmp_dir, f"{name}.dylib")
        with open(m, "w") as f:
            f.write(str(mod))
        subprocess.run(
            [_find_tool("mlir-translate"), "--mlir-to-llvmir", m, "-o", l],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        subprocess.run(
            [_find_tool("cc"), "-c", l, "-o", o, "-O0"],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        free_o = _compile_serveforge_free(tmp_dir)
        link_dylib([o, free_o], d)
        return d


@pytest.mark.integration
@pytest.mark.timeout(60)
class TestEmbeddingIsolation:
    @pytest.mark.parametrize(
        "batch,seq,vocab,hidden",
        [
            (2, 4, 50272, 768),
            (2, 4, 100, 64),
            (1, 8, 50, 32),
        ],
    )
    def test_embedding_standalone_dylib(self, batch, seq, vocab, hidden):
        """sf.embedding compiled as standalone dylib matches numpy reference."""
        rng = np.random.RandomState(42)
        weight = rng.randn(vocab, hidden).astype(np.float32)
        indices = rng.randint(0, vocab, size=(batch, seq), dtype=np.int64)

        mlir = f"""module {{
  func.func @main_0(%ids: tensor<{batch}x{seq}xi64>, %w: tensor<{vocab}x{hidden}xf32>) -> tensor<{batch}x{seq}x{hidden}xf32> {{
    %0 = "sf.embedding"(%w, %ids) : (tensor<{vocab}x{hidden}xf32>, tensor<{batch}x{seq}xi64>) -> tensor<{batch}x{seq}x{hidden}xf32>
    return %0 : tensor<{batch}x{seq}x{hidden}xf32>
  }}
}}"""
        with tempfile.TemporaryDirectory() as td:
            dylib = _compile_sf_to_dylib(mlir, td, "test_emb")
            lib = ctypes.CDLL(dylib)
            in_m = _memref(indices.ctypes.data, 2, indices.shape)
            w_m = _memref(weight.ctypes.data, 2, weight.shape)
            sret = (ctypes.c_uint8 * DEFAULT_SRET_SIZE)()
            args = [ctypes.byref(sret), ctypes.byref(in_m), ctypes.byref(w_m)]
            lib._mlir_ciface_main_0.argtypes = [ctypes.c_void_p] * 3
            lib._mlir_ciface_main_0.restype = None
            lib._mlir_ciface_main_0(*args)

            sb = bytes(sret)
            al = struct.unpack_from("<Q", sb, 8)[0]
            sz = tuple(struct.unpack_from("<q", sb, 24 + 8 * i)[0] for i in range(3))
            assert sz == (batch, seq, hidden), f"Wrong output shape: {sz}"
            n = int(np.prod(sz))
            actual = np.array((ctypes.c_float * n).from_address(al), dtype=np.float32).reshape(sz)
            expected = weight[indices % vocab]
            cos = _cos(actual, expected)
            assert cos >= 0.9999, f"Embedding standalone cos={cos:.8f} < 0.9999 shape=({batch},{seq},{vocab},{hidden})"


@pytest.mark.integration
@pytest.mark.timeout(60)
class TestEmbeddingWithIdentityWeights:
    def test_embedding_plus_10_identity_weights(self):
        """sf.embedding + 10 identity weight copies in same function."""
        rng = np.random.RandomState(42)
        vocab, hidden = 100, 64
        weight = rng.randn(vocab, hidden).astype(np.float32)
        indices = np.array([[2, 3, 1, 5], [0, 0, 0, 0]], dtype=np.int64)
        id_weights = [rng.randn(64, 64).astype(np.float32) for _ in range(10)]

        args = ["%ids: tensor<2x4xi64>", f"%emb_w: tensor<{vocab}x{hidden}xf32>"]
        for i in range(10):
            args.append(f"%id{i}: tensor<64x64xf32>")
        lines = ["module {"]
        lines.append(f"  func.func @main_0({', '.join(args)}) -> (")
        lines.append("    tensor<2x4x64xf32>")
        for _ in range(10):
            lines.append(", tensor<64x64xf32>")
        lines.append("  ) {")
        lines.append(
            f'    %emb = "sf.embedding"(%emb_w, %ids) : (tensor<{vocab}x{hidden}xf32>, tensor<2x4xi64>) -> tensor<2x4x{hidden}xf32>'
        )
        rv = ["%emb"]
        for i in range(10):
            lines.append(
                f"    %c{i} = linalg.copy ins(%id{i} : tensor<64x64xf32>) outs(%id{i} : tensor<64x64xf32>) -> tensor<64x64xf32>"
            )
            rv.append(f"%c{i}")
        lines.append(
            f"    return {', '.join(rv)} : tensor<2x4x64xf32>" + "".join(", tensor<64x64xf32>" for _ in range(10))
        )
        lines.append("  }")
        lines.append("}")

        with tempfile.TemporaryDirectory() as td:
            dylib = _compile_sf_to_dylib("\n".join(lines), td, "test_embid")
            lib = ctypes.CDLL(dylib)
            mrs = [_memref(indices.ctypes.data, 2, indices.shape), _memref(weight.ctypes.data, 2, weight.shape)]
            for w in id_weights:
                mrs.append(_memref(w.ctypes.data, 2, w.shape))
            sret = (ctypes.c_uint8 * DEFAULT_SRET_SIZE)()
            args2 = [ctypes.byref(sret)] + [ctypes.byref(m) for m in mrs]
            lib._mlir_ciface_main_0.argtypes = [ctypes.c_void_p] * len(args2)
            lib._mlir_ciface_main_0.restype = None
            lib._mlir_ciface_main_0(*args2)

            # output[0] = embedding result
            sb = bytes(sret)
            al = struct.unpack_from("<Q", sb, 8)[0]
            sz = tuple(struct.unpack_from("<q", sb, 24 + 8 * i)[0] for i in range(3))
            assert sz == (2, 4, hidden), f"Wrong shape: {sz}"
            n = int(np.prod(sz))
            actual = np.array((ctypes.c_float * n).from_address(al), dtype=np.float32).reshape(sz)
            expected = weight[indices % vocab]
            cos = _cos(actual, expected)
            assert cos >= 0.9999, f"Embedding+10id cos={cos:.8f} < 0.9999"


@pytest.mark.integration
@pytest.mark.timeout(300)
class TestFullModelMain0Isolation:
    def test_standalone_main0_vs_fulldylib_main0(self):
        """RED: standalone sf dialect main_0 vs full-dylib libmodel.dylib main_0.

        Compiles the exact sf dialect main_0 (model.mlir) as standalone dylib
        via the SAME pipeline as per-op tests. If standalone output matches
        Python executor but full-dylib doesn't, the bug is in the full-dylib
        compilation path (compile_dylib.py). If both are wrong, the bug is in
        the pipeline stages for this specific IR.
        """
        import re

        from compiler.serialize import load_artifact
        from scripts._cos import cosine_similarity

        ARTIFACT_DIR = "outputs/compiled/opt_125m_fresh"

        # Load sf dialect model.mlir, extract only main_0
        sf_mlir = open(f"{ARTIFACT_DIR}/model.mlir").read()
        end_idx = sf_mlir.find("\n  func.func @main_1")
        if end_idx > 0:
            sf_mlir = sf_mlir[:end_idx].strip() + "\n}"

        # Load artifact weights (needed for weight ordering)
        artifact = load_artifact(ARTIFACT_DIR)
        all_w: dict[str, np.ndarray] = {}
        for func in artifact.functions:
            for wname, wtensor in func.weights.items():
                if wname not in all_w:
                    all_w[wname] = np.ascontiguousarray(wtensor.numpy())

        with tempfile.TemporaryDirectory() as td:
            # Compile standalone main_0 via _compile_sf_to_dylib
            dylib = _compile_sf_to_dylib(sf_mlir, td, "main0_standalone")

            # The lowered MLIR should have sf.weight_names with argument order
            lowered = _apply_sf_to_linalg(sf_mlir)
            wm = re.search(r"sf\.weight_names\s*=\s*\[(.*?)\]", lowered, re.DOTALL)
            assert wm, "No sf.weight_names in lowered MLIR"
            names = [w.strip().strip('"') for w in wm.group(1).split(",")]
            w_arrs = [all_w.get(n, np.zeros((1,), dtype=np.float32)) for n in names]

            input_ids = np.array([[2, 32826, 85, 4129], [0, 0, 0, 0]], dtype=np.int64)
            all_inputs = [input_ids] + w_arrs

            lib = ctypes.CDLL(dylib)
            mrs = [_memref(a.ctypes.data, a.ndim, a.shape) for a in all_inputs]
            sret = (ctypes.c_uint8 * DEFAULT_SRET_SIZE)()
            args = [ctypes.byref(sret)] + [ctypes.byref(m) for m in mrs]
            k = lib._mlir_ciface_main_0
            k.argtypes = [ctypes.c_void_p] * len(args)
            k.restype = None
            k(*args)

            sb = bytes(sret)
            off = 12 * 40  # output[12] offset
            al = struct.unpack_from("<Q", sb, off + 8)[0]
            sz = tuple(struct.unpack_from("<q", sb, off + 24 + 8 * i)[0] for i in range(3))
            assert sz == (2, 4, 768), f"Wrong standalone shape: {sz}"
            n = int(np.prod(sz))
            standalone_emb = np.array((ctypes.c_float * n).from_address(al), dtype=np.float32).reshape(sz)

            # Compare with Python executor
            from scripts.ctypes_forward import run_python_executor

            py_result = run_python_executor(ARTIFACT_DIR)
            py_layer0 = py_result.func_outputs[1][0]
            standalone_cos = cosine_similarity(standalone_emb.ravel(), py_layer0.ravel())
            print(f"\n  Standalone main_0 vs py: cos={standalone_cos:.8f}")

            # Compare with full-dylib main_0
            from scripts.ctypes_forward import run_ctypes

            ctypes_result = run_ctypes(ARTIFACT_DIR, dylib_path=f"{ARTIFACT_DIR}/libmodel.dylib")
            full_emb = ctypes_result.func_outputs[0][12]
            full_cos = cosine_similarity(standalone_emb.ravel(), full_emb.ravel())
            print(f"  Standalone vs full-dylib: cos={full_cos:.8f}")

            if standalone_cos >= 0.9999:
                print("  ✓ Standalone main_0 pipeline is CORRECT")
                print("  → Bug is in full-dylib compilation path (compile_dylib.py)")
            else:
                print("  ✗ Standalone main_0 pipeline is ALSO WRONG")
                print("  → Bug is in pipeline stages for this specific IR")

            assert standalone_cos >= 0.9999, (
                f"Standalone main_0 cos={standalone_cos:.8f} < 0.9999. Bug is in pipeline stages."
            )


def _apply_sf_to_linalg(sf_mlir: str) -> str:
    from compiler.pipeline import _apply_sf_to_linalg as _f

    return _f(sf_mlir)
