"""A.2 SDPA attention contract goldens.

The LLaMA no-mask path is defined by three co-occurring contracts:

1. ``scalar-mask``: a rank-0/1 mask operand (exported from SDPA's scalar
   ``dropout_p`` / ``attn_mask`` slot) is an additive scalar mask.  It must
   be broadcast to scores and must not enter the rank>=3 boolean-mask path.
2. ``enable_gqa=true``: K/V arrive at native KV-head shape and SDPA lowering
   expands them to the query-head count before attention math.
3. ``is_causal=true``: with no explicit boolean mask, SDPA lowering must
   synthesize the causal lower-triangular additive mask itself.

These tests compile minimal ``sf.scaled_dot_product_attention`` functions to
dylibs and compare against PyTorch's SDPA.  They are independent of runtime.
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
import torch
import torch.nn.functional as F  # noqa: N812

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def _find_tool(name: str) -> str:
    candidates = [name]
    if name in ("cc", "clang"):
        candidates.insert(0, "/usr/local/opt/llvm/bin/clang")
    candidates.append(str(ROOT / "llvm-project" / "build" / "bin" / name))
    for p in candidates:
        if Path(p).is_file():
            return str(p)
        try:
            if subprocess.run([p, "--version"], capture_output=True, timeout=5).returncode == 0:
                return p
        except FileNotFoundError:
            continue
    raise RuntimeError(f"{name} not found")


def _compile(sf_mlir: str, tmp_dir: str, name: str) -> str:
    """Compile sf MLIR -> lowered -> LLVM -> cc -> dylib."""
    import mlir.ir as ir
    from mlir_sf._mlir_libs._sfDialectsNanobind import sf  # noqa: F401

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
        ll_file = os.path.join(tmp_dir, "m.ll")
        o = os.path.join(tmp_dir, "m.o")
        d = os.path.join(tmp_dir, f"{name}.dylib")
        with open(m, "w") as f:
            f.write(str(mod))
        subprocess.run(
            [_find_tool("mlir-translate"), "--mlir-to-llvmir", m, "-o", ll_file],
            capture_output=True, text=True, check=True, timeout=60,
        )
        subprocess.run(
            [_find_tool("cc"), "-c", ll_file, "-o", o, "-O0"],
            capture_output=True, text=True, check=True, timeout=60,
        )
        free_o = _compile_serveforge_free(tmp_dir)
        link_dylib([o, free_o], d)
        return d


def _sret(size: int = 4096) -> ctypes.Array:
    return (ctypes.c_uint8 * size)()


def _memref(arr: np.ndarray) -> ctypes.Structure:
    arr = np.ascontiguousarray(arr)
    ndim = arr.ndim
    elem_strides = tuple(s // arr.itemsize for s in arr.strides)

    class M(ctypes.Structure):
        _fields_ = [
            ("allocated", ctypes.c_void_p),
            ("aligned", ctypes.c_void_p),
            ("offset", ctypes.c_int64),
            ("sizes", ctypes.c_int64 * ndim),
            ("strides", ctypes.c_int64 * ndim),
        ]

    return M(
        ctypes.c_void_p(arr.ctypes.data),
        ctypes.c_void_p(arr.ctypes.data),
        0,
        (ctypes.c_int64 * ndim)(*arr.shape),
        (ctypes.c_int64 * ndim)(*elem_strides),
    )


def _parse_sret_f32(sret_bytes: bytes, rank: int) -> np.ndarray:
    aligned = struct.unpack_from("<Q", sret_bytes, 8)[0]
    sizes = []
    for i in range(rank):
        s = struct.unpack_from("<q", sret_bytes, 24 + 8 * i)[0]
        sizes.append(s if s > 0 else 1)
    n = int(np.prod(sizes))
    if n > 0 and aligned != 0:
        buf = (ctypes.c_float * n).from_address(aligned)
        return np.array(buf, dtype=np.float32).reshape(sizes)
    return np.array([], dtype=np.float32)


def _call_main0(dylib: str, inputs: list[np.ndarray], output_rank: int) -> np.ndarray:
    lib = ctypes.CDLL(dylib)
    sret = _sret()
    kernel = lib._mlir_ciface_main_0
    kernel.argtypes = [ctypes.c_void_p] * (1 + len(inputs))
    kernel.restype = None
    args = [ctypes.byref(sret)] + [ctypes.byref(_memref(a)) for a in inputs]
    kernel(*args)
    return _parse_sret_f32(bytes(sret), output_rank)


def _make_sdpa_mlir(
    q_type: str,
    k_type: str,
    v_type: str,
    out_type: str,
    *,
    scale: float | None = None,
    scalar_mask_type: str | None = None,
    attrs: tuple[tuple[str, object], ...] = (),
) -> str:
    attr_items = []
    if scale is not None:
        attr_items.append(f"scale = {scale} : f64")
    for k, v in attrs:
        if isinstance(v, bool):
            attr_items.append(f"{k} = {'true' if v else 'false'}")
        elif isinstance(v, float):
            attr_items.append(f"{k} = {v} : f64")
        else:
            attr_items.append(f'{k} = "{v}"')
    attrs_str = f" {{{', '.join(attr_items)}}}" if attr_items else ""

    args = [f"%q: {q_type}", f"%k: {k_type}", f"%v: {v_type}"]
    operands = ["%q", "%k", "%v"]
    types = [q_type, k_type, v_type]
    if scalar_mask_type is not None:
        args.append(f"%mask: {scalar_mask_type}")
        operands.append("%mask")
        types.append(scalar_mask_type)
    return f"""module {{
  func.func @main_0({', '.join(args)}) -> {out_type} {{
    %0 = "sf.scaled_dot_product_attention"({', '.join(operands)}){attrs_str} : ({', '.join(types)}) -> {out_type}
    func.return %0 : {out_type}
  }}
}}"""


def _sdpa_mlir_op_attrs(module) -> dict:
    for func in module.functions:
        for op in func.ops:
            if op.op_name == "scaled_dot_product_attention":
                return {
                    "operands": list(op.operands),
                    "attributes": {k: v for k, v in op.attributes.items() if k != "source_node"},
                }
    raise AssertionError("no sdpa op in module")


class TestSdpaConverterContract:
    """FX positional SDPA kwargs must become attributes, not scalar operands."""

    def _export_and_convert(self, module: torch.nn.Module):
        from torch.export import export

        from compiler.fx.converter import fx_graph_to_mlir

        q = torch.randn(1, 2, 3, 4)
        k = torch.randn(1, 2, 3, 4)
        v = torch.randn(1, 2, 3, 4)
        program = export(module.eval(), (q, k, v))
        return fx_graph_to_mlir(program)

    def test_is_causal_and_dropout_are_attributes_not_operands(self) -> None:

        class M(torch.nn.Module):
            def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
                return F.scaled_dot_product_attention(
                    q, k, v, dropout_p=0.0, scale=1.0, is_causal=True
                )

        info = _sdpa_mlir_op_attrs(self._export_and_convert(M()))
        assert len(info["operands"]) == 3, info
        assert info["attributes"]["is_causal"] is True, info
        assert info["attributes"]["dropout_p"] == 0.0, info

    def test_explicit_mask_keeps_mask_operand(self) -> None:

        class M(torch.nn.Module):
            def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
                mask = torch.ones(1, 1, 3, 3, dtype=q.dtype)
                return F.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask, dropout_p=0.0, scale=1.0, is_causal=False
                )

        info = _sdpa_mlir_op_attrs(self._export_and_convert(M()))
        assert len(info["operands"]) == 4, info
        assert info["attributes"].get("is_causal", False) is False, info


@pytest.mark.integration
@pytest.mark.timeout(120)
class TestSdpaAttentionContract:
    def _run(self, sf_mlir: str, inputs: list[np.ndarray], output_rank: int) -> np.ndarray:
        with tempfile.TemporaryDirectory() as td:
            dylib = _compile(sf_mlir, td, "sdpa_contract")
            return _call_main0(dylib, inputs, output_rank)

    def test_scalar_mask_zero_is_additive_noop(self) -> None:
        """Rank-1 scalar mask 0.0 must broadcast additively and be a no-op."""
        rng = np.random.RandomState(0)
        q = rng.randn(1, 2, 3, 4).astype(np.float32)
        k = rng.randn(1, 2, 3, 4).astype(np.float32)
        v = rng.randn(1, 2, 3, 4).astype(np.float32)
        scalar_mask = np.zeros((1,), dtype=np.float32)

        actual = self._run(
            _make_sdpa_mlir(
                "tensor<1x2x3x4xf32>",
                "tensor<1x2x3x4xf32>",
                "tensor<1x2x3x4xf32>",
                "tensor<1x2x3x4xf32>",
                scale=1.0,
                scalar_mask_type="tensor<1xf32>",
            ),
            [q, k, v, scalar_mask],
            4,
        )


        expected = F.scaled_dot_product_attention(
            torch.from_numpy(q),
            torch.from_numpy(k),
            torch.from_numpy(v),
            scale=1.0,
        ).numpy().astype(np.float32)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_is_causal_without_explicit_mask(self) -> None:
        """``is_causal=true`` + no mask must equal PyTorch causal SDPA."""
        rng = np.random.RandomState(1)
        q = rng.randn(1, 2, 3, 4).astype(np.float32)
        k = rng.randn(1, 2, 3, 4).astype(np.float32)
        v = rng.randn(1, 2, 3, 4).astype(np.float32)

        actual = self._run(
            _make_sdpa_mlir(
                "tensor<1x2x3x4xf32>",
                "tensor<1x2x3x4xf32>",
                "tensor<1x2x3x4xf32>",
                "tensor<1x2x3x4xf32>",
                scale=1.0,
                attrs=(("is_causal", True),),
            ),
            [q, k, v],
            4,
        )


        expected = F.scaled_dot_product_attention(
            torch.from_numpy(q),
            torch.from_numpy(k),
            torch.from_numpy(v),
            scale=1.0,
            is_causal=True,
        ).numpy().astype(np.float32)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_enable_gqa_expands_native_kv_heads(self) -> None:
        """``enable_gqa=true`` must expand 1 KV head to 4 query heads."""
        rng = np.random.RandomState(2)
        q = rng.randn(1, 4, 3, 4).astype(np.float32)
        k = rng.randn(1, 1, 3, 4).astype(np.float32)
        v = rng.randn(1, 1, 3, 4).astype(np.float32)

        actual = self._run(
            _make_sdpa_mlir(
                "tensor<1x4x3x4xf32>",
                "tensor<1x1x3x4xf32>",
                "tensor<1x1x3x4xf32>",
                "tensor<1x4x3x4xf32>",
                scale=1.0,
                attrs=(("enable_gqa", True),),
            ),
            [q, k, v],
            4,
        )


        q_t = torch.from_numpy(q)
        k_t = torch.from_numpy(k).repeat_interleave(4, dim=1)
        v_t = torch.from_numpy(v).repeat_interleave(4, dim=1)
        expected = F.scaled_dot_product_attention(q_t, k_t, v_t, scale=1.0).numpy().astype(np.float32)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_llama_nomask_dynamic_gqa_causal_contract(self) -> None:
        """Combined actual LLaMA contract: dynamic seq + scalar mask + GQA + causal."""
        rng = np.random.RandomState(3)
        q = rng.randn(1, 4, 2, 4).astype(np.float32)
        k = rng.randn(1, 1, 2, 4).astype(np.float32)
        v = rng.randn(1, 1, 2, 4).astype(np.float32)
        scalar_mask = np.zeros((1,), dtype=np.float32)

        actual = self._run(
            _make_sdpa_mlir(
                "tensor<1x4x?x4xf32>",
                "tensor<1x1x?x4xf32>",
                "tensor<1x1x?x4xf32>",
                "tensor<1x4x?x4xf32>",
                scale=0.5,
                scalar_mask_type="tensor<1xf32>",
                attrs=(("enable_gqa", True), ("is_causal", True)),
            ),
            [q, k, v, scalar_mask],
            4,
        )


        q_t = torch.from_numpy(q)
        k_t = torch.from_numpy(k).repeat_interleave(4, dim=1)
        v_t = torch.from_numpy(v).repeat_interleave(4, dim=1)
        expected = F.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            scale=0.5,
            is_causal=True,
        ).numpy().astype(np.float32)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
