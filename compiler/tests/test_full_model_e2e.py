"""Contract test: full model compilation through SF→linalg→LLVM→dylib
vs HuggingFace reference.

Validates that the complete compiler pipeline (including dynamic dimensions,
16 functions, 198 args) produces numerically correct output (cos >= 0.9999)
when compiled to a shared library.

This is the contract test that links the compiler to the runtime —
if it passes, the dylib cos vs HF is guaranteed >= 0.9999.
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import safetensors.torch
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from compiler.backend.compile_utils import (
    _compile_serveforge_free,
    _find_llc,
    _setup_mlir_path,
    link_dylib,
)
from compiler.backend.fixups import _fixup_unrealized_casts_pass
from compiler.backend.llvm_backend import lower_linalg_to_llvm_ir
from compiler.pipeline.lowering import SF_LOWERING_PIPELINE
from compiler.dylib_ffi import DEFAULT_SRET_SIZE

_setup_mlir_path()
import mlir.ir as ir  # noqa: E402
import mlir.passmanager as pm  # noqa: E402
from mlir_sf._mlir_libs._sfDialectsNanobind import sf  # noqa: E402

# --- constants ---

MODEL_DIR = ROOT / "outputs" / "compiled" / "opt_125m_fresh"
SAFETENSORS_PATH = Path(
    "/Users/zeng/.cache/huggingface/hub/models--facebook--opt-125m/"
    "snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6/model.safetensors"
)

# Prompt that produces non-trivial token activations
TEST_PROMPT = "The quick brown fox jumps over the lazy dog"


def _memref(ptr, ndim, shape):
    """Create a memref descriptor struct matching MLIR's ABI."""
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


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two arrays."""
    a_flat = a.astype(np.float64).ravel()
    b_flat = b.astype(np.float64).ravel()
    dot = float(np.dot(a_flat, b_flat))
    norm_a = float(np.linalg.norm(a_flat))
    norm_b = float(np.linalg.norm(b_flat))
    if norm_a == 0 and norm_b == 0:
        return 1.0
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _unpack_sret(sret_bytes: bytes) -> np.ndarray:
    """Unpack output from the sret buffer (memref descriptor + float data)."""
    sb = bytes(sret_bytes)
    # sret layout: 8 bytes padding + 8 bytes allocated ptr + 8 bytes aligned ptr
    # + 8 bytes offset + rank*8 sizes + rank*8 strides + float data
    al = struct.unpack_from("<Q", sb, 8)[0]  # allocated pointer
    # Determine rank from output — for logits it's always 3D (batch, seq, vocab)
    rank = 3
    sz = tuple(struct.unpack_from("<q", sb, 24 + 8 * i)[0] for i in range(rank))
    n = int(np.prod(sz))
    arr = np.array((ctypes.c_float * n).from_address(al), dtype=np.float32)
    return arr.reshape(sz)


def _compile_dylib(mlir_text: str, td: str, dylib_name: str = "test.dylib") -> str:
    """Compile MLIR text to a .dylib shared library.

    Returns the path to the compiled dylib.
    """
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    sf.register_dialects(ctx._CAPIPtr, load=True)
    with ir.Location.unknown(ctx):
        mod = ir.Module.parse(mlir_text, ctx)
        pman = pm.PassManager.parse(f"builtin.module({SF_LOWERING_PIPELINE})", ctx)
        pman.enable_verifier(True)
        pman.run(mod.operation)
        lower_linalg_to_llvm_ir(mod)
        _fixup_unrealized_casts_pass(mod)

        m_path = os.path.join(td, "m.mlir")
        ll_path = os.path.join(td, "m.ll")
        o_path = os.path.join(td, "m.o")
        dylib_path = os.path.join(td, dylib_name)

        with open(m_path, "w") as f:
            f.write(str(mod))

        subprocess.run(
            ["./llvm-project/build/bin/mlir-translate", "--mlir-to-llvmir", m_path, "-o", ll_path],
            capture_output=True,
            check=True,
            timeout=120,
        )
        subprocess.run(
            [_find_llc(), "-filetype=obj", ll_path, "-o", o_path],
            capture_output=True,
            check=True,
            timeout=120,
        )
        free_o = _compile_serveforge_free(td)
        link_dylib([o_path, free_o], dylib_path)

        return dylib_path


def _extract_lowered_weight_names(mlir_text: str) -> list[str]:
    """Extract non-const weight names from sf.weight_names in lowered MLIR."""
    # Match the sf.weight_names format in generic MLIR:
    # {sf.weight_names = #sf<weight_names["name1", "name2", ...]>}
    match = re.search(r"sf\.weight_names\s*=\s*#sf<weight_names\[([^\]]+)\]", mlir_text)
    if not match:
        raise ValueError("Could not find sf.weight_names in lowered MLIR")
    names = re.findall(r'"([^"]+)"', match.group(1))
    return [w for w in names if "_const_" not in w]


@pytest.mark.integration
@pytest.mark.timeout(300)
class TestFullModelE2E:
    """Compile full opt-125m model through complete pipeline and compare with HF."""

    def test_dylib_vs_hf_cosine(self):
        """RED: Full model dylib cos vs HF should be >= 0.9999."""
        # 1. Load model artifacts
        mlir_path = MODEL_DIR / "model.mlir"
        if not mlir_path.exists():
            pytest.skip(f"model.mlir not found at {mlir_path} — run compile.py first")
        model_mlir = mlir_path.read_text()

        meta_path = MODEL_DIR / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)
            hf_key_map = metadata.get("hf_key_map", {})
        else:
            hf_key_map = {}

        # 2. Load safetensors and HF model
        st = safetensors.torch.load_file(str(SAFETENSORS_PATH))
        st_numpy = {k: v.numpy().astype(np.float32) for k, v in st.items()}

        hf_model = AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m",
            local_files_only=True,
            device_map="cpu",
            torch_dtype=torch.float32,  # match dylib fp32
        )
        hf_model.eval()
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", local_files_only=True)

        # 3. Prepare reference: HF forward pass
        inputs = tokenizer(TEST_PROMPT, return_tensors="pt")
        input_ids_hf = inputs["input_ids"]
        batch, seq = input_ids_hf.shape
        with torch.no_grad():
            hf_output = hf_model(input_ids=input_ids_hf)
            hf_logits = hf_output.logits.numpy().astype(np.float32)

        # 4. Prepare dylib inputs
        # Run SF lowering on model.mlir to extract weight order
        ctx_tmp = ir.Context()
        ctx_tmp.allow_unregistered_dialects = True
        sf.register_dialects(ctx_tmp._CAPIPtr, load=True)
        with ir.Location.unknown(ctx_tmp):
            tmp_mod = ir.Module.parse(model_mlir, ctx_tmp)
            tmp_pm = pm.PassManager.parse(f"builtin.module({SF_LOWERING_PIPELINE})", ctx_tmp)
            tmp_pm.run(tmp_mod.operation)
            lowered_text = str(tmp_mod)

        weight_names = _extract_lowered_weight_names(lowered_text)
        print(f"\nExtracted {len(weight_names)} non-const weight names from lowered MLIR")

        # Build input arrays in the correct order
        # arg 0: input_ids (int64, batch x seq)
        input_ids_np = input_ids_hf.numpy().astype(np.int64)
        inputs = [input_ids_np]

        # args 1..N: weights in lowered weight_names order
        for wn in weight_names:
            hf_key = hf_key_map.get(wn, wn)
            if hf_key in st_numpy:
                inputs.append(st_numpy[hf_key])
            else:
                raise KeyError(
                    f"Weight '{wn}' (HF key: '{hf_key}') not found in safetensors. "
                    f"Available keys: {list(st_numpy.keys())[:5]}..."
                )

        print(f"Total dylib inputs: {len(inputs)} (1 input_ids + {len(inputs) - 1} weights)")

        # 5. Compile and run dylib
        with tempfile.TemporaryDirectory() as td:
            dylib_path = _compile_dylib(model_mlir, td)
            print(f"Compiled dylib: {os.path.getsize(dylib_path)} bytes")

            lib = ctypes.CDLL(dylib_path)
            mrs = [_memref(a.ctypes.data, a.ndim, a.shape) for a in inputs]
            sret = (ctypes.c_uint8 * DEFAULT_SRET_SIZE)()
            args = [ctypes.byref(sret)] + [ctypes.byref(mr) for mr in mrs]
            lib._mlir_ciface_main_0.argtypes = [ctypes.c_void_p] * len(args)
            lib._mlir_ciface_main_0.restype = None
            lib._mlir_ciface_main_0(*args)

            dylib_logits = _unpack_sret(sret)
            # Reshape to match HF logits shape: (batch, seq, vocab)
            if dylib_logits.shape != hf_logits.shape:
                # sret might return a larger tensor (allocated size vs actual dims)
                # Trim to match HF shape
                dylib_logits = dylib_logits[: hf_logits.shape[0], : hf_logits.shape[1], : hf_logits.shape[2]]

        # 6. Compare
        cos = _cosine_similarity(dylib_logits, hf_logits)
        mae = float(np.abs(dylib_logits.astype(np.float64) - hf_logits.astype(np.float64)).mean())
        max_abs_err = float(np.abs(dylib_logits.astype(np.float64) - hf_logits.astype(np.float64)).max())

        print(
            f"\nComparison results:\n"
            f"  cos={cos:.8f}\n"
            f"  MAE={mae:.6e}\n"
            f"  max_abs_err={max_abs_err:.6e}\n"
            f"  dylib mean={dylib_logits.mean():.6f}\n"
            f"  HF mean={hf_logits.mean():.6f}\n"
            f"  dylib shape={dylib_logits.shape}\n"
            f"  HF shape={hf_logits.shape}"
        )

        assert cos >= 0.9999, (
            f"Full model cos={cos:.8f} < 0.9999 threshold.\n"
            f"MAE={mae:.2e}, max_abs_err={max_abs_err:.2e}\n"
            f"Gap indicates a compiler bug in the full-model dynamic-dimension path."
        )
