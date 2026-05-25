#!/usr/bin/env python3
"""Trace func[0]'s mask construction op-by-op: new_ones → logical_and → index → logical_and → expand.

Compares JIT-compiled MLIR vs PyTorch reference at each step.
Starts from a known-correct causal mask (sf.le output).

Usage:
    source .venv/bin/activate
    python scripts/trace_mask_chain.py
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import numpy as np
import torch

sys.path.insert(0, ".")

# ── Suppress verbose runner logging ───────────────────────────────────
logging.basicConfig(level=logging.WARNING)
for name in (
    "tests.op_correctness.runner",
    "compiler.mlir_dialect.fixups",
    "compiler.fixups",
):
    logging.getLogger(name).setLevel(logging.WARNING)

from hal.pytorch_backend import PyTorchBackend  # noqa: E402
from scripts._cos import cosine_similarity  # noqa: E402
from tests.op_correctness.runner import invoke_and_extract, lower_and_jit  # noqa: E402

_log = logging.getLogger("trace_mask_chain")


# ══════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════


def _mlir_type(shape: tuple[int, ...], dtype: str = "f32") -> str:
    """Build an MLIR tensor type string."""
    if not shape:
        return f"tensor<{dtype}>"
    dims = "x".join(str(d) for d in shape)
    return f"tensor<{dims}x{dtype}>"


def make_mlir(
    op_name: str,
    input_shapes: list[tuple[int, ...]],
    input_dtypes: list[str],
    output_shape: tuple[int, ...],
    output_dtype: str = "f32",
    kwargs: dict[str, Any] | None = None,
) -> str:
    """Generate MLIR text for a single sf op with custom dtypes."""
    input_types = [_mlir_type(s, d) for s, d in zip(input_shapes, input_dtypes, strict=True)]
    output_type = _mlir_type(output_shape, output_dtype)

    inputs_str = ", ".join(f"%arg{i}: {t}" for i, t in enumerate(input_types))
    input_vals = ", ".join(f"%arg{i}" for i in range(len(input_types)))
    input_type_str = ", ".join(input_types)

    attrs_str = ""
    if kwargs:
        parts: list[str] = []
        for k, v in kwargs.items():
            if isinstance(v, bool):
                parts.append(f'{k} = {"true" if v else "false"}')
            elif isinstance(v, int):
                parts.append(f"{k} = {v} : i64")
            elif isinstance(v, float):
                parts.append(f"{k} = {v} : f64")
            elif isinstance(v, str):
                parts.append(f'{k} = "{v}"')
            elif isinstance(v, (list, tuple)):
                elts = ", ".join(str(x) for x in v)
                parts.append(f"{k} = [{elts}]")
        if parts:
            attrs_str = "{" + ", ".join(parts) + "} "

    return (
        f"module {{\n"
        f"  func.func @main({inputs_str}) -> {output_type} {{\n"
        f"    %0 = \"{op_name}\"({input_vals}) {attrs_str}: "
        f"({input_type_str}) -> {output_type}\n"
        f"    return %0 : {output_type}\n"
        f"  }}\n"
        f"}}"
    )


def run_jit(
    mlir_text: str, input_arrays: list[np.ndarray[Any, Any]], output_shape: tuple[int, ...]
) -> np.ndarray[Any, Any]:
    """Lower and JIT compile MLIR, invoke with inputs."""
    engine, _ = lower_and_jit(mlir_text)
    try:
        return invoke_and_extract(engine, "main", input_arrays, output_shape)
    finally:
        del engine


def test_op(
    name: str,
    mlir_text: str,
    input_arrays: list[np.ndarray[Any, Any]],
    output_shape: tuple[int, ...],
    py_reference: torch.Tensor,
) -> tuple[float, np.ndarray[Any, Any], torch.Tensor]:
    """Test a single op comparing JIT vs PyTorch reference.

    Args:
        name: Display name for the test.
        mlir_text: MLIR module text.
        input_arrays: Numpy input arrays for JIT.
        output_shape: Expected output shape.
        py_reference: PyTorch reference output tensor.

    Returns:
        Tuple of (cosine_similarity, jit_output, py_reference).
    """
    jit_out = run_jit(mlir_text, input_arrays, output_shape)
    py_np = py_reference.detach().cpu().float().numpy()
    cos = cosine_similarity(jit_out.astype(np.float32), py_np)
    print(
        f"  {name:30s}  cos={cos:.8f}  "
        f"jit={list(jit_out.shape)}  py={list(py_np.shape)}"
    )
    if cos < 0.9999:
        print(f"    ⚠  WARNING: cos={cos:.6f} < 0.9999")
        print(f"    JIT sample: {jit_out.ravel()[:10]}")
        print(f"    PY  sample: {py_np.ravel()[:10]}")
    return cos, jit_out, py_reference


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════


def main() -> None:
    backend = PyTorchBackend("cpu")

    # ── 1. Starting point: causal mask from sf.le ─────────────
    seq_len = 4
    arange = backend.execute("arange", [torch.tensor([seq_len])], device="cpu", pin_memory=False)
    row = backend.execute("unsqueeze", [arange], dim=0)        # [1, 4]
    col = backend.execute("unsqueeze", [arange], dim=1)        # [4, 1]
    causal = backend.execute("le", [col, row]).float()          # [4, 4]

    print("=" * 72)
    print("[1/6] Causal mask (sf.le(col, row)):")
    print(f"      shape={list(causal.shape)}")
    for i in range(min(4, causal.shape[0])):
        vals = " ".join(f"{v:.1f}" for v in causal[i].numpy())
        print(f"        row[{i}]: {vals}")
    print()

    # ══════════════════════════════════════════════════════════════
    #  2. sf.new_ones — create all-ones with same shape as input
    # ══════════════════════════════════════════════════════════════
    #
    # In func[0]: new_ones(unsqueeze_8) → creates [1,1,4,1] all-ones
    # Simpler: new_ones([4,4] causal) → [4,4] all-ones
    print("[2/6] sf.new_ones — all-ones tensor from shape ─────")
    ones_mlir = make_mlir("sf.new_ones", [(4, 4)], ["f32"], (4, 4))
    causal_np = causal.numpy()
    cos_ones, jit_ones, _ = test_op(
        "new_ones",
        ones_mlir,
        [causal_np],
        (4, 4),
        torch.ones_like(causal),
    )

    # ══════════════════════════════════════════════════════════════
    #  3. sf.logical_and — combine padding mask + causal mask
    # ══════════════════════════════════════════════════════════════
    #
    # In func[0]: logical_and(new_ones, to_2) → and_1
    # where to_2 = identity(le_result)
    # Simpler: logical_and(ones, causal_mask)
    print("[3/6] sf.logical_and — element-wise AND ─────────────")
    ones = torch.ones_like(causal)
    and1_mlir = make_mlir(
        "sf.logical_and", [(4, 4), (4, 4)], ["f32", "f32"], (4, 4)
    )
    cos_and1, jit_and1, _ = test_op(
        "logical_and(ones, causal)",
        and1_mlir,
        [ones.numpy(), causal_np],
        (4, 4),
        torch.logical_and(ones, causal.bool()).float(),
    )

    # ══════════════════════════════════════════════════════════════
    #  4. sf.index — gather from data tensor using index tensors
    # ══════════════════════════════════════════════════════════════
    #
    # In func[0]: index(to_1, unsqueeze_2, slice_5)
    #   to_1 = identity(ones_like) → [2, 4] all-ones
    #   unsqueeze_2 → [2,1,1,1] batch indices
    #   slice_5 → [1,1,4,?] row indices
    #
    # Simpler: index(data=[4], idx=[2]) → data[idx] = [2]
    print("[4/6] sf.index — gather (1D) ───────────────────────")
    data_1d = torch.tensor([10.0, 20.0, 30.0, 40.0])
    idx_1d = torch.tensor([2, 0], dtype=torch.int64)
    index_mlir = make_mlir(
        "sf.index", [(4,), (2,)], ["f32", "i64"], (2,)
    )
    cos_idx1, jit_idx1, _ = test_op(
        "index([10,20,30,40], [2,0])",
        index_mlir,
        [data_1d.numpy(), idx_1d.numpy()],
        (2,),
        data_1d[idx_1d],
    )



    # ══════════════════════════════════════════════════════════════
    #  5. sf.expand — broadcast tensor to larger shape
    # ══════════════════════════════════════════════════════════════
    #
    # In func[0]: expand(and_2, [2,1,4,4]) → final mask
    # Simpler: expand([1,4] → [2,4]) broadcasts dim 0
    print("[5/6] sf.expand — broadcast ─────────────────────────")
    expand_in = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # [1, 4]
    expand_mlir = make_mlir("sf.expand", [(1, 4)], ["f32"], (2, 4))
    cos_exp, jit_exp, _ = test_op(
        "expand [1,4] → [2,4]",
        expand_mlir,
        [expand_in.numpy()],
        (2, 4),
        expand_in.expand(2, 4),
    )

    # ══════════════════════════════════════════════════════════════
    #  6. Chain: new_ones → logical_and → expand
    # ══════════════════════════════════════════════════════════════
    #
    # This is the core mask chain from func[0], simplified:
    #   new_ones(causal) → ones
    #   logical_and(ones, causal) → masked (causal preserved)
    #   expand(masked) → broadcast to final shape
    print("[6/6] Chain: new_ones → logical_and → expand ───────")
    # causal is [4,4]; need 3D for expand to make sense
    causal_3d = causal.unsqueeze(0)  # [1, 4, 4]

    chain_mlir = (
        "module {\n"
        "  func.func @main(%arg0: tensor<1x4x4xf32>) -> tensor<2x4x4xf32> {\n"
        '    %0 = "sf.new_ones"(%arg0) : (tensor<1x4x4xf32>) -> tensor<1x4x4xf32>\n'
        '    %1 = "sf.logical_and"(%0, %arg0) : '
        "(tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<1x4x4xf32>\n"
        '    %2 = "sf.expand"(%1) : (tensor<1x4x4xf32>) -> tensor<2x4x4xf32>\n'
        "    return %2 : tensor<2x4x4xf32>\n"
        "  }\n"
        "}"
    )
    engine, out_shape = lower_and_jit(chain_mlir)
    try:
        jit_chain = invoke_and_extract(engine, "main", [causal_3d.numpy()], (2, 4, 4))
        # PyTorch reference
        py_chain_ones = torch.ones_like(causal_3d)
        py_chain_and = torch.logical_and(py_chain_ones, causal_3d.bool()).float()
        py_chain = py_chain_and.expand(2, 4, 4)
        cos_chain = cosine_similarity(
            jit_chain.astype(np.float32), py_chain.float().numpy()
        )
        print(
            f"  chain(new_ones→and→expand)  cos={cos_chain:.8f}  "
            f"jit={list(jit_chain.shape)}"
        )
        if cos_chain < 0.9999:
            print(f"    ⚠  WARNING: cos={cos_chain:.6f} < 0.9999")
    finally:
        del engine

    # ══════════════════════════════════════════════════════════
    #  Summary
    # ══════════════════════════════════════════════════════════
    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    all_cos = [
        ("new_ones", cos_ones),
        ("logical_and", cos_and1),
        ("index (1D)", cos_idx1),
        ("expand", cos_exp),
        ("chain", cos_chain),
    ]
    for name, cos in all_cos:
        status = "✅" if cos >= 0.9999 else "⚠️"
        print(f"  {status}  {name:25s}  cos={cos:.8f}")
    print()
    if all(c >= 0.9999 for _, c in all_cos):
        print("✅  All ops match between JIT and PyTorch reference.")
    else:
        print("⚠️  Some ops have cos < 0.9999 — investigate above.")


if __name__ == "__main__":
    main()
