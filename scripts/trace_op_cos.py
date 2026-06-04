#!/usr/bin/env python3
"""Trace per-op cosine similarity through func[1]'s op chain.

For each compute op in func[1], runs both PyTorchBackend and JIT-compiled
versions with the REAL inputs (from dylib func[0] outputs + weights), then
compares cosine similarity.

Usage:
    source .venv/bin/activate
    python scripts/trace_op_cos.py
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Suppress verbose runner logging ───────────────────────────────────
logging.basicConfig(level=logging.WARNING)
for name in (
    "tests.op_correctness.runner",
    "compiler.mlir_dialect.fixups",
    "compiler.fixups",
):
    logging.getLogger(name).setLevel(logging.WARNING)


from compiler.mlir_artifact import load_mlir_artifact
from engine.mlir_executor import _WEIGHT_OPS
from hal.pytorch_backend import PyTorchBackend
from scripts._cos import cosine_similarity
from scripts.ctypes_forward import run_ctypes
from tests.op_correctness.registry import OpCase
from tests.op_correctness.runner import generate_mlir, invoke_and_extract, lower_and_jit

_log = logging.getLogger("trace_op_cos")


# ══════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════


def _strip_debug_attrs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Remove debug-only attributes that generate_mlir can't serialize."""
    skip = {"dump_layer"}
    return {k: v for k, v in kwargs.items() if k not in skip}


def _resolve_view_shape(
    kwargs: dict[str, Any], ssa: dict[str, torch.Tensor]
) -> list[int]:
    """Resolve symbolic shape attrs for sf.view ops to actual ints.

    The parsed MLIR shape attr may contain strings like ``'%sym_size_int_32'``
    referencing SSA values (tensor<1xf32> scalars).  Replace them with the
    actual integer value.
    """
    raw = kwargs.get("shape", [])
    resolved: list[int] = []
    for s in raw:
        if isinstance(s, str):
            clean = s.replace("%", "")
            if clean in ssa:
                val = int(ssa[clean].item())
                resolved.append(val)
            else:
                resolved.append(s)  # keep as-is (shouldn't happen)
        else:
            resolved.append(s)
    return resolved


def _build_weight_dict(artifact: Any) -> dict[str, torch.Tensor]:
    """Build flattenend weight dict matching mlir_executor logic."""
    w: dict[str, torch.Tensor] = {}
    for func in artifact.functions:
        for name, tensor in func.weights.items():
            if name not in w:
                w[name] = tensor.float() if tensor.is_floating_point() else tensor

    # hf_key_map: compiled_name → HF key
    hfk = artifact.metadata.get("hf_key_map", {})
    for compiled_name, hf_key in hfk.items():
        if compiled_name not in w and hf_key in w:
            w[compiled_name] = w[hf_key]

    # Bare constant names (strip function prefix)
    for func in artifact.functions:
        prefix = func.name + "."
        for key in list(w.keys()):
            if key.startswith(prefix):
                bare = key[len(prefix) :]
                if bare not in w:
                    w[bare] = w[key]

    return w


def _resolve_inputs(
    op: Any,
    ssa: dict[str, torch.Tensor],
    all_weights: dict[str, torch.Tensor],
) -> list[torch.Tensor]:
    """Resolve an op's operands to actual torch tensors."""
    tensors: list[torch.Tensor] = []
    for inp_name in op.operands:
        clean = inp_name.replace("%", "")
        if inp_name in ssa:
            tensors.append(ssa[inp_name])
        elif clean in ssa:
            tensors.append(ssa[clean])
        elif inp_name in all_weights:
            tensors.append(all_weights[inp_name])
        elif clean in all_weights:
            tensors.append(all_weights[clean])
        else:
            raise KeyError(
                f"Cannot resolve operand '{inp_name}' (clean='{clean}')"
            )
    return tensors


def _jit_compare_op(
    i: int,
    op: Any,
    input_tensors: list[torch.Tensor],
    py_result: torch.Tensor,
    ssa: dict[str, torch.Tensor],
) -> float | None:
    """JIT-compile a single op, compare output vs PyTorchBackend.

    Returns cosine similarity, or ``None`` on failure.
    """
    kwargs = _strip_debug_attrs(dict(op.attributes))

    # ── sf.view: filter extra symbolic-scalar operands ──────────
    if op.op_name == "view":
        kwargs["shape"] = _resolve_view_shape(kwargs, ssa)
        jit_inputs = input_tensors[:1]  # tensor only
    else:
        jit_inputs = input_tensors

    input_shapes = [list(t.shape) for t in jit_inputs]
    output_shape = list(py_result.shape)

    # Handle scalar output (rank-0)
    if not output_shape:
        output_shape = [1]

    case = OpCase(
        f"sf.{op.op_name}",
        lambda *args: None,
        input_shapes,
        kwargs=kwargs,
        output_shapes=[tuple(output_shape)],
    )

    mlir_text = generate_mlir(case)
    engine, _ = lower_and_jit(mlir_text)

    try:
        input_arrays = [t.detach().cpu().numpy() for t in jit_inputs]
        jit_output = invoke_and_extract(
            engine, "main", input_arrays, tuple(output_shape)
        )

        py_np = py_result.detach().cpu().numpy()

        # Handle shape mismatches (rank-0 vs rank-1)
        if jit_output.shape != py_np.shape:
            if jit_output.ndim == 0 and py_np.ndim == 1 and py_np.shape[0] == 1:
                py_np = py_np.reshape(())
            elif py_np.ndim == 0 and jit_output.ndim == 1 and jit_output.shape[0] == 1:
                jit_output = jit_output.reshape(())
            else:
                _log.warning(
                    "op[%d] %s: shape mismatch JIT=%s vs PY=%s",
                    i, op.op_name, jit_output.shape, py_np.shape,
                )

        cos = cosine_similarity(jit_output, py_np)
        return cos
    finally:
        del engine


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════


def main() -> None:
    model_dir = "compiled/opt_125m_fresh"

    # 1. Load artifact (MLIR text + weights)
    artifact = load_mlir_artifact(model_dir)
    all_weights = _build_weight_dict(artifact)

    # 2. Run dylib to seed dynamic function inputs
    dylib = run_ctypes(model_dir, dylib_path=f"{model_dir}/libopt_125m.dylib")
    dylib_outs = dylib._func_outputs  # list[list[np.ndarray]]

    func = artifact.functions[1]  # main_1 — first decoder layer

    # 3. Bind dylib func[0] outputs → func[1] inputs (from ctypes_forward)
    #    (0,209)=sym_size_int_32, (0,13)=expand, (0,12)=add_5, (0,10)=_const_57,
    #    (0,17..32)=weight tensors, (0,210)=sym_size_int_33
    bindings = (
        [(0, 209), (0, 13), (0, 12), (0, 10)]
        + [(0, o) for o in range(17, 33)]
        + [(0, 210)]
    )

    ssa: dict[str, torch.Tensor] = {}
    for ai, (name, _typ) in enumerate(func.inputs):
        src_func, src_out = bindings[ai]
        arr = dylib_outs[src_func][src_out]
        t = torch.from_numpy(arr.copy()).float()
        clean = name.replace("%", "")
        ssa[name] = t
        ssa[clean] = t

    backend = PyTorchBackend("cpu")

    print(f"func[1] ({func.name}): {len(func.ops)} ops")
    print(f"{'op#':>4s} {'name':30s} {'cos':>10s} {'shape':>28s}")
    print("-" * 78)

    for i, op in enumerate(func.ops):
        # ── Weight ops (if any) — store in SSA, skip comparison ──
        if op.name in _WEIGHT_OPS:
            wname = op.attributes.get("name", "") or op.attributes.get('"name"', "")
            if wname in all_weights and op.results:
                ssa[op.results[0]] = all_weights[wname]
            continue

        # ── Resolve input tensors ────────────────────────────────
        try:
            input_tensors = _resolve_inputs(op, ssa, all_weights)
        except KeyError as e:
            print(f"{i:4d} {op.op_name:30s} {'NO_INPUT':>10s}  [{e}]")
            continue

        # ── Run PyTorchBackend ───────────────────────────────────
        try:
            py_result = backend.execute(op.op_name, input_tensors, **op.attributes)
        except Exception as e:
            print(f"{i:4d} {op.op_name:30s} {'PY_ERR':>10s}  [{type(e).__name__}]")
            continue

        if py_result is None or not op.results:
            continue

        # ── Store in SSA for downstream ops ──────────────────────
        ssa[op.results[0]] = py_result

        # ── JIT-compile & compare ────────────────────────────────
        cos = _jit_compare_op(i, op, input_tensors, py_result, ssa)
        if cos is not None:
            shape_str = str(list(py_result.shape))
            print(f"{i:4d} {op.op_name:30s} {cos:10.6f} {shape_str:>28s}")
        else:
            shape_str = str(list(py_result.shape))
            print(f"{i:4d} {op.op_name:30s} {'JIT_ERR':>10s} {shape_str:>28s}")


if __name__ == "__main__":
    main()
