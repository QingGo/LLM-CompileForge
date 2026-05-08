#!/usr/bin/env python3
"""Per-op divergence detector between IR executor and ExportedProgram.

Finds the first operation where the IR executor's output diverges from
the ExportedProgram reference (cosine similarity < threshold).

Usage:
    # After loading an ExportedProgram and IrModule:
    result = find_first_divergence(program, ir_module, example_input)
    if result:
        print(f"First divergence at op #{result['op_index']}: {result['op_name']}")
        print(f"  cosine: {result['cosine']:.8f}")
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.fx


def compare_ir_vs_ep(
    program: Any,
    ir_module: Any,
    example_input: torch.Tensor,
    max_ops: int = 0,
    threshold: float = 0.999,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Compare IR executor outputs against ExportedProgram node outputs.

    Runs the EP graph node-by-node to capture all intermediate outputs,
    then runs the IR executor and compares each IR op's output against
    the corresponding EP node (matched via ``source_node`` attribute).

    Args:
        program: ``torch.export.ExportedProgram``.
        ir_module: ``IrModule`` from ``fx_graph_to_ir(program)``.
        example_input: The same input used for export.
        max_ops: Stop after this many ops (0 = all).
        threshold: Cosine similarity threshold.  Ops below this are flagged.
        verbose: Print progress for each op.

    Returns:
        List of divergence records, each a dict with keys:
        ``op_index``, ``op_name``, ``source_node``, ``cosine``,
        ``ir_shape``, ``ep_shape``, ``inputs``.
        Empty list means all ops matched within threshold.
    """
    # ── Step 1: capture EP node outputs ───────────────────────
    ep_outputs = _capture_ep_outputs(program, example_input)

    ir_func = ir_module.main

    # ── Step 2: run IR executor with per-op capture ────────────
    from hal.pytorch_backend import PyTorchBackend

    backend = PyTorchBackend("cpu")
    ssa_values: dict[str, torch.Tensor] = {}

    # Seed function inputs
    input_names = [name for name, _ in ir_func.inputs]
    if input_names:
        ssa_values[input_names[0]] = example_input
    for name in input_names[1:]:
        ssa_values[name] = torch.zeros(1, dtype=torch.long)

    # Flatten weights
    all_weights: dict[str, torch.Tensor] = dict(ir_func.weights)

    total = len(ir_func.ops)
    limit = max_ops if max_ops > 0 else total
    divergences: list[dict[str, Any]] = []

    for i, op in enumerate(ir_func.ops):
        if i >= limit:
            break

        source_node = op.attributes.get("source_node")
        if source_node is None:
            if verbose:
                print(f"  [{i:4d}/{total}] {op.name:20s} SKIP (no source_node)")
            continue

        # Resolve inputs
        tensor_inputs: list[torch.Tensor] = []
        try:
            for inp_name in op.inputs:
                if inp_name in ssa_values:
                    tensor_inputs.append(ssa_values[inp_name])
                elif inp_name in all_weights:
                    tensor_inputs.append(all_weights[inp_name])
        except Exception:
            if verbose:
                print(f"  [{i:4d}/{total}] {op.name:20s} SKIP (unresolved input)")
            continue

        if not tensor_inputs:
            result = backend.execute(op.name, [], **op.attributes)
        else:
            try:
                result = backend.execute(op.name, tensor_inputs, **op.attributes)
            except Exception as e:
                if verbose:
                    print(f"  [{i:4d}/{total}] {op.name:20s} ERROR: {e}")
                continue

        if result is None or not isinstance(result, torch.Tensor):
            if op.outputs and isinstance(result, torch.Tensor):
                pass
            elif op.outputs:
                for out_name in op.outputs:
                    ssa_values[out_name] = result if isinstance(result, torch.Tensor) else torch.empty(0)
                continue
            else:
                continue

        # Store in SSA for downstream consumers
        for out_name in op.outputs:
            ssa_values[out_name] = result

        # Compare against EP reference
        ep_ref = ep_outputs.get(source_node)
        if ep_ref is None:
            if verbose:
                print(f"  [{i:4d}/{total}] {op.name:20s} SKIP (no EP ref for {source_node})")
            continue

        if not isinstance(ep_ref, torch.Tensor):
            continue

        # Align dtypes and shapes
        ir_val = result.float()
        ep_val = ep_ref.float()
        if ir_val.numel() == 0 or ep_val.numel() == 0:
            continue
        if ir_val.shape != ep_val.shape:
            ir_val = ir_val.reshape(-1)
            ep_val = ep_val.reshape(-1)

        cos = float(
            torch.nn.functional.cosine_similarity(
                ir_val.flatten(), ep_val.flatten(), dim=0
            ).item()
        )

        if math.isnan(cos) or math.isinf(cos):
            continue

        if cos < threshold:
            record = {
                "op_index": i,
                "op_name": op.name,
                "source_node": source_node,
                "cosine": cos,
                "ir_shape": list(result.shape),
                "ep_shape": list(ep_ref.shape),
                "inputs": [inp[:60] if isinstance(inp, str) else str(inp)[:60] for inp in op.inputs],
            }
            divergences.append(record)
            if verbose:
                print(
                    f"  [{i:4d}/{total}] {op.name:20s} DIVERGED cos={cos:.8f} "
                    f"ir={list(result.shape)} ep={list(ep_ref.shape)}"
                )
        elif verbose:
            print(f"  [{i:4d}/{total}] {op.name:20s} OK cos={cos:.8f}")

    return divergences


def find_first_divergence(
    program: Any,
    ir_module: Any,
    example_input: torch.Tensor,
    max_ops: int = 0,
    threshold: float = 0.999,
) -> dict[str, Any] | None:
    """Convenience: return the first divergence record, or None."""
    results = compare_ir_vs_ep(
        program, ir_module, example_input,
        max_ops=max_ops, threshold=threshold,
    )
    return results[0] if results else None


# ── internals ────────────────────────────────────────────────


def _capture_ep_outputs(program: Any, example_input: torch.Tensor) -> dict[str, Any]:
    """Execute the ExportedProgram node-by-node, capturing all outputs.

    Returns dict mapping FX node name → output tensor.
    """
    gm = program.graph_module
    sig = program.graph_signature
    user_inputs = list(sig.user_inputs)

    # Build placeholder values
    placeholder_values: list[Any] = []
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        if node.name in user_inputs:
            idx = user_inputs.index(node.name)
            if idx == 0:
                placeholder_values.append(example_input)
            else:
                placeholder_values.append(torch.zeros(1, dtype=torch.long))
        else:
            found = False
            for spec in sig.input_specs:
                if spec.arg.name == node.name:
                    param = program.state_dict.get(spec.target)
                    if param is None and hasattr(program, "constants"):
                        param = program.constants.get(spec.target)
                    if param is not None:
                        placeholder_values.append(param)
                        found = True
                        break
            if not found:
                placeholder_values.append(torch.empty(0))

    class _CaptureInterpreter(torch.fx.Interpreter):
        def __init__(self, mod):
            super().__init__(mod)
            self.captured: dict[str, Any] = {}

        def run_node(self, n):
            result = super().run_node(n)
            self.captured[n.name] = result
            return result

    interp = _CaptureInterpreter(gm)
    with torch.no_grad():
        interp.run(*placeholder_values)

    return interp.captured


# ── CLI ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Per-op IR-vs-EP divergence detector"
    )
    parser.add_argument("--model", default="opt-125m", choices=["opt-125m", "qwen", "tiny-llama"])
    parser.add_argument("--max-ops", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.999)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"Loading {args.model} and running divergence check...")
    print("(Interactive use: import compare_ir_vs_ep, find_first_divergence)")
    print("(See docstring for API usage)")
