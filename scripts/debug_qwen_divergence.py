#!/usr/bin/env python3
"""EP-to-IR op-by-op divergence debugger for Qwen3.5-0.8B.

Finds the first operation where the IR executor's output diverges from the
ExportedProgram reference (cosine similarity < 0.999).

Usage:
    python scripts/debug_qwen_divergence.py [--max-ops N] [--threshold T]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

import torch
import torch.fx

# ── Ensure project root is on path ────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _patch_transformers_torch() -> None:
    """Patch transformers to work with the conda-symlinked torch installation."""
    try:
        from importlib import metadata as _meta

        import transformers  # type: ignore[import-untyped]

        for _pkg_name in list(_meta.packages_distributions().keys()):
            if _pkg_name == "torch":
                if not hasattr(transformers, "is_torch_available"):
                    transformers.is_torch_available = lambda: True  # type: ignore[attr-defined]
                if not hasattr(transformers.utils, "is_torch_available"):
                    transformers.utils.is_torch_available = lambda: True  # type: ignore[attr-defined]

        _has_flag = getattr(transformers, "_torch_available", None)
        if _has_flag is not None and callable(_has_flag):
            transformers._torch_available = True  # type: ignore[assignment]
        if hasattr(transformers.utils, "_torch_available"):
            transformers.utils._torch_available = True  # type: ignore[assignment]
        if hasattr(transformers, "is_torch_available"):
            if not transformers.is_torch_available():  # type: ignore[attr-defined]
                transformers.is_torch_available = lambda: True  # type: ignore[assignment]

        from torch._functorch import pytree
        if not hasattr(transformers, "_torch_pytree"):
            transformers._torch_pytree = pytree  # type: ignore[assignment]
        if hasattr(transformers, "modeling_utils"):
            if not getattr(transformers.modeling_utils, "_torch_pytree", None):
                transformers.modeling_utils._torch_pytree = pytree  # type: ignore[attr-defined]

        if hasattr(transformers, "utils"):
            if hasattr(transformers.utils, "is_torch_fx_available"):
                transformers.utils.is_torch_fx_available = lambda: True  # type: ignore[assignment]
            if hasattr(transformers.utils, "is_torch_export_available"):
                transformers.utils.is_torch_export_available = lambda: True  # type: ignore[assignment]

        _maybe_fx_imp = getattr(transformers, "_torch_fx_imported", None)
        if _maybe_fx_imp is not None:
            transformers._torch_fx_imported = True  # type: ignore[assignment]
        _has_dyn = hasattr(transformers, "dynamic_module_utils")
        if _has_dyn and hasattr(transformers.dynamic_module_utils, "_torch_fx_imported"):
            transformers.dynamic_module_utils._torch_fx_imported = True  # type: ignore[attr-defined]

        if hasattr(transformers, "modeling_utils"):
            if hasattr(transformers.modeling_utils, "_model_output_unflatten"):
                def _noop_flatten(x: Any, context: Any = None) -> Any:
                    return x
                transformers.modeling_utils._model_output_flatten = _noop_flatten  # type: ignore[attr-defined]
                transformers.modeling_utils._model_output_unflatten = _noop_flatten  # type: ignore[attr-defined]

    except ImportError:
        pass


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two tensors.

    Returns -1.0 for NaN (zero vectors), so it doesn't trigger false divergence.
    """
    a_f = a.float().reshape(-1)
    b_f = b.float().reshape(-1)
    if a_f.numel() == 0 or b_f.numel() == 0:
        return 1.0  # empty tensors are treated as matching
    cos = torch.nn.functional.cosine_similarity(a_f, b_f, dim=0)
    if torch.isnan(cos) or torch.isinf(cos):
        return -1.0
    return float(cos)


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Maximum absolute difference between two tensors."""
    return float((a.float() - b.float()).abs().max())


def _resolve_node_arg(
    arg: Any, node_outputs: dict[str, Any], gm: Any
) -> Any:
    """Resolve a node argument to a concrete value."""
    if isinstance(arg, torch.fx.Node):
        return node_outputs.get(arg.name)
    if isinstance(arg, (list, tuple)):
        return type(arg)(_resolve_node_arg(a, node_outputs, gm) for a in arg)
    return arg


def _run_ep_capture(
    program: Any, example_input: torch.Tensor
) -> dict[str, Any]:
    """Execute the ExportedProgram graph node-by-node, capturing all outputs.

    Returns dict mapping FX node name → output tensor (or tuple of tensors).
    Uses torch.fx.Interpreter for reliable graph execution.
    """
    gm = program.graph_module
    sig = program.graph_signature

    user_inputs = list(sig.user_inputs)

    # Build placeholder values in graph order
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

    class CaptureInterpreter(torch.fx.Interpreter):
        def __init__(self, mod):
            super().__init__(mod)
            self.captured: dict[str, Any] = {}
            self.node_inputs: dict[str, list[Any]] = {}

        def run_node(self, n):
            if n.op == "call_function":
                args_resolved: list[Any] = []
                for a in n.args:
                    if isinstance(a, torch.fx.Node):
                        args_resolved.append(self.env[a])
                    else:
                        args_resolved.append(a)
                self.node_inputs[n.name] = args_resolved

            result = super().run_node(n)
            if isinstance(result, torch.Tensor):
                self.captured[n.name] = result.detach()
            elif isinstance(result, (tuple, list)):
                self.captured[n.name] = result
                for i, elem in enumerate(result):
                    if isinstance(elem, torch.Tensor):
                        self.captured[f"{n.name}__elem_{i}"] = elem.detach()
            return result

    print(f"  EP: executing graph with {len(list(gm.graph.nodes))} nodes...")
    print(f"  EP: {len(placeholder_values)} placeholders to feed")
    t0 = time.time()
    with torch.no_grad():
        interp = CaptureInterpreter(gm)
        interp.run(*placeholder_values)
    elapsed = time.time() - t0
    print(f"  EP: done in {elapsed:.1f}s, captured {len(interp.captured)} values")

    # Debug: check neg_1 inputs vs masked_fill
    if "neg_1" in interp.node_inputs:
        ni = interp.node_inputs["neg_1"]
        print(f"  EP debug: neg_1 received {len(ni)} input(s)")
        for idx, inp in enumerate(ni):
            if isinstance(inp, torch.Tensor):
                print(f"    input[{idx}]: Tensor shape={list(inp.shape)}, id={id(inp)}")
            else:
                print(f"    input[{idx}]: {type(inp).__name__}")
        if "masked_fill" in interp.captured:
            mf = interp.captured["masked_fill"]
            if isinstance(mf, torch.Tensor):
                print(f"    masked_fill output: Tensor shape={list(mf.shape)}, id={id(mf)}")
                if ni and isinstance(ni[0], torch.Tensor):
                    print(f"    same object? {ni[0] is mf}")
                    # Check if manual negation matches captured neg
                    manual_neg = -ni[0].float()
                    if "neg_1" in interp.captured:
                        neg_out = interp.captured["neg_1"]
                        if isinstance(neg_out, torch.Tensor):
                            _cos = _cosine_similarity(manual_neg, neg_out)
                            print(f"    manual -input vs captured output: cos={_cos:.10f}")

    return interp.captured


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EP-to-IR op-by-op divergence debugger for Qwen3.5-0.8B"
    )
    parser.add_argument(
        "--max-ops", type=int, default=0,
        help="Max IR ops to check (0 = all)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.999,
        help="Cosine similarity threshold for divergence detection"
    )
    parser.add_argument(
        "--stop-at-first", action="store_true", default=True,
        help="Stop at first divergence (default: True)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print status of every checked/skipped op"
    )
    args = parser.parse_args()

    _patch_transformers_torch()

    # ── Load Qwen model ───────────────────────────────────────
    from transformers import AutoConfig, AutoModelForCausalLM  # type: ignore[import-untyped]

    model_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "Qwen", "Qwen3.5-0.8B",
    )
    model_dir = os.path.abspath(model_dir)

    if not os.path.isdir(model_dir):
        print(f"ERROR: Model directory not found: {model_dir}")
        print("Download with: huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir models/Qwen/Qwen3.5-0.8B")
        sys.exit(1)

    print(f"Loading Qwen3.5-0.8B from: {model_dir}")
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    if hasattr(config, "text_config") and hasattr(config.text_config, "use_cache"):
        config.text_config.use_cache = False
    elif hasattr(config, "use_cache"):
        config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    model.eval()
    print("  Model loaded.")

    # ── Export ────────────────────────────────────────────────
    example_input = torch.randint(0, 248320, (1, 64), dtype=torch.long)
    print(f"Exporting with example input shape: {list(example_input.shape)}...")
    t0 = time.time()
    with torch.no_grad():
        program = torch.export.export(model, (example_input,))
    print(f"  Export done in {time.time() - t0:.1f}s")

    # ── Step 1: Run EP and capture all node outputs ──────────
    print("\n=== Step 1: Capturing ExportedProgram node outputs ===")
    ep_outputs = _run_ep_capture(program, example_input)

    # Get reference logits from EP output node
    ep_final_output = None
    for key in ("output__elem_0", "output"):
        if key in ep_outputs:
            val = ep_outputs[key]
            if isinstance(val, torch.Tensor):
                ep_final_output = val
                break
            elif isinstance(val, (tuple, list)) and len(val) > 0 and isinstance(val[0], torch.Tensor):
                ep_final_output = val[0]
                break
    if ep_final_output is not None and isinstance(ep_final_output, torch.Tensor):
        print(f"  EP final output shape: {list(ep_final_output.shape)}")

    # ── Step 2: Convert EP to IR (no optimization passes) ────
    print("\n=== Step 2: Converting EP to IR ===")
    from compiler.fx_to_mlir import fx_graph_to_mlir
    from hal.pytorch_backend import PyTorchBackend

    t0 = time.time()
    ir_module = fx_graph_to_mlir(program)
    ir_func = ir_module.functions[0]
    print(f"  IR conversion done in {time.time() - t0:.1f}s")
    print(f"  IR ops: {len(ir_func.ops)}, weights: {len(ir_func.weights)}")

    # ── Step 3: Run IR ops and compare ───────────────────────
    print("\n=== Step 3: Executing IR ops and comparing against EP ===")

    backend = PyTorchBackend("cpu")
    ssa_values: dict[str, torch.Tensor] = {}

    # Seed input values
    input_names = [name for name, _ in ir_func.inputs]
    if input_names:
        ssa_values[input_names[0]] = example_input
    for named_input in input_names[1:]:
        ssa_values[named_input] = torch.zeros(1, dtype=torch.long)

    # Flatten weights for lookup
    all_weights: dict[str, torch.Tensor] = {}
    for name, tensor in ir_func.weights.items():
        all_weights[name] = tensor

    total_ops = len(ir_func.ops)
    checked = 0
    skipped = 0
    diverged = 0
    first_diverged_op: int = -1
    divergence_details: list[dict[str, Any]] = []
    op_status: dict[int, str] = {}  # track status of each op

    max_ops = args.max_ops if args.max_ops > 0 else total_ops
    threshold = args.threshold

    print(f"  Threshold: cos < {threshold}, max ops: {max_ops}")
    t0 = time.time()

    for i, op in enumerate(ir_func.ops):
        if i >= max_ops:
            break

        source_node = op.attributes.get("source_node")

        # Resolve inputs
        tensor_inputs: list[torch.Tensor] = []
        try:
            for inp_name in op.inputs:
                if inp_name in ssa_values:
                    tensor_inputs.append(ssa_values[inp_name])
                elif inp_name in all_weights:
                    tensor_inputs.append(all_weights[inp_name])
                elif inp_name in ir_func.weights:
                    tensor_inputs.append(ir_func.weights[inp_name])
                else:
                    raise KeyError(f"Unknown input '{inp_name}'")
        except KeyError as e:
            if args.verbose or i < 200:
                print(f"  [{i:5d}] MISSING INPUT: {op.name} needs '{e}'  "
                      f"(source={source_node}, all_inputs={op.inputs}, "
                      f"attrs={ {k: v for k,v in op.attributes.items() if k!='source_node'} })")
            skipped += 1
            op_status[i] = "MISSING_INPUT"
            continue

        # Execute op
        try:
            if tensor_inputs:
                result = backend.execute(op.name, tensor_inputs, **{
                    k: v for k, v in op.attributes.items() if k != "source_node"
                })
            else:
                result = backend.execute(op.name, [], **{
                    k: v for k, v in op.attributes.items() if k != "source_node"
                })
        except Exception as e:
            print(f"  [{i:5d}] EXEC ERROR: {op.name} - {e}")
            skipped += 1
            continue

        if result is not None and op.outputs:
            ssa_values[op.outputs[0]] = result

        # Compare against EP output
        if source_node and source_node in ep_outputs:
            ep_val = ep_outputs[source_node]
            if isinstance(result, torch.Tensor) and isinstance(ep_val, torch.Tensor):
                # Normalize to float for comparison (handles bool, int, etc.)
                r_cmp = result.float() if result.dtype != torch.float32 else result
                e_cmp = ep_val.float() if ep_val.dtype != torch.float32 else ep_val

                if result.shape == ep_val.shape or result.numel() == ep_val.numel():
                    try:
                        cos = _cosine_similarity(r_cmp, e_cmp)
                        max_diff = _max_abs_diff(r_cmp, e_cmp)
                    except Exception:
                        print("  WARNING: cosine similarity calculation failed, skipping")
                        skipped += 1
                        continue
                    checked += 1

                    # Zero vectors: cosine is meaningless, use abs diff
                    ir_all_zero = (r_cmp.abs().max().item() == 0.0)
                    ep_all_zero = (e_cmp.abs().max().item() == 0.0)
                    is_zero_pair = ir_all_zero and ep_all_zero
                    is_diverged = False
                    if is_zero_pair:
                        is_diverged = False  # both zero, matched
                    elif ir_all_zero != ep_all_zero:
                        is_diverged = True  # one is zero, one isn't
                    elif cos < 0.0:
                        is_diverged = False  # NaN cos → zero vectors, handled above
                    else:
                        is_diverged = cos < threshold or max_diff > 1e-3

                    if is_diverged:
                        diverged += 1
                        if first_diverged_op < 0:
                            first_diverged_op = i
                        op_status[i] = "DIVERGED"

                        detail: dict[str, Any] = {
                            "op_idx": i,
                            "op_name": op.name,
                            "source_node": source_node,
                            "ir_output_name": op.outputs[0] if op.outputs else "?",
                            "ir_shape": list(result.shape),
                            "ep_shape": list(ep_val.shape),
                            "ir_dtype": str(result.dtype).replace("torch.", ""),
                            "ep_dtype": str(ep_val.dtype).replace("torch.", ""),
                            "cosine": cos,
                            "max_abs_diff": max_diff,
                            "ir_stats": {
                                "min": float(r_cmp.min()),
                                "max": float(r_cmp.max()),
                                "mean": float(r_cmp.mean()),
                            },
                            "ep_stats": {
                                "min": float(e_cmp.min()),
                                "max": float(e_cmp.max()),
                                "mean": float(e_cmp.mean()),
                            },
                        }
                        divergence_details.append(detail)

                        if cos < 0.99 or max_diff > 1e-6:
                            print(
                                f"  [{i:5d}] DIVERGENCE: {op.name:25s} "
                                f"cos={cos:.6f}  max_diff={max_diff:.2e}  "
                                f"ir_shape={list(result.shape)}  ep_shape={list(ep_val.shape)}"
                            )
                            print(
                                f"          ir:  min={r_cmp.min():.6e}  max={r_cmp.max():.6e}  "
                                f"mean={r_cmp.mean():.6e}"
                            )
                            print(
                                f"          ep:  min={e_cmp.min():.6e}  max={e_cmp.max():.6e}  "
                                f"mean={e_cmp.mean():.6e}"
                            )

                        if args.stop_at_first and first_diverged_op >= 0:
                            break
            elif isinstance(result, torch.Tensor) and isinstance(ep_val, (tuple, list)):
                skipped += 1
        else:
            skipped += 1

    elapsed = time.time() - t0

    # ── Report ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Summary: {total_ops} ops, {checked} checked, {skipped} skipped, {diverged} diverged")
    print(f"Time: {elapsed:.1f}s")

    if first_diverged_op >= 0:
        print(f"\nFirst divergence at IR op index {first_diverged_op} (cosine < {threshold}):")
        for d in divergence_details[:5]:
            print(f"  [{d['op_idx']:5d}] {d['op_name']:25s}  cos={d['cosine']:.6f}  "
                  f"ir={d['ir_shape']}  ep={d['ep_shape']}  "
                  f"node={d['source_node']}")

        # Show context: ops around the first divergence
        first = divergence_details[0]["op_idx"]
        print(f"\nContext around first divergence (ops {max(0, first-5)} to {min(total_ops-1, first+3)}):")
        for j in range(max(0, first - 5), min(total_ops, first + 4)):
            op = ir_func.ops[j]
            source_node = op.attributes.get("source_node", "?")
            status = op_status.get(j, "checked" if j < first else "?")
            marker = " <-- FIRST DIVERGENCE" if j == divergence_details[0]["op_idx"] else ""
            if status == "MISSING_INPUT":
                marker = " <-- MISSING INPUT" + marker
            elif status == "DIVERGED":
                pass  # already marked
            print(f"  [{j:5d}] {op.name:25s}  inputs={op.inputs[:4]}{'...' if len(op.inputs)>4 else ''}  "
                  f"output={op.outputs}  source={source_node}  [{status}]{marker}")

        # Also try to find the op after which everything breaks
        print("\nLast divergence details:")
        for d in divergence_details[-3:]:
            print(f"  [{d['op_idx']:5d}] {d['op_name']:25s}  cos={d['cosine']:.6f}")
    else:
        print(f"\nNo divergence found among {checked} checked ops (cosine >= {threshold}).")

    # ── Compute final output cosine ──────────────────────────
    if ep_final_output is not None and ssa_values:
        ir_output_names = [name for name, _, _ in ir_func.outputs]
        if ir_output_names and ir_output_names[0] in ssa_values:
            ir_final = ssa_values[ir_output_names[0]]
            if isinstance(ir_final, torch.Tensor):
                final_cos = _cosine_similarity(ep_final_output, ir_final)
                print(f"\nIR executor final output vs EP: cosine = {final_cos:.6f}")
        elif ssa_values:
            ir_final = list(ssa_values.values())[-1]
            if isinstance(ir_final, torch.Tensor):
                final_cos = _cosine_similarity(ep_final_output, ir_final)
                print(f"\nIR executor final output vs EP: cosine = {final_cos:.6f}")


if __name__ == "__main__":
    main()
