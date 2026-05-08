"""Fuse Q/K/V projection pass.

Detects linear (or matmul) ops that share the same input tensor and
whose weight names suggest attention projections (q_proj / k_proj / v_proj).
Replaces them with a single fused linear + split pattern.

Before:
    q = linear(x, W_q)   ─┐
    k = linear(x, W_k)   ─┤  sharing input x
    v = linear(x, W_v)   ─┘

After:
    qkv = linear(x, cat([W_q, W_k, W_v], dim=0))
    q = slice(qkv, dim=-1, start=0, end=q_dim)
    k = slice(qkv, dim=-1, start=q_dim, end=q_dim+k_dim)
    v = slice(qkv, dim=-1, start=q_dim+k_dim, end=q_dim+k_dim+v_dim)

Works with GQA (Grouped Query Attention) where K/V heads differ from Q heads.
"""

from __future__ import annotations

import torch

from compiler.ir import IrFunction, IrModule, IrOp
from compiler.passes.base import Pass

# Weight-name substrings that identify attention projection weights
_Q_PROJ_TOKENS = ("q_proj", "query", "q_linear")
_K_PROJ_TOKENS = ("k_proj", "key", "k_linear")
_V_PROJ_TOKENS = ("v_proj", "value", "v_linear")


class FuseQKVProjection(Pass):
    """Fuse Q/K/V linear/matmul ops that share the same input into a single
    fused matmul + split pattern.

    The pass groups linear or matmul ops by their first input, then fuses
    groups whose weight names contain q_proj/k_proj/v_proj (or query/key/value).
    Weights from each group member are concatenated along dim=0 (output features)
    and slice ops are emitted to recover the original per-projection outputs.
    """

    def apply(self, module: IrModule) -> IrModule:
        for func in module.functions:
            self._fuse_function(func)
        return module

    # ── main dispatch ──────────────────────────────────────

    @staticmethod
    def _fuse_function(func: IrFunction) -> None:
        groups = _collect_fusible_groups(func)
        if not groups:
            return

        new_ops: list[IrOp] = []
        for op in func.ops:
            handled = False
            for group in groups:
                if op is group[0]:
                    _emit_fused_group(group, func, new_ops)
                    handled = True
                    break
                if op in group:
                    handled = True  # skip remaining group members
                    break
            if not handled:
                new_ops.append(op)

        func.ops = new_ops



# ── internal helpers ───────────────────────────────────────

def _collect_fusible_groups(func: IrFunction) -> list[list[IrOp]]:
    """Collect groups of linear/matmul ops that share the same first input
    and whose weight names suggest they are attention projections."""
    # Group by first input SSA name
    input_to_ops: dict[str, list[IrOp]] = {}
    for op in func.ops:
        if op.name not in ("linear", "matmul"):
            continue
        if len(op.inputs) < 2:
            continue
        first_in = op.inputs[0]
        input_to_ops.setdefault(first_in, []).append(op)

    # Filter: keep only groups of size >= 2 with weight-name signals
    fusible: list[list[IrOp]] = []
    for ops in input_to_ops.values():
        if len(ops) < 2:
            continue
        if _has_attention_projection_weights(ops):
            # Sort for deterministic output order: q, k, v (by weight name)
            ops.sort(key=lambda op: _projection_order(op))
            fusible.append(ops)

    return fusible


def _has_attention_projection_weights(ops: list[IrOp]) -> bool:
    """Return True if the group's weight names suggest Q/K/V projections."""
    weight_names: list[str] = []
    for op in ops:
        if len(op.inputs) >= 2:
            weight_names.append(op.inputs[1].lower())
    if not weight_names:
        return False
    has_q = any(tok in w for w in weight_names for tok in _Q_PROJ_TOKENS)
    has_k = any(tok in w for w in weight_names for tok in _K_PROJ_TOKENS)
    has_v = any(tok in w for w in weight_names for tok in _V_PROJ_TOKENS)
    return has_q or has_k or has_v


def _projection_order(op: IrOp) -> int:
    """Return a sort key: q=0, k=1, v=2, other=3."""
    wname = op.inputs[1].lower() if len(op.inputs) >= 2 else ""
    for tok in _Q_PROJ_TOKENS:
        if tok in wname:
            return 0
    for tok in _K_PROJ_TOKENS:
        if tok in wname:
            return 1
    for tok in _V_PROJ_TOKENS:
        if tok in wname:
            return 2
    return 3


def _emit_fused_group(group: list[IrOp], func: IrFunction, new_ops: list[IrOp]) -> None:
    """Concatenate weights from *group*, emit a single fused linear op,
    then emit slice ops to recover the original per-projection outputs."""
    if len(group) < 2:
        return  # degenerate — nothing to fuse

    # Collect weight tensors (and biases if all members have them)
    weights_t: list[torch.Tensor] = []
    biases_t: list[torch.Tensor | None] = []
    for op in group:
        w_name = op.inputs[1]
        w = func.weights[w_name]
        weights_t.append(w)
        if len(op.inputs) > 2:
            b_name = op.inputs[2]
            biases_t.append(func.weights[b_name])
        else:
            biases_t.append(None)

    # Validate all weights have same dtype
    dtype0 = weights_t[0].dtype
    for i, w in enumerate(weights_t):
        if w.dtype != dtype0:
            raise RuntimeError(
                f"FuseQKVProjection: weight dtype mismatch in group at index {i}: "
                f"{dtype0} vs {w.dtype}"
            )

    # Concatenate along dim=0 (output features)
    fused_w = torch.cat(weights_t, dim=0)
    fused_w_name = f"__fused_qkv_w_{group[0].outputs[0]}"
    func.weights[fused_w_name] = fused_w

    # Build fused op inputs
    fused_inputs: list[str] = [group[0].inputs[0], fused_w_name]
    all_have_bias = all(b is not None for b in biases_t)
    if all_have_bias:
        fused_b = torch.cat([b for b in biases_t if b is not None], dim=0)
        fused_b_name = f"__fused_qkv_b_{group[0].outputs[0]}"
        func.weights[fused_b_name] = fused_b
        fused_inputs.append(fused_b_name)

    # Pick the op name from the first group member (linear or matmul)
    op_name = group[0].name

    fused_out = f"__fused_qkv_out_{group[0].outputs[0]}"
    fused_op = IrOp(
        name=op_name,
        inputs=fused_inputs,
        outputs=[fused_out],
        attributes={"folded": True},
    )
    new_ops.append(fused_op)

    # Emit slice ops to recover individual outputs
    offset = 0
    for op in group:
        w = func.weights[op.inputs[1]]
        out_dim = w.shape[0]
        slice_op = IrOp(
            name="slice",
            inputs=[fused_out],
            outputs=op.outputs,  # reuse original output SSA names
            attributes={
                "dim": -1,
                "start": offset,
                "end": offset + out_dim,
                "step": 1,
                "folded": True,
            },
        )
        new_ops.append(slice_op)
        offset += out_dim
