"""Fuse Attention Pattern pass.

Fuses the attention output chain into a single fused op:

Before:
    scaled_dot_product_attention(Q, K, V) → transpose → reshape → linear(O_w)

After:
    fused_attention_output(Q, K, V, O_w) → result

This reduces dispatch overhead (4 ops → 1) and enables future GPU kernel fusion.
The transpose+reshape are typically "recombine heads" operations that undo the
pre-attention Q/K/V head splitting.

Phase 3 — extends the fusion framework established by FuseQKV/FuseRMSNorm/FuseSiLU.
"""

from __future__ import annotations

from typing import Any

from compiler.ir import IrFunction, IrModule, IrOp
from compiler.passes.base import Pass


class FuseAttentionPattern(Pass):
    """Fuse the attention output chain: SDPA → transpose → view → linear.

    Pattern matching walks forward from each SDPA op looking for the
    canonical output chain.  The matched ops are replaced by a single
    fused_attention_output op that bundles the SDPA computation with
    the head-recombine and output projection.

    This is the output-side counterpart to FuseQKVProjection which fuses
    the input-side Q/K/V projections.
    """

    def apply(self, module: IrModule) -> IrModule:
        for func in module.functions:
            self._fuse_function(func)
        return module

    @staticmethod
    def _fuse_function(func: IrFunction) -> None:
        # Build a map: op output name → op index
        producer: dict[str, int] = {}
        for idx, op in enumerate(func.ops):
            for out in op.outputs:
                producer[out] = idx

        consumed: set[str] = set()  # outputs consumed by the match
        new_ops: list[IrOp] = []

        idx = 0
        while idx < len(func.ops):
            op = func.ops[idx]
            match = _match_attention_output(func.ops, idx, producer)
            if match is not None:
                sdpa_op, trans_op, view_op, linear_op = match
                _emit_fused_attention_output(
                    func, sdpa_op, trans_op, view_op, linear_op, new_ops
                )
                # Mark all matched op outputs as consumed
                for mop in (sdpa_op, trans_op, view_op, linear_op):
                    for out in mop.outputs:
                        consumed.add(out)
                # Skip past the matched ops
                idx = max(
                    producer.get(sdpa_op.outputs[0], idx),
                    producer.get(trans_op.outputs[0], idx),
                    producer.get(view_op.outputs[0], idx),
                    producer.get(linear_op.outputs[0], idx),
                ) + 1
            else:
                new_ops.append(op)
                idx += 1

        func.ops = new_ops

    @property
    def name(self) -> str:
        return "FuseAttentionPattern"


# ── pattern matching ──────────────────────────────────────────


def _match_attention_output(
    ops: list[IrOp], sdpa_idx: int, producer: dict[str, int]
) -> tuple[IrOp, IrOp, IrOp, IrOp] | None:
    """Try to match: SDPA → transpose → view → linear at *sdpa_idx*.

    Returns (sdpa_op, transpose_op, view_op, linear_op) or None.
    """
    if sdpa_idx >= len(ops):
        return None

    sdpa_op = ops[sdpa_idx]
    if sdpa_op.name != "scaled_dot_product_attention":
        return None
    if not sdpa_op.outputs:
        return None

    # Step 1: find the transpose consumer of SDPA output
    sdpa_out = sdpa_op.outputs[0]
    trans_op = _find_single_consumer(ops, sdpa_idx + 1, sdpa_out, "transpose")
    if trans_op is None:
        trans_op = _find_single_consumer(ops, sdpa_idx + 1, sdpa_out, "permute")
    if trans_op is None or not trans_op.outputs:
        return None

    # Step 2: find the view consumer of transpose output
    trans_out = trans_op.outputs[0]
    view_op = _find_single_consumer(ops, sdpa_idx + 1, trans_out, "view")
    if view_op is None or not view_op.outputs:
        return None

    # Step 3: find the linear consumer of view output
    view_out = view_op.outputs[0]
    linear_op = _find_single_consumer(ops, sdpa_idx + 1, view_out, "linear")
    if linear_op is None:
        linear_op = _find_single_consumer(ops, sdpa_idx + 1, view_out, "matmul")
    if linear_op is None or not linear_op.outputs:
        return None

    # Verify the weight of the linear is an attention output projection
    if len(linear_op.inputs) < 2:
        return None
    wname = linear_op.inputs[1].lower()
    if "o_proj" not in wname and "out_proj" not in wname:
        return None

    # Verify that intermediate outputs (sdpa, transpose, view) are not
    # consumed by any op outside the matched chain.  The final linear
    # output is the fused op's output — its downstream consumers are fine.
    intermediate_outputs: set[str] = set()
    for mop in (sdpa_op, trans_op, view_op):
        intermediate_outputs.update(mop.outputs)

    for i in range(sdpa_idx + 1, len(ops)):
        o = ops[i]
        if o in (sdpa_op, trans_op, view_op, linear_op):
            continue
        for inp in o.inputs:
            if inp in intermediate_outputs:
                return None  # intermediate used elsewhere — can't fuse

    return sdpa_op, trans_op, view_op, linear_op


def _find_single_consumer(
    ops: list[IrOp], start: int, value: str, op_name: str
) -> IrOp | None:
    """Find the first op after *start* that consumes *value* and has name *op_name*."""
    for i in range(start, len(ops)):
        op = ops[i]
        if value in op.inputs and op.name == op_name:
            return op
    return None


# ── emission ──────────────────────────────────────────────────


def _emit_fused_attention_output(
    func: IrFunction,
    sdpa_op: IrOp,
    trans_op: IrOp,
    view_op: IrOp,
    linear_op: IrOp,
    new_ops: list[IrOp],
) -> None:
    """Emit a single fused_attention_output op replacing the matched chain.

    The fused op reuses the SDPA inputs (Q, K, V, [attn_mask]) plus the
    output projection weight, and produces the linear op's output.
    """
    # Collect inputs: SDPA inputs + O_proj weight (+ optional bias)
    fused_inputs: list[str] = list(sdpa_op.inputs)
    # Add the out_proj weight (2nd input of linear)
    if len(linear_op.inputs) >= 2:
        wname = linear_op.inputs[1]
        if wname not in fused_inputs:
            fused_inputs.append(wname)
    # Add optional bias
    if len(linear_op.inputs) >= 3:
        bname = linear_op.inputs[2]
        if bname not in fused_inputs:
            fused_inputs.append(bname)

    # Collect attributes from SDPA
    fused_attrs: dict[str, Any] = dict(sdpa_op.attributes)
    fused_attrs["folded"] = True
    # Carry transpose/view info
    if trans_op.attributes:
        fused_attrs["fuse_transpose_attrs"] = dict(trans_op.attributes)
    if view_op.attributes:
        fused_attrs["fuse_view_attrs"] = dict(view_op.attributes)

    # Reuse the linear op's outputs as the fused op's outputs
    fused_op = IrOp(
        name="fused_attention_output",
        inputs=fused_inputs,
        outputs=list(linear_op.outputs),
        attributes=fused_attrs,
    )
    new_ops.append(fused_op)
