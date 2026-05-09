"""Fuse Attention Block pass — extends FuseAttentionPattern to include RMSNorm.

Runs AFTER FuseAttentionPattern.  Matches fused_attention_output ops and
walks backward from the Q input to fuse the RMSNorm + QKV linear into
the attention block.

Before:
    rms_norm(input) → linear(QKV_fused_w) → slice → ... → sdpa → ... → fused_attention_output

After:
    fused_attention_block(input, rms_w, QKV_w, O_w, mask, ...) → output

Phase 3 — extends FuseAttentionPattern to cover the full input-to-output chain.
"""

from __future__ import annotations

from typing import Any

from compiler.ir import IrFunction, IrModule, IrOp
from compiler.passes.base import Pass


class FuseAttentionBlock(Pass):
    """Fuse RMSNorm + QKV linear + attention output into a single fused block."""

    def apply(self, module: IrModule) -> IrModule:
        for func in module.functions:
            self._fuse_function(func)
        return module

    @staticmethod
    def _fuse_function(func: IrFunction) -> None:
        # Build op index maps
        producer: dict[str, int] = {}
        op_by_output: dict[str, IrOp] = {}
        for idx, op in enumerate(func.ops):
            producer[op.outputs[0]] = idx if op.outputs else -1
            for out in op.outputs:
                op_by_output[out] = op

        new_ops: list[IrOp | None] = list(func.ops)  # mutable list for replacement

        for idx, op in enumerate(func.ops):
            if op.name != "fused_attention_output":
                continue

            match = _match_rms_norm_input(func.ops, idx, op, op_by_output, producer)
            if match is None:
                continue

            rms_op, qkv_op = match
            new_op = _emit_fused_attention_block(func, op, rms_op, qkv_op)
            # Replace: mark old ops as None, insert new op at the rms_op position
            new_ops[idx] = None  # old fused_attention_output
            new_ops[producer.get(qkv_op.outputs[0], idx)] = None  # QKV linear
            new_ops[producer.get(rms_op.outputs[0], idx)] = None  # RMS mul
            new_ops[producer.get(rms_op.outputs[0], idx) if producer.get(rms_op.outputs[0], idx) > 0 else idx] = new_op

        func.ops = [o for o in new_ops if o is not None]

    @property
    def name(self) -> str:
        return "FuseAttentionBlock"


def _match_rms_norm_input(
    ops: list[IrOp],
    fused_idx: int,
    fused_op: IrOp,
    op_by_output: dict[str, IrOp],
    producer: dict[str, int],
) -> tuple[IrOp, IrOp] | None:
    """Given a fused_attention_output op, trace Q backward to find RMSNorm.

    Returns (rms_mul_op, qkv_linear_op) or None.
    """
    if len(fused_op.inputs) < 4:
        return None

    q_name = fused_op.inputs[0]

    # Step 1: walk backward from Q to find a linear/matmul with fused QKV weight
    qkv_op = None
    current = q_name
    for _ in range(15):
        op = op_by_output.get(current)
        if op is None:
            break
        if op.name in ("linear", "matmul"):
            # Check if this is a fused QKV projection
            if len(op.inputs) >= 2:
                wname = op.inputs[1].lower()
                if "q_proj" in wname or "fused_qkv" in wname:
                    qkv_op = op
                    break
            # Also accept if it feeds into QVA slices
            if op.outputs:
                # Check if output consumers include slices
                out_name = op.outputs[0]
                consumers = [o for o in ops if out_name in o.inputs]
                if any(o.name == "slice" for o in consumers):
                    qkv_op = op
                    break
        current = op.inputs[0] if op.inputs else ""
        if current == q_name:  # prevent infinite loop
            break
        q_name = current

    if qkv_op is None or not qkv_op.inputs:
        return None

    # Step 2: walk backward from QKV linear to find RMSNorm
    # RMSNorm pattern: mul(rms_weight, identity(mul(input, rsqrt)))
    # The QKV linear's first input should be the RMSNorm output
    rms_in = qkv_op.inputs[0]
    rms_op = op_by_output.get(rms_in)
    if rms_op is None:
        return None

    # Check if rms_op is a mul with a weight that looks like a norm weight
    if rms_op.name == "mul" and len(rms_op.inputs) >= 2:
        wname = rms_op.inputs[0].lower()
        if any(tok in wname for tok in ("layernorm", "norm", "rms")):
            return rms_op, qkv_op

    # Check if there's an identity + mul chain
    if rms_op.name == "identity" and rms_op.inputs:
        pre_mul = op_by_output.get(rms_op.inputs[0])
        if pre_mul and pre_mul.name == "mul" and len(pre_mul.inputs) >= 2:
            wname = pre_mul.inputs[0].lower()
            if any(tok in wname for tok in ("layernorm", "norm", "rms")):
                return pre_mul, qkv_op

    # Direct RMSNorm: the QKV input is from mul with norm weight
    if rms_op.name == "mul":
        # Check if the first input is a norm weight
        for inp in rms_op.inputs:
            il = inp.lower()
            if any(tok in il for tok in ("layernorm", "norm", "rms")) and "weight" in il:
                return rms_op, qkv_op

    return None


def _emit_fused_attention_block(
    func: IrFunction,
    fused_op: IrOp,
    rms_op: IrOp,
    qkv_op: IrOp,
) -> IrOp:
    """Emit a fused_attention_block combining RMSNorm + QKV + attention + out_proj."""
    # Build new inputs: rms_input, rms_weight, qkv_weight, o_proj_weight + SDPA inputs
    fused_inputs: list[str] = []

    # RMS input (from rms_op minus the weight)
    # rms_op has inputs like [rms_weight, rms_input]
    rms_weight_name = ""
    rms_input_name = ""
    for inp in rms_op.inputs:
        inp_lower = inp.lower()
        if any(tok in inp_lower for tok in ("layernorm", "norm", "rms")):
            rms_weight_name = inp
        else:
            rms_input_name = inp

    fused_inputs.append(
        rms_input_name if rms_input_name else (
            rms_op.inputs[1] if len(rms_op.inputs) > 1 else rms_op.inputs[0]
        )
    )
    fused_inputs.append(rms_weight_name if rms_weight_name else rms_op.inputs[0])

    # QKV weight
    if len(qkv_op.inputs) >= 2:
        fused_inputs.append(qkv_op.inputs[1])

    # O_proj weight + SDPA inputs from the original fused op
    for inp in fused_op.inputs:
        if inp not in fused_inputs:
            fused_inputs.append(inp)

    # Build attributes
    fused_attrs: dict[str, Any] = dict(fused_op.attributes)
    fused_attrs["folded"] = True
    fused_attrs["fused_block"] = True

    return IrOp(
        name="fused_attention_block",
        inputs=fused_inputs,
        outputs=list(fused_op.outputs),
        attributes=fused_attrs,
    )
