"""MLIR fusion passes — run on MlirModule via official MLIR bindings.

Each pass is a ``RewritePatternSet`` callback that:
  1. Matches a specific op (the root of a fusion chain)
  2. Walks operands/producers to find the full pattern
  3. Creates a fused op using ``Operation.create()`` inside ``rewriter.ip``
  4. Calls ``rewriter.replace_op()`` to replace the root
  5. Calls ``rewriter.erase_op()`` to remove consumed intermediate ops

Requires the official MLIR Python bindings (``import mlir.ir``).
When bindings are unavailable, these functions are no-ops (skip gracefully).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from compiler.mlir_dialect.compile_utils import _setup_mlir_path


def _run_pattern(
    mlir_text: str,
    pattern_name: str,
    callback: Callable[..., Any],
    max_iterations: int = 5,
) -> str:
    """Run a single RewritePatternSet callback on MLIR text.

    Returns the modified MLIR text.
    """
    _setup_mlir_path()
    import mlir.ir as ir
    from mlir.rewrite import (
        GreedyRewriteConfig,
        GreedyRewriteStrictness,
        RewritePatternSet,
        apply_patterns_and_fold_greedily,
    )

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    try:
        from mlir_sf._mlir_libs._sfDialectsNanobind import sf
        sf.register_dialects(ctx._CAPIPtr, load=True)
    except ImportError:
        pass

    with ctx, ir.Location.unknown(ctx):
        module = ir.Module.parse(mlir_text, ctx)
        pattern_set = RewritePatternSet(ctx)
        pattern_set.add(pattern_name, callback)
        config = GreedyRewriteConfig()
        config.max_iterations = max_iterations
        config.strictness = GreedyRewriteStrictness.EXISTING_OPS
        config.enable_folding = False

        apply_patterns_and_fold_greedily(module, pattern_set.freeze(), config)

        return str(module)


# ── FuseSiLU: sf.silu(x) + sf.mul(silu_out, y) → sf.fused_silu_mul(x, y) ──


def fuse_silu_pass(mlir_text: str) -> str:
    """Fuse sf.silu → sf.mul chains into sf.fused_silu_mul."""
    import mlir.ir as ir

    def callback(op: Any, rewriter: Any) -> Any:
        if op.name != "sf.mul" or len(op.operands) < 2:
            return True  # signal: no match
        silu_val = op.operands[0]
        block = op.operation.parent.regions[0].blocks[0]
        silu_found = None
        for o in block:
            if o.name == "sf.silu" and len(o.operation.results) == 1 and o.result == silu_val:
                silu_found = o
                break
        if silu_found is None:
            return True  # signal: no match
        with ir.Location.unknown(op.operation.context), rewriter.ip:
            fused = ir.Operation.create(
                "sf.fused_silu_mul",
                results=[op.result.type],
                operands=[silu_found.operands[0], op.operands[1]],
            )
        rewriter.replace_op(op.operation, fused)
        rewriter.erase_op(silu_found.operation)
        # returning None signals success (match + modification)

    return _run_pattern(mlir_text, "sf.mul", callback)


# ── FuseQKV: sf.q_linear + sf.k_linear + sf.v_linear → sf.fused_qkv ──


def fuse_qkv_pass(mlir_text: str) -> str:
    """Fuse three adjacent linear ops (Q/K/V) into sf.fused_qkv."""

    def callback(op: Any, rewriter: Any) -> Any:
        if op.name not in ("sf.linear", "sf.matmul"):
            return True
        if len(op.operands) < 2:
            return True
        block = op.operation.parent.regions[0].blocks[0]
        ops_list = list(block)

        # Find op position
        idx = next((i for i, o in enumerate(ops_list) if o is op), None)
        if idx is None:
            return True

        # Look for two more linear/matmul ops with same first input
        first_input = op.operands[0]
        siblings = []
        for i in range(idx + 1, min(idx + 10, len(ops_list))):
            sib = ops_list[i]
            if sib.name in ("sf.linear", "sf.matmul") and len(sib.operands) >= 2:
                if sib.operands[0] == first_input:
                    siblings.append(sib)
            if len(siblings) >= 2:
                break

        if len(siblings) < 2:
            return True
        qkv_ops = [op] + siblings[:2]
        qkv_weights = [o.operands[1] for o in qkv_ops if len(o.operands) >= 2]
        if len(qkv_weights) != 3:
            return True

        import mlir.ir as ir

        with ir.Location.unknown(op.operation.context), rewriter.ip:
            fused = ir.Operation.create(
                "sf.fused_qkv",
                results=[ir.F32Type.get() for _ in range(3)],
                operands=[first_input] + qkv_weights,
            )

        # Replace all three, create slice ops for each result
        for qkv_op in qkv_ops:
            rewriter.replace_op(qkv_op.operation, fused.operation)
        return

    return _run_pattern(mlir_text, "sf.linear", callback, max_iterations=3)


# ── FuseRMSNorm: sf.rms_norm(x,w) + sf.mul(x_norm, rms_w) + sf.matmul → fused ──


def fuse_rms_norm_pass(mlir_text: str) -> str:
    """Fuse sf.rms_norm → sf.mul → sf.matmul into sf.fused_rms_norm_matmul."""

    def callback(op: Any, rewriter: Any) -> Any:
        if op.name not in ("sf.matmul", "sf.linear"):
            return
        if len(op.operands) < 2:
            return
        block = op.operation.parent.regions[0].blocks[0]
        norm_out = op.operands[0]

        mul_op = None
        rms_op = None

        for o in block:
            if o.name != "sf.mul" and o.name != "sf.rms_norm":
                continue
            if len(o.operation.results) != 1:
                continue
            if o.result == norm_out and o.name == "sf.mul":
                mul_op = o
                # mul's first input should be rms_norm output
                rms_out = mul_op.operands[0]
                for o2 in block:
                    if o2.name == "sf.rms_norm" and len(o2.operation.results) == 1:
                        if o2.result == rms_out:
                            rms_op = o2
                            break

        if rms_op is None or mul_op is None:
            return True  # signal: no match

        import mlir.ir as ir

        with ir.Location.unknown(op.operation.context), rewriter.ip:
            fused = ir.Operation.create(
                "sf.fused_rms_norm_matmul",
                results=[op.result.type],
                operands=[
                    rms_op.operands[0],  # rms input
                    mul_op.operands[1],    # rms weight (second input to mul)
                    op.operands[1],        # matmul weight
                ],
            )
        rewriter.replace_op(op.operation, fused)
        return

    return _run_pattern(mlir_text, "sf.matmul", callback)


# ── FuseAttentionPattern: sdpa → transpose → view → linear → fused_attention_output ──


def fuse_attention_pass(mlir_text: str) -> str:
    """Fuse SDPA chain into sf.fused_attention_output."""

    def callback(op: Any, rewriter: Any) -> Any:
        if op.name not in ("sf.matmul", "sf.linear"):
            return
        if len(op.operands) < 2:
            return
        block = op.operation.parent.regions[0].blocks[0]
        ops_list = list(block)

        # Find the linear (out_proj) op and walk backward
        idx = next((i for i, o in enumerate(ops_list) if o is op), None)
        if idx is None or idx < 3:
            return True

        view_op = ops_list[idx - 1]
        trans_op = ops_list[idx - 2]
        sdpa_op = ops_list[idx - 3]

        if view_op.name != "sf.view":
            return True
        if trans_op.name not in ("sf.transpose", "sf.permute"):
            return True
        if sdpa_op.name != "sf.scaled_dot_product_attention":
            return True

        if len(sdpa_op.operands) < 3:
            return True

        import mlir.ir as ir

        with ir.Location.unknown(op.operation.context), rewriter.ip:
            operands = list(sdpa_op.operands) + [op.operands[0]]
            fused = ir.Operation.create(
                "sf.fused_attention_output",
                results=[op.result.type],
                operands=operands,
            )
        rewriter.replace_op(op.operation, fused)
        return

    return _run_pattern(mlir_text, "sf.linear", callback)

