"""Shared sf→linalg lowering pipeline — single source of truth.

Every caller that runs the sf dialect lowering must use run_sf_lowering_pipeline().
This prevents pipeline drift (10 files had independent pass strings, some missing
sf-chain-wrapper, some with different pass ordering).
"""

from __future__ import annotations

from typing import Any

SF_LOWERING_PIPELINE = "sf-promote-weights,canonicalize,cse,sf-chain-wrapper,sf-lower-to-linalg,canonicalize,cse"


def run_sf_lowering_pipeline(ir_mod: Any, ctx: Any, verify: bool = True) -> str:
    """Run the complete sf→linalg lowering pipeline on an ir.Module.

    Pipeline order:
      sf-chain-wrapper    → generates public main() calling sub-funcs in order
      sf-promote-weights  → promotes sf.weight ops to function arguments
      canonicalize        → dead code elimination, constant folding
      cse                 → common subexpression elimination
      sf-lower-to-linalg  → lowers all sf ops to linalg/arith/math

    Note: wrapper runs before promotion.  After promotion modifies function
    signatures, the call ops in main() may reference pre-promotion types.
    Set verify=False if the downstream pipeline (e.g. llc) can tolerate
    minor type mismatches that do not affect the generated machine code.

    Args:
        ir_mod: mlir.ir.Module created by mlir_module_to_ir_module() or ir.Module.parse().
        ctx: mlir.ir.Context with sf dialect registered.
        verify: Enable MLIR verifier after each pass (default True).

    Returns:
        Lowered MLIR text (mixed linalg/arith/math/...).
    """
    import mlir.passmanager as pm

    pman = pm.PassManager.parse(f"builtin.module({SF_LOWERING_PIPELINE})", ctx)
    pman.enable_verifier(verify)
    pman.enable_timing()
    pman.run(ir_mod.operation)
    return ir_mod.operation.get_asm(print_generic_op_form=True)
