"""linalg → LLVM IR lowering pipeline.

Uses standard MLIR passes: one-shot-bufferize → convert-linalg-to-loops
→ lower-affine → convert-scf-to-cf → finalize-memref-to-llvm
→ convert-arith/math/cf/func-to-llvm → reconcile-unrealized-casts.

The pipeline requires that ALL sf dialect ops have been eliminated
before bufferization (sf→linalg lowering must be complete).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def _setup_mlir_path() -> None:
    _mlir_pkg = Path(__file__).resolve().parent.parent.parent / "mlir_binding" / "mlir_package"
    if _mlir_pkg.is_dir() and str(_mlir_pkg) not in sys.path:
        sys.path.insert(0, str(_mlir_pkg))


def _has_bindings() -> bool:
    _setup_mlir_path()
    try:
        import mlir.ir  # noqa: F401
        return True
    except ImportError:
        return False


def lower_linalg_to_llvm_ir(ir_module: Any) -> str:
    """Run full linalg→LLVM lowering pipeline on an ir.Module.

    All ops must already be lowered to linalg/arith/math/tensor dialect.
    Any remaining sf.* or other unregistered dialect ops will cause
    bufferization failures.

    Returns LLVM IR text.
    """
    if not _has_bindings():
        raise RuntimeError("MLIR Python bindings not available")

    import mlir.ir as ir
    import mlir.passmanager as pm

    ctx = ir_module.operation.context
    ctx.allow_unregistered_dialects = True

    with ir.Location.unknown(ctx):
        # First, finalize sf.weight → func.func extra arguments
        _promote_sf_weight_to_args(ir_module)

        pipeline = (
            "builtin.module("
            "func.func(linalg-fuse-elementwise-ops),"
            "func.func(linalg-generalize-named-ops),"
            "one-shot-bufferize{bufferize-function-boundaries},"
            "convert-linalg-to-loops,"
            "lower-affine,"
            "convert-scf-to-cf,"
            "expand-strided-metadata,"
            "finalize-memref-to-llvm,"
            "convert-cf-to-llvm,"
            "convert-math-to-llvm,"
            "convert-arith-to-llvm,"
            "convert-func-to-llvm,"
            "reconcile-unrealized-casts"
            ")"
        )
        pman = pm.PassManager.parse(pipeline, ctx)
        pman.run(ir_module.operation)

    return str(ir_module)


def _promote_sf_weight_to_args(ir_module: Any) -> None:
    """Promote sf.weight ops to func.func arguments.

    Each sf.weight is replaced by a new memref argument added to the
    enclosing function. The Rust runtime will pass mmap'd weight data
    as these arguments at call time. This preserves the 2-file deployment
    model (binary + weights.bin).
    """
    import mlir.ir as ir

    weight_ops: list[Any] = []

    def _collect(inner: Any) -> Any:
        if getattr(inner, 'name', '') in ("sf.weight", "sf.constant"):
            weight_ops.append(inner)
        return ir.WalkResult.ADVANCE

    ir_module.operation.walk(_collect)

    if not weight_ops:
        return

    for w_op in weight_ops:
        if not w_op.operation.results:
            continue
        result = w_op.operation.results[0]
        rt = result.type
        if isinstance(rt, ir.UnrankedTensorType):
            rt = ir.RankedTensorType.get([1], rt.element_type)

        # Find enclosing func.func block
        func_block = None
        for region in ir_module.operation.regions:
            for block in region.blocks:
                for maybe_func in block:
                    if getattr(maybe_func, 'name', '') != 'func.func':
                        continue
                    func_region = maybe_func.operation.regions[0]
                    if func_region.blocks:
                        func_block = func_region.blocks[0]
                    break
                if func_block is not None:
                    break
            if func_block is not None:
                break

        if func_block is None:
            continue

        # Add new block argument for this weight
        new_arg = func_block.add_argument(rt)
        result.replace_all_uses_with(new_arg)
        w_op.operation.erase()

    # Update function type to reflect new arguments
    for region in ir_module.operation.regions:
        for block in region.blocks:
            for func_op in block:
                if getattr(func_op, 'name', '') != 'func.func':
                    continue
                func_region = func_op.operation.regions[0]
                if not func_region.blocks:
                    continue
                func_block = func_region.blocks[0]
                arg_types = [a.type for a in func_block.arguments]
                ret_types: list[ir.Type] = []
                for op_in_block in func_block:
                    if getattr(op_in_block, 'name', '') == 'func.return':
                        ret_types = [o.type for o in op_in_block.operands]
                        break
                func_type = ir.FunctionType.get(arg_types, ret_types)
                func_op.operation.attributes["function_type"] = ir.TypeAttr.get(func_type)


def lower_linalg_to_llvm_ir_text(mlir_text: str) -> str:
    """Parse MLIR text and run linalg→LLVM lowering."""
    if not _has_bindings():
        raise RuntimeError("MLIR Python bindings not available")

    import mlir.ir as ir

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True

    with ir.Location.unknown(ctx):
        module = ir.Module.parse(mlir_text, ctx)
        return lower_linalg_to_llvm_ir(module)


def jit_compile_and_run(ir_module: Any, func_name: str = "main") -> Any:
    """JIT-compile an LLVM IR module and return a callable wrapper.

    Args:
        ir_module: ir.Module with LLVM dialect.
        func_name: Name of the function to look up.

    Returns:
        An ExecutionEngine that can invoke the function.
    """
    if not _has_bindings():
        raise RuntimeError("MLIR Python bindings not available")

    from mlir.execution_engine import ExecutionEngine

    engine = ExecutionEngine(ir_module, opt_level=2)
    return engine
