# ruff: noqa: E501 — long lines in MLIR transform script f-strings

"""Custom stage actions for the MLIR lowering pipeline.

Currently contains the tiling action (transform-dialect based), the
identity-copies insertion action, and framework for other custom actions.
FMA fusion is handled by llc -O3. Matmul output filling is handled by
C++ lowering patterns in SfLowerMatmul.cpp.
"""

from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger(__name__)


def tile_matmuls_action(module: Any, tile_k: int = 64) -> None:
    """Tile ``linalg.matmul`` and ``linalg.batch_matmul`` K dim by tile_k.

    Applies the transform dialect ONCE per func.func (to avoid the
    ``tile_using_for`` multi-handle limitation).  Each func is wrapped in
    a temporary module, tiled, and the result is cloned back.
    """
    import mlir.ir as ir
    import mlir.passmanager as pm

    ctx = module.operation.context
    ctx.load_all_available_dialects()

    script = (
        "module attributes {transform.with_named_sequence} {\n"
        "  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {\n"
        '    %mats = transform.structured.match ops{["linalg.matmul"]} in %arg0\n'
        "      : (!transform.any_op) -> !transform.any_op\n"
        "    transform.structured.tile_using_for %mats\n"
        "      tile_sizes [0, " + str(tile_k) + ", " + str(tile_k) + "]\n"
        "      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)\n"
        '    %batch_mats = transform.structured.match ops{["linalg.batch_matmul"]} in %arg0\n'
        "      : (!transform.any_op) -> !transform.any_op\n"
        "    transform.structured.tile_using_for %batch_mats\n"
        "      tile_sizes [0, 0, " + str(tile_k) + ", " + str(tile_k) + "]\n"
        "      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)\n"
        "    transform.yield\n"
        "  }\n"
        "}\n"
    )

    block = module.operation.regions[0].blocks[0]
    for func in list(block):
        if str(func.operation.name) != "func.func":
            continue
        ftxt = str(func)
        if "linalg.matmul" not in ftxt and "linalg.batch_matmul" not in ftxt:
            continue

        combined = ir.Module.parse(script + "\n" + ftxt, ctx)
        try:
            pm.PassManager.parse("builtin.module(transform-interpreter)", ctx).run(combined.operation)
        except Exception as e:
            _log.warning(
                "  tile_matmuls: func %s skipped (%s)",
                str(func.operation.name) if hasattr(func, "operation") else "?",
                str(e).split("\n")[0] if "\n" in str(e) else str(e),
                exc_info=True,
            )
            continue

        for op in list(combined.operation.regions[0].blocks[0]):
            name = str(op.operation.name)
            src = None
            if name == "func.func":
                src = op
            elif name == "builtin.module":
                for inner in op.operation.regions[0].blocks[0]:
                    if str(inner.operation.name) == "func.func":
                        src = inner
                        break
            if src is not None:
                cloned = src.operation.clone()
                func.operation.erase()
                block.append(cloned)
                break


def insert_identity_copies_action(module: Any) -> None:
    """Insert ``tensor.insert_slice`` copies for identity pass-through returns.

    In tensor semantics, ``func.return %arg0`` is a valid zero-copy reference.
    But after bufferization, the output memref is never allocated or written,
    producing garbage values at runtime.

    This action walks all ``func.func`` ops, finds ``func.return`` operands
    that are direct ``BlockArgument`` values, and inserts a
    ``tensor.empty`` + ``tensor.insert_slice`` pair to force a materialized copy.

    Only inserts copies for identity pass-throughs. Computed output values
    (results of ops like ``linalg.matmul``, ``linalg.add``, etc.) are left
    untouched.
    """
    import mlir.dialects.func as func
    import mlir.dialects.tensor as tensor
    import mlir.ir as ir

    ctx = module.operation.context
    total_inserted = 0

    main_block = module.operation.regions[0].blocks[0]
    for func_op in list(main_block):
        if str(func_op.operation.name) != "func.func":
            continue

        func_region = func_op.operation.regions[0]
        func_body = func_region.blocks[0]
        func_args = list(func_body.arguments)

        if not func_args:
            continue

        return_op = None
        for op in func_body.operations:
            if str(op.operation.name) == "func.return":
                return_op = op
                break

        if return_op is None:
            continue

        with ctx, ir.Location.unknown(ctx):
            modified = False
            new_operands = list(return_op.operation.operands)

            for idx, operand in enumerate(new_operands):
                if not isinstance(operand, ir.BlockArgument):
                    continue

                block_arg = operand
                arg_type = block_arg.type
                if not isinstance(arg_type, ir.RankedTensorType):
                    continue

                shape = arg_type.shape
                element_type = arg_type.element_type

                ip = ir.InsertionPoint(return_op.operation)

                empty = tensor.empty(sizes=list(shape), element_type=element_type, ip=ip)
                rank = len(shape)
                offsets = [0] * rank
                sizes = list(shape)
                strides = [1] * rank

                copied = tensor.insert_slice(
                    source=block_arg,
                    dest=empty,
                    static_offsets=offsets,
                    static_sizes=sizes,
                    static_strides=strides,
                    offsets=[],
                    sizes=[],
                    strides=[],
                    ip=ip,
                )

                new_operands[idx] = copied
                modified = True

            if modified:
                ip = ir.InsertionPoint(return_op.operation)
                func.ReturnOp(operands_=new_operands, ip=ip)
                return_op.operation.erase()
                total_inserted += 1

    if total_inserted > 0:
        _log.info(
            "  insert_identity_copies: inserted copies in %d function(s)",
            total_inserted,
        )


def insert_unsqueeze_copies_action(module: Any) -> None:
    """Stage C2.6 placeholder — bufferization fix is in C3 options."""
    pass
