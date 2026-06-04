# ruff: noqa: E501 — long lines in MLIR transform script f-strings

"""Custom stage actions for the MLIR lowering pipeline.

Currently contains only the tiling action (transform-dialect based).
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
        'module attributes {transform.with_named_sequence} {\n'
        '  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {\n'
        '    %mats = transform.structured.match ops{["linalg.matmul"]} in %arg0\n'
        '      : (!transform.any_op) -> !transform.any_op\n'
        '    transform.structured.tile_using_for %mats\n'
        '      tile_sizes [0, ' + str(tile_k) + ', ' + str(tile_k) + ']\n'
        '      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)\n'
        '    %batch_mats = transform.structured.match ops{["linalg.batch_matmul"]} in %arg0\n'
        '      : (!transform.any_op) -> !transform.any_op\n'
        '    transform.structured.tile_using_for %batch_mats\n'
        '      tile_sizes [0, 0, ' + str(tile_k) + ', ' + str(tile_k) + ']\n'
        '      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)\n'
        '    transform.yield\n'
        '  }\n'
        '}\n'
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
            pm.PassManager.parse("builtin.module(transform-interpreter)", ctx).run(
                combined.operation
            )
        except Exception as e:
            _log.warning("  tile_matmuls: func %s skipped (%s)",
                         str(func.operation.name) if hasattr(func, "operation") else "?",
                         str(e).split("\n")[0] if "\n" in str(e) else str(e),
                         exc_info=True)
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
