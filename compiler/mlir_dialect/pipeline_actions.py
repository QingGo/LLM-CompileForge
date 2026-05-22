# ruff: noqa: E501 — long lines in MLIR transform script f-strings

"""Custom stage actions for the MLIR lowering pipeline.

Contains the action callables used by ``Stage`` objects — tiling, FMA
fusion, matmul output filling.  Split from ``pipeline_stages.py`` to
keep each file under 500 lines.
"""

from __future__ import annotations

import logging
import re
from typing import Any

_log = logging.getLogger(__name__)


# ── Custom stage actions ──────────────────────────────────────────────


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


def _op_name(op):
    try:
        return op.operation.name
    except Exception as e:
        _log.warning("  _op_name failed: %s", e, exc_info=True)
        return ""


def _def_op(value):
    try:
        own = value.owner
        if own is not None and hasattr(own, 'operation'):
            _op_name(own)
            return own
    except Exception as e:
        _log.warning("  _def_op failed: %s", e, exc_info=True)
    return None


def fuse_fma_action(module: Any) -> int:
    """Replace ``llvm.fmul + llvm.{fadd,fsub}`` with ``llvm.intr.fmuladd``.

    Handles three patterns on LLVM dialect ops:
      (1) ``%s = fmul a, b ; %r = fadd %s, c``   →  ``%r = fmuladd(a, b, c)``
      (2) ``%s = fmul a, b ; %r = fsub c, %s``   →  ``%r = fmuladd(a, fneg(b), c)``
      (3) ``%s = fmul a, b ; %r = fsub %s, c``   →  ``%r = fmuladd(a, b, fneg(c))``

    Returns number of fusions.
    """
    import mlir.ir as ir

    candidates: list = []
    for region in [module.operation.regions[0]]:
        for block in region.blocks:
            for func_op in block:
                if _op_name(func_op) not in ("func.func", "llvm.func"):
                    continue
                func_region = func_op.operation.regions[0]
                for body_block in func_region.blocks:
                    for op in list(body_block):
                        name = _op_name(op)
                        if name not in ("llvm.fadd", "llvm.fsub"):
                            continue
                        is_fsub = (name == "llvm.fsub")
                        for idx in range(2):
                            src = op.operation.operands[idx]
                            if src is None:
                                continue
                            d = _def_op(src)
                            if d is None or _op_name(d) != "llvm.fmul":
                                continue
                            candidates.append((d, op, idx, is_fsub))
                            break

    if not candidates:
        return 0

    f32_type = ir.F32Type.get(context=module.operation.context)
    n_fused = 0

    for fmul_op, tgt_op, fmul_pos, is_fsub in candidates:
        a = fmul_op.operation.operands[0]
        b = fmul_op.operation.operands[1]

        with ir.InsertionPoint(tgt_op):
            try:
                if is_fsub and fmul_pos == 1:
                    c_src = tgt_op.operation.operands[0]
                    neg = ir.Operation.create(
                        "llvm.fneg", operands=[b], results=[f32_type],
                        ip=ir.InsertionPoint(tgt_op),
                    )
                    a_fma, b_fma = a, neg.results[0]
                elif is_fsub and fmul_pos == 0:
                    c_src = tgt_op.operation.operands[1]
                    neg = ir.Operation.create(
                        "llvm.fneg", operands=[c_src], results=[f32_type],
                        ip=ir.InsertionPoint(tgt_op),
                    )
                    a_fma, b_fma = a, b
                    c_src = neg.results[0]
                else:
                    c_src = tgt_op.operation.operands[1 - fmul_pos]
                    a_fma, b_fma = a, b

                new_op = ir.Operation.create(
                    "llvm.intr.fmuladd",
                    operands=[a_fma, b_fma, c_src],
                    results=[f32_type],
                    ip=ir.InsertionPoint(tgt_op),
                )

                tgt_op.operation.result.replace_all_uses_with(new_op.results[0])
                tgt_op.erase()
                try:
                    fmul_op.erase()
                except Exception as e:
                    _log.debug("  FMA: could not erase fmul (multi-use) — %s", e, exc_info=True)
                n_fused += 1
            except Exception as e:
                _log.debug("  FMA: fusion failed for a candidate — %s", e, exc_info=True)

    return n_fused


def ensure_filled_matmul_outputs_action(module: Any) -> int:
    """Insert ``linalg.fill(0.0)`` before matmuls using ``tensor.empty`` output.

    Operates on the MLIR **text** (``str(module)``) to insert fills between
    ``tensor.empty`` and ``linalg.{matmul,batch_matmul}`` that lack them.
    After text modification, re-parses the module in-place.

    Returns the number of fills inserted.
    """
    import mlir.ir as ir

    txt = str(module)
    modified = False

    empty_pattern = re.compile(
        r'%\w+\s*=\s*tensor\.empty\(\)\s*:\s*tensor<([^>]+)>'
    )

    for m in list(empty_pattern.finditer(txt)):
        empty_name = m.group(0).split('=')[0].strip()
        empty_inner_type = m.group(1)

        pos = m.end()
        chunk = txt[pos:pos + 4000]

        use_line = None
        for line_idx, line in enumerate(chunk.split('\n')):
            stripped = line.strip()
            if ('linalg.matmul' in stripped or 'linalg.batch_matmul' in stripped):
                if empty_name in stripped.replace('%__TMP_', ''):
                    use_line = line_idx
                    break

        if use_line is None:
            continue

        has_fill = any(
            'linalg.fill' in chunk.split('\n')[i]
            for i in range(min(use_line, len(chunk.split('\n'))))
        )
        if has_fill:
            continue

        outs_match = re.search(
            rf'outs\(\s*{re.escape(empty_name)}\s*:',
            chunk.split('\n')[use_line],
        )
        if not outs_match:
            continue

        indent = '    '
        fill_lines = [
            f'{indent}%__FILL_{empty_name.lstrip("%")}__ = arith.constant 0.000000e+00 : f32',
            f'{indent}%__FILLED_{empty_name.lstrip("%")}__ = linalg.fill '
            f'ins(%__FILL_{empty_name.lstrip("%")}__ : f32) '
            f'outs({empty_name} : tensor<{empty_inner_type}>) '
            f'-> tensor<{empty_inner_type}>',
        ]

        actual_line_pos = pos
        for _ in range(use_line):
            actual_line_pos = txt.index('\n', actual_line_pos) + 1

        insert_pos = actual_line_pos
        txt = txt[:insert_pos] + '\n'.join(fill_lines) + '\n' + txt[insert_pos:]
        modified = True

    if not modified:
        return 0

    ctx = module.operation.context
    new_mod = ir.Module.parse(txt, ctx)
    old_block = module.operation.regions[0].blocks[0]
    new_block = new_mod.operation.regions[0].blocks[0]
    for op in list(old_block):
        op.erase()
    for op in list(new_block):
        cloned = op.operation.clone()
        old_block.append(cloned)

    return 1
