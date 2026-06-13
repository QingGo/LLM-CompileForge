"""MLIR Quantize/Dequantize pass — insert Q/DQ nodes into MLIR text.

Reference: design-phase2.md §2.1.4, §2.1.5
"""

from __future__ import annotations

import re

from compiler.quantize.mixed_precision import MixedPrecisionConfig

_WEIGHT_RE = re.compile(r'(%\w+)\s*=\s*"sf\.weight"\(\)\s*\{name\s*=\s*"([^"]+)"')
_DQ_RE = re.compile(r'"sf\.dequantize"')


def _parse_weight_names(mlir_text: str) -> dict[str, str]:
    """Extract weight SSA names from MLIR text.

    Returns:
        dict mapping weight name attribute → SSA name (e.g. "%1").
    """
    result: dict[str, str] = {}
    for match in _WEIGHT_RE.finditer(mlir_text):
        ssa_name, weight_name = match.groups()
        result[weight_name] = ssa_name
    return result


def insert_quantize_dequantize(
    mlir_text: str,
    config: MixedPrecisionConfig | None = None,
) -> str:
    """Insert Q/DQ nodes into MLIR text based on a precision strategy.

    For each weight marked as w8a8 or w4a16, inserts a dequantize op
    after the weight declaration and replaces all SSA references with
    the dequantized result.

    Uses the MLIR Python API for correct SSA value renaming, avoiding
    substring collisions (e.g., ``%1`` vs ``%10``) that would occur with
    ``str.replace()``.

    Args:
        mlir_text: Valid MLIR module text.
        config: Per-layer precision strategy.  If None, uses defaults.

    Returns:
        MLIR text with Q/DQ nodes inserted.
    """
    import mlir.ir as ir
    from mlir.ir import InsertionPoint, Location, StringAttr

    if config is None:
        config = MixedPrecisionConfig()

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True

    with ctx, Location.unknown(ctx):
        module = ir.Module.parse(mlir_text, ctx)

        for top_op in module.body.operations:
            for region in top_op.regions:
                for block in region.blocks:
                    ops = list(block.operations)
                    for i, op in enumerate(ops):
                        if str(op.name) == "sf.weight":
                            weight_name = op.attributes["name"].value
                            precision = config.get_precision(weight_name)
                            if precision in ("w8a8", "w4a16", "int8"):
                                weight_result = op.results[0]

                                if i + 1 < len(ops):
                                    ip = InsertionPoint(ops[i + 1])
                                else:
                                    ip = InsertionPoint.at_block_terminator(block)

                                dq_op = ir.Operation.create(
                                    "sf.dequantize",
                                    operands=[weight_result],
                                    results=[weight_result.type],
                                    attributes={
                                        "precision": StringAttr.get(precision),
                                        "weight": StringAttr.get(weight_name),
                                    },
                                )
                                ip.insert(dq_op)

                                weight_result.replace_all_uses_except(dq_op.results[0], dq_op)

        return str(module)


def count_dq_ops(mlir_text: str) -> int:
    """Count the number of sf.dequantize operations in MLIR text."""
    return len(_DQ_RE.findall(mlir_text))
