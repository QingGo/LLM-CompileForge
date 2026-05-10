"""MLIR Quantize/Dequantize pass — insert Q/DQ nodes into MLIR text.

Reference: design-phase2.md §2.1.4, §2.1.5
"""

from __future__ import annotations

import re

from compiler.quantize.mixed_precision import MixedPrecisionConfig

_WEIGHT_RE = re.compile(
    r'(%\w+)\s*=\s*"sf\.weight"\(\)\s*\{name\s*=\s*"([^"]+)"'
)
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
    the dequantized SSA name.

    Args:
        mlir_text: Valid MLIR module text.
        config: Per-layer precision strategy.  If None, uses defaults.

    Returns:
        MLIR text with Q/DQ nodes inserted.
    """
    if config is None:
        config = MixedPrecisionConfig()

    lines = mlir_text.split("\n")

    quant_weights: dict[str, tuple[str, str, str]] = {}
    for _i, line in enumerate(lines):
        m = _WEIGHT_RE.search(line)
        if m:
            ssa_name = m.group(1)
            weight_name = m.group(2)
            precision = config.get_precision(weight_name)
            if precision in ("w8a8", "w4a16", "int8"):
                dq_ssa = f"%dq_{len(quant_weights)}"
                quant_weights[ssa_name] = (dq_ssa, precision, weight_name)

    output: list[str] = []
    for line in lines:
        output.append(line)

        m = _WEIGHT_RE.search(line)
        if m:
            ssa_name = m.group(1)
            if ssa_name in quant_weights:
                dq_ssa, precision, wname = quant_weights[ssa_name]
                dq_line = (
                    f'    {dq_ssa} = "sf.dequantize"({ssa_name}) '
                    f'{{precision = "{precision}", weight = "{wname}"}}'
                    f' : () -> tensor<*xf32>'
                )
                output.append(dq_line)

    result_text = "\n".join(output)

    for old_ssa, (new_ssa, _prec, _wname) in quant_weights.items():
        result_text = result_text.replace(f"( {old_ssa} ", f"( {new_ssa} ")
        result_text = result_text.replace(f"({old_ssa} ", f"({new_ssa} ")
        result_text = result_text.replace(f" {old_ssa} ", f" {new_ssa} ")
        result_text = result_text.replace(f" {old_ssa})", f" {new_ssa})")
        if result_text.rstrip().endswith(f" {old_ssa}"):
            idx = result_text.rfind(f" {old_ssa}")
            result_text = result_text[:idx] + f" {new_ssa}" + result_text[idx + len(old_ssa) + 1:]

    return result_text


def count_dq_ops(mlir_text: str) -> int:
    """Count the number of sf.dequantize operations in MLIR text."""
    return len(_DQ_RE.findall(mlir_text))
