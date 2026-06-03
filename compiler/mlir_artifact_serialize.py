"""MLIR artifact serialization — save compiled models in MLIR format.

Section A: Serialization functions from the original mlir_artifact.py.

Output structure:
  compiled/<model>/
    model.mlir       — MLIR text (primary format)
    weights.pth      — PyTorch state dict
    metadata.json    — compilation metadata
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

from compiler.mlir_artifact_binary import (
    _build_constants_binary,
    _build_name_mapping,
)
from compiler.mlir_dialect.mlir_op_types import (
    MlirModule,
)
from compiler.mlir_dialect.mlir_op_types import (
    ssa as _ssa,
)

_log = logging.getLogger(__name__)


def mlir_module_to_text(module: MlirModule) -> str:
    """Serialize an MlirModule to standard MLIR text.

    This is the reverse of _parse_mlir_text — generates model.mlir format.
    """
    lines: list[str] = []
    lines.append("module {")

    for func in module.functions:
        # Function arguments
        arg_strs = []
        for name, tp in func.inputs:
            arg_strs.append(f"{_ssa(name)}: {tp}")
        args_str = ", ".join(arg_strs)

        # Return type
        out_types = [tp for _, tp, _ in func.outputs]
        ret = out_types[0] if len(out_types) == 1 else f"({', '.join(out_types)})"

        lines.append(f"  func.func @{func.name}({args_str}) -> {ret} {{")

        # Ops
        for op in func.ops:
            # Weight constants: no operands, just attribute
            if op.op_name == "weight":
                wname = op.attributes.get("name", "")
                tp = op.output_types[0] if op.output_types else "tensor<f32>"
                attrs = f'{{name = "{wname}"}}' if wname else ""
                lines.append(f'    {_ssa(op.results[0])} = "{op.name}"() {attrs} : '
                             f'() -> {tp}')
                continue

            results_str = ", ".join(_ssa(r) for r in op.results)
            operands_str = ", ".join(_ssa(o) for o in op.operands)
            attrs_str = ""
            if op.attributes:
                attr_parts = []
                for k, v in op.attributes.items():
                    if k == "source_node":
                        continue
                    attr_parts.append(_format_attr(k, v))
                if attr_parts:
                    attrs_str = " {" + ", ".join(attr_parts) + "}"
            # Type signature
            if op.input_types and op.output_types:
                in_types = ", ".join(op.input_types)
                out_tp = op.output_types[0] if len(op.output_types) == 1 else f"({', '.join(op.output_types)})"
                type_sig = f"({in_types}) -> {out_tp}"
            else:
                type_sig = ""
            lines.append(
                f'    {results_str} = "{op.name}"({operands_str}){attrs_str}'
                f'{" : " + type_sig if type_sig else ""}'
            )

        # Return — use explicit SSA names.  If any output name is empty
        # (legacy pre-lowering artifact), derive from op results by type.
        ret_names: list[str] = []
        ret_types = [tp for _, tp, _ in func.outputs]
        if any(not name for name, _, _ in func.outputs):
            type_to_results: dict[str, list[str]] = {}
            for op in func.ops:
                for j, r in enumerate(op.results):
                    if j < len(op.output_types):
                        type_to_results.setdefault(op.output_types[j], []).append(r)
            type_positions: dict[str, int] = {}
            for name, tp, _ in func.outputs:
                results = type_to_results.get(tp, [])
                pos = type_positions.get(tp, 0)
                if pos < len(results):
                    ret_names.append(f"%{results[pos]}")
                    type_positions[tp] = pos + 1
                else:
                    ret_names.append("")
            ret_str = ", ".join(ret_names)
        else:
            ret_str = ", ".join(_ssa(name) for name, _, _ in func.outputs)
        if len(ret_types) == 1:
            ret_type_str = f" : {ret_types[0]}"
        else:
            ret_type_str = f" : {', '.join(ret_types)}"
        lines.append(f"    func.return {ret_str}{ret_type_str}")
        lines.append("  }")

    lines.append("}")
    return "\n".join(lines) + "\n"


def _unranked_tensor_type(tp: str) -> str:
    """Convert a ranked tensor type to unranked form to avoid type conflicts.

    tensor<1xi64> -> tensor<*xi64>
    tensor<2x64xf32> -> tensor<*xf32>
    tensor<f32> -> tensor<*xf32>
    tensor<*xf32> -> tensor<*xf32>  (already unranked, no change)
    """
    import re
    if tp.startswith("tensor<*x"):
        return tp  # already unranked
    m = re.match(r"tensor<(?:[\d?*]+x)*(.+)>", tp)
    if m:
        return f"tensor<*x{m.group(1)}>"
    return tp


def _format_attr(key: str, value: Any) -> str:
    """Format a single attribute key=value pair for MLIR text."""
    if isinstance(value, bool):
        return f'{key} = {"true" if value else "false"}'
    if isinstance(value, int):
        return f"{key} = {value} : i64"
    if isinstance(value, float):
        return f"{key} = {value} : f64"
    if isinstance(value, str):
        return f'{key} = "{value}"'
    if isinstance(value, (list, tuple)):
        items = ", ".join(
            f'"{v}"' if isinstance(v, str) else str(v)
            for v in value
        )
        return f"{key} = [{items}]"
    if value is None:
        return f"{key} = none"
    return f'{key} = "{value}"'


def save_mlir_module_artifact(module: MlirModule, directory: str) -> None:
    """Persist an MlirModule as MLIR artifact.

    Output:
      model.mlir         — MLIR text
      constants.pth      — export-time constants (scalars, fused weights) only
      constants.bin      — embedded data for .dylib: name_mapping + constants
      metadata.json      — compilation metadata + weight source + classification

    Model parameters are NOT duplicated — weight_source in metadata points
    to the original safetensors file for mmap loading at runtime.
    """
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    mlir_text = mlir_module_to_text(module)
    with open(out_dir / "model.mlir", "w") as f:
        f.write(mlir_text)

    has_classification = any(
        func.param_weight_names or func.const_weight_names
        for func in module.functions
    )
    if has_classification:
        const_state: dict[str, torch.Tensor] = {}
        for func in module.functions:
            for wname, tensor in func.weights.items():
                if wname in func.const_weight_names:
                    key = f"{func.name}.{wname}" if func.name != "main" else wname
                    const_state[key] = tensor
        if const_state:
            torch.save(const_state, out_dir / "constants.pth")
    else:
        # Backward compat: no classification info → write all weights
        weight_state: dict[str, torch.Tensor] = {}
        for func in module.functions:
            for wname, tensor in func.weights.items():
                key = f"{func.name}.{wname}" if func.name != "main" else wname
                weight_state[key] = tensor
        torch.save(weight_state, out_dir / "weights.pth")

    # ── Build name mapping and embedded constants binary ──
    name_mapping = _build_name_mapping(module)
    if name_mapping:
        const_bin = _build_constants_binary(module, name_mapping)
        with open(out_dir / "constants.bin", "wb") as f:
            f.write(const_bin)
        module.metadata["weight_source"] = module.metadata.get("weight_source", {})
        module.metadata["weight_source"]["embedded_data"] = "constants.bin"
        module.metadata["weight_source"]["name_mapping"] = name_mapping

    classification: dict[str, dict[str, list[str]]] = {}
    for func in module.functions:
        if func.param_weight_names or func.const_weight_names:
            classification[func.name] = {
                "params": sorted(func.param_weight_names),
                "constants": sorted(func.const_weight_names),
            }
    if classification:
        module.metadata["weight_classification"] = classification

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(module.metadata, f, indent=2)
