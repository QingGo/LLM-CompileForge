"""MLIR artifact serialization — save/load compiled models in MLIR format.

Replaces the JSON-based model.ir with standard-compliant MLIR text.

Output structure:
  compiled/<model>/
    model.mlir       — MLIR text (primary format)
    weights.pth      — PyTorch state dict
    metadata.json    — compilation metadata

The model.mlir passes `mlir-opt --allow-unregistered-dialect` and can be
parsed by `mlir.parse_string()` (pymlir) for downstream processing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from compiler.ir import IrModule, pack_weights
from compiler.mlir_emitter import ir_module_to_mlir


@dataclass
class MlirOp:
    """A single MLIR operation parsed from model.mlir.

    Mirrors IrOp structure so the executor can walk the graph
    without depending on IrModule.
    """

    name: str  # full qualified name: "sf.linear", "sf.weight", etc.
    dialect: str  # "sf", "arith", etc.
    op_name: str  # "linear", "matmul", "weight", etc.
    operands: list[str]  # SSA names of inputs
    results: list[str]  # SSA names of outputs
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class MlirFunction:
    """A parsed MLIR function (func.func)."""

    name: str
    inputs: list[tuple[str, str]]  # (ssa_name, mlir_type_string)
    outputs: list[tuple[str, str]]  # (ssa_name, mlir_type_string)
    ops: list[MlirOp] = field(default_factory=list)
    weights: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class MlirModule:
    """A parsed MLIR module containing functions and weights."""

    functions: list[MlirFunction] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def main(self) -> MlirFunction:
        """Return the main (first) function."""
        if not self.functions:
            raise ValueError("MlirModule has no functions")
        return self.functions[0]


# ── Public API ────────────────────────────────────────────────


def save_mlir_artifact(module: IrModule, directory: str) -> None:
    """Persist a compiled IrModule as MLIR artifact.

    Writes model.mlir (primary), weights.pth, and metadata.json.
    """
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stamp schema version
    module.metadata.setdefault("ir_schema_version", 1)
    module.metadata["artifact_format"] = "mlir"

    # model.mlir (primary format)
    mlir_text = ir_module_to_mlir(module)
    with open(out_dir / "model.mlir", "w") as f:
        f.write(mlir_text)

    # weights.pth
    all_weights = pack_weights(module)
    weight_state: dict[str, torch.Tensor] = {}
    for func_name, func_weights in all_weights.items():
        for wname, tensor in func_weights.items():
            key = f"{func_name}.{wname}" if func_name != "main" else wname
            weight_state[key] = tensor
    torch.save(weight_state, out_dir / "weights.pth")

    # metadata.json
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(module.metadata, f, indent=2)


def load_mlir_artifact(directory: str) -> MlirModule:
    """Load a compiled MLIR artifact from disk.

    Parses model.mlir (via pymlir) and loads weights from weights.pth.
    Falls back to model.ir (legacy JSON) if model.mlir is missing.

    Returns an MlirModule with ops, weights, and metadata.
    """
    in_dir = Path(directory)
    if not in_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    mlir_path = in_dir / "model.mlir"
    weights_path = in_dir / "weights.pth"
    meta_path = in_dir / "metadata.json"

    if not mlir_path.exists():
        raise FileNotFoundError(
            f"model.mlir not found at {directory}. "
            "Re-compile the model with the current toolchain."
        )

    with open(mlir_path) as f:
        mlir_text = f.read()
    module = _parse_mlir_text(mlir_text)

    # Load weights
    if weights_path.exists():
        raw_weights: dict[str, torch.Tensor] = torch.load(
            weights_path, map_location="cpu", weights_only=True
        )
        # Distribute to functions
        for key, tensor in raw_weights.items():
            if "." in key and not key.startswith("_"):
                func_name, wname = key.split(".", 1)
            else:
                func_name = "main"
                wname = key
            for func in module.functions:
                if func.name == func_name:
                    func.weights[wname] = tensor
                    break
            else:
                if module.functions:
                    module.functions[0].weights[wname] = tensor

    # Load metadata
    if meta_path.exists():
        with open(meta_path) as f:
            module.metadata = json.load(f)

    return module


# ── Internal parsing ──────────────────────────────────────────


def _parse_mlir_text(text: str) -> MlirModule:
    """Parse MLIR text into an MlirModule.

    Uses a lightweight line-based parser that handles the subset of MLIR
    generated by ir_module_to_mlir().  This avoids the overhead of pymlir
    for large files and gives us direct control over the AST structure.
    """
    functions: list[MlirFunction] = []
    current_func: MlirFunction | None = None
    ssa_to_name: dict[str, str] = {}  # SSA → weight name
    ssa_types: dict[str, str] = {}  # SSA → MLIR type

    for line in text.splitlines():
        stripped = line.strip()

        # Skip comments and empty lines
        if not stripped or stripped.startswith("//"):
            continue

        # Module / function boundaries
        if stripped == "module {":
            continue
        if stripped == "}":
            if current_func is not None:
                functions.append(current_func)
                current_func = None
            continue

        # Function declaration: func.func @name(...) -> ... {
        if stripped.startswith("func.func @"):
            # Parse function name and signature
            rest = stripped[len("func.func @"):]
            name_end = rest.find("(")
            func_name = rest[:name_end].strip()
            args_str = rest[name_end + 1:rest.find(")")]
            # Parse args: %arg0: tensor<...>, %arg1: tensor<...>
            inputs: list[tuple[str, str]] = []
            for arg in _split_comma(args_str):
                arg = arg.strip()
                if ":" in arg:
                    ssa, tp = arg.split(":", 1)
                    inputs.append((ssa.strip(), tp.strip()))
            current_func = MlirFunction(name=func_name, inputs=inputs, outputs=[])
            ssa_to_name = {}
            ssa_types = {}
            continue

        # Skip func.return lines for now
        if stripped.startswith("func.return"):
            continue

        # Wrapper lines: { or } within function
        if stripped in ("{", "}"):
            continue

        # Parse operations: %r = "dialect.op"(...) {attrs} : (in_types) -> out_type
        if "=" in stripped and "\"" in stripped:
            _parse_mlir_op(
                stripped, current_func, ssa_to_name, ssa_types,
            )

    if current_func is not None:
        functions.append(current_func)

    return MlirModule(functions=functions)


def _parse_mlir_op(
    line: str,
    func: MlirFunction | None,
    ssa_to_name: dict[str, str],
    ssa_types: dict[str, str],
) -> None:
    """Parse a single MLIR operation line."""
    if func is None:
        return

    # Split: %r1, %r2 = "sf.op"(%a, %b) {attr} : (t1, t2) -> (t3, t4)
    eq_idx = line.index("=")
    results_part = line[:eq_idx].strip()
    rest = line[eq_idx + 1:].strip()

    # Parse results (strip % prefix)
    results = [r.strip().lstrip("%") for r in results_part.split(",")]

    # Parse op name
    quote_start = rest.index('"')
    quote_end = rest.index('"', quote_start + 1)
    qualified = rest[quote_start + 1:quote_end]
    dialect, op_name = _split_qualified(qualified)

    # Parse operands: everything between (...) after the op name
    paren_open = rest.index("(", quote_end)
    paren_close = rest.index(")", paren_open)
    operands_str = rest[paren_open + 1:paren_close]
    operands = [o.strip() for o in operands_str.split(",") if o.strip()]

    # Parse attributes: everything between { and } before the :
    attrs: dict[str, Any] = {}
    brace_open = rest.find("{", paren_close)
    if brace_open >= 0 and brace_open < rest.find(":", brace_open):
        brace_close = rest.index("}", brace_open)
        attrs_str = rest[brace_open + 1:brace_close]
        attrs = _parse_attrs(attrs_str)

    # Record weight SSA mapping for sf.weight ops
    if qualified == "sf.weight" and "name" in attrs:
        if results:
            ssa_to_name[results[0]] = attrs["name"]

    # Record SSA types
    colon_idx = rest.rfind(":")
    if colon_idx >= 0:
        type_part = rest[colon_idx + 1:].strip()
        # Parse output types after -> or single type
        arrow_idx = type_part.find("->")
        if arrow_idx >= 0:
            out_types_str = type_part[arrow_idx + 2:].strip()
            out_types = _parse_type_list(out_types_str)
        else:
            out_types = [type_part]

        for i, r in enumerate(results):
            if i < len(out_types):
                ssa_types[r] = out_types[i]

    # Resolve operand names from SSA mapping, strip % prefix
    resolved_operands: list[str] = []
    for opnd in operands:
        clean = opnd.lstrip("%")
        if clean in ssa_to_name:
            resolved_operands.append(ssa_to_name[clean])
        else:
            resolved_operands.append(clean)

    func.ops.append(
        MlirOp(
            name=qualified,
            dialect=dialect,
            op_name=op_name,
            operands=resolved_operands,
            results=results,
            attributes=attrs,
        )
    )


def _split_qualified(qualified: str) -> tuple[str, str]:
    """Split 'sf.weight' → ('sf', 'weight')."""
    if "." in qualified:
        parts = qualified.split(".", 1)
        return parts[0], parts[1]
    return "sf", qualified


def _split_comma(text: str) -> list[str]:
    """Split by commas, respecting nested angle brackets."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in text:
        if ch in "<(":
            depth += 1
        elif ch in ">)":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return parts


def _parse_attrs(attrs_str: str) -> dict[str, Any]:
    """Parse MLIR attribute string like 'dim = 0, name = "foo"'."""
    result: dict[str, Any] = {}
    if not attrs_str.strip():
        return result

    parts = _split_attrs(attrs_str)
    for part in parts:
        part = part.strip()
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()
        # Parse value
        if v == "true":
            result[k] = True
        elif v == "false":
            result[k] = False
        elif v == "none":
            result[k] = None
        elif v.startswith('"') and v.endswith('"'):
            result[k] = v[1:-1]
        elif v.startswith("[") and v.endswith("]"):
            inner = v[1:-1]
            items = _split_comma(inner)
            result[k] = [_parse_attr_value(item.strip()) for item in items if item.strip()]
        else:
            try:
                result[k] = int(v)
            except ValueError:
                try:
                    result[k] = float(v)
                except ValueError:
                    result[k] = v
    return result


def _split_attrs(text: str) -> list[str]:
    """Split attribute string respecting nested brackets and quotes."""
    parts: list[str] = []
    depth = 0
    in_quote = False
    current: list[str] = []
    for ch in text:
        if ch == '"':
            in_quote = not in_quote
        if not in_quote:
            if ch in "[{(":
                depth += 1
            elif ch in "]})":
                depth -= 1
        if ch == "," and depth == 0 and not in_quote:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return parts


def _parse_attr_value(v: str) -> Any:
    """Parse a single attribute value."""
    v = v.strip()
    if v == "true":
        return True
    if v == "false":
        return False
    if v == "none":
        return None
    if v.startswith('"') and v.endswith('"'):
        return v[1:-1]
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def _parse_type_list(text: str) -> list[str]:
    """Parse a comma-separated list of MLIR types, respecting angle brackets."""
    text = text.strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    if not text:
        return []
    return [t.strip() for t in _split_comma(text)]
