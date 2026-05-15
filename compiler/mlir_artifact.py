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
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import mlir.ir as ir

import torch


@dataclass
class MlirOp:
    """A single MLIR operation parsed from model.mlir."""

    name: str  # full qualified name: "sf.linear", "sf.weight", etc.
    dialect: str  # "sf", "arith", etc.
    op_name: str  # "linear", "matmul", "weight", etc.
    operands: list[str]  # SSA names of inputs
    results: list[str]  # SSA names of outputs
    attributes: dict[str, Any] = field(default_factory=dict)
    input_types: list[str] = field(default_factory=list)  # MLIR type strings for each operand
    output_types: list[str] = field(default_factory=list)  # MLIR type strings for each result


@dataclass
class MlirFunction:
    """A parsed MLIR function (func.func)."""

    name: str
    inputs: list[tuple[str, str]]  # (ssa_name, mlir_type_string)
    outputs: list[tuple[str, str]]  # (ssa_name, mlir_type_string)
    ops: list[MlirOp] = field(default_factory=list)
    weights: dict[str, torch.Tensor] = field(default_factory=dict)
    param_weight_names: set[str] = field(default_factory=set)
    const_weight_names: set[str] = field(default_factory=set)


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
            arg_strs.append(f"{name}: {tp}")
        args_str = ", ".join(arg_strs)

        # Return type
        out_types = [tp for _, tp in func.outputs]
        ret = out_types[0] if len(out_types) == 1 else f"({', '.join(out_types)})"

        lines.append(f"  func.func @{func.name}({args_str}) -> {ret} {{")

        # Ops
        for op in func.ops:
            # Weight constants: no operands, just attribute
            if op.op_name == "weight":
                wname = op.attributes.get("name", "")
                tp = op.output_types[0] if op.output_types else "tensor<f32>"
                attrs = f'{{name = "{wname}"}}' if wname else ""
                lines.append(f'    %{op.results[0]} = "{op.name}"() {attrs} : '
                             f'() -> {tp}')
                continue

            results_str = ", ".join(f"%{r}" if not r.startswith("%") else r for r in op.results)
            operands_str = ", ".join(f"%{o}" if not o.startswith("%") else o for o in op.operands)
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

        # Return
        ret_names = [name for name, _ in func.outputs]
        ret_str = ", ".join(ret_names)
        ret_types = [tp for _, tp in func.outputs]
        if len(ret_types) == 1:
            ret_type_str = f" : {ret_types[0]}"
        else:
            ret_type_str = f" : ({', '.join(ret_types)})"
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
        items = ", ".join(str(v) for v in value)
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



def load_mlir_artifact(directory: str) -> MlirModule:
    """Load a compiled MLIR artifact from disk.

    Parses model.mlir and loads weights via one of two paths:

    1. New path (preferred): If metadata.json has ``weight_source`` pointing
       to a safetensors file, mmap model parameters from that file and load
       export constants from constants.pth.  No redundant copy on disk.

    2. Old path (backward compat): If ``weight_source`` is absent or the
       source file is missing, loads weights.pth as before.

    Returns an MlirModule with ops, weights, and metadata.
    """
    in_dir = Path(directory)
    if not in_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    mlir_path = in_dir / "model.mlir"
    const_path = in_dir / "constants.pth"
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

    if meta_path.exists():
        with open(meta_path) as f:
            module.metadata = json.load(f)

    # ── Determine loading path ──────────────────────────
    ws = module.metadata.get("weight_source", {})
    ws_path = ws.get("path", "")
    use_new_path = bool(ws_path and os.path.isfile(ws_path))

    if use_new_path and ws.get("format") == "safetensors":
        _load_weights_via_mmap(module, ws_path, const_path)
    elif use_new_path and ws.get("format") == "safetensors_sharded":
        _load_weights_via_sharded(module, ws_path, const_path)
    elif use_new_path and ws.get("format") == "pytorch_bin":
        _load_weights_via_bin(module, ws_path, const_path, ws.get("name_mapping"))
    elif weights_path.exists():
        _load_weights_legacy(module, weights_path)
    elif const_path.exists():
        # constants-only artifact (new path without weight_source)
        raw_c: dict[str, torch.Tensor] = torch.load(
            str(const_path), map_location="cpu", weights_only=True
        )
        for key, tensor in raw_c.items():
            module.functions[0].weights[key] = tensor
            module.functions[0].const_weight_names.add(key)

    # ── Handle tied weights: lm_head_weight → model_embed_tokens_weight ──
    tied = ws.get("tied_weights", {})
    for tied_key, src_key in tied.items():
        for func in module.functions:
            if src_key in func.weights and tied_key not in func.weights:
                func.weights[tied_key] = func.weights[src_key]
                func.param_weight_names.add(tied_key)
    classification = module.metadata.get("weight_classification", {})
    for func in module.functions:
        info = classification.get(func.name, {})
        func.param_weight_names = set(info.get("params", []))
        func.const_weight_names = set(info.get("constants", []))

    return module


def _load_weights_via_mmap(
    module: MlirModule,
    ws_path: str,
    const_path: Path,
) -> None:
    import safetensors
    import safetensors.torch

    with safetensors.safe_open(ws_path, framework="pt", device="cpu") as f:  # type: ignore[no-untyped-call]
        for key in f.keys():
            wname = key.replace(".", "_")
            func_name = _guess_func(wname, module)
            for func in module.functions:
                if func.name == func_name:
                    func.weights[wname] = f.get_tensor(key)
                    func.param_weight_names.add(wname)
                    break
            else:
                if module.functions:
                    module.functions[0].weights[wname] = f.get_tensor(key)
                    module.functions[0].param_weight_names.add(wname)

    if const_path.exists():
        raw_c: dict[str, torch.Tensor] = torch.load(
            str(const_path), map_location="cpu", weights_only=True
        )
        for key, tensor in raw_c.items():
            func_name = _guess_func(key, module)
            for func in module.functions:
                if func.name == func_name:
                    func.weights[key] = tensor
                    func.const_weight_names.add(key)
                    break
            else:
                if module.functions:
                    module.functions[0].weights[key] = tensor
                    module.functions[0].const_weight_names.add(key)


def _load_weights_via_sharded(
    module: MlirModule,
    ws_path: str,
    const_path: Path,
) -> None:
    import json
    import os as _os

    ws_dir = _os.path.dirname(ws_path)
    with open(ws_path) as f:
        index = json.load(f)
    weight_map = index.get("weight_map", {})

    shard_files: set[str] = set()
    for shard_name in weight_map.values():
        shard_files.add(shard_name)

    for shard_file in sorted(shard_files):
        sf_path = _os.path.join(ws_dir, shard_file)
        import safetensors
        with safetensors.safe_open(sf_path, framework="pt", device="cpu") as f:  # type: ignore[no-untyped-call]
            for key in f.keys():
                wname = key.replace(".", "_")
                tensor = f.get_tensor(key)
                candidates = _candidate_names(wname)
                # Store under all candidate names for maximum compatibility
                stored = False
                for try_name in candidates:
                    func_name = _guess_func(try_name, module)
                    for func in module.functions:
                        if func.name == func_name:
                            func.weights[try_name] = tensor
                            func.param_weight_names.add(try_name)
                            stored = True
                    if not stored and module.functions:
                        module.functions[0].weights[try_name] = tensor
                        module.functions[0].param_weight_names.add(try_name)
                        stored = True
                if not stored and module.functions:
                    module.functions[0].weights[wname] = tensor
                    module.functions[0].param_weight_names.add(wname)

    if const_path.exists():
        raw_c = torch.load(str(const_path), map_location="cpu", weights_only=True)
        for key, tensor in raw_c.items():
            func_name = _guess_func(key, module)
            for func in module.functions:
                if func.name == func_name:
                    func.weights[key] = tensor
                    func.const_weight_names.add(key)
                    break
            else:
                if module.functions:
                    module.functions[0].weights[key] = tensor
                    module.functions[0].const_weight_names.add(key)


def _load_weights_via_bin(
    module: MlirModule,
    ws_path: str,
    const_path: Path,
    name_mapping: dict[str, str] | None = None,
) -> None:
    raw_bin: dict[str, torch.Tensor] = torch.load(
        ws_path, map_location="cpu", weights_only=True
    )
    name_map = name_mapping or {}
    for key, tensor in raw_bin.items():
        wname = key.replace(".", "_")
        if wname in name_map:
            wname = name_map[wname]
        func_name = _guess_func(wname, module)
        for func in module.functions:
            if func.name == func_name:
                func.weights[wname] = tensor
                func.param_weight_names.add(wname)
                break
        else:
            if module.functions:
                module.functions[0].weights[wname] = tensor
                module.functions[0].param_weight_names.add(wname)

    if const_path.exists():
        raw_c: dict[str, torch.Tensor] = torch.load(
            str(const_path), map_location="cpu", weights_only=True
        )
        for key, tensor in raw_c.items():
            func_name = _guess_func(key, module)
            for func in module.functions:
                if func.name == func_name:
                    func.weights[key] = tensor
                    func.const_weight_names.add(key)
                    break
            else:
                if module.functions:
                    module.functions[0].weights[key] = tensor
                    module.functions[0].const_weight_names.add(key)


def _load_weights_legacy(module: MlirModule, weights_path: Path) -> None:
    raw_weights: dict[str, torch.Tensor] = torch.load(
        str(weights_path), map_location="cpu", weights_only=True
    )
    for key, tensor in raw_weights.items():
        func_name = _guess_func(key, module)
        for func in module.functions:
            if func.name == func_name:
                func.weights[key] = tensor
                break
        else:
            if module.functions:
                module.functions[0].weights[key] = tensor


def _guess_func(wname: str, module: MlirModule) -> str:
    if "." in wname and not wname.startswith("_"):
        return wname.split(".", 1)[0]
    return "main"


def _candidate_names(wname: str) -> list[str]:
    """Generate possible cleaned names from a safetensors key.

    Handles model-specific naming mismatches between safetensors and
    PyTorch state_dict (e.g. Qwen: model.language_model. → model.).
    """
    names = [wname]
    parts = wname.split("_")
    for i in range(1, min(4, len(parts))):
        names.append("_".join(parts[i:]))
    return names


_DTYPE_TO_CODE: dict[Any, int] = {}


def _init_dtype_codes() -> dict[Any, int]:
    global _DTYPE_TO_CODE
    if _DTYPE_TO_CODE:
        return _DTYPE_TO_CODE
    _DTYPE_TO_CODE = {
        torch.float32: 0,
        torch.float16: 1,
        torch.bfloat16: 2,
        torch.int64: 3,
        torch.int32: 4,
        torch.int8: 5,
        torch.uint8: 6,
    }
    return _DTYPE_TO_CODE


def _build_name_mapping(module: MlirModule) -> dict[str, str]:
    """Build a compiled weight name → original HF safetensors key mapping.

    Uses ``hf_key_map`` from module metadata (stored at compile time by
    ``fx_graph_to_mlir``) to obtain the original HF key for each weight.
    """
    hf_key_map: dict[str, str] = module.metadata.get("hf_key_map", {})
    mapping: dict[str, str] = {}

    for func in module.functions:
        for op in func.ops:
            if op.op_name != "weight":
                continue
            short = op.attributes.get("name", "")
            if not short or short in mapping:
                continue
            for full, hf in hf_key_map.items():
                candidates = _candidate_names(full)
                if short in candidates or full.endswith(short):
                    mapping[short] = hf
                    break
    return mapping


def _build_constants_binary(module: MlirModule, name_mapping: dict[str, str]) -> bytes:
    """Build a self-contained binary blob with name mapping + constants + compute graph.

    Format (all integers little-endian):
        Magic:    4 bytes  "SFCF"
        Version:  u32      = 2
        ── name mapping ──
        Mapping entries: u32
        For each entry:
            compiled_name_len: u32
            compiled_name:     UTF-8
            hf_key_len:        u32
            hf_key:            UTF-8
        ── constants ──
        Constant tensors: u32
        For each tensor:
            name_len:   u32
            name:       UTF-8
            dtype:      u8  (0=f32,1=f16,2=bf16,3=i64,4=i32,5=i8,6=u8)
            ndim:       u8
            shape[ndim]: u64 repeated
            data_len:   u64
            data:       raw bytes
        ── compute graph ──
        num_functions: u32
        For each function:
            symbol: string
            num_inputs: u32
            num_outputs: u32
            For each input:
                binding_type: u8 (0=weight, 1=ssa, 2=global_input)
                if weight: key: string
                if ssa: producer_func: u32, output_idx: u32
                rank: u8
                num_dims: u32
                shape: [u64; num_dims] (0 = dynamic)
            For each output:
                rank: u8
                num_dims: u32
                shape: [u64; num_dims]
        global_input: func_idx: u32, arg_idx: u32
        global_output: func_idx: u32, output_idx: u32
    """
    _init_dtype_codes()
    parts: list[bytes] = []

    # Header
    parts.append(b"SFCF")
    parts.append(struct.pack("<I", 2))  # version

    # Name mapping
    parts.append(struct.pack("<I", len(name_mapping)))
    for short, full in sorted(name_mapping.items()):
        s = short.encode("utf-8")
        f = full.encode("utf-8")
        parts.append(struct.pack("<I", len(s)))
        parts.append(s)
        parts.append(struct.pack("<I", len(f)))
        parts.append(f)

    # Constants (const_weight_names tensors only)
    const_tensors: list[tuple[str, torch.Tensor]] = []
    for func in module.functions:
        for wname in func.const_weight_names:
            if wname in func.weights:
                t = func.weights[wname]
                # Store short name for constants too
                candidates = _candidate_names(wname)
                short_name = candidates[-1] if len(candidates) > 1 else candidates[0]
                const_tensors.append((short_name, t))

    parts.append(struct.pack("<I", len(const_tensors)))
    for name, tensor in const_tensors:
        n = name.encode("utf-8")
        parts.append(struct.pack("<I", len(n)))
        parts.append(n)
        dtype_code = _DTYPE_TO_CODE.get(tensor.dtype, 0)
        parts.append(struct.pack("<B", dtype_code))
        shape = tuple(tensor.shape)
        parts.append(struct.pack("<B", len(shape)))
        for dim in shape:
            parts.append(struct.pack("<Q", dim))
        data = tensor.detach().cpu().numpy().tobytes()
        parts.append(struct.pack("<Q", len(data)))
        parts.append(data)

    # ── Compute graph section ──
    _emit_compute_graph_section(parts, module, name_mapping)

    return b"".join(parts)


def _emit_compute_graph_section(
    parts: list[bytes], module: MlirModule, name_mapping: dict[str, str]
) -> None:
    """Emit the compute graph section: function list with I/O bindings."""

    # Build producer map: SSA name → (func_idx, output_idx)
    producer: dict[str, tuple[int, int]] = {}
    all_weight_names: set[str] = set(name_mapping.keys())
    for func in module.functions:
        all_weight_names |= set(func.weights.keys())

    for fi, func in enumerate(module.functions):
        for oi, (out_name, _out_type) in enumerate(func.outputs):
            clean = out_name.lstrip("%")
            producer[clean] = (fi, oi)

    # Determine global input: first input of first function
    global_input_func: int = 0
    global_input_arg: int = 0
    global_output_func: int = len(module.functions) - 1
    global_output_idx: int = 0

    num_funcs = len(module.functions)
    parts.append(struct.pack("<I", num_funcs))

    for fi, func in enumerate(module.functions):
        _emit_string(parts, f"_mlir_ciface_{func.name}")

        # Collect weight/constant ops in this function — these become additional
        # function arguments after C++ lowering promotes them from ops to args.
        weight_ops = [op for op in func.ops if op.op_name in ("weight", "constant")]
        weight_ops_with_names = [op for op in weight_ops if op.attributes.get("name", "")]

        num_inputs = len(func.inputs) + len(weight_ops_with_names)
        num_outputs = len(func.outputs)
        parts.append(struct.pack("<I", num_inputs))
        parts.append(struct.pack("<I", num_outputs))

        # 1. Emit bindings for real function parameters (from func.inputs)
        for in_idx, (in_name, in_type_str) in enumerate(func.inputs):
            clean = in_name.lstrip("%")
            rank, shape_dims = _parse_type_shape(in_type_str)

            if fi == 0 and in_idx == 0:
                # First input of first function → global input (token ids)
                parts.append(struct.pack("<B", 2))  # global_input
            elif clean in producer:
                parts.append(struct.pack("<B", 1))  # ssa
                pfi, poi = producer[clean]
                parts.append(struct.pack("<I", pfi))
                parts.append(struct.pack("<I", poi))
            else:
                # Fallback: treat as ssa from previous function
                parts.append(struct.pack("<B", 1))  # ssa
                parts.append(struct.pack("<I", 0))
                parts.append(struct.pack("<I", 0))

            parts.append(struct.pack("<B", rank))
            parts.append(struct.pack("<I", len(shape_dims)))
            for d in shape_dims:
                parts.append(struct.pack("<Q", d))

        # 2. Emit bindings for weight/constant ops (promoted to func args in C++ pass)
        #    These appear AFTER the real function parameters in the arg list.
        for wop in weight_ops_with_names:
            wname = wop.attributes.get("name", "")
            wtype_str = wop.output_types[0] if wop.output_types else "tensor<f32>"
            rank, shape_dims = _parse_type_shape(wtype_str)
            parts.append(struct.pack("<B", 0))  # weight
            _emit_string(parts, wname)
            parts.append(struct.pack("<B", rank))
            parts.append(struct.pack("<I", len(shape_dims)))
            for d in shape_dims:
                parts.append(struct.pack("<Q", d))

        for _out_idx, (_out_name, out_type_str) in enumerate(func.outputs):
            rank, shape_dims = _parse_type_shape(out_type_str)
            parts.append(struct.pack("<B", rank))
            parts.append(struct.pack("<I", len(shape_dims)))
            for d in shape_dims:
                parts.append(struct.pack("<Q", d))

    parts.append(struct.pack("<I", global_input_func))
    parts.append(struct.pack("<I", global_input_arg))
    parts.append(struct.pack("<I", global_output_func))
    parts.append(struct.pack("<I", global_output_idx))


def _emit_string(parts: list[bytes], s: str) -> str:
    """Emit a u32-prefixed UTF-8 string. Returns the string for chaining."""
    encoded = s.encode("utf-8")
    parts.append(struct.pack("<I", len(encoded)))
    parts.append(encoded)
    return s


def _parse_type_shape(type_str: str) -> tuple[int, list[int]]:
    """Return (rank, shape_dims) from MLIR type string. 0 = dynamic dim."""
    import re as _re
    m = _re.match(r"(?:tensor|memref)<\s*(.+?)\s*>$", type_str.strip())
    if not m:
        return (0, [])
    inner = m.group(1).strip()
    parts = inner.split("x")
    dims: list[int] = []
    for p in parts:
        p = p.strip()
        if p == "?" or p.startswith("?"):
            dims.append(0)
            continue
        try:
            dims.append(int(p))
        except ValueError:
            break
    rank = len(dims)
    return (rank, [0 if d < 0 else d for d in dims])


# ── Internal parsing ──────────────────────────────────────────

_ELT_MAP: dict[str, ir.Type] = {}
_DYNAMIC_DIM: int | None = None


def _get_dynamic_dim() -> int:
    global _DYNAMIC_DIM
    if _DYNAMIC_DIM is None:
        import mlir.ir as _ir
        _DYNAMIC_DIM = _ir.ShapedType.get_dynamic_size()
    return _DYNAMIC_DIM


def _init_elt_map() -> dict[str, ir.Type]:
    global _ELT_MAP
    if _ELT_MAP:
        return _ELT_MAP
    import mlir.ir as _ir
    _ELT_MAP = {
        "f32": _ir.F32Type.get(), "f64": _ir.F64Type.get(),
        "f16": _ir.F16Type.get(), "bf16": _ir.BF16Type.get(),
        "i1": _ir.IntegerType.get_signless(1),
        "i8": _ir.IntegerType.get_signless(8),
        "i32": _ir.IntegerType.get_signless(32),
        "i64": _ir.IntegerType.get_signless(64),
        "ui8": _ir.IntegerType.get_unsigned(8),
    }
    return _ELT_MAP


def _type_str_to_ir_type(type_str: str) -> ir.Type:
    """Convert MLIR type string to ir.Type object.

    tensor<1x64xf32>  → RankedTensorType([1, 64], f32)
    tensor<f32>        → RankedTensorType([], f32)  (scalar tensor)
    tensor<?x64xf32>   → RankedTensorType([dynamic], f32)
    """
    import re as _re

    import mlir.ir as _ir

    elt_map = _init_elt_map()

    m = _re.match(r"tensor<\s*(.+?)\s*>$", type_str.strip())
    if not m:
        elt = elt_map.get(type_str.strip())
        if elt is not None:
            return _ir.UnrankedTensorType.get(elt)
        return _ir.UnrankedTensorType.get(_ir.F32Type.get())

    inner = m.group(1).strip()
    parts = inner.split("x")
    dtype_str = parts[-1].strip()
    elt_type: ir.Type = elt_map.get(dtype_str, _ir.F32Type.get())
    dim_strs = parts[:-1]

    for d in dim_strs:
        d = d.strip()
        if d == "*":
            return _ir.UnrankedTensorType.get(elt_type)

    dyn_dim = _get_dynamic_dim()
    dims: list[int] = []
    for d in dim_strs:
        d = d.strip()
        try:
            dims.append(int(d))
        except ValueError:
            dims.append(dyn_dim)

    if not dims:
        return _ir.RankedTensorType.get([], elt_type)
    try:
        return _ir.RankedTensorType.get(dims, elt_type)
    except Exception as e:
        raise RuntimeError(
            f"Failed to create RankedTensorType for '{type_str}': dims={dims}, elt={elt_type}"
        ) from e


def mlir_module_to_ir_module(module: MlirModule, ctx: Any = None) -> Any:
    """Build an ir.Module from an MlirModule using MLIR Python API.

    This bypasses the MLIR text round-trip entirely, creating a valid
    ir.Module that can be passed directly to PassManager-based passes.

    Weight ops are emitted with unranked tensor types to avoid type
    conflicts between the actual tensor shape and inferred consumer types.
    """
    import mlir.ir as _ir

    if ctx is None:
        ctx = _ir.Context()
        ctx.allow_unregistered_dialects = True

    with ctx, _ir.Location.unknown(ctx):
        ir_mod = _ir.Module.create()

        for func in module.functions:
            # ── Build function ──────────────────────────────────
            arg_types: list[ir.Type] = []
            for _, tp in func.inputs:
                arg_types.append(_type_str_to_ir_type(tp))

            func_type = _ir.FunctionType.get(arg_types, [])
            func_op = _ir.Operation.create(
                "func.func",
                attributes={
                    "function_type": _ir.TypeAttr.get(func_type),
                    "sym_name": _ir.StringAttr.get(func.name),
                },
                regions=1,
            )
            ir_mod.body.append(func_op.operation)

            body_region = func_op.operation.regions[0]
            body_blk = _ir.Block.create_at_start(body_region, arg_types)
            arg_values: list[ir.Value] = list(body_blk.arguments)

            # ── SSA name → ir.Value mapping ─────────────────────
            ssa_map: dict[str, ir.Value] = {}
            for i, (name, _) in enumerate(func.inputs):
                ssa_map[name] = arg_values[i]
                ssa_map[name.lstrip("%")] = arg_values[i]

            # ── Build ops ───────────────────────────────────────
            output_values: list[ir.Value] = []

            for op in func.ops:
                if op.op_name in ("weight", "constant"):
                    w_attrs: dict[str, ir.Attribute] = {}
                    for k, v in op.attributes.items():
                        if k == "source_node":
                            continue
                        w_attrs[k] = _python_to_attr_ir(v)

                    if op.output_types:
                        w_result_types = [_type_str_to_ir_type(t) for t in op.output_types]
                    else:
                        w_result_types = [_ir.UnrankedTensorType.get(_ir.F32Type.get(ctx))]

                    with _ir.InsertionPoint(body_blk):
                        ir_op = _ir.Operation.create(
                            op.name,
                            results=w_result_types,
                            attributes=w_attrs if w_attrs else {},
                        )
                    for i, rname in enumerate(op.results):
                        if i < len(ir_op.operation.results):
                            val = ir_op.operation.results[i]
                            ssa_map[rname] = val
                            ssa_map[rname.lstrip("%")] = val
                    # Also map the weight name attribute so operand resolution works
                    wname = op.attributes.get("name")
                    if wname:
                        ssa_map[wname] = ir_op.operation.results[0]
                    continue

                # Regular compute ops: resolve operands from ssa_map
                operands: list[ir.Value] = []
                for o in op.operands:
                    key = o
                    if key in ssa_map:
                        operands.append(ssa_map[key])
                    elif key.lstrip("%") in ssa_map:
                        operands.append(ssa_map[key.lstrip("%")])
                    else:
                        raise KeyError(
                            f"ssa_map missing operand '{key}' for op '{op.name}'. "
                            f"Known: {list(ssa_map.keys())[:10]}"
                        )

                try:
                    # Build attributes as MLIR attributes
                    mlir_attrs: dict[str, ir.Attribute] = {}
                    for k, v in op.attributes.items():
                        if k == "source_node":
                            continue
                        mlir_attrs[k] = _python_to_attr_ir(v)

                    # Determine result types
                    result_types: list[ir.Type] = []
                    if op.output_types:
                        result_types = [_type_str_to_ir_type(t) for t in op.output_types]
                    elif op.op_name == "sym_size":
                        result_types = [_ir.RankedTensorType.get([], _ir.F32Type.get(ctx))]
                    elif operands:
                        opnd_type = operands[0].type
                        try:
                            _rank = len(opnd_type.shape)
                            result_types = [opnd_type]
                        except Exception:
                            try:
                                elt = opnd_type.element_type
                            except Exception:
                                elt = _ir.F32Type.get(ctx)
                            result_types = [_ir.RankedTensorType.get([1], elt)]
                    else:
                        result_types = [_ir.RankedTensorType.get([1], _ir.F32Type.get(ctx))]

                    with _ir.InsertionPoint(body_blk):
                        ir_op = _ir.Operation.create(
                            op.name,
                            operands=operands,
                            results=result_types,
                            attributes=mlir_attrs if mlir_attrs else {},
                        )
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to build op '{op.name}' (result '{op.results[0] if op.results else '?'}'): "
                        f"output_types={op.output_types}, operands_count={len(op.operands)}, "
                        f"attr_keys={list(op.attributes.keys())[:5]}, error={e}"
                    ) from e

                with _ir.InsertionPoint(body_blk):
                    ir_op = _ir.Operation.create(
                        op.name,
                        operands=operands,
                        results=result_types,
                        attributes=mlir_attrs if mlir_attrs else {},
                    )

                # Map results
                for i, rname in enumerate(op.results):
                    if i < len(ir_op.operation.results):
                        val = ir_op.operation.results[i]
                        ssa_map[rname] = val
                        ssa_map[rname.lstrip("%")] = val

            # Collect output values
            for out_name, _ in func.outputs:
                if out_name in ssa_map:
                    output_values.append(ssa_map[out_name])
                elif out_name.lstrip("%") in ssa_map:
                    output_values.append(ssa_map[out_name.lstrip("%")])

            # ── Return op ───────────────────────────────────────
            if not output_values and func.ops:
                last_op = func.ops[-1]
                if last_op.results:
                    rname = last_op.results[-1]
                    if rname in ssa_map:
                        output_values.append(ssa_map[rname])

            with _ir.InsertionPoint(body_blk):
                _ir.Operation.create("func.return", operands=output_values)

            # Update function signature with actual return types
            ret_types = [v.type for v in output_values]
            arg_types_list = [a.type for a in arg_values]
            new_func_type = _ir.FunctionType.get(arg_types_list, ret_types)
            func_op.operation.attributes["function_type"] = _ir.TypeAttr.get(new_func_type)

        return ir_mod


def _python_to_attr_ir(value: Any) -> ir.Attribute:
    """Convert Python value to ir.Attribute for ir.Operation.create()."""
    import mlir.ir as _ir

    if isinstance(value, bool):
        return _ir.BoolAttr.get(value)
    if isinstance(value, int):
        return _ir.IntegerAttr.get(_ir.IntegerType.get_signless(64), value)
    if isinstance(value, float):
        return _ir.FloatAttr.get(_ir.F64Type.get(), value)
    if isinstance(value, str):
        return _ir.StringAttr.get(value)
    if isinstance(value, (list, tuple)):
        items = [_python_to_attr_ir(v) for v in value]
        return _ir.ArrayAttr.get(items)
    if value is None:
        return _ir.UnitAttr.get()
    return _ir.StringAttr.get(str(value))


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
            # Parse return type: -> type after )
            ret_start = rest.find("->")
            ret_types: list[str] = []
            if ret_start > 0:
                ret_part = rest[ret_start + 2:].strip()
                if ret_part.endswith("{"):
                    ret_part = ret_part[:-1].strip()
                if ret_part and ret_part != "()":
                    # Could be single type or (type1, type2, ...)
                    if ret_part.startswith("("):
                        ret_part = ret_part[1:-1]  # strip parens
                    for tp in _split_comma(ret_part):
                        ret_types.append(tp.strip())
            outputs: list[tuple[str, str]] = [("", tp) for tp in ret_types]
            current_func = MlirFunction(name=func_name, inputs=inputs, outputs=outputs)
            ssa_to_name = {}
            ssa_types = {}
            continue

        # Parse func.return: extract output names and match with output types
        if stripped.startswith("func.return"):
            if current_func is not None:
                ret_content = stripped[len("func.return"):].strip()
                if ":" in ret_content:
                    ret_content = ret_content[:ret_content.rfind(":")].strip()
                ret_names = [n.strip() for n in _split_comma(ret_content) if n.strip()]
                if current_func.outputs and ret_names:
                    new_outputs = []
                    for i, name in enumerate(ret_names):
                        tp = current_func.outputs[i][1] if i < len(current_func.outputs) else "tensor<f32>"
                        new_outputs.append((name, tp))
                    if new_outputs:
                        current_func.outputs = new_outputs
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
    colon_idx_for_attrs = rest.find(":", brace_open) if brace_open >= 0 else -1
    if brace_open >= 0 and (colon_idx_for_attrs < 0 or brace_open < colon_idx_for_attrs):
        brace_close = rest.index("}", brace_open)
        attrs_str = rest[brace_open + 1:brace_close]
        attrs = _parse_attrs(attrs_str)

    # Record weight SSA mapping for sf.weight ops
    if qualified == "sf.weight" and "name" in attrs:
        if results:
            ssa_to_name[results[0]] = attrs["name"]

    # Record SSA types and collect input/output type strings
    in_type_strs: list[str] = []
    out_type_strs: list[str] = []
    colon_idx = rest.rfind(":")
    if colon_idx >= 0:
        type_part = rest[colon_idx + 1:].strip()
        arrow_idx = type_part.find("->")
        if arrow_idx >= 0:
            in_types_part = type_part[:arrow_idx].strip()
            if in_types_part.startswith("(") and in_types_part.endswith(")"):
                in_type_strs = _parse_type_list(in_types_part[1:-1])
            elif in_types_part:
                in_type_strs = [in_types_part]
            # Filter out '...' placeholder
            in_type_strs = [t for t in in_type_strs if t != "..."]
            out_types_str = type_part[arrow_idx + 2:].strip()
            out_type_strs = _parse_type_list(out_types_str)
            # Filter out '...' placeholder
            out_type_strs = [t for t in out_type_strs if t != "..."]
        else:
            out_type_strs = [type_part]

        for i, r in enumerate(results):
            if i < len(out_type_strs):
                ssa_types[r] = out_type_strs[i]

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
            input_types=in_type_strs,
            output_types=out_type_strs,
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
        # Strip quotes from attribute keys (emitters quote them)
        if k.startswith('"') and k.endswith('"'):
            k = k[1:-1]
        v = v.strip()
        # Parse value
        if v == "true":
            result[k] = True
        elif v == "false":
            result[k] = False
        elif v == "none":
            result[k] = None
        elif v.startswith('"') and v.endswith('"'):
            raw = v[1:-1]
            if " : " in raw:
                raw = raw.split(" : ")[0]
                try:
                    result[k] = int(raw)
                except ValueError:
                    try:
                        result[k] = float(raw)
                    except ValueError:
                        result[k] = raw
            else:
                result[k] = raw
        elif v.startswith("[") and v.endswith("]"):
            inner = v[1:-1]
            items = _split_comma(inner)
            result[k] = [_parse_attr_value(item.strip()) for item in items if item.strip()]
        else:
            raw = v
            if " : " in v:
                raw = v.split(" : ")[0]
            try:
                result[k] = int(raw)
            except ValueError:
                try:
                    result[k] = float(raw)
                except ValueError:
                    result[k] = raw
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
    """Parse a single attribute value, stripping MLIR type annotations."""
    v = v.strip()
    if v == "true":
        return True
    if v == "false":
        return False
    if v == "none":
        return None
    if v.startswith('"') and v.endswith('"'):
        raw = v[1:-1]
        if " : " in raw:
            raw = raw.split(" : ")[0]
            try:
                return int(raw)
            except ValueError:
                try:
                    return float(raw)
                except ValueError:
                    return raw
        return raw
    if " : " in v:
        v = v.split(" : ")[0]
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
