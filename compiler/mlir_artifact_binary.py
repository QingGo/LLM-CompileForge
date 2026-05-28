"""MLIR artifact binary generation — embedded constants binary format.

Section C: Binary generation functions from the original mlir_artifact.py.

Builds the ``constants.bin`` format used by the Rust runtime (.dylib) for
name mapping, constant tensors, and compute graph description.
"""

from __future__ import annotations

import logging
import struct
import subprocess
from typing import Any

import torch

from compiler.mlir_artifact_load import _candidate_names
from compiler.mlir_dialect.mlir_op_types import (
    MlirModule,
)

_log = logging.getLogger(__name__)

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
    Resolves tied weights: if two weights share the same data tensor,
    both map to the SINGLE survivor HF key in the safetensors file.
    """
    hf_key_map: dict[str, str] = module.metadata.get("hf_key_map", {})
    tied_weights: dict[str, str] = module.metadata.get("weight_source", {}).get("tied_weights", {})
    # tied_weights maps alias → survivor, e.g. "lm_head_weight" → "model_decoder_embed_tokens_weight"
    # meaning lm_head_weight is an alias for model_decoder_embed_tokens_weight (same tensor data).
    #
    # NOTE: The survivor's hf_key exists in safetensors. Both should use that key.
    # In OPT-125m: lm_head.weight = model.decoder.embed_tokens.weight (tied),
    # safetensors stores under "lm_head.weight" only.

    # Build reverse map: survivor → alias (so we redirect aliases to the survivor's HF key)
    alias_to_survivor: dict[str, str] = {}
    for alias_name, surv_name in tied_weights.items():
        alias_to_survivor[surv_name] = alias_name

    mapping: dict[str, str] = {}

    for func in module.functions:
        for op in func.ops:
            if op.op_name != "weight":
                continue
            short = op.attributes.get("name", "")
            if not short or short in mapping:
                continue

            # If this weight is an alias for a tied weight, use the survivor's HF key
            if short in alias_to_survivor:
                survivor = alias_to_survivor[short]
                if survivor in hf_key_map:
                    mapping[short] = hf_key_map[survivor]
                    continue

            # Normal lookup
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
        Version:  u32      = 3
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
            For each output (v3+):
                consumed_internally: u8 (0=external, 1=internal)
                rank: u8
                num_dims: u32
                shape: [u64; num_dims]
            For each output (v2):
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
    parts.append(struct.pack("<I", 4))  # version (v4 adds contract section with metadata)

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
                # Use full wname (e.g. _const_7) — runtime looks up by this name
                const_tensors.append((wname, t))

    parts.append(struct.pack("<I", len(const_tensors)))
    for name, tensor in const_tensors:
        encoded = name.encode("utf-8")
        parts.append(struct.pack("<I", len(encoded)))
        parts.append(encoded)
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

    # ── Contract section (v4+) — appended after compute graph trailer ──
    # Count global inputs using same logic as _emit_compute_graph_section
    global_input_names: list[str] = []
    for in_idx, (in_name, _in_type_str) in enumerate(module.functions[0].inputs):
        if in_idx in (0, 1):
            global_input_names.append(in_name.lstrip("%"))

    try:
        compiler_version = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        compiler_version = "unknown"

    contract_entries = [
        ("sfcf_version", "4"),
        ("num_global_inputs", str(len(global_input_names))),
        ("global_input_names", ",".join(global_input_names)),
        ("compiler_version", compiler_version),
    ]

    parts.append(struct.pack("<I", len(contract_entries)))
    for key, val in contract_entries:
        key_bytes = key.encode("utf-8")
        val_bytes = val.encode("utf-8")
        parts.append(struct.pack("<I", len(key_bytes)))
        parts.append(key_bytes)
        parts.append(struct.pack("<I", len(val_bytes)))
        parts.append(val_bytes)

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
        for oi, (out_name, _out_type, _consumed_internally) in enumerate(func.outputs):
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
        # emit_c_interface is added during LLVM lowering (after constants.bin is built).
        # The Rust runtime uses _mlir_ciface_* wrappers, so prefix the symbol.
        _emit_string(parts, f"_mlir_ciface_{func.name}")

        # Collect weight ops in this function — these become additional
        # function arguments after C++ promotion. Scalar constant weight ops
        # (with _const_ prefix) are inlined as arith.constant during C++
        # conversion and are NOT function parameters.
        weight_ops = [op for op in func.ops
                      if op.op_name == "weight"
                      and not op.attributes.get("name", "").startswith("_const_")]
        weight_ops_with_names = [op for op in weight_ops if op.attributes.get("name", "")]

        num_inputs = len(func.inputs) + len(weight_ops_with_names)
        num_outputs = len(func.outputs)
        parts.append(struct.pack("<I", num_inputs))
        parts.append(struct.pack("<I", num_outputs))

        # 1. Emit bindings for real function parameters (from func.inputs)
        for in_idx, (in_name, in_type_str) in enumerate(func.inputs):
            clean = in_name.lstrip("%")
            rank, shape_dims = _parse_type_shape(in_type_str)

            if fi == 0 and in_idx in (0, 1):
                # First two inputs of first function → global inputs
                # in_idx=0: input_ids, in_idx=1: position_ids (when present)
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

        for _out_idx, (_out_name, out_type_str, consumed_internally) in enumerate(func.outputs):
            rank, shape_dims = _parse_type_shape(out_type_str)
            parts.append(struct.pack("<B", 1 if consumed_internally else 0))
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
