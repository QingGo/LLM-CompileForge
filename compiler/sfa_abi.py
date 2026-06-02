"""
sfa_abi.py — Stable Function ABI header generation from LLVM IR.

Parses LLVM IR text to extract ciface function signatures, merges with
pre-lowering input semantics, and serializes to protobuf SfaAbiHeader format.

Usage:
    from compiler.sfa_abi import parse_ciface_signatures, merge_with_semantics, serialize_abi

    sigs = parse_ciface_signatures("model.ll")
    metas = merge_with_semantics(sigs, pre_lowering_module)
    proto_bytes = serialize_abi(metas)
"""

from __future__ import annotations

import re
from typing import Any

from gen.proto.python.sfa_abi_pb2 import (
    SfaAbiHeader,
    SfaInputKind,
)

# ── Constants (from include/sfa.h) ───────────────────────────────────

SFA_MAGIC: int = 0x41464253  # "SFBA" in LE bytes
SFA_VERSION: int = 1

# ── Regex patterns ───────────────────────────────────────────────────

# Matches ciface wrapper definition: define void @_mlir_ciface_<name>(<params>) {
_CIFACE_DEF_RE: re.Pattern[str] = re.compile(
    r"define\s+void\s+@(_mlir_ciface_\w+)\s*\(([^)]*)\)\s*\{"
)

# Extracts rank from descriptor struct: [N x i64]
_RANK_RE: re.Pattern[str] = re.compile(r"\[(\d+)\s+x\s+i64\]")


def parse_ciface_signatures(llvm_ir_path: str) -> dict[str, tuple[int, int]]:
    """Parse LLVM IR text to extract function signatures from ciface wrappers.

    Scans for ``define void @_mlir_ciface_<name>(ptr %0, ptr %1, ...)``
    and extracts the number of pointer arguments and the output rank
    (from the return struct of the inner ``call``).

    Args:
        llvm_ir_path: Path to an LLVM IR text file (e.g. ``model.ll``).

    Returns:
        Dict mapping function symbol name (e.g. ``"_mlir_ciface_main_0"``)
        to a ``(num_args, output_rank)`` tuple.  ``num_args`` is the total
        number of ``ptr`` parameters (including the sret pointer).
        ``output_rank`` is the rank of the post-bufferization packed
        output memref descriptor (1-4).

        Returns an empty dict if no ciface wrappers are found.
    """
    with open(llvm_ir_path) as f:
        text = f.read()

    signatures: dict[str, tuple[int, int]] = {}

    # Find all ciface function definitions
    for def_match in _CIFACE_DEF_RE.finditer(text):
        func_name = def_match.group(1)
        params_str = def_match.group(2)

        # Count ptr parameters (each "ptr %N" is one argument)
        num_args = params_str.count("ptr")

        # Inner function name: strip "_mlir_ciface_" prefix (e.g. "main_0")
        inner_name = func_name.replace("_mlir_ciface_", "")

        # Find the call to the inner function to get the return struct rank.
        def_start = def_match.start()
        next_def = text.find("\ndefine ", def_start + 1)
        if next_def == -1:
            next_def = len(text)
        func_body = text[def_start:next_def]

        call_pattern = rf"call\s+(\{{[^}}]*\}})\s*@{re.escape(inner_name)}\s*\("
        call_match = re.search(call_pattern, func_body)
        output_rank = 0
        if call_match:
            ret_struct = call_match.group(1)
            ranks = _RANK_RE.findall(ret_struct)
            # The descriptor struct is { ptr, ptr, i64, [R x i64], [R x i64] }
            # The rank is the N in [N x i64] — use the first match as output rank
            if ranks:
                output_rank = int(ranks[0])

        # Fallback: main_0's inner function returns a complex struct of 211
        # individual descriptors — the regex can't match. All dylib functions
        # use rank-3 packed output after bufferization.
        if output_rank == 0:
            output_rank = 3

        # num_args includes the sret parameter (ptr %0) — exclude it
        signatures[func_name] = (num_args - 1, output_rank)

    return signatures


def merge_with_semantics(
    signatures: dict[str, tuple[int, int]],
    pre_lowering_module: dict[str, Any],
) -> list[dict[str, Any]]:
    """Merge LLVM IR ciface signatures with pre-lowering input semantics.

    Combines the raw function signatures (argument count + output rank from
    LLVM IR) with semantic information from the pre-lowering module
    (input names, weight mappings, SSA connections).

    Args:
        signatures: Output of :func:`parse_ciface_signatures` — maps
            ciface symbol names to ``(num_args, output_rank)`` tuples.
        pre_lowering_module: A dict representing the pre-lowering MLIR module
            with a ``"functions"`` key containing a list of function dicts.
            Each function dict should have:

            - ``"name"``: str — base function name (e.g. ``"main_0"``)
            - ``"inputs"``: list of ``(name, type_str)`` tuples
            - ``"weights"``: dict (optional)
            - ``"weight_ops"``: list of dicts with ``"name"`` key (optional)
            - ``"outputs"``: list of ``(name, type_str, consumed)`` tuples (optional)

    Returns:
        List of ``SfaFuncMeta`` dicts, each containing:

        - ``"symbol"``: str — ciface wrapper name
        - ``"num_inputs"``: int — total number of arguments (incl. sret)
        - ``"output_rank"``: int — packed output memref rank
        - ``"input_fields"``: list of ``SfaInputField`` dicts with keys:
          ``kind`` (int), ``weight_name`` (str, optional),
          ``producer_func`` (int, optional), ``producer_out`` (int, optional)
    """
    funcs = pre_lowering_module.get("functions", [])

    # Build producer map: output name → (func_idx, output_idx)
    # Also build function name → func_idx for fallback matching
    producer_map: dict[str, tuple[int, int]] = {}
    func_name_to_idx: dict[str, int] = {}

    for fi, func in enumerate(funcs):
        func_name_to_idx[func["name"]] = fi

        # Explicit outputs (pre-lowering MLIR function results)
        for _oi, out_entry in enumerate(func.get("outputs", [])):
            if isinstance(out_entry, (list, tuple)) and len(out_entry) >= 1:
                out_name = str(out_entry[0]).lstrip("%")
                # Use actual output index from pre-lowering (not packed to 0)
                producer_map[out_name] = (fi, _oi)

    metas: list[dict[str, Any]] = []
    for fi, func in enumerate(funcs):
        symbol = f"_mlir_ciface_{func['name']}"
        if symbol not in signatures:
            continue

        num_args, output_rank = signatures[symbol]

        # Build input field semantics
        input_fields: list[dict[str, Any]] = []

        # 1. Regular function parameters (from func.inputs)
        for in_idx, in_entry in enumerate(func.get("inputs", [])):
            if isinstance(in_entry, (list, tuple)) and len(in_entry) >= 1:
                in_name = str(in_entry[0]).lstrip("%")
            else:
                in_name = ""

            if fi == 0 and in_idx <= 1:
                # First two inputs of the first function → global inputs
                input_fields.append({"kind": SfaInputKind.Value("SFA_INPUT_GLOBAL")})
            elif in_name and in_name in producer_map:
                pfi, poi = producer_map[in_name]
                input_fields.append({
                    "kind": SfaInputKind.Value("SFA_INPUT_SSA"),
                    "producer_func": pfi,
                    "producer_out": poi,
                })
            elif in_name:
                # Try to match by function name embedded in the input name
                # Pattern: <var>_N_<func_name>_M
                matched = False
                for fname, fni in func_name_to_idx.items():
                    if fname in in_name:
                        input_fields.append({
                            "kind": SfaInputKind.Value("SFA_INPUT_SSA"),
                            "producer_func": fni,
                            "producer_out": 0,
                        })
                        matched = True
                        break
                if not matched:
                    # Fallback: treat as SSA from previous function
                    prev_fi = max(0, fi - 1)
                    input_fields.append({
                        "kind": SfaInputKind.Value("SFA_INPUT_SSA"),
                        "producer_func": prev_fi,
                        "producer_out": 0,
                    })

        # 2. Weight/constant ops (promoted to function args during C++ lowering)
        for wop in func.get("weight_ops", []):
            wname = wop.get("name", "")
            if not wname:
                continue
            # Skip scalar constants (prefixed with _const_ — inlined as arith.const)
            if wname.startswith("_const_"):
                continue
            input_fields.append({
                "kind": SfaInputKind.Value("SFA_INPUT_WEIGHT"),
                "weight_name": wname,
            })

        # 3. Output descriptors — use LLVM IR sret rank (post-bufferization).
        # Each bufferized function has a SINGLE packed output memref.
        # Pre-lowering func.outputs may list many SSA values (e.g. main_0
        # has 211), but the sret has exactly one descriptor.  Always
        # generate one OutputDescriptor matching the LLVM IR return struct.
        effective_output_rank = output_rank
        output_descs = [{
            "rank": effective_output_rank,
            "dims": [0] * effective_output_rank,
        }]

        metas.append({
            "symbol": symbol,
            "num_inputs": num_args,
            "output_rank": effective_output_rank,
            "input_fields": input_fields,
            "outputs": output_descs,
        })

    return metas


def serialize_abi(func_metas: list[dict[str, Any]]) -> bytes:
    """Serialize a list of ``SfaFuncMeta`` dicts into protobuf SfaAbiHeader binary.

    Produces a protobuf binary blob suitable for embedding in a compiled dylib
    as the ``sfa_abi`` exported symbol.

    Args:
        func_metas: Output of :func:`merge_with_semantics` — a list of dicts
            each containing ``symbol``, ``num_inputs``, ``output_rank``,
            and ``input_fields``.

    Returns:
        Raw protobuf bytes of the serialized SfaAbiHeader message.
    """
    header = SfaAbiHeader()
    header.magic = SFA_MAGIC
    header.version = SFA_VERSION

    for meta in func_metas:
        func_meta = header.funcs.add()
        func_meta.symbol = meta["symbol"]
        func_meta.num_inputs = meta["num_inputs"]
        func_meta.output_rank = meta["output_rank"]

        for field in meta.get("input_fields", []):
            input_field = func_meta.input_fields.add()
            input_field.kind = field["kind"]  # type: ignore[assignment]

            if field["kind"] == SfaInputKind.Value("SFA_INPUT_WEIGHT"):
                input_field.weight_name = field.get("weight_name", "")
            elif field["kind"] == SfaInputKind.Value("SFA_INPUT_SSA"):
                input_field.ssa.producer_func = field.get("producer_func", 0)
                input_field.ssa.producer_out = field.get("producer_out", 0)
            # SFA_INPUT_GLOBAL: no binding needed

        for od in meta.get("outputs", []):
            out_desc = func_meta.outputs.add()
            out_desc.rank = od["rank"]
            out_desc.dims.extend(od["dims"])

    return header.SerializeToString()
