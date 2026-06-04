"""MLIR artifact loading — load compiled models from disk.

Section B: Weight loading functions from the original mlir_artifact.py.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import torch

from compiler.mlir_artifact._utils import _candidate_names
from compiler.mlir_artifact.parse import _parse_mlir_text
from compiler.mlir_dialect.sf.mlir_op_types import (
    MlirModule,
)

_log = logging.getLogger(__name__)


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
    # NOTE: Backward-compat fallback for artifacts that have weights.pth
    # without weight_source metadata. Kept for safety — newer artifacts use
    # _load_weights_via_mmap / _load_weights_via_sharded / _load_weights_via_bin.
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



