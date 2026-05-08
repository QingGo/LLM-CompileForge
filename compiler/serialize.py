"""Model artifact serialization / deserialization.

Saves and loads compiled IrModules with their weight tensors to/from disk.
Output structure (per design doc):
  compiled/<model_name>/
    model.ir        — IrModule structure (JSON)
    weights.pth     — PyTorch state dict of all weight tensors
    metadata.json   — compilation metadata
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from compiler.ir import IrModule, pack_weights


def save_artifact(module: IrModule, directory: str) -> None:
    """Persist a compiled IrModule to disk.

    Output structure (per design doc §4.2):
      compiled/<model_name>/
        model.ir        — IrModule structure (JSON)
        model.mlir      — MLIR canonical representation (text)
        weights.pth     — PyTorch state dict of all weight tensors
        metadata.json   — compilation metadata

    Args:
        module: The compiled IrModule to save.
        directory: Target directory (created if it doesn't exist).
    """
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stamp schema version for forward compatibility
    module.metadata.setdefault("ir_schema_version", 1)

    # model.ir
    ir_dict = module.to_dict()
    with open(out_dir / "model.ir", "w") as f:
        json.dump(ir_dict, f, indent=2, default=str)

    # model.mlir (canonical MLIR representation)
    mlir_text = module.metadata.get("mlir", "")
    if mlir_text:
        with open(out_dir / "model.mlir", "w") as f:
            f.write(mlir_text)

    # weights.pth
    all_weights: dict[str, dict[str, torch.Tensor]] = pack_weights(module)
    weight_state: dict[str, torch.Tensor] = {}
    for func_name, func_weights in all_weights.items():
        for wname, tensor in func_weights.items():
            key = f"{func_name}.{wname}" if func_name != "main" else wname
            weight_state[key] = tensor
    torch.save(weight_state, out_dir / "weights.pth")

    # metadata.json
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(module.metadata, f, indent=2)


def load_artifact(directory: str) -> IrModule:
    """Load a compiled IrModule from disk.

    Args:
        directory: Path to the compiled artifact directory.

    Returns:
        The reconstructed IrModule with weights.

    Raises:
        FileNotFoundError: If the directory or required files are missing.
    """
    in_dir = Path(directory)
    if not in_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    ir_path = in_dir / "model.ir"
    weights_path = in_dir / "weights.pth"

    if not ir_path.exists():
        raise FileNotFoundError(f"IR file not found: {ir_path}")

    # Load IR structure
    with open(ir_path) as f:
        ir_dict = json.load(f)

    # Schema version check
    schema_ver = (ir_dict.get("metadata", {}) or {}).get("ir_schema_version")
    if schema_ver is not None and schema_ver != 1:
        raise RuntimeError(
            f"Unsupported IR schema version {schema_ver}. "
            f"This runtime supports version 1. Re-compile the model."
        )

    # Load weights
    weights: dict[str, dict[str, torch.Tensor]] = {}
    if weights_path.exists():
        raw_weights: dict[str, torch.Tensor] = torch.load(weights_path, map_location="cpu", weights_only=True)
        # Distribute weights back to functions
        for key, tensor in raw_weights.items():
            if "." in key and not key.startswith("_"):
                func_name, wname = key.split(".", 1)
            else:
                func_name = "main"
                wname = key
            weights.setdefault(func_name, {})[wname] = tensor

    return IrModule.from_dict(ir_dict, all_weights=weights)
