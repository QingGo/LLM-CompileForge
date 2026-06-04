"""Model artifact serialization / deserialization.

The sole artifact format is MLIR (model.mlir) with weights (weights.pth)
and metadata (metadata.json).

Output structure:
  outputs/compiled/<model_name>/
    model.mlir       — MLIR text (standard-compliant)
    weights.pth      — PyTorch state dict
    metadata.json    — compilation metadata
"""

from __future__ import annotations

from pathlib import Path

from compiler.mlir_artifact import MlirModule  # type: ignore[attr-defined]


def load_artifact(directory: str) -> MlirModule:
    """Load a compiled model as an MlirModule (canonical MLIR artifact).

    Parses model.mlir and loads weights from weights.pth.  Returns an
    MlirModule suitable for MlirExecutor or LLMEngine.
    """
    in_dir = Path(directory)
    if not in_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    from compiler.mlir_artifact import load_mlir_artifact  # type: ignore[attr-defined]

    return load_mlir_artifact(str(in_dir))
