"""Shared utility functions for mlir_artifact sub-package."""

from __future__ import annotations


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
