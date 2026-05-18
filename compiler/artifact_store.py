"""Artifact path management — single source of truth for compiled model paths.

All code that needs to locate compiled model artifacts should use this
module instead of hardcoding ``compiled/<model>/`` paths.  The default
artifact root can be overridden via the ``SERVE_FORGE_COMPILED_DIR``
environment variable.

Usage::

    from compiler.artifact_store import artifact_dir, default_compiled_dir

    paths = artifact_dir("opt_125m_fresh")
    print(paths.model_mlir)    # compiled/opt_125m_fresh/model.mlir
    print(paths.dylib)         # compiled/opt_125m_fresh/libopt_125m_fresh.dylib
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def default_compiled_dir() -> str:
    """Return the default compiled artifacts directory.

    Checks ``SERVE_FORGE_COMPILED_DIR`` env var first, then falls back
    to ``./compiled`` (relative to project root).
    """
    return os.environ.get("SERVE_FORGE_COMPILED_DIR", "compiled")


def _resolve(base: str, model_name: str) -> str:
    return os.path.join(base, model_name)


@dataclass(frozen=True)
class ArtifactPaths:
    """All known paths for a compiled model."""
    model_dir: str
    model_mlir: str
    model_lowered_mlir: str
    dylib: str
    safetensors: str
    constants_bin: str
    metadata_json: str
    ll_file: str
    o_file: str
    tokenizer_json: str
    tokenizer_config: str

    @classmethod
    def for_model(cls, model_name: str, compiled_dir: Optional[str] = None) -> ArtifactPaths:
        base = compiled_dir if compiled_dir is not None else default_compiled_dir()
        model_dir = _resolve(base, model_name)
        return cls(
            model_dir=model_dir,
            model_mlir=os.path.join(model_dir, "model.mlir"),
            model_lowered_mlir=os.path.join(model_dir, "model.lowered.mlir"),
            dylib=os.path.join(model_dir, f"lib{model_name}.dylib"),
            safetensors=os.path.join(model_dir, "model.safetensors"),
            constants_bin=os.path.join(model_dir, "constants.bin"),
            metadata_json=os.path.join(model_dir, "metadata.json"),
            ll_file=os.path.join(model_dir, f"{model_name}.ll"),
            o_file=os.path.join(model_dir, f"{model_name}.o"),
            tokenizer_json=os.path.join(model_dir, "tokenizer.json"),
            tokenizer_config=os.path.join(model_dir, "tokenizer_config.json"),
        )

    def ensure_dir(self) -> None:
        """Create the artifact directory if it doesn't exist."""
        os.makedirs(self.model_dir, exist_ok=True)


def artifact_dir(model_name: str) -> ArtifactPaths:
    """Shorthand: return artifact paths for a model using the default compiled dir."""
    return ArtifactPaths.for_model(model_name)
