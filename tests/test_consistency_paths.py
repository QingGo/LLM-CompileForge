"""Path-resolution guards for the 4-way logit alignment check.

``scripts/checks/test_consistency.py`` (make test-consistency) compares
HF ↔ Python Executor ↔ ctypes dylib ↔ Rust forward_check logits.  Stale
artifact paths silently break the check before any cosine is computed:

  * DYLIB_PATH pointed at ``libopt_125m.dylib`` while the build emits
    ``libopt_125m_fresh.dylib``.
  * FORWARD_CHECK_BIN pointed at ``rust/target/...`` while the crate
    lives in ``runtime/``.
  * An ``ensure(weights.safetensors)`` guard required a file the build
    no longer emits — weights come from the HF cache.

These tests fail fast (no model load) whenever the constants drift from
the real artifacts again.
"""

from __future__ import annotations

import os
import pathlib

import pytest

import scripts.checks.test_consistency as tc


class TestConsistencyPaths:
    """Constants in test_consistency.py must resolve to real artifacts."""

    def test_dylib_path_exists(self) -> None:
        assert os.path.isfile(tc.DYLIB_PATH), (
            f"DYLIB_PATH does not exist: {tc.DYLIB_PATH}. "
            "Build the dylib first (make rebuild-dylib) or fix the constant."
        )

    def test_forward_check_binary_exists(self) -> None:
        binary = os.path.join(tc._PROJECT_ROOT, tc.FORWARD_CHECK_BIN)
        assert os.path.isfile(binary), (
            f"FORWARD_CHECK_BIN does not exist: {binary}. "
            "Build it (cd runtime && cargo build --release --bin forward_check) "
            "or fix the constant."
        )

    def test_weights_resolution_matches_rust_fallback(self) -> None:
        """Weights must resolve from the artifact dir or the HF cache.

        Mirrors ``forward_check_runner::find_safetensors``: local
        safetensors first, then ``~/.cache/huggingface/hub/.../snapshots``.
        """
        artifact_dir = pathlib.Path(tc.ARTIFACT_DIR)
        local = artifact_dir / "model.safetensors"
        if local.is_file():
            return

        hub = pathlib.Path(tc.HF_CACHE) / "snapshots"
        if hub.is_dir():
            for snap in sorted(hub.iterdir(), reverse=True):
                if (snap / "model.safetensors").is_file():
                    return
        pytest.fail(
            "No model.safetensors in artifact dir or HF cache — "
            "the Rust forward_check will fail to find weights."
        )
