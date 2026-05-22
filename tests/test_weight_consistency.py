"""Tests for weight consistency verification.

Three-way comparison: ground truth, Python artifact, Rust-dumped weights.
"""

import json
import os
import subprocess

import numpy as np
import pytest

skip_no_model = pytest.mark.skipif(
    not os.path.isdir("compiled/opt_125m_fresh"),
    reason="Requires compiled model at compiled/opt_125m_fresh",
)

skip_no_rust = pytest.mark.skipif(
    not os.path.isdir("rust/target/debug"),
    reason="Rust not built",
)


def test_weight_script_importable() -> None:
    """Verify check_weights.py is importable/executable."""
    result = subprocess.run(
        ["python", "scripts/check_weights.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Three-way weight consistency" in result.stdout


@skip_no_model
@pytest.mark.timeout(30)
def test_weight_consistency_opt125m() -> None:
    """Run check_weights on compiled model, verify all pass."""
    result = subprocess.run(
        ["python", "scripts/check_weights.py", "--model", "compiled/opt_125m_fresh"],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
    assert result.returncode == 0
    assert "All" in result.stdout and "passed" in result.stdout


def test_weight_skip_no_model() -> None:
    """Verify graceful skip (exit 0) when model doesn't exist."""
    result = subprocess.run(
        ["python", "scripts/check_weights.py", "--model", "compiled/nonexistent"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "not found" in result.stdout or "Skipping" in result.stdout


@pytest.mark.slow
@skip_no_rust
def test_dump_weights_binary_builds() -> None:
    """Verify dump_weights binary builds."""
    result = subprocess.run(
        ["cargo", "build", "--bin", "dump_weights"],
        capture_output=True,
        text=True,
        cwd="rust",
    )
    assert result.returncode == 0


@skip_no_model
@pytest.mark.slow
@skip_no_rust
def test_dump_weights_produces_valid_npy() -> None:
    """Run dump_weights and verify .npy files loadable by numpy."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                "cargo",
                "run",
                "--bin",
                "dump_weights",
                "--",
                "--compiled-dir",
                "compiled/opt_125m_fresh",
                "--output-dir",
                tmpdir,
            ],
            capture_output=True,
            text=True,
            cwd="rust",
        )
        assert result.returncode == 0, f"dump_weights failed:\n{result.stderr}"

        # Check npy files exist and are loadable
        npy_files = [f for f in os.listdir(tmpdir) if f.endswith(".npy")]
        assert len(npy_files) > 0, "No .npy files produced"

        for f in npy_files[:5]:  # Check first 5
            arr = np.load(os.path.join(tmpdir, f))
            assert arr.dtype == np.float32

        # Check index
        index_path = os.path.join(tmpdir, "weights_index.json")
        assert os.path.exists(index_path)
        with open(index_path) as fh:
            index = json.load(fh)
        assert len(index) > 0
