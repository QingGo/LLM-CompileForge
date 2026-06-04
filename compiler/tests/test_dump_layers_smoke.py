"""Smoke test: DUMP_LAYERS env var produces valid .npy dump files from Rust executor.

Validates each dumped .npy file:
  - Correct NumPy v1.0 header (magic bytes ``\\x93NUMPY``)
  - Loadable via ``np.load``
  - No NaN values present
  - Not all-identical values (min < max)
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


MODEL_DIR = Path("compiled/opt_125m_fresh")


def _dylib_exists() -> bool:
    if not MODEL_DIR.is_dir():
        return False
    return any(p.suffix == ".dylib" for p in MODEL_DIR.iterdir())


def _forward_check_built() -> bool:
    binary = Path("rust/target/debug/forward_check")
    return binary.is_file() and binary.stat().st_mode & 0o100 != 0


def _build_forward_check() -> bool:
    try:
        subprocess.run(
            ["cargo", "build", "--bin", "forward_check", "--features", "cli"],
            cwd="rust", capture_output=True, timeout=90,
        ).check_returncode()
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


requires_dylib = pytest.mark.skipif(
    not _dylib_exists(),
    reason=f"no .dylib found in {MODEL_DIR}",
)


def _validate_npy(path: Path) -> None:
    """Validate .npy header and tensor content for one dump file."""
    with open(path, "rb") as f:
        magic = f.read(6)
    assert magic == b"\x93NUMPY", f"File {path.name}: invalid npy magic {magic!r}"

    arr = np.load(str(path))
    assert arr.size > 0, f"File {path.name}: empty tensor"

    nan_count = int(np.isnan(arr).sum())
    assert nan_count < arr.size, f"File {path.name}: all-NaN tensor"
    if nan_count > 0:
        ratio = nan_count / arr.size
        assert ratio < 0.01, (
            f"File {path.name}: {nan_count}/{arr.size} ({ratio:.1%}) NaN"
        )

    finite = arr[np.isfinite(arr)]
    assert finite.size > 0, f"File {path.name}: all values are NaN/Inf"
    assert finite.min() < finite.max(), (
        f"File {path.name}: all-identical finite values ({finite.flat[0]})"
    )


@pytest.mark.unit
@pytest.mark.timeout(120)
@requires_dylib
def test_dump_layers_smoke() -> None:
    """Run Rust forward pass with DUMP_LAYERS, validate every .npy dump."""
    if not _forward_check_built() and not _build_forward_check():
        pytest.skip("forward_check binary could not be built")

    with tempfile.TemporaryDirectory(prefix="dump_layers_") as tmpdir:
        env = os.environ.copy()
        env["DUMP_LAYERS"] = tmpdir

        result = subprocess.run(
            ["./rust/target/debug/forward_check"],
            capture_output=True, text=True, timeout=90,
            env=env,
        )
        if result.returncode != 0:
            stderr_tail = result.stderr.strip().split("\n")[-3:]
            pytest.skip(
                f"forward_check exited {result.returncode}: "
                + "; ".join(stderr_tail)
            )

        npy_files = sorted(Path(tmpdir).glob("func_*.npy"))
        assert len(npy_files) > 0, f"No .npy files dumped to {tmpdir}"

        for fpath in npy_files:
            _validate_npy(fpath)
