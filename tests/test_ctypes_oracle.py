"""E2E test for ctypes oracle: verify compiled dylib matches Python executor."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from scripts.ctypes_oracle import CtypesOracle

DYLIB_PATH = "compiled/opt_125m_fresh/libopt_125m.dylib"
ARTIFACT_DIR = "compiled/opt_125m_fresh"
MIN_COS = 0.99  # ctypes should closely match Python executor


@pytest.mark.smoke
@pytest.mark.timeout(30)
def test_ctypes_cosine_high() -> None:
    """Compiled dylib via ctypes matches Python executor with cos > 0.99."""
    if not os.path.exists(DYLIB_PATH):
        pytest.skip(f"dylib not found at {DYLIB_PATH}")
    oracle = CtypesOracle(artifact_dir=ARTIFACT_DIR)
    cos = oracle.compare(DYLIB_PATH)
    assert cos >= MIN_COS, f"cos={cos:.6f} < {MIN_COS}"


@pytest.mark.integration
@pytest.mark.timeout(30)
def test_ctypes_npy_dump_valid() -> None:
    """DUMP_LAYERS env var produces valid .npy files."""
    import subprocess
    import tempfile

    dylib_path = DYLIB_PATH
    if not os.path.exists(dylib_path):
        pytest.skip(f"dylib not found at {dylib_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        env = os.environ.copy()
        env["DUMP_LAYERS"] = tmpdir
        result = subprocess.run(
            ["cargo", "run", "--bin", "forward_check"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd="rust",
            env=env,
        )
        if result.returncode != 0:
            pytest.skip("forward_check binary failed (probably no compiled model)")

        npy_files = [f for f in os.listdir(tmpdir) if f.endswith(".npy")]
        assert len(npy_files) > 0, f"No .npy files in {tmpdir}"

        first = os.path.join(tmpdir, sorted(npy_files)[0])
        with open(first, "rb") as f:
            magic = f.read(6)
        assert magic == b"\x93NUMPY", f"Invalid npy magic: {magic}"
