"""Golden test data I/O: save/load float32 tensor dicts as .npz files.

Contract: Shared between compiler and runtime sub-projects.
- .npz key = output name (e.g. "output_0", "func_main_1_output")
- .npz value = float32 numpy array
- Flat dict only — no nested structures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_npz(path: str | Path, tensors: dict[str, np.ndarray]) -> None:
    """Save a dict of {name: float32 numpy array} as a .npz file.

    Each array is stored as ``name.npy`` inside the zip archive via
    ``np.savez()``.
    """
    path = Path(path)
    # Validate all arrays are float32
    for name, arr in tensors.items():
        if arr.dtype != np.float32:
            raise TypeError(
                f"golden_io: tensor '{name}' has dtype {arr.dtype}, "
                f"expected float32"
            )
    np.savez(path, **tensors)


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load an .npz file and return a dict of {name: float32 numpy array}.

    Returns an empty dict if the archive contains no arrays.
    All arrays are guaranteed to be float32 (validated on load).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"golden_io: .npz file not found: {path}")
    npz = np.load(path, allow_pickle=False)
    result: dict[str, np.ndarray] = {}
    for key in npz.files:
        arr = npz[key]
        if arr.dtype != np.float32:
            raise TypeError(
                f"golden_io: array '{key}' in {path} has dtype {arr.dtype}, "
                f"expected float32"
            )
        result[key] = arr
    npz.close()
    return result


# ── Roundtrip test (run with: pytest compiler/tests/golden_io.py) ──────────


def test_save_load_roundtrip(tmp_path: Path) -> None:
    """Verify save_npz → load_npz preserves shapes and values."""
    tensors = {
        "output_0": np.arange(24, dtype=np.float32).reshape(1, 6, 4),
        "output_1": np.ones((12, 64), dtype=np.float32),
        "output_2": np.zeros((768,), dtype=np.float32),
    }
    npz_path = tmp_path / "test_golden.npz"

    save_npz(npz_path, tensors)
    assert npz_path.exists(), "save_npz: output file not created"

    loaded = load_npz(npz_path)
    assert set(loaded.keys()) == set(tensors.keys()), (
        f"key mismatch: expected {set(tensors.keys())}, got {set(loaded.keys())}"
    )

    for name, expected in tensors.items():
        got = loaded[name]
        assert got.shape == expected.shape, (
            f"shape mismatch for '{name}': "
            f"expected {expected.shape}, got {got.shape}"
        )
        assert got.dtype == np.float32, (
            f"dtype mismatch for '{name}': expected float32, got {got.dtype}"
        )
        assert np.allclose(got, expected, atol=0, rtol=0), (
            f"value mismatch for '{name}'"
        )


def test_load_npz_missing_file() -> None:
    """load_npz raises FileNotFoundError for missing file."""
    try:
        load_npz("/tmp/_golden_io_nonexistent_.npz")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected FileNotFoundError for missing .npz")
