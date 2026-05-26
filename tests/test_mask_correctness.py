"""Mask correctness tests for compiled dylib — TDD RED phase.

Expected failures against current (buggy) dylib where mask rows 2-3 = 0,
breaking the causal attention pattern. Tests follow TDD discipline:
write the failing test first (RED), fix the code (GREEN), then refactor.

See: .omo/plans/mask-residual-fix.md — Task 3
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

from scripts.ctypes_forward import run_ctypes  # noqa: E402

DYLIB_DIR = "compiled/opt_125m_fresh"


# ── Skip guard (dylib must exist) ─────────────────────────────────


def _dylib_exists() -> bool:
    try:
        return os.path.isdir(DYLIB_DIR) and any(
            f.endswith(".dylib") for f in os.listdir(DYLIB_DIR)
        )
    except OSError:
        return False


requires_dylib = pytest.mark.skipif(
    not _dylib_exists(),
    reason=f"compiled opt_125m_fresh dylib not found at {DYLIB_DIR}",
)


# ── Helpers ───────────────────────────────────────────────────────


def _load_mask(
    input_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Load mask tensor (func_outputs[0][13]) from compiled dylib.

    Returns:
        4D numpy array of shape (batch, 1, seq_len, seq_len).
    """
    result = run_ctypes(artifact_dir=DYLIB_DIR, input_ids=input_ids)
    mask: np.ndarray = result.func_outputs[0][13]
    return mask


# ── Mask value conventions ─────────────────────────────────────────
# The compiled model uses float32 mask with:
#   1.0  = attend to this position
#   0.0  = mask out this position (don't attend)
# This is equivalent to a boolean mask where True=1.0=attend.


_MASK_ATTEND: float = 1.0
_MASK_BLOCK: float = 0.0


# ── Tests ─────────────────────────────────────────────────────────


class TestMaskCorrectness:
    """Mask correctness tests against compiled dylib."""

    # ── Test 1: Not all zeros ──────────────────────────────────────

    @pytest.mark.unit
    @pytest.mark.timeout(120)
    @requires_dylib
    def test_mask_not_all_zeros(self) -> None:
        """Mask tensor must contain non-zero values.

        A valid causal mask has both 1.0 (attend) and 0.0 (mask)
        entries, so it should never be entirely ones or entirely zeros.
        """
        mask = _load_mask()
        assert mask.size > 0, f"Mask tensor is empty (shape={mask.shape})"
        unique_vals = np.unique(mask)
        assert 0 in unique_vals, (
            f"Mask has NO mask-out entries (0.0) — all values are attend. "
            f"Unique values: {unique_vals}"
        )
        assert 1 in unique_vals, (
            f"Mask has NO attend entries (1.0) — all values are blocked. "
            f"Unique values: {unique_vals}"
        )

    # ── Test 2: Causal pattern ──────────────────────────────────

    @pytest.mark.unit
    @pytest.mark.timeout(120)
    @requires_dylib
    def test_mask_causal_pattern(self) -> None:
        """Verify causal mask: 1 for j≤i (attend), 0 for j>i (mask).

        A correct causal causal mask for OPT-125M when embedded at
        ``sf.le`` satisfies::

            mask[b, h, i, j] = 1.0   for j ≤ i  (can attend)
            mask[b, h, i, j] = 0.0   for j > i  (blocked)
        """
        mask = _load_mask()
        assert mask.ndim == 4, f"Expected 4D mask, got ndim={mask.ndim} (shape={mask.shape})"

        batch, heads, seq_len, _ = mask.shape
        assert heads == 1, f"Expected single head, got {heads} heads"
        assert seq_len >= 2, f"Expected seq_len >= 2, got {seq_len}"

        for b in range(batch):
            for i in range(seq_len):
                for j in range(seq_len):
                    val = float(mask[b, 0, i, j])
                    if j <= i:
                        # Attend: should be 1.0
                        assert val == pytest.approx(1.0, abs=1e-6), (
                            f"batch={b}, pos=({i},{j}): attend should be "
                            f"~1.0, got {val}"
                        )
                    else:
                        # Mask: should be 0.0
                        assert val == pytest.approx(0.0, abs=1e-6), (
                            f"batch={b}, pos=({i},{j}): masked position "
                            f"should be 0.0, got {val}"
                        )

    # ── Test 3: Determinism ────────────────────────────────────────

    @pytest.mark.unit
    @pytest.mark.timeout(300)
    @requires_dylib
    def test_mask_deterministic(self) -> None:
        """Mask values must be bit-identical across 3 consecutive runs."""
        masks: list[np.ndarray] = []
        for _ in range(3):
            masks.append(_load_mask())

        for i in range(1, 3):
            np.testing.assert_array_equal(
                masks[0],
                masks[i],
                err_msg=(
                    f"Mask differs between run 0 and run {i} — "
                    f"non-determinism detected in compiled dylib"
                ),
            )

    # ── Test 4: Both batch items have valid masks ─────────────────

    @pytest.mark.unit
    @pytest.mark.timeout(30)
    @requires_dylib
    def test_mask_both_batches_valid(self) -> None:
        """Both batch items must produce valid causal masks.

        The model is statically compiled for batch=2. Both batch items
        (tokens [2, 32826, 85, 4129] and [0, 0, 0, 0]) should have
        the same causal mask shape and value range.
        """
        mask = _load_mask()
        assert mask.shape == (2, 1, 4, 4), f"Expected (2,1,4,4), got {mask.shape}"

        # Both batch items must have the same unique value set
        for b in range(2):
            unique_vals = np.unique(mask[b])
            assert 0 in unique_vals, f"Batch {b}: no 0.0 entries: {unique_vals}"
            assert 1 in unique_vals, f"Batch {b}: no 1.0 entries: {unique_vals}"


# ── Standalone execution ───────────────────────────────────────────


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
