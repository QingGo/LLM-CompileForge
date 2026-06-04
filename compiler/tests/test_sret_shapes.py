# ruff: noqa: E501
"""Unit tests for verify_output_shapes — structural sret output shape verification.

Tests cover:
1. Happy path: all shapes match
2. Output count mismatch
3. Rank mismatch
4. Static shape dimension mismatch
5. Null data pointer (aligned==0) detection
6. Multiple functions with mixed results
7. Edge case: scalar (rank-0) output
8. Edge case: fully dynamic shape (all dims = 0) — no false positive
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from compiler.sfcf_parser import parse_sret_outputs, verify_output_shapes

# ── Helpers ──────────────────────────────────────────────────────────


def _make_sret_descriptor(arr: np.ndarray) -> bytes:
    """Build an sret descriptor for a numpy array.

    Descriptor layout (per output, 24 + 16*rank bytes):
        offset 0:  allocated (u64)  — pointer to backing memory
        offset 8:  aligned   (u64)  — pointer to actual data
        offset 16: offset    (i64)
        offset 24: sizes[i64] * rank
        after sizes: strides[i64] * rank
    """
    rank = arr.ndim
    buf = struct.pack("<Q", arr.ctypes.data)  # allocated
    buf += struct.pack("<Q", arr.ctypes.data)  # aligned
    buf += struct.pack("<q", 0)                # offset
    for s in arr.shape:
        buf += struct.pack("<q", s)            # sizes
    elem_strides = arr.strides
    for s in elem_strides:
        buf += struct.pack("<q", s // arr.itemsize)  # strides (in elements)
    assert len(buf) == 24 + 16 * rank, f"bad descriptor size: {len(buf)} vs {24 + 16 * rank}"
    return buf


def _make_sret_descriptor_null(rank: int) -> bytes:
    """Build an sret descriptor with aligned==0 (null pointer).

    The descriptor has valid size/strides but aligned=0, simulating
    a kernel that failed to write its output.
    """
    buf = struct.pack("<Q", 0)  # allocated = 0
    buf += struct.pack("<Q", 0)  # aligned = 0 (null)
    buf += struct.pack("<q", 0)  # offset = 0
    for _ in range(rank):
        buf += struct.pack("<q", 0)  # sizes = 0 (dynamic, will be filled from graph)
    for _ in range(rank):
        buf += struct.pack("<q", 0)  # strides = 0
    assert len(buf) == 24 + 16 * rank
    return buf


def _make_tensors_and_sret(
    arrays: list[np.ndarray],
) -> tuple[list[np.ndarray], bytes]:
    """Create sret bytes from a list of numpy arrays.

    Returns the arrays (to keep them alive) and the concatenated sret bytes.
    """
    descs = [_make_sret_descriptor(arr) for arr in arrays]
    return arrays, b"".join(descs)


# ── Tests ────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestVerifyOutputShapes:

    def test_happy_path(self):
        """All shapes match — no errors."""
        arr1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        arr2 = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
        keep, sret = _make_tensors_and_sret([arr1, arr2])

        output_defs = [
            {"rank": 2, "shape": [2, 3]},
            {"rank": 2, "shape": [1, 5]},
        ]
        tensors = parse_sret_outputs(sret, output_defs)
        graph_functions = [{"symbol": "test_func", "outputs": output_defs}]
        errors = verify_output_shapes([tensors], graph_functions)
        assert errors == [], f"expected no errors, got: {errors}"

    def test_count_mismatch(self):
        """Fewer outputs than graph declares."""
        arr1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        keep, sret = _make_tensors_and_sret([arr1])

        # Graph declares 2 outputs, but sret only has 1
        output_defs = [
            {"rank": 2, "shape": [2, 2]},
            {"rank": 1, "shape": [5]},
        ]
        tensors = parse_sret_outputs(sret, output_defs[:1])  # only parse 1
        graph_functions = [{"symbol": "test_func", "outputs": output_defs}]
        errors = verify_output_shapes([tensors], graph_functions)
        assert len(errors) == 1
        assert "expected 2 output(s), got 1" in errors[0]

    def test_rank_mismatch_via_dynamic_shape(self):
        """Rank mismatch detected when null pointer yields empty array.

        With a fully dynamic graph shape (all dims = 0), the null pointer
        check has no static dims to compare, so the rank check fires:
        empty array has ndim=1, but graph declares rank=2.
        """
        desc = _make_sret_descriptor_null(rank=2)
        # Fully dynamic shape — no static dims to trigger null pointer check
        output_defs = [{"rank": 2, "shape": [0, 0]}]
        tensors = parse_sret_outputs(desc, output_defs)
        graph_functions = [{"symbol": "test_func", "outputs": output_defs}]
        errors = verify_output_shapes([tensors], graph_functions)
        assert len(errors) == 1
        assert "expected rank 2, got rank 1" in errors[0]

    def test_shape_dim_mismatch(self):
        """Static shape dimension does not match."""
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)  # [2, 3]
        keep, sret = _make_tensors_and_sret([arr])

        # Graph says [2, 5] — dim[1] mismatch
        output_defs = [{"rank": 2, "shape": [2, 5]}]
        tensors = parse_sret_outputs(sret, output_defs)
        graph_functions = [{"symbol": "test_func", "outputs": output_defs}]
        errors = verify_output_shapes([tensors], graph_functions)
        assert len(errors) == 1
        assert "dim[1] expected 5, got 3" in errors[0]

    def test_null_pointer_detection(self):
        """Null data pointer (aligned==0) is detected."""
        # Descriptor with aligned=0
        desc = _make_sret_descriptor_null(rank=2)
        output_defs = [{"rank": 2, "shape": [2, 768]}]
        tensors = parse_sret_outputs(desc, output_defs)
        graph_functions = [{"symbol": "test_func", "outputs": output_defs}]
        errors = verify_output_shapes([tensors], graph_functions)
        assert len(errors) == 1
        assert "null data pointer" in errors[0]

    def test_scalar_output(self):
        """Rank-0 (scalar) output — no error if empty but rank=0."""
        # Scalar tensors have ndim=0, shape=()
        arr = np.array(3.14, dtype=np.float32)
        keep, sret = _make_tensors_and_sret([arr])

        output_defs = [{"rank": 0, "shape": []}]
        tensors = parse_sret_outputs(sret, output_defs)
        graph_functions = [{"symbol": "test_func", "outputs": output_defs}]
        errors = verify_output_shapes([tensors], graph_functions)
        assert errors == [], f"expected no errors for scalar, got: {errors}"

    def test_dynamic_shape_skipped(self):
        """Dynamic dims (0 in graph) are skipped — no false positives."""
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        keep, sret = _make_tensors_and_sret([arr])

        # Graph says shape [0, 3] — first dim is dynamic
        output_defs = [{"rank": 2, "shape": [0, 3]}]
        tensors = parse_sret_outputs(sret, output_defs)
        graph_functions = [{"symbol": "test_func", "outputs": output_defs}]
        errors = verify_output_shapes([tensors], graph_functions)
        assert errors == [], f"expected no errors for dynamic dims, got: {errors}"

    def test_multiple_functions(self):
        """Multiple functions — one passes, one fails (null pointer)."""
        # Func 0: good
        arr0 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        keep0, sret0 = _make_tensors_and_sret([arr0])

        # Func 1: null pointer (aligned=0)
        desc1 = _make_sret_descriptor_null(rank=2)

        output_defs_0 = [{"rank": 2, "shape": [2, 2]}]
        output_defs_1 = [{"rank": 2, "shape": [4, 128]}]

        tensors_0 = parse_sret_outputs(sret0, output_defs_0)
        tensors_1 = parse_sret_outputs(desc1, output_defs_1)
        all_tensors = [tensors_0, tensors_1]
        graph_functions = [
            {"symbol": "func_0", "outputs": output_defs_0},
            {"symbol": "func_1", "outputs": output_defs_1},
        ]
        errors = verify_output_shapes(all_tensors, graph_functions)
        assert len(errors) == 1
        assert "func_1" in errors[0]
        assert "null data pointer" in errors[0]


@pytest.mark.unit
class TestParseSretOutputsEdgeCases:

    def test_null_pointer_returns_empty(self):
        """parse_sret_outputs returns empty array when aligned==0."""
        desc = _make_sret_descriptor_null(rank=2)
        output_defs = [{"rank": 2, "shape": [4, 128]}]
        tensors = parse_sret_outputs(desc, output_defs)
        assert len(tensors) == 1
        assert tensors[0].size == 0
        assert tensors[0].dtype == np.float32
