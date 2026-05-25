"""RED-phase tests for index shape inference (_infer_index_pure).

``infer_output_shape('index', ...)`` currently dispatches to
``_infer_elementwise_pure`` (wrong semantics for index ops — it broadcasts ALL
inputs).  These tests will FAIL until ``_infer_index_pure`` is implemented.
"""

from __future__ import annotations

import pytest

from compiler.mlir_dialect.shape_inference_pure import infer_output_shape


@pytest.mark.unit
def test_infer_index_pure_single():
    result = infer_output_shape('index', [(4, 768), (2,)], ['f32', 'i64'])
    assert result == [((2, 768), 'f32')], f"got {result}"


@pytest.mark.unit
def test_infer_index_pure_multi_broadcast():
    result = infer_output_shape('index', [(2, 4), (2, 1, 1, 1), (1, 1, 1, 4)], ['f32', 'i64', 'i64'])
    assert result == [((2, 1, 1, 4), 'f32')], f"got {result}"


@pytest.mark.unit
def test_infer_index_pure_trailing():
    result = infer_output_shape('index', [(4, 768, 64), (2,), (3,)], ['f32', 'i64', 'i64'])
    assert result == [((2, 3, 64), 'f32')], f"got {result}"


@pytest.mark.unit
def test_infer_index_pure_dynamic():
    result = infer_output_shape('index', [(None, None), (None, 1, 1, 1), (1, 1, 1, None)], ['f32', 'i64', 'i64'])
    assert result[0][0] == (None, 1, 1, None), f"got {result[0][0]}"
    assert result[0][1] == 'f32', f"got {result[0][1]}"


@pytest.mark.unit
def test_infer_index_pure_single_trailing():
    result = infer_output_shape('index', [(4, 768, 64), (2,)], ['f32', 'i64'])
    assert result == [((2, 768, 64), 'f32')], f"got {result}"
