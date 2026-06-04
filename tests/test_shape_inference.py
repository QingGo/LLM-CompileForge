"""RED-phase tests for index shape inference (_infer_index_pure).

``infer_output_shape('index', ...)`` currently dispatches to
``_infer_elementwise_pure`` (wrong semantics for index ops — it broadcasts ALL
inputs).  These tests will FAIL until ``_infer_index_pure`` is implemented.
"""

from __future__ import annotations

import pytest

from compiler.mlir_dialect.shape.shape_inference_pure import infer_output_shape


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


@pytest.mark.unit
@pytest.mark.timeout(30)
def test_index_dim_identity_guard():
    """Verify dim_names attribute flows through sf.index lowering without crashing.

    This tests that:
    1. The sf.index op carries dim_names attribute with symbolic names
    2. The C++ assertion in SfIndexOpLowering (NDEBUG-guarded) compiles
    3. Lowering produces correct linalg/tensor ops (no regression in dim sourcing)
    """
    try:
        from tests.lowering_test_helpers import check_lowered, lower
    except ImportError:
        pytest.skip("mlir-core or lowering_test_helpers not available")

    lowered = lower("""module {
  func.func @test(%data: tensor<?x?xf32>, %idx0: tensor<?x1x1x1xi64>,
                  %idx1: tensor<1x1x1x?xi64>) -> tensor<?x1x1x?xf32> {
    %0 = "sf.index"(%data, %idx0, %idx1) {dim_names = ["s0", "", "", "s1"]}
       : (tensor<?x?xf32>, tensor<?x1x1x1xi64>, tensor<1x1x1x?xi64>)
       -> tensor<?x1x1x?xf32>
    return %0 : tensor<?x1x1x?xf32>
  }
}""")
    check_lowered(lowered)
