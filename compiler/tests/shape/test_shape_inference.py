"""Seam tests for shape inference through public compiler API.

Tests ``infer_output_shape()`` from ``compiler.shape.shape_inference``
— a pure-Python function that requires NO MLIR context, NO compiled dylib,
and NO model artifacts.
"""

from compiler.shape.shape_inference import infer_output_shape


class TestMatmul:
    """matmul: 2D matmul (a,b) @ (b,c) → (a,c); 3D+batch preservd."""

    def test_2d_matmul(self) -> None:
        out = infer_output_shape("matmul", [(4, 8), (8, 16)], ["f32", "f32"])
        assert out == [((4, 16), "f32")]

    def test_batched_matmul(self) -> None:
        out = infer_output_shape("matmul", [(2, 4, 8), (8, 16)], ["f32", "f32"])
        assert out == [((2, 4, 16), "f32")]

    def test_unknown_1st_dim(self) -> None:
        out = infer_output_shape("matmul", [(None, 8), (8, 16)], ["f32", "f32"])
        assert out == [((None, 16), "f32")]


class TestElementWise:
    """Element-wise ops broadcast inputs and pass-thru first element type."""

    def test_add_same_shape(self) -> None:
        out = infer_output_shape("add", [(2, 64), (2, 64)], ["f32", "f32"])
        assert out == [((2, 64), "f32")]

    def test_add_broadcast(self) -> None:
        out = infer_output_shape("add", [(2, 64), (1, 64)], ["f32", "f32"])
        assert out == [((2, 64), "f32")]

    def test_relu_shape(self) -> None:
        out = infer_output_shape("relu", [(3, 128)], ["f32"])
        assert out == [((3, 128), "f32")]

    def test_gelu_shape(self) -> None:
        out = infer_output_shape("gelu", [(None, 256)], ["f16"])
        assert out == [((None, 256), "f16")]

    def test_tanh_shape(self) -> None:
        out = infer_output_shape("tanh", [(5, 10)], ["f32"])
        assert out == [((5, 10), "f32")]


class TestSoftmax:
    """softmax is element-wise for shape purposes."""

    def test_softmax_2d(self) -> None:
        out = infer_output_shape("softmax", [(2, 128)], ["f32"])
        assert out == [((2, 128), "f32")]

    def test_softmax_3d(self) -> None:
        out = infer_output_shape("softmax", [(4, 8, 128)], ["f16"])
        assert out == [((4, 8, 128), "f16")]


class TestShapeManip:
    """view/unsqueeze/transpose: explicit shape changes."""

    def test_view(self) -> None:
        out = infer_output_shape("view", [(2, 64)], ["f32"], shape=(4, 32))
        assert out == [((4, 32), "f32")]

    def test_unsqueeze_dim0(self) -> None:
        out = infer_output_shape("unsqueeze", [(64,)], ["f32"], dim=0)
        assert out == [((1, 64), "f32")]

    def test_transpose(self) -> None:
        out = infer_output_shape("transpose", [(2, 3, 4)], ["f32"], dim0=0, dim1=2)
        assert out == [((4, 3, 2), "f32")]

    def test_slice(self) -> None:
        out = infer_output_shape(
            "slice",
            [(10, 64)],
            ["f32"],
            dim=0,
            start=2,
            end=8,
            step=1,
        )
        assert out == [((6, 64), "f32")]


class TestReduce:
    """mean/sum reduce dim; keepdim preserves reduced axis."""

    def test_mean_reduce(self) -> None:
        out = infer_output_shape("mean", [(2, 3, 4)], ["f32"], dim=1)
        assert out == [((2, 4), "f32")]

    def test_mean_keepdim(self) -> None:
        out = infer_output_shape("mean", [(2, 3, 4)], ["f32"], dim=1, keepdim=True)
        assert out == [((2, 1, 4), "f32")]

    def test_sum_keepdim(self) -> None:
        out = infer_output_shape("sum", [(2, 3, 4)], ["f32"], dim=1, keepdim=True)
        assert out == [((2, 1, 4), "f32")]


class TestLinear:
    """Linear = matmul with weight broadcast."""

    def test_linear_2d(self) -> None:
        # input (batch, in_features), weight (out_features, in_features)
        out = infer_output_shape("linear", [(2, 768), (512, 768)], ["f32", "f32"])
        assert out == [((2, 512), "f32")]

    def test_linear_3d(self) -> None:
        out = infer_output_shape("linear", [(2, 16, 768), (512, 768)], ["f32", "f32"])
        assert out == [((2, 16, 512), "f32")]


class TestEdgeCases:
    """Graceful fallbacks and unknown ops."""

    def test_unknown_op_fallback(self) -> None:
        out = infer_output_shape("nonexistent_op", [(4, 8)], ["f32"])
        assert out == [((4, 8), "f32")]

    def test_empty_input_fallback(self) -> None:
        out = infer_output_shape("relu", [], [])
        assert out == [((1,), "f32")]

    def test_dynamic_shape_softmax(self) -> None:
        out = infer_output_shape("softmax", [(None, None, 128)], ["f32"])
        assert out == [((None, None, 128), "f32")]
