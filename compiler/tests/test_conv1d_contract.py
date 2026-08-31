"""Conv1d contract goldens.

Covers the sf.conv1d lowering used by Qwen3.5 GatedDeltaNet short-conv
(depthwise) and the generic grouped Conv1d path.  These tests compile a
minimal sf.conv1d function to a dylib and compare against PyTorch.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F  # noqa: N812

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from compiler.tests.test_sdpa_attention_contract import _call_main0, _compile  # noqa: E402


def _conv_mlir(
    input_type: str,
    weight_type: str,
    out_type: str,
    *,
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    groups: int,
) -> str:
    attrs = (
        "stride = " + str(stride) + ", padding = " + str(padding)
        + ", dilation = " + str(dilation) + ", groups = " + str(groups) + " : i64"
    )
    return f"""module {{
  func.func @main_0(%x: {input_type}, %w: {weight_type}) -> {out_type} {{
    %0 = "sf.conv1d"(%x, %w) {{{attrs}}} : ({input_type}, {weight_type}) -> {out_type}
    func.return %0 : {out_type}
  }}
}}"""


class TestConv1dShapeInference:
    def test_depthwise_output_shape(self) -> None:
        from compiler.shape.shape_inference_pure import _infer_conv1d_pure

        out = _infer_conv1d_pure(
            [(2, 4, 5), (4, 1, 3)],
            ["f32", "f32"],
            stride=[1],
            padding=[2],
            dilation=[1],
            groups=4,
        )
        assert out == [((2, 4, 7), "f32")]

    def test_grouped_output_shape(self) -> None:
        from compiler.shape.shape_inference_pure import _infer_conv1d_pure

        out = _infer_conv1d_pure(
            [(2, 4, 5), (4, 2, 3)],
            ["f32", "f32"],
            stride=[2],
            padding=[1],
            dilation=[2],
            groups=2,
        )
        assert out == [((2, 4, 2), "f32")]


@pytest.mark.integration
@pytest.mark.timeout(120)
class TestConv1dDylibContract:
    def _run(
        self,
        input_type: str,
        weight_type: str,
        out_type: str,
        x: np.ndarray,
        w: np.ndarray,
        *,
        stride: list[int],
        padding: list[int],
        dilation: list[int],
        groups: int,
    ) -> np.ndarray:
        import tempfile

        mlir = _conv_mlir(
            input_type,
            weight_type,
            out_type,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        with tempfile.TemporaryDirectory() as td:
            dylib = _compile(mlir, td, "conv1d_contract")
            return _call_main0(dylib, [x, w], 3)

    def test_depthwise_padded_conv1d(self) -> None:
        rng = np.random.RandomState(0)
        x = rng.randn(2, 4, 5).astype(np.float32)
        w = rng.randn(4, 1, 3).astype(np.float32)
        actual = self._run(
            "tensor<2x4x5xf32>",
            "tensor<4x1x3xf32>",
            "tensor<2x4x7xf32>",
            x,
            w,
            stride=[1],
            padding=[2],
            dilation=[1],
            groups=4,
        )
        expected = F.conv1d(
            torch.from_numpy(x),
            torch.from_numpy(w),
            padding=2,
            groups=4,
        ).numpy().astype(np.float32)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_grouped_padded_conv1d(self) -> None:
        rng = np.random.RandomState(1)
        x = rng.randn(2, 4, 5).astype(np.float32)
        w = rng.randn(4, 2, 3).astype(np.float32)
        actual = self._run(
            "tensor<2x4x5xf32>",
            "tensor<4x2x3xf32>",
            "tensor<2x4x5xf32>",
            x,
            w,
            stride=[1],
            padding=[1],
            dilation=[1],
            groups=2,
        )
        expected = F.conv1d(
            torch.from_numpy(x),
            torch.from_numpy(w),
            padding=1,
            groups=2,
        ).numpy().astype(np.float32)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
