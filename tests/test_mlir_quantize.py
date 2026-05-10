"""Tests for MLIR Q/DQ pass.

Verifies that sf.quantize and sf.dequantize ops are correctly inserted
into MLIR text according to precision strategy configs.
"""

from __future__ import annotations

import pytest


@pytest.mark.unit
class TestWeightNameParsing:
    def test_parse_single_weight(self) -> None:
        from compiler.mlir_passes.quantize import _parse_weight_names

        mlir = '''module {
  func.func @main() -> tensor<f32> {
    %0 = "sf.weight"() {name = "q_proj_weight"} : () -> tensor<64x32xf32>
    return %0 : tensor<64x32xf32>
  }
}'''
        names = _parse_weight_names(mlir)
        assert names == {"q_proj_weight": "%0"}

    def test_parse_multiple_weights(self) -> None:
        from compiler.mlir_passes.quantize import _parse_weight_names

        mlir = '''module {
  func.func @main() -> tensor<f32> {
    %w = "sf.weight"() {name = "q_proj_weight"} : () -> tensor<64x32xf32>
    %x = "sf.weight"() {name = "k_proj_weight"} : () -> tensor<16x16xf32>
    return %w : tensor<64x32xf32>
  }
}'''
        names = _parse_weight_names(mlir)
        assert names == {"q_proj_weight": "%w", "k_proj_weight": "%x"}

    def test_no_weights_returns_empty(self) -> None:
        from compiler.mlir_passes.quantize import _parse_weight_names

        mlir = 'module {\n  func.func @main() -> tensor<f32> {\n    return %0 : tensor<f32>\n  }\n}'
        names = _parse_weight_names(mlir)
        assert names == {}


@pytest.mark.unit
class TestDQInsertion:
    def test_inserts_dq_for_w8a8_weight(self) -> None:
        from compiler.mlir_passes.quantize import count_dq_ops, insert_quantize_dequantize
        from compiler.quantize.mixed_precision import MixedPrecisionConfig

        mlir = '''module {
  func.func @main(%0: tensor<4x32xf32>) -> tensor<4x64xf32> {
    %w = "sf.weight"() {name = "q_proj_weight"} : () -> tensor<64x32xf32>
    %1 = "sf.linear"(%0, %w) {source_node = "linear"} : (tensor<4x32xf32>, tensor<64x32xf32>) -> tensor<4x64xf32>
    return %1 : tensor<4x64xf32>
  }
}'''
        config = MixedPrecisionConfig.from_dict({"q_proj_weight": "w8a8"})
        result = insert_quantize_dequantize(mlir, config)

        assert count_dq_ops(result) >= 1
        assert "sf.dequantize" in result

    def test_no_dq_for_fp16_weight(self) -> None:
        from compiler.mlir_passes.quantize import count_dq_ops, insert_quantize_dequantize
        from compiler.quantize.mixed_precision import MixedPrecisionConfig

        mlir = '''module {
  func.func @main(%0: tensor<4x32xf32>) -> tensor<4x64xf32> {
    %w = "sf.weight"() {name = "embed_tokens_weight"} : () -> tensor<50000x32xf32>
    %1 = "sf.embedding"(%w, %0) : (tensor<50000x32xf32>, tensor<4x32xf32>) -> tensor<4x32xf32>
    return %1 : tensor<4x32xf32>
  }
}'''
        config = MixedPrecisionConfig()  # embed_tokens → fp16
        result = insert_quantize_dequantize(mlir, config)

        assert count_dq_ops(result) == 0

    def test_empty_mlir_unchanged(self) -> None:
        from compiler.mlir_passes.quantize import insert_quantize_dequantize

        mlir = 'module {\n}'
        result = insert_quantize_dequantize(mlir)
        assert result == mlir


@pytest.mark.unit
class TestDQCount:
    def test_count_zero_for_no_dq_ops(self) -> None:
        from compiler.mlir_passes.quantize import count_dq_ops

        assert count_dq_ops("no ops here") == 0

    def test_count_multiple_dq_ops(self) -> None:
        from compiler.mlir_passes.quantize import count_dq_ops

        mlir = '''module {
  %a = "sf.dequantize"(%0) : () -> tensor<f32>
  %b = "sf.dequantize"(%1) : () -> tensor<f32>
}'''
        assert count_dq_ops(mlir) == 2
