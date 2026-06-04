# ruff: noqa: E501  # MLIR text fixtures exceed 120-char limit
"""Seam tests for MLIR text parsing through public compiler API.

Tests ``_parse_mlir_text()`` from ``compiler.mlir_artifact_parse``
— a lightweight line-based MLIR parser that does NOT require PyMLIR,
C++ bindings, or any compiled artifacts.
"""

from compiler.mlir_artifact import _parse_mlir_text
from compiler.mlir_dialect.sf.mlir_op_types import MlirModule

# ── Hand-crafted MLIR snippet (valid sf dialect) ─────────────────────

MLIR_SIMPLE = """\
module {
  func.func @simple(%arg0: tensor<2x64xf32>, %arg1: tensor<64x32xf32>) -> tensor<2x32xf32> {
    %0 = "sf.matmul"(%arg0, %arg1) : (tensor<2x64xf32>, tensor<64x32xf32>) -> tensor<2x32xf32>
    func.return %0 : tensor<2x32xf32>
  }
}"""

MLIR_WEIGHT = """\
module {
  func.func @weighted(%arg0: tensor<2x8xf32>, %arg1: tensor<8x4xf32>) -> tensor<2x8xf32> {
    %0 = "sf.weight"() {name = "w_proj"} : () -> tensor<4x8xf32>
    %1 = "sf.linear"(%arg0, %0) : (tensor<2x8xf32>, tensor<4x8xf32>) -> tensor<2x4xf32>
    %2 = "sf.relu"(%1) : (tensor<2x4xf32>) -> tensor<2x4xf32>
    func.return %2 : tensor<2x4xf32>
  }
}"""

MLIR_MULTI_RESULT = """\
module {
  func.func @multi(%arg0: tensor<4x8xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
    %0 = "sf.split"(%arg0) {split_sizes = [4, 4], dim = 1 : i64} : (tensor<4x8xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>)
    func.return %0#0, %0#1 : tensor<4x4xf32>, tensor<4x4xf32>
  }
}"""

MLIR_ATTRS = """\
module {
  func.func @with_attrs(%arg0: tensor<2x64xf32>) -> tensor<2x64xf32> {
    %0 = "sf.slice"(%arg0) {dim = 1 : i64, end = 32 : i64, start = 0 : i64, step = 1 : i64} : (tensor<2x64xf32>) -> tensor<2x32xf32>
    func.return %0 : tensor<2x32xf32>
  }
}"""


# ── Tests ────────────────────────────────────────────────────────────


class TestParseSimple:
    """Parse a minimal sf.matmul MLIR module."""

    def test_returns_mlir_module(self) -> None:
        mod = _parse_mlir_text(MLIR_SIMPLE)
        assert isinstance(mod, MlirModule)
        assert len(mod.functions) == 1

    def test_function_name(self) -> None:
        mod = _parse_mlir_text(MLIR_SIMPLE)
        assert mod.functions[0].name == "simple"

    def test_inputs_parsed(self) -> None:
        mod = _parse_mlir_text(MLIR_SIMPLE)
        fn = mod.functions[0]
        assert len(fn.inputs) == 2
        assert fn.inputs[0][0] == "%arg0"
        assert "tensor<2x64xf32>" in fn.inputs[0][1]

    def test_outputs_parsed(self) -> None:
        mod = _parse_mlir_text(MLIR_SIMPLE)
        fn = mod.functions[0]
        assert len(fn.outputs) == 1
        assert fn.outputs[0][0] == "%0"  # SSA name from func.return

    def test_ops_parsed(self) -> None:
        mod = _parse_mlir_text(MLIR_SIMPLE)
        fn = mod.functions[0]
        assert len(fn.ops) == 1
        op = fn.ops[0]
        assert op.name == "sf.matmul"
        assert op.dialect == "sf"
        assert op.op_name == "matmul"


class TestParseWeights:
    """Parse ops with sf.weight and sf.linear."""

    def test_weight_recorded_in_ssa_map(self) -> None:
        mod = _parse_mlir_text(MLIR_WEIGHT)
        fn = mod.functions[0]
        assert len(fn.ops) == 3

    def test_weight_op_attributes(self) -> None:
        mod = _parse_mlir_text(MLIR_WEIGHT)
        fn = mod.functions[0]
        weight_op = fn.ops[0]
        assert weight_op.name == "sf.weight"
        assert weight_op.attributes.get("name") == "w_proj"

    def test_linear_operand_resolved(self) -> None:
        mod = _parse_mlir_text(MLIR_WEIGHT)
        fn = mod.functions[0]
        linear_op = fn.ops[1]
        assert linear_op.name == "sf.linear"
        # Operands resolved via SSA→weight mapping
        assert len(linear_op.operands) == 2


class TestParseMultiResult:
    """Parse op with multiple results (sf.split)."""

    def test_multi_result_function(self) -> None:
        mod = _parse_mlir_text(MLIR_MULTI_RESULT)
        fn = mod.functions[0]
        assert fn.name == "multi"
        assert len(fn.outputs) == 2

    def test_split_op_has_two_output_types(self) -> None:
        mod = _parse_mlir_text(MLIR_MULTI_RESULT)
        fn = mod.functions[0]
        split_op = fn.ops[0]
        assert split_op.name == "sf.split"
        assert len(split_op.output_types) == 2


class TestParseAttributes:
    """Parse integer-typed MLIR attributes."""

    def test_slice_attrs(self) -> None:
        mod = _parse_mlir_text(MLIR_ATTRS)
        fn = mod.functions[0]
        slice_op = fn.ops[0]
        assert slice_op.name == "sf.slice"
        assert slice_op.attributes.get("dim") == 1
        assert slice_op.attributes.get("start") == 0
        assert slice_op.attributes.get("end") == 32
        assert slice_op.attributes.get("step") == 1


class TestEmptyAndEdgeCases:
    """Robustness: comments, empty module."""

    def test_empty_module(self) -> None:
        mod = _parse_mlir_text("module {\n}")
        assert isinstance(mod, MlirModule)
        assert len(mod.functions) == 0

    def test_comment_lines_ignored(self) -> None:
        text = """\
// This is a comment
module {
  // Another comment
  func.func @with_comment(%arg0: tensor<2x64xf32>) -> tensor<2x64xf32> {
    %0 = "sf.relu"(%arg0) : (tensor<2x64xf32>) -> tensor<2x64xf32>
    func.return %0 : tensor<2x64xf32>
  }
}"""
        mod = _parse_mlir_text(text)
        assert len(mod.functions) == 1
        assert mod.functions[0].name == "with_comment"

    def test_multiple_functions(self) -> None:
        text = """\
module {
  func.func @a(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = "sf.relu"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
    func.return %0 : tensor<4xf32>
  }
  func.func @b(%arg0: tensor<8xf32>) -> tensor<8xf32> {
    %0 = "sf.gelu"(%arg0) : (tensor<8xf32>) -> tensor<8xf32>
    func.return %0 : tensor<8xf32>
  }
}"""
        mod = _parse_mlir_text(text)
        assert len(mod.functions) == 2
        assert {f.name for f in mod.functions} == {"a", "b"}
