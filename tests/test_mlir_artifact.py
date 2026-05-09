"""Tests for compiler/mlir_artifact.py — MLIR emitter/parser roundtrip."""

from __future__ import annotations

import pytest
import torch

from compiler.mlir_artifact import (
    MlirFunction,
    MlirModule,
    MlirOp,
    _parse_attrs,
    _parse_mlir_text,
    load_mlir_artifact,
    mlir_module_to_text,
    save_mlir_module_artifact,
)


@pytest.mark.unit
class TestMlirRoundtrip:
    def test_empty_module_roundtrip(self) -> None:
        mod = MlirModule(functions=[MlirFunction(name="main", inputs=[], outputs=[])])
        mlir_text = mlir_module_to_text(mod)
        assert "module" in mlir_text
        assert "func.func @main" in mlir_text

    def test_single_op_roundtrip(self) -> None:
        func = MlirFunction(
            name="main",
            inputs=[("%x", "tensor<4xf32>")],
            outputs=[("%y", "tensor<4xf32>")],
            ops=[MlirOp(name="sf.relu", dialect="sf", op_name="relu",
                        operands=["%x"], results=["%y"])],
        )
        mod = MlirModule(functions=[func])
        mlir_text = mlir_module_to_text(mod)
        assert '"sf.relu"' in mlir_text

    def test_weight_constants_roundtrip(self) -> None:
        w = torch.randn(4, 8)
        func = MlirFunction(
            name="main",
            inputs=[("%x", "tensor<?x4xf32>")],
            outputs=[("%y", "tensor<?x8xf32>")],
            ops=[
                MlirOp(name="sf.weight", dialect="sf", op_name="weight",
                       operands=["w"], results=["%w"], attributes={"name": "w"}),
                MlirOp(name="sf.linear", dialect="sf", op_name="linear",
                       operands=["%x", "%w"], results=["%y"]),
            ],
            weights={"w": w},
        )
        mod = MlirModule(functions=[func])
        mlir_text = mlir_module_to_text(mod)
        assert '"sf.weight"' in mlir_text
        assert '"sf.linear"' in mlir_text

    def test_dynamic_dimensions_preserved(self) -> None:
        func = MlirFunction(
            name="main",
            inputs=[("%x", "tensor<?x4xf32>")],
            outputs=[("%y", "tensor<?x4xf32>")],
            ops=[MlirOp(name="sf.relu", dialect="sf", op_name="relu",
                        operands=["%x"], results=["%y"])],
        )
        mod = MlirModule(functions=[func])
        mlir_text = mlir_module_to_text(mod)
        assert "tensor<?x4xf32>" in mlir_text

    def test_attributes_preserved(self) -> None:
        func = MlirFunction(
            name="main",
            inputs=[("%x", "tensor<4xf32>")],
            outputs=[("%y", "tensor<4xf32>")],
            ops=[MlirOp(name="sf.softmax", dialect="sf", op_name="softmax",
                        operands=["%x"], results=["%y"],
                        attributes={"dim": -1})],
        )
        mod = MlirModule(functions=[func])
        mlir_text = mlir_module_to_text(mod)
        assert "dim" in mlir_text

    def test_ssa_chaining(self) -> None:
        func = MlirFunction(
            name="main",
            inputs=[("%x", "tensor<4xf32>")],
            outputs=[("%z", "tensor<4xf32>")],
            ops=[
                MlirOp(name="sf.relu", dialect="sf", op_name="relu",
                       operands=["%x"], results=["%y"]),
                MlirOp(name="sf.relu", dialect="sf", op_name="relu",
                       operands=["%y"], results=["%z"]),
            ],
        )
        mod = MlirModule(functions=[func])
        mlir_text = mlir_module_to_text(mod)
        assert "%y" in mlir_text
        assert "%z" in mlir_text

    def test_multiple_outputs_handled(self) -> None:
        func = MlirFunction(
            name="main",
            inputs=[("%x", "tensor<4xf32>")],
            outputs=[("%y", "tensor<2xf32>"), ("%z", "tensor<2xf32>")],
            ops=[MlirOp(name="sf.split", dialect="sf", op_name="split",
                        operands=["%x"], results=["%y", "%z"])],
        )
        mod = MlirModule(functions=[func])
        mlir_text = mlir_module_to_text(mod)
        assert '"sf.split"' in mlir_text


@pytest.mark.unit
class TestMlirParsing:
    def test_parse_empty_module(self) -> None:
        mod = MlirModule(functions=[MlirFunction(name="main", inputs=[], outputs=[])])
        text = mlir_module_to_text(mod)
        parsed = _parse_mlir_text(text)
        assert len(parsed.functions) == 1

    def test_parse_module_with_weights(self) -> None:
        w = torch.randn(4, 8)
        func = MlirFunction(
            name="main",
            inputs=[("%x", "tensor<4xf32>")],
            outputs=[("%y", "tensor<8xf32>")],
            ops=[
                MlirOp(name="sf.weight", dialect="sf", op_name="weight",
                       operands=["w"], results=["%w"], attributes={"name": "w"}),
                MlirOp(name="sf.linear", dialect="sf", op_name="linear",
                       operands=["%x", "%w"], results=["%y"]),
            ],
            weights={"w": w},
        )
        mod = MlirModule(functions=[func])
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            save_mlir_module_artifact(mod, d)
            parsed = load_mlir_artifact(d)
            assert len(parsed.functions) == 1
            assert len(parsed.main.ops) > 0
            assert "w" in parsed.main.weights
            assert torch.allclose(parsed.main.weights["w"], w)

    def test_parse_preserves_ssa_chain(self) -> None:
        func = MlirFunction(
            name="main",
            inputs=[("%x", "tensor<4xf32>")],
            outputs=[("%z", "tensor<4xf32>")],
            ops=[
                MlirOp(name="sf.relu", dialect="sf", op_name="relu",
                       operands=["%x"], results=["%y"]),
                MlirOp(name="sf.add", dialect="sf", op_name="add",
                       operands=["%y", "b"], results=["%z"]),
            ],
        )
        mod = MlirModule(functions=[func])
        mlir_text = mlir_module_to_text(mod)
        lines = mlir_text.splitlines()
        assert any("relu" in line for line in lines)
        assert any("add" in line for line in lines)

    def test_fused_op_names_preserved(self) -> None:
        func = MlirFunction(
            name="main",
            inputs=[("%x", "tensor<4xf32>")],
            outputs=[("%y", "tensor<4xf32>")],
            ops=[MlirOp(
                name="sf.fused_rms_norm_matmul", dialect="sf",
                op_name="fused_rms_norm_matmul",
                operands=["%x", "w"], results=["%y"],
                attributes={"eps": 1e-5},
            )],
        )
        mod = MlirModule(functions=[func])
        mlir_text = mlir_module_to_text(mod)
        assert '"sf.fused_rms_norm_matmul"' in mlir_text


@pytest.mark.unit
class TestMlirAttributeParsing:
    def test_bool_attr(self) -> None:
        attrs = _parse_attrs("folded = true, is_causal = false")
        assert attrs["folded"] is True
        assert attrs["is_causal"] is False

    def test_int_attr(self) -> None:
        attrs = _parse_attrs("dim = 0, start = 42")
        assert attrs["dim"] == 0
        assert attrs["start"] == 42

    def test_string_attr(self) -> None:
        attrs = _parse_attrs('name = "hello world"')
        assert attrs["name"] == "hello world"

    def test_list_attr(self) -> None:
        attrs = _parse_attrs('shape = [1, 2, 3]')
        assert attrs["shape"] == [1, 2, 3]

    def test_mixed_attrs(self) -> None:
        attrs = _parse_attrs('dim = 0, folded = true, name = "test", shape = [1, 2]')
        assert attrs["dim"] == 0
        assert attrs["folded"] is True
        assert attrs["name"] == "test"
        assert attrs["shape"] == [1, 2]

    def test_none_attr(self) -> None:
        attrs = _parse_attrs("end = none")
        assert attrs["end"] is None
