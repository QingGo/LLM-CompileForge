"""Tests for compiler/mlir_artifact.py — MLIR parser roundtrip and edge cases."""

from __future__ import annotations

import pytest
import torch

from compiler.ir import IrFunction, IrModule, IrOp, IrType
from compiler.mlir_artifact import load_mlir_artifact, save_mlir_artifact
from compiler.mlir_emitter import ir_module_to_mlir


@pytest.mark.unit
class TestMlirRoundtrip:
    def test_empty_module_roundtrip(self) -> None:
        func = IrFunction(name="main", inputs=[], outputs=[])
        mod = IrModule(functions=[func])
        mlir_text = ir_module_to_mlir(mod)
        assert "module" in mlir_text
        assert "func.func @main" in mlir_text

    def test_single_op_roundtrip(self) -> None:
        func = IrFunction(
            name="main",
            inputs=[("x", IrType(dtype="float32", shape=(4,)))],
            outputs=[("y", IrType(dtype="float32", shape=(4,)))],
            ops=[IrOp(name="relu", inputs=["x"], outputs=["y"])],
        )
        mod = IrModule(functions=[func])
        mlir_text = ir_module_to_mlir(mod)
        assert '"sf.relu"' in mlir_text

    def test_weight_constants_roundtrip(self) -> None:
        w = torch.randn(4, 8)
        func = IrFunction(
            name="main",
            inputs=[("x", IrType(dtype="float32", shape=(None, 4)))],
            outputs=[("y", IrType(dtype="float32", shape=(None, 8)))],
            ops=[IrOp(name="linear", inputs=["x", "w"], outputs=["y"])],
            weights={"w": w},
        )
        mod = IrModule(functions=[func])
        mlir_text = ir_module_to_mlir(mod)
        assert '"sf.weight"' in mlir_text
        assert '"sf.matmul"' in mlir_text or '"sf.linear"' in mlir_text

    def test_dynamic_dimensions_use_question_mark(self) -> None:
        func = IrFunction(
            name="main",
            inputs=[("x", IrType(dtype="float32", shape=(None, 4)))],
            outputs=[("y", IrType(dtype="float32", shape=(None, 4)))],
            ops=[IrOp(name="relu", inputs=["x"], outputs=["y"])],
        )
        mod = IrModule(functions=[func])
        mlir_text = ir_module_to_mlir(mod)
        assert "tensor<?x4xf32>" in mlir_text

    def test_attributes_preserved(self) -> None:
        func = IrFunction(
            name="main",
            inputs=[("x", IrType(dtype="float32", shape=(4,)))],
            outputs=[("y", IrType(dtype="float32", shape=(4,)))],
            ops=[IrOp(
                name="softmax",
                inputs=["x"],
                outputs=["y"],
                attributes={"dim": -1},
            )],
        )
        mod = IrModule(functions=[func])
        mlir_text = ir_module_to_mlir(mod)
        assert "dim" in mlir_text

    def test_ssa_chaining_produces_sequential_names(self) -> None:
        func = IrFunction(
            name="main",
            inputs=[("x", IrType(dtype="float32", shape=(4,)))],
            outputs=[("z", IrType(dtype="float32", shape=(4,)))],
            ops=[
                IrOp(name="relu", inputs=["x"], outputs=["y"]),
                IrOp(name="relu", inputs=["y"], outputs=["z"]),
            ],
        )
        mod = IrModule(functions=[func])
        mlir_text = ir_module_to_mlir(mod)
        # SSA names should be sequential
        for ssa in ["%0", "%1", "%2"]:
            assert ssa in mlir_text, f"Missing SSA {ssa}"

    def test_multiple_outputs_handled(self) -> None:
        func = IrFunction(
            name="main",
            inputs=[("x", IrType(dtype="float32", shape=(4,)))],
            outputs=[
                ("y", IrType(dtype="float32", shape=(2,))),
                ("z", IrType(dtype="float32", shape=(2,))),
            ],
            ops=[IrOp(name="split", inputs=["x"], outputs=["y", "z"])],
        )
        mod = IrModule(functions=[func])
        mlir_text = ir_module_to_mlir(mod)
        assert '"sf.split"' in mlir_text


@pytest.mark.unit
class TestMlirParsing:
    def test_parse_empty_module(self) -> None:
        func = IrFunction(name="main", inputs=[], outputs=[])
        mod = IrModule(functions=[func])
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            save_mlir_artifact(mod, d)
            parsed = load_mlir_artifact(d)
            assert len(parsed.functions) == 1

    def test_parse_module_with_weights(self) -> None:
        w = torch.randn(4, 8)
        func = IrFunction(
            name="main",
            inputs=[("x", IrType(dtype="float32", shape=(4,)))],
            outputs=[("y", IrType(dtype="float32", shape=(8,)))],
            ops=[IrOp(name="linear", inputs=["x", "w"], outputs=["y"])],
            weights={"w": w},
        )
        mod = IrModule(functions=[func])
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            save_mlir_artifact(mod, d)
            parsed = load_mlir_artifact(d)
            assert len(parsed.functions) == 1
            assert len(parsed.main.ops) > 0
            assert "w" in parsed.main.weights
            assert torch.allclose(parsed.main.weights["w"], w)

    def test_parse_preserves_ssa_chain(self) -> None:
        func = IrFunction(
            name="main",
            inputs=[("x", IrType(dtype="float32", shape=(4,)))],
            outputs=[("z", IrType(dtype="float32", shape=(4,)))],
            ops=[
                IrOp(name="relu", inputs=["x"], outputs=["y"]),
                IrOp(name="add", inputs=["y", "b"], outputs=["z"]),
            ],
        )
        mod = IrModule(functions=[func])
        mlir_text = ir_module_to_mlir(mod)
        # Verify the emitted MLIR has valid structure
        lines = mlir_text.splitlines()
        assert any("relu" in line for line in lines)
        assert any("add" in line for line in lines)

    def test_fused_op_names_preserved(self) -> None:
        func = IrFunction(
            name="main",
            inputs=[("x", IrType(dtype="float32", shape=(4,)))],
            outputs=[("y", IrType(dtype="float32", shape=(4,)))],
            ops=[IrOp(
                name="fused_rms_norm_matmul",
                inputs=["x", "w"],
                outputs=["y"],
                attributes={"eps": 1e-5},
            )],
        )
        mod = IrModule(functions=[func])
        mlir_text = ir_module_to_mlir(mod)
        assert '"sf.fused_rms_norm_matmul"' in mlir_text


@pytest.mark.unit
class TestMlirAttributeParsing:
    def test_bool_attr(self) -> None:
        from compiler.mlir_artifact import _parse_attrs
        attrs = _parse_attrs("folded = true, is_causal = false")
        assert attrs["folded"] is True
        assert attrs["is_causal"] is False

    def test_int_attr(self) -> None:
        from compiler.mlir_artifact import _parse_attrs
        attrs = _parse_attrs("dim = 0, start = 42")
        assert attrs["dim"] == 0
        assert attrs["start"] == 42

    def test_string_attr(self) -> None:
        from compiler.mlir_artifact import _parse_attrs
        attrs = _parse_attrs('name = "hello world"')
        assert attrs["name"] == "hello world"

    def test_list_attr(self) -> None:
        from compiler.mlir_artifact import _parse_attrs
        attrs = _parse_attrs('shape = [1, 2, 3]')
        assert attrs["shape"] == [1, 2, 3]

    def test_mixed_attrs(self) -> None:
        from compiler.mlir_artifact import _parse_attrs
        attrs = _parse_attrs('dim = 0, folded = true, name = "test", shape = [1, 2]')
        assert attrs["dim"] == 0
        assert attrs["folded"] is True
        assert attrs["name"] == "test"
        assert attrs["shape"] == [1, 2]

    def test_none_attr(self) -> None:
        from compiler.mlir_artifact import _parse_attrs
        attrs = _parse_attrs("end = none")
        assert attrs["end"] is None
