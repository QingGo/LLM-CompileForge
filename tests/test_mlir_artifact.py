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

    def test_quoted_key_attr(self) -> None:
        """P0-1: MLIR attribute key like "dim" should be stripped to dim."""
        attrs = _parse_attrs('"dim" = 0 : i64, "name" = "weight"')
        assert attrs["dim"] == 0
        assert attrs["name"] == "weight"

    def test_type_suffix_on_int(self) -> None:
        """P0-2: MLIR attribute value '2 : i64' should resolve to int 2."""
        attrs = _parse_attrs('"dim" = 2 : i64')
        assert attrs["dim"] == 2
        assert isinstance(attrs["dim"], int)

    def test_type_suffix_on_float(self) -> None:
        """P0-2: MLIR attribute value '5.0e-01 : f64' should resolve to float."""
        attrs = _parse_attrs('"scale" = 5.000000e-01 : f64')
        assert abs(attrs["scale"] - 0.5) < 1e-6
        assert isinstance(attrs["scale"], float)

    def test_mixed_quoted_and_type_suffix(self) -> None:
        """P0-1+P0-2 combined: realistic MLIR attribute string."""
        attrs = _parse_attrs(
            '"dim" = 0 : i64, "folded" = true, '
            '"name" = "lm_head_weight", "shape" = [1, 32, 64, 64]'
        )
        assert attrs["dim"] == 0
        assert attrs["folded"] is True
        assert attrs["name"] == "lm_head_weight"
        assert attrs["shape"] == [1, 32, 64, 64]


@pytest.mark.unit
class TestWeightLoadTied:
    """P0-3: Tied weights should be filled from source on load."""

    def test_tied_weight_filled_on_load(self) -> None:
        """lm_head_weight missing from source → filled from embed_tokens via tied_weights."""
        import tempfile

        w = torch.randn(128, 64)
        f = MlirFunction(
            name="main",
            inputs=[("%x", "tensor<4xf32>")],
            outputs=[("%o", "tensor<4xf32>")],
            ops=[MlirOp(name="sf.linear", dialect="sf", op_name="linear",
                         operands=["%x", "w"], results=["%o"], attributes={})],
            weights={"model_embed_tokens_weight": w},
        )
        # Use backward compat: no classification → all weights go to weights.pth
        mod = MlirModule(functions=[f], metadata={
            "weight_classification": {
                "main": {"params": ["model_embed_tokens_weight", "lm_head_weight"],
                         "constants": []},
            },
            "weight_source": {
                "path": "/nonexistent.safetensors",
                "format": "safetensors",
                "tied_weights": {"lm_head_weight": "model_embed_tokens_weight"},
            },
        })

        with tempfile.TemporaryDirectory() as d:
            save_mlir_module_artifact(mod, d)
            # weights.pth is written (backward compat), then load_mlir_artifact
            # reads it + applies tied_weights from metadata
            loaded = load_mlir_artifact(d)
            assert "model_embed_tokens_weight" in loaded.main.weights
            assert "lm_head_weight" in loaded.main.weights
            assert loaded.main.weights["lm_head_weight"] is loaded.main.weights["model_embed_tokens_weight"]


@pytest.mark.unit
class TestSynthConstClassification:
    """P0-6: Synthesized constants (_const_*) should be classified as consts."""

    def test_synth_const_in_weight_classification(self) -> None:
        """Verify fx_to_mlir classifies non-param weights as consts."""
        from compiler.fx_to_mlir import fx_graph_to_mlir

        class FakeSpec:
            kind = type("k", (), {"value": -1})()  # not a weight kind
        class FakeSig:
            user_inputs = ["input_ids"]
            input_specs = [FakeSpec()]
        class FakeConst:
            pass
        class FakeProgram:
            graph_module = type("g", (), {"graph": type("g2", (), {"nodes": []})()})()
            state_dict = {}
            graph_signature = FakeSig()
            constants = FakeConst()

        program = FakeProgram()
        program.constants = {}  # No export constants
        program.graph_module.graph.nodes = []

        mod = fx_graph_to_mlir(program)
        assert len(mod.main.ops) == 0
        assert len(mod.main.weights) == 0
        assert len(mod.main.param_weight_names) == 0
        assert len(mod.main.const_weight_names) == 0  # no synth consts either


@pytest.mark.unit
class TestCandidateNames:
    """Verify suffix-based name matching for model prefix variations."""

    def test_simple_name_unchanged(self) -> None:
        from compiler.mlir_artifact import _candidate_names
        names = _candidate_names("model_embed_tokens_weight")
        assert "model_embed_tokens_weight" in names

    def test_qwen_prefix_stripped(self) -> None:
        """model_language_model_embed_tokens → model_embed_tokens via suffix."""
        from compiler.mlir_artifact import _candidate_names
        names = _candidate_names("model_language_model_embed_tokens_weight")
        assert "model_embed_tokens_weight" in names

    def test_full_suffix_chain(self) -> None:
        from compiler.mlir_artifact import _candidate_names
        names = _candidate_names("a_b_c_d_e")
        assert "a_b_c_d_e" in names
        assert "b_c_d_e" in names
        assert "c_d_e" in names
        assert "d_e" in names


@pytest.mark.unit
class TestMultiFunctionRoundtrip:
    """P0: verify MlirModule emit→parse roundtrip preserves multi-function outputs."""

    @pytest.mark.timeout(5)
    def test_roundtrip_preserves_function_outputs(self) -> None:
        from compiler.mlir_artifact import MlirFunction, MlirModule, MlirOp, _parse_mlir_text, mlir_module_to_text

        mod = MlirModule(functions=[
            MlirFunction(name="f0", inputs=[("%x", "tensor<2x4xf32>")],
                         outputs=[("%mid", "tensor<2x4xf32>")],
                         ops=[MlirOp(name="sf.add", dialect="sf", op_name="add",
                                     operands=["%x", "%x"], results=["%mid"],
                                     output_types=["tensor<2x4xf32>"], attributes={})]),
            MlirFunction(name="f1", inputs=[("%mid", "tensor<2x4xf32>")],
                         outputs=[("%out", "tensor<2x4xf32>")],
                         ops=[MlirOp(name="sf.mul", dialect="sf", op_name="mul",
                                     operands=["%mid", "%mid"], results=["%out"],
                                     output_types=["tensor<2x4xf32>"], attributes={})]),
        ])
        text = mlir_module_to_text(mod)
        reparsed = _parse_mlir_text(text)
        assert len(reparsed.functions) == 2
        for orig, parsed in zip(mod.functions, reparsed.functions, strict=True):
            assert len(parsed.outputs) == len(orig.outputs), \
                f"{parsed.name}: expected {len(orig.outputs)} outputs, got {len(parsed.outputs)}"

    @pytest.mark.timeout(5)
    def test_roundtrip_preserves_function_inputs(self) -> None:
        from compiler.mlir_artifact import MlirFunction, MlirModule, MlirOp, _parse_mlir_text, mlir_module_to_text

        mod = MlirModule(functions=[
            MlirFunction(name="f0", inputs=[("%in", "tensor<1x64xf32>")],
                         outputs=[("%out", "tensor<1x64xf32>")],
                         ops=[MlirOp(name="sf.relu", dialect="sf", op_name="relu",
                                     operands=["%in"], results=["%out"],
                                     output_types=["tensor<1x64xf32>"], attributes={})]),
        ])
        text = mlir_module_to_text(mod)
        reparsed = _parse_mlir_text(text)
        assert len(reparsed.functions) == 1
        assert len(reparsed.functions[0].inputs) == 1
        assert reparsed.functions[0].inputs[0][0] == "%in"
