import pytest
import torch

from compiler.ir import IrFunction, IrModule, IrOp
from compiler.passes.base import Pass, PassManager
from compiler.passes.constant_fold import ConstantFold
from compiler.passes.cse_pass import CommonSubexpressionElimination
from compiler.passes.dce_pass import DeadCodeElimination
from compiler.passes.fuse_qkv import FuseQKVProjection
from compiler.passes.fuse_rms_norm import FuseRMSNorm
from compiler.passes.fuse_silu import FuseSiLU

# ═══════════════════════════════════════════════════════════
# PassManager
# ═══════════════════════════════════════════════════════════


class _NoopPass(Pass):
    def apply(self, module: IrModule) -> IrModule:
        module.metadata["noop_called"] = True
        return module


@pytest.mark.unit
class TestPassManager:
    def test_empty_pipeline(self):
        pm = PassManager()
        mod = IrModule(metadata={"key": "val"})
        result = pm.run(mod)
        assert result.metadata["key"] == "val"
        assert pm.num_passes == 0

    def test_single_pass(self):
        pm = PassManager()
        pm.add(_NoopPass())
        mod = IrModule()
        result = pm.run(mod)
        assert result.metadata.get("noop_called") is True

    def test_passes_applied_recorded(self):
        pm = PassManager()
        pm.add(_NoopPass())
        mod = IrModule()
        result = pm.run(mod)
        assert "passes_applied" in result.metadata
        assert "_NoopPass" in result.metadata["passes_applied"]


# ═══════════════════════════════════════════════════════════
# CommonSubexpressionElimination
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
class TestCSE:
    def test_no_duplicates(self):
        func = IrFunction(
            name="main",
            ops=[
                IrOp(name="add", inputs=["a", "b"], outputs=["c"]),
                IrOp(name="mul", inputs=["c", "d"], outputs=["e"]),
            ],
        )
        mod = IrModule(functions=[func])
        cse = CommonSubexpressionElimination()
        result = cse.apply(mod)
        assert len(result.main.ops) == 2

    def test_eliminates_duplicate(self):
        func = IrFunction(
            name="main",
            ops=[
                IrOp(name="add", inputs=["a", "b"], outputs=["c"]),
                IrOp(name="add", inputs=["a", "b"], outputs=["d"]),  # Duplicate
                IrOp(name="mul", inputs=["c", "d"], outputs=["e"]),
            ],
        )
        mod = IrModule(functions=[func])
        cse = CommonSubexpressionElimination()
        result = cse.apply(mod)
        assert len(result.main.ops) == 2
        # The mul op should use the same output for both inputs
        mul_op = result.main.ops[1]
        assert mul_op.inputs[0] == mul_op.inputs[1]


# ═══════════════════════════════════════════════════════════
# FuseRMSNorm
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
class TestFuseRMSNorm:
    def test_no_pattern_no_change(self):
        func = IrFunction(
            name="main",
            ops=[
                IrOp(name="rms_norm", inputs=["x"], outputs=["n"]),
                IrOp(name="add", inputs=["n", "b"], outputs=["y"]),
            ],
        )
        mod = IrModule(functions=[func])
        fuse = FuseRMSNorm()
        result = fuse.apply(mod)
        # Neither rms_norm nor add match the fusion pattern
        assert len(result.main.ops) == 2

    def test_fuses_rms_norm_matmul_chain(self):
        func = IrFunction(
            name="main",
            ops=[
                IrOp(name="rms_norm", inputs=["x"], outputs=["n"]),
                IrOp(name="mul", inputs=["n", "g"], outputs=["normed"]),
                IrOp(name="matmul", inputs=["normed", "w"], outputs=["y"]),
            ],
        )
        mod = IrModule(functions=[func])
        fuse = FuseRMSNorm()
        result = fuse.apply(mod)
        assert len(result.main.ops) == 1
        assert result.main.ops[0].name == "fused_rms_norm_matmul"


# ═══════════════════════════════════════════════════════════
# FuseSiLU
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
class TestFuseSiLU:
    def test_no_pattern_no_change(self):
        func = IrFunction(
            name="main",
            ops=[
                IrOp(name="silu", inputs=["x"], outputs=["a"]),
                IrOp(name="add", inputs=["a", "b"], outputs=["c"]),
            ],
        )
        mod = IrModule(functions=[func])
        fuse = FuseSiLU()
        result = fuse.apply(mod)
        assert len(result.main.ops) == 2

    def test_fuses_silu_mul_chain(self):
        func = IrFunction(
            name="main",
            ops=[
                IrOp(name="silu", inputs=["x"], outputs=["a"]),
                IrOp(name="mul", inputs=["a", "y"], outputs=["z"]),
            ],
        )
        mod = IrModule(functions=[func])
        fuse = FuseSiLU()
        result = fuse.apply(mod)
        assert len(result.main.ops) == 1
        assert result.main.ops[0].name == "fused_silu_mul"



# ═══════════════════════════════════════════════════════════
# FuseQKVProjection
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
class TestFuseQKVProjection:
    def test_no_pattern_no_change(self):
        func = IrFunction(
            name="main",
            ops=[
                IrOp(name="linear", inputs=["x", "w_q"], outputs=["q"]),
                IrOp(name="add", inputs=["q", "b"], outputs=["y"]),
            ],
            weights={"w_q": torch.randn(4, 8)},
        )
        mod = IrModule(functions=[func])
        fuse = FuseQKVProjection()
        result = fuse.apply(mod)
        assert len(result.main.ops) == 2

    def test_fuses_qkv_linear(self):
        w_q = torch.randn(8, 16)
        w_k = torch.randn(4, 16)
        w_v = torch.randn(4, 16)
        func = IrFunction(
            name="main",
            ops=[
                IrOp(name="linear", inputs=["x", "q_proj_w"], outputs=["q"]),
                IrOp(name="linear", inputs=["x", "k_proj_w"], outputs=["k"]),
                IrOp(name="linear", inputs=["x", "v_proj_w"], outputs=["v"]),
                IrOp(name="add", inputs=["q", "k"], outputs=["qk"]),
            ],
            weights={"q_proj_w": w_q, "k_proj_w": w_k, "v_proj_w": w_v},
        )
        mod = IrModule(functions=[func])
        fuse = FuseQKVProjection()
        result = fuse.apply(mod)
        ops = result.main.ops
        assert len(ops) == 5
        fused_op = ops[0]
        assert fused_op.name == "linear"
        assert fused_op.inputs[0] == "x"
        assert fused_op.inputs[1].startswith("__fused_qkv_w_")
        assert fused_op.outputs[0].startswith("__fused_qkv_out_")
        assert ops[1].name == "slice"
        assert ops[1].outputs == ["q"]
        assert ops[2].name == "slice"
        assert ops[2].outputs == ["k"]
        assert ops[3].name == "slice"
        assert ops[3].outputs == ["v"]
        assert ops[4].name == "add"
        fused_weight = result.main.weights[fused_op.inputs[1]]
        assert fused_weight.shape == (16, 16)
        assert torch.equal(fused_weight[:8], w_q)
        assert torch.equal(fused_weight[8:12], w_k)
        assert torch.equal(fused_weight[12:], w_v)

    def test_fuses_qkv_with_bias(self):
        w_q = torch.randn(8, 16)
        w_k = torch.randn(4, 16)
        w_v = torch.randn(4, 16)
        b_q = torch.randn(8)
        b_k = torch.randn(4)
        b_v = torch.randn(4)
        func = IrFunction(
            name="main",
            ops=[
                IrOp(name="linear", inputs=["x", "q_proj_w", "b_q"], outputs=["q"]),
                IrOp(name="linear", inputs=["x", "k_proj_w", "b_k"], outputs=["k"]),
                IrOp(name="linear", inputs=["x", "v_proj_w", "b_v"], outputs=["v"]),
            ],
            weights={"q_proj_w": w_q, "k_proj_w": w_k, "v_proj_w": w_v, "b_q": b_q, "b_k": b_k, "b_v": b_v},
        )
        mod = IrModule(functions=[func])
        fuse = FuseQKVProjection()
        result = fuse.apply(mod)
        ops = result.main.ops
        assert len(ops) == 4
        fused_op = ops[0]
        assert len(fused_op.inputs) == 3
        fused_b_name = fused_op.inputs[2]
        fused_b = result.main.weights[fused_b_name]
        assert fused_b.shape == (16,)

    def test_fuses_two_projections(self):
        w_k = torch.randn(4, 16)
        w_v = torch.randn(4, 16)
        func = IrFunction(
            name="main",
            ops=[
                IrOp(name="linear", inputs=["x", "k_proj_w"], outputs=["k"]),
                IrOp(name="linear", inputs=["x", "v_proj_w"], outputs=["v"]),
            ],
            weights={"k_proj_w": w_k, "v_proj_w": w_v},
        )
        mod = IrModule(functions=[func])
        fuse = FuseQKVProjection()
        result = fuse.apply(mod)
        assert len(result.main.ops) == 3
        assert result.main.ops[0].name == "linear"
        assert result.main.ops[1].outputs == ["k"]
        assert result.main.ops[2].outputs == ["v"]

    def test_no_fuse_unrelated_matmul(self):
        w_a = torch.randn(8, 16)
        w_b = torch.randn(4, 16)
        func = IrFunction(
            name="main",
            ops=[
                IrOp(name="linear", inputs=["x", "w_a"], outputs=["a"]),
                IrOp(name="linear", inputs=["x", "w_b"], outputs=["b"]),
            ],
            weights={"w_a": w_a, "w_b": w_b},
        )
        mod = IrModule(functions=[func])
        fuse = FuseQKVProjection()
        result = fuse.apply(mod)
        assert len(result.main.ops) == 2

    def test_no_fuse_different_inputs(self):
        w_q = torch.randn(8, 16)
        w_k = torch.randn(4, 16)
        func = IrFunction(
            name="main",
            ops=[
                IrOp(name="linear", inputs=["x", "w_q"], outputs=["q"]),
                IrOp(name="linear", inputs=["y", "w_k"], outputs=["k"]),
            ],
            weights={"w_q": w_q, "w_k": w_k},
        )
        mod = IrModule(functions=[func])
        fuse = FuseQKVProjection()
        result = fuse.apply(mod)
        assert len(result.main.ops) == 2

    def test_slice_attributes_correct(self):
        w_q = torch.randn(8, 16)
        w_k = torch.randn(4, 16)
        w_v = torch.randn(4, 16)
        func = IrFunction(
            name="main",
            ops=[
                IrOp(name="linear", inputs=["x", "q_proj_w"], outputs=["q"]),
                IrOp(name="linear", inputs=["x", "k_proj_w"], outputs=["k"]),
                IrOp(name="linear", inputs=["x", "v_proj_w"], outputs=["v"]),
            ],
            weights={"q_proj_w": w_q, "k_proj_w": w_k, "v_proj_w": w_v},
        )
        mod = IrModule(functions=[func])
        fuse = FuseQKVProjection()
        result = fuse.apply(mod)
        slices = [op for op in result.main.ops if op.name == "slice"]
        assert len(slices) == 3
        assert slices[0].attributes["start"] == 0
        assert slices[0].attributes["end"] == 8
        assert slices[1].attributes["start"] == 8
        assert slices[1].attributes["end"] == 12
        assert slices[2].attributes["start"] == 12
        assert slices[2].attributes["end"] == 16

    def test_fuses_matmul_ops(self):
        w_q = torch.randn(8, 16)
        w_k = torch.randn(4, 16)
        func = IrFunction(
            name="main",
            ops=[
                IrOp(name="matmul", inputs=["x", "q_proj_w"], outputs=["q"]),
                IrOp(name="matmul", inputs=["x", "k_proj_w"], outputs=["k"]),
            ],
            weights={"q_proj_w": w_q, "k_proj_w": w_k},
        )
        mod = IrModule(functions=[func])
        fuse = FuseQKVProjection()
        result = fuse.apply(mod)
        assert result.main.ops[0].name == "matmul"

    def test_alternate_naming_tokens(self):
        w_query = torch.randn(4, 16)
        w_key = torch.randn(4, 16)
        w_value = torch.randn(4, 16)
        func = IrFunction(
            name="main",
            ops=[
                IrOp(name="linear", inputs=["x", "w_query"], outputs=["q"]),
                IrOp(name="linear", inputs=["x", "w_key"], outputs=["k"]),
                IrOp(name="linear", inputs=["x", "w_value"], outputs=["v"]),
            ],
            weights={"w_query": w_query, "w_key": w_key, "w_value": w_value},
        )
        mod = IrModule(functions=[func])
        fuse = FuseQKVProjection()
        result = fuse.apply(mod)
        assert len(result.main.ops) == 4

    def test_q_linear_k_linear_tokens(self):
        w_q = torch.randn(4, 16)
        w_k = torch.randn(4, 16)
        func = IrFunction(
            name="main",
            ops=[
                IrOp(name="linear", inputs=["x", "w_q_linear"], outputs=["q"]),
                IrOp(name="linear", inputs=["x", "w_k_linear"], outputs=["k"]),
            ],
            weights={"w_q_linear": w_q, "w_k_linear": w_k},
        )
        mod = IrModule(functions=[func])
        fuse = FuseQKVProjection()
        result = fuse.apply(mod)
        assert len(result.main.ops) == 3



# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
class TestConstantFold:
    def test_folds_add_of_constants(self):
        w1 = torch.tensor([1.0, 2.0])
        w2 = torch.tensor([3.0, 4.0])
        func = IrFunction(
            name="main",
            ops=[IrOp(name="add", inputs=["w1", "w2"], outputs=["y"])],
            weights={"w1": w1, "w2": w2},
        )
        mod = IrModule(functions=[func])
        cf = ConstantFold()
        result = cf.apply(mod)
        assert result.main.ops[0].name == "constant"
        # Check folded value
        folded_name = result.main.ops[0].inputs[0]
        expected = w1 + w2
        assert torch.equal(result.main.weights[folded_name], expected)


# ═══════════════════════════════════════════════════════════
# DeadCodeElimination
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
class TestDCE:
    def test_keeps_live_ops(self):
        func = IrFunction(
            name="main",
            inputs=[("x", None)],  # type: ignore[arg-type]
            outputs=[("z", None)],  # type: ignore[arg-type]
            ops=[
                IrOp(name="add", inputs=["x", "y"], outputs=["z"]),
            ],
        )
        mod = IrModule(functions=[func])
        dce = DeadCodeElimination()
        result = dce.apply(mod)
        assert len(result.main.ops) == 1

    def test_eliminates_dead_op(self):
        func = IrFunction(
            name="main",
            inputs=[("x", None)],  # type: ignore[arg-type]
            outputs=[("z", None)],  # type: ignore[arg-type]
            ops=[
                IrOp(name="add", inputs=["a", "b"], outputs=["dead"]),
                IrOp(name="mul", inputs=["x", "w"], outputs=["z"]),
            ],
        )
        mod = IrModule(functions=[func])
        dce = DeadCodeElimination()
        result = dce.apply(mod)
        assert len(result.main.ops) == 1
        assert result.main.ops[0].name == "mul"

    def test_transitive_dead_chain(self):
        func = IrFunction(
            name="main",
            inputs=[("x", None)],  # type: ignore[arg-type]
            outputs=[("z", None)],  # type: ignore[arg-type]
            ops=[
                IrOp(name="add1", inputs=["a", "b"], outputs=["dead1"]),
                IrOp(name="add2", inputs=["dead1", "c"], outputs=["dead2"]),
                IrOp(name="mul", inputs=["x", "w"], outputs=["z"]),
            ],
        )
        mod = IrModule(functions=[func])
        dce = DeadCodeElimination()
        result = dce.apply(mod)
        assert len(result.main.ops) == 1
        assert result.main.ops[0].name == "mul"

    def test_input_consumed_by_output_is_live(self):
        func = IrFunction(
            name="main",
            inputs=[("x", None)],  # type: ignore[arg-type]
            outputs=[("live", None)],  # type: ignore[arg-type]
            ops=[
                IrOp(name="add", inputs=["a", "b"], outputs=["dead"]),
                IrOp(name="mul", inputs=["dead", "x"], outputs=["live"]),
            ],
        )
        mod = IrModule(functions=[func])
        dce = DeadCodeElimination()
        result = dce.apply(mod)
        # "add" produces "dead" which IS consumed by mul
        # mul's output "live" IS a function output → live
        # Therefore "add" is live transitively through mul
        assert len(result.main.ops) == 2

    def test_does_not_fold_runtime_input(self):
        w = torch.tensor([1.0])
        func = IrFunction(
            name="main",
            ops=[IrOp(name="add", inputs=["x", "w"], outputs=["y"])],
            weights={"w": w},
        )
        mod = IrModule(functions=[func])
        cf = ConstantFold()
        result = cf.apply(mod)
        # x is not a constant, so add should NOT be folded
        assert result.main.ops[0].name == "add"

    def test_folds_matmul_of_constants(self):
        w1 = torch.randn(2, 3)
        w2 = torch.randn(3, 4)
        func = IrFunction(
            name="main",
            ops=[IrOp(name="matmul", inputs=["w1", "w2"], outputs=["y"])],
            weights={"w1": w1, "w2": w2},
        )
        mod = IrModule(functions=[func])
        cf = ConstantFold()
        result = cf.apply(mod)
        assert result.main.ops[0].name == "constant"


# ═══════════════════════════════════════════════════════════
# ValidateIR
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
class TestValidateIR:
    def test_valid_module_passes(self):
        from compiler.passes.validate_ir import ValidateIR
        func = IrFunction(
            name="main",
            inputs=[("x", None)],  # type: ignore[arg-type]
            outputs=[("y", None)],  # type: ignore[arg-type]
            ops=[
                IrOp(name="add", inputs=["x", "w"], outputs=["y"]),
            ],
            weights={"w": torch.ones(3)},
        )
        mod = IrModule(functions=[func])
        ValidateIR().apply(mod)  # Should not raise

    def test_undefined_input_raises(self):
        from compiler.passes.validate_ir import IRValidationError, ValidateIR
        func = IrFunction(
            name="main",
            ops=[
                IrOp(name="add", inputs=["x", "missing_name"], outputs=["y"]),
            ],
        )
        mod = IrModule(functions=[func])
        with pytest.raises(IRValidationError, match="Undefined SSA inputs"):
            ValidateIR().apply(mod)

    def test_duplicate_output_raises(self):
        from compiler.passes.validate_ir import IRValidationError, ValidateIR
        func = IrFunction(
            name="main",
            ops=[
                IrOp(name="add", inputs=["a", "b"], outputs=["y"]),
                IrOp(name="mul", inputs=["y", "c"], outputs=["y"]),
            ],
            weights={"a": torch.ones(1), "b": torch.ones(1), "c": torch.ones(1)},
        )
        mod = IrModule(functions=[func])
        with pytest.raises(IRValidationError, match="Duplicate SSA output names"):
            ValidateIR().apply(mod)

    def test_missing_output_raises(self):
        from compiler.passes.validate_ir import IRValidationError, ValidateIR
        func = IrFunction(
            name="main",
            outputs=[("no_such_name", None)],  # type: ignore[arg-type]
            ops=[
                IrOp(name="add", inputs=["a", "b"], outputs=["y"]),
            ],
            weights={"a": torch.ones(1), "b": torch.ones(1)},
        )
        mod = IrModule(functions=[func])
        with pytest.raises(IRValidationError, match="has no producer"):
            ValidateIR().apply(mod)

    def test_input_as_output_passes(self):
        from compiler.passes.validate_ir import ValidateIR
        func = IrFunction(
            name="main",
            inputs=[("x", None)],  # type: ignore[arg-type]
            outputs=[("x", None)],  # type: ignore[arg-type]
            ops=[],
        )
        mod = IrModule(functions=[func])
        ValidateIR().apply(mod)  # Should not raise


# ═══════════════════════════════════════════════════════════
# PassManager immutability
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
class TestPassManagerImmutability:
    def test_original_module_not_mutated(self):
        func = IrFunction(
            name="main",
            ops=[
                IrOp(name="add", inputs=["a", "b"], outputs=["y"]),
            ],
            weights={"a": torch.ones(1), "b": torch.ones(1)},
        )
        original = IrModule(functions=[func])
        original_ops_count = len(original.main.ops)

        pm = PassManager()
        pm.add(DeadCodeElimination())
        result = pm.run(original)

        assert len(original.main.ops) == original_ops_count
        assert result is not original
        assert result.main.ops is not original.main.ops

    def test_structural_copy_preserves_weights(self):
        from compiler.passes.base import _structural_copy
        w = torch.ones(3)
        func = IrFunction(
            name="main",
            ops=[],
            weights={"w": w},
        )
        mod = IrModule(functions=[func])
        copymod = _structural_copy(mod)

        assert copymod.main.weights["w"] is mod.main.weights["w"]
        assert copymod.main.ops is not mod.main.ops
