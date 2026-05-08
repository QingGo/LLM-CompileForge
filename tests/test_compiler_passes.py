import pytest
import torch

from compiler.ir import IrFunction, IrModule, IrOp
from compiler.passes.base import Pass, PassManager
from compiler.passes.constant_fold import ConstantFold
from compiler.passes.cse_pass import CommonSubexpressionElimination
from compiler.passes.dce_pass import DeadCodeElimination
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
# ConstantFold
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
