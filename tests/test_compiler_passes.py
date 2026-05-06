import pytest
import torch

from compiler.ir import IrFunction, IrModule, IrOp
from compiler.passes.base import Pass, PassManager
from compiler.passes.constant_fold import ConstantFold
from compiler.passes.cse_pass import CommonSubexpressionElimination
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
        # add should be replaced by constant
        assert result.main.ops[0].name == "constant"
        # Check folded value
        folded_name = result.main.ops[0].inputs[0]
        expected = w1 + w2
        assert torch.equal(result.main.weights[folded_name], expected)

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
