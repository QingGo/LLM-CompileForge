import pytest
import torch

from compiler.ir import IrFunction, IrModule, IrOp, IrType
from engine.executor import Executor
from hal.pytorch_backend import PyTorchBackend

# ═══════════════════════════════════════════════════════════
# Minimal model IR for testing
# ═══════════════════════════════════════════════════════════


def _make_test_module() -> IrModule:
    """Create a minimal IrModule: x → gelu → output."""
    return IrModule(
        functions=[
            IrFunction(
                name="main",
                inputs=[("input_ids", IrType("int64", (None,)))],
                outputs=[("logits", IrType("float32", (None,)))],
                ops=[IrOp(name="gelu", inputs=["input_ids"], outputs=["logits"])],
            )
        ]
    )


def _make_add_module() -> IrModule:
    """Create an IrModule: x,y → add → output."""
    w = torch.tensor([1.0, 2.0, 3.0])
    return IrModule(
        functions=[
            IrFunction(
                name="main",
                inputs=[
                    ("x", IrType("float32", (None,))),
                    ("y", IrType("float32", (None,))),
                ],
                outputs=[("z", IrType("float32", (None,)))],
                ops=[IrOp(name="add", inputs=["x", "y"], outputs=["z"])],
                weights={"w": w},
            )
        ]
    )


# ═══════════════════════════════════════════════════════════
# Executor
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
class TestExecutor:
    def test_creation(self):
        mod = _make_test_module()
        backend = PyTorchBackend("cpu")
        executor = Executor(mod, backend)
        assert executor.function.name == "main"

    def test_forward_gelu(self):
        mod = _make_test_module()
        backend = PyTorchBackend("cpu")
        executor = Executor(mod, backend)

        input_ids = torch.randn(1, 4)
        output = executor.forward(input_ids)
        assert output.shape == input_ids.shape

    def test_forward_with_kwargs(self):
        mod = _make_add_module()
        backend = PyTorchBackend("cpu")
        executor = Executor(mod, backend)

        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([4.0, 5.0, 6.0])
        output = executor.forward(x, y=y)
        assert torch.allclose(output, x + y)

    def test_prepare_kv_blocks(self):
        mod = _make_test_module()
        backend = PyTorchBackend("cpu")
        executor = Executor(mod, backend)

        kv = executor.prepare_kv_blocks(
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            block_size=16,
            num_blocks=10,
        )
        assert kv.shape == (10, 32, 2, 16, 8, 128)
