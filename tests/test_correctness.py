"""Correctness tests — compare compiled model logits against HuggingFace baseline.

Uses a simple deterministic IrModule for fast CI-level correctness checks.
For real model correctness, use tests/test_correctness_real_models.py.
"""

from __future__ import annotations

import pytest
import torch

from compiler.ir import IrFunction, IrModule, IrOp, IrType
from engine.executor import Executor
from hal.pytorch_backend import PyTorchBackend


def _make_matmul_model() -> IrModule:
    """Create a simple model: input → matmul(weight) → output."""
    weight = torch.randn(4, 8, dtype=torch.float32)
    return IrModule(
        functions=[
            IrFunction(
                name="main",
                inputs=[("x", IrType("float32", (None, 4)))],
                outputs=[("y", IrType("float32", (None, 8)))],
                ops=[IrOp(name="matmul", inputs=["x", "w"], outputs=["y"])],
                weights={"w": weight},
            )
        ]
    )


@pytest.mark.unit
class TestCorrectnessMatmul:
    def test_cosine_similarity_is_one(self):
        """Simple matmul should match exactly."""
        module = _make_matmul_model()
        backend = PyTorchBackend("cpu")
        executor = Executor(module, backend)

        x = torch.randn(2, 4)
        compiled_out = executor.forward(x)

        # Reference: direct matmul
        weight = module.main.weights["w"]
        ref_out = torch.matmul(x, weight)

        cos_sim = torch.nn.functional.cosine_similarity(
            compiled_out.flatten(), ref_out.flatten(), dim=0
        )
        assert cos_sim.item() > 0.9999
        assert torch.allclose(compiled_out, ref_out, atol=1e-6)


@pytest.mark.unit
class TestCorrectnessGelu:
    def test_gelu_matches_torch(self):
        """GeLU activation should match torch.nn.functional.gelu."""
        func = IrFunction(
            name="main",
            inputs=[("x", IrType("float32", (None,)))],
            outputs=[("y", IrType("float32", (None,)))],
            ops=[IrOp(name="gelu", inputs=["x"], outputs=["y"])],
        )
        module = IrModule(functions=[func])
        backend = PyTorchBackend("cpu")
        executor = Executor(module, backend)

        x = torch.randn(4, 8)
        compiled_out = executor.forward(x)

        import torch.nn.functional as F  # noqa: N812
        ref_out = F.gelu(x)

        cos_sim = torch.nn.functional.cosine_similarity(
            compiled_out.flatten(), ref_out.flatten(), dim=0
        )
        assert cos_sim.item() > 0.9999


@pytest.mark.unit
class TestCorrectnessRMSNorm:
    def test_rms_norm_matches_reference(self):
        """RMS norm should match the reference formula."""
        weight = torch.randn(8, dtype=torch.float32)
        func = IrFunction(
            name="main",
            inputs=[("x", IrType("float32", (None, 8)))],
            outputs=[("y", IrType("float32", (None, 8)))],
            ops=[IrOp(name="rms_norm", inputs=["x", "w"], outputs=["y"], attributes={"eps": 1e-5})],
            weights={"w": weight},
        )
        module = IrModule(functions=[func])
        backend = PyTorchBackend("cpu")
        executor = Executor(module, backend)

        x = torch.randn(2, 8)
        compiled_out = executor.forward(x)

        # Reference
        eps = 1e-5
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        ref_out = x * torch.rsqrt(variance + eps)
        ref_out = ref_out * weight

        assert torch.allclose(compiled_out, ref_out, atol=1e-5)


@pytest.mark.unit
class TestCorrectnessSoftmax:
    def test_softmax_matches_reference(self):
        """Softmax should match torch.nn.functional.softmax."""
        func = IrFunction(
            name="main",
            inputs=[("x", IrType("float32", (None,)))],
            outputs=[("y", IrType("float32", (None,)))],
            ops=[IrOp(name="softmax", inputs=["x"], outputs=["y"], attributes={"dim": -1})],
        )
        module = IrModule(functions=[func])
        backend = PyTorchBackend("cpu")
        executor = Executor(module, backend)

        x = torch.randn(2, 10)
        compiled_out = executor.forward(x)

        import torch.nn.functional as F  # noqa: N812
        ref_out = F.softmax(x, dim=-1)

        assert torch.allclose(compiled_out, ref_out, atol=1e-6)
