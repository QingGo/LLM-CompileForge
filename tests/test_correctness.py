"""Correctness tests — compare MlirExecutor outputs against reference.

Uses MlirModule for CI-level correctness checks.
"""

from __future__ import annotations

import pytest
import torch

from compiler.mlir_artifact import MlirFunction, MlirModule, MlirOp
from engine.mlir_executor import MlirExecutor
from hal.pytorch_backend import PyTorchBackend


@pytest.mark.unit
class TestCorrectnessMlirExecutor:
    """Correctness checks via MlirExecutor (MLIR-native path)."""

    def test_matmul_matches_reference(self):
        w = torch.randn(4, 8, dtype=torch.float32)
        mod = MlirModule(
            functions=[MlirFunction(
                name="main",
                inputs=[("%x", "tensor<?x4xf32>")],
                outputs=[("%y", "tensor<?x8xf32>", False)],
                ops=[MlirOp(
                    name="sf.matmul", dialect="sf", op_name="matmul",
                    operands=["%x", "w_tensor"], results=["%y"],
                )],
                weights={"w_tensor": w},
            )]
        )
        ex = MlirExecutor(mod, PyTorchBackend("cpu"))
        x = torch.randn(2, 4)
        out = ex.forward(x)
        ref = torch.matmul(x, w)
        assert torch.allclose(out, ref, atol=1e-6)

    def test_gelu_matches_torch(self):
        mod = MlirModule(
            functions=[MlirFunction(
                name="main",
                inputs=[("%x", "tensor<?x8xf32>")],
                outputs=[("%y", "tensor<?x8xf32>", False)],
                ops=[MlirOp(
                    name="sf.gelu", dialect="sf", op_name="gelu",
                    operands=["%x"], results=["%y"],
                )],
            )]
        )
        ex = MlirExecutor(mod, PyTorchBackend("cpu"))
        x = torch.randn(4, 8)
        out = ex.forward(x)
        ref = torch.nn.functional.gelu(x)
        assert torch.allclose(out, ref, atol=1e-6)

    def test_rms_norm_matches_reference(self):
        w = torch.randn(8, dtype=torch.float32)
        mod = MlirModule(
            functions=[MlirFunction(
                name="main",
                inputs=[("%x", "tensor<?x8xf32>")],
                outputs=[("%y", "tensor<?x8xf32>", False)],
                ops=[MlirOp(
                    name="sf.rms_norm", dialect="sf", op_name="rms_norm",
                    operands=["%x", "w"], results=["%y"],
                    attributes={"eps": 1e-5},
                )],
                weights={"w": w},
            )]
        )
        ex = MlirExecutor(mod, PyTorchBackend("cpu"))
        x = torch.randn(2, 8)
        out = ex.forward(x)
        eps = 1e-5
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        ref = x * torch.rsqrt(variance + eps) * w
        assert torch.allclose(out, ref, atol=1e-5)

    def test_softmax_matches_reference(self):
        mod = MlirModule(
            functions=[MlirFunction(
                name="main",
                inputs=[("%x", "tensor<?x10xf32>")],
                outputs=[("%y", "tensor<?x10xf32>", False)],
                ops=[MlirOp(
                    name="sf.softmax", dialect="sf", op_name="softmax",
                    operands=["%x"], results=["%y"],
                    attributes={"dim": -1},
                )],
            )]
        )
        ex = MlirExecutor(mod, PyTorchBackend("cpu"))
        x = torch.randn(2, 10)
        out = ex.forward(x)
        ref = torch.nn.functional.softmax(x, dim=-1)
        assert torch.allclose(out, ref, atol=1e-6)
