"""Tests for multi-output op support and getitem resolution in fx_to_ir.py.

Phase 1.5 legacy fixes:
  1. IR multi-output op support — split_with_sizes/chunk → slice ops
  2. getitem resolution — tuple indexing vs sym_size fallback
  3. conv1d attribute extraction (stride, padding, dilation)
  4. torch.dtype argument handling
"""

from __future__ import annotations

import pytest
import torch

from compiler.fx_to_ir import _map_aten_op

# ═══════════════════════════════════════════════════════════
# _map_aten_op — mapping correctness (unit: no export needed)
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
class TestMapAtenOp:
    def test_getitem_not_mapped_to_sym_size(self):
        assert _map_aten_op("getitem") == "getitem"

    def test_aten_sym_size_still_maps_to_sym_size(self):
        assert _map_aten_op("aten.sym_size.int") == "sym_size"

    def test_split_with_sizes_maps_to_split(self):
        assert _map_aten_op("aten.split_with_sizes.default") == "split"

    def test_chunk_maps_to_chunk(self):
        assert _map_aten_op("aten.chunk.default") == "chunk"

    def test_conv1d_maps_to_conv1d(self):
        assert _map_aten_op("aten.conv1d.default") == "conv1d"

    def test_identity_maps_to_identity(self):
        assert _map_aten_op("aten.to.dtype") == "identity"


# ═══════════════════════════════════════════════════════════
# dtype promotion in matmul/linear/conv1d (unit: no export needed)
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
class TestDtypePromotion:
    """matmul, linear, conv1d should handle mixed dtype inputs gracefully."""

    def test_matmul_mixed_dtype(self):
        from hal.pytorch_backend import PyTorchBackend

        backend = PyTorchBackend("cpu")
        a = torch.randn(2, 4, dtype=torch.float32)
        b = torch.randn(4, 3, dtype=torch.bfloat16)
        result = backend._op_matmul([a, b])
        assert result.dtype == torch.float32
        assert result.shape == (2, 3)

    def test_linear_mixed_dtype(self):
        from hal.pytorch_backend import PyTorchBackend

        backend = PyTorchBackend("cpu")
        x = torch.randn(1, 4, dtype=torch.float32)
        w = torch.randn(8, 4, dtype=torch.bfloat16)
        result = backend._op_linear([x, w])
        assert result.dtype == torch.float32
        assert result.shape == (1, 8)

    def test_conv1d_mixed_dtype(self):
        from hal.pytorch_backend import PyTorchBackend

        backend = PyTorchBackend("cpu")
        x = torch.randn(1, 1, 8, dtype=torch.float32)
        w = torch.randn(1, 1, 3, dtype=torch.bfloat16)
        result = backend._op_conv1d([x, w], padding=1, groups=1)
        assert result.dtype == torch.float32
        assert result.shape == (1, 1, 8)


# ═══════════════════════════════════════════════════════════
# Integration tests — require torch.export (slower, >1s)
# ═══════════════════════════════════════════════════════════


@pytest.mark.integration
class TestConv1dArgsIntegration:
    """Verify that conv1d's stride, padding, dilation list args are extracted
    from a real torch.export FX graph."""

    def _run_compile(self, model, example_input):
        from compiler.fx_to_ir import fx_graph_to_ir

        program = torch.export.export(model, (example_input,))
        module = fx_graph_to_ir(program)
        return module.main

    def test_conv1d_with_padding(self):
        class Conv1dModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv1d(1, 1, kernel_size=4, padding=3)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.conv(x)

        model = Conv1dModel()
        example = torch.randn(1, 1, 10)
        func = self._run_compile(model, example)

        conv1d_ops = [op for op in func.ops if op.name == "conv1d"]
        assert len(conv1d_ops) >= 1
        attrs = conv1d_ops[0].attributes
        assert attrs.get("padding") == [3], f"padding: {attrs}"

    def test_conv1d_with_stride_and_groups(self):
        class Conv1dModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv1d(4, 4, kernel_size=3, stride=2, padding=0, groups=4)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.conv(x)

        model = Conv1dModel()
        example = torch.randn(1, 4, 16)
        func = self._run_compile(model, example)

        conv1d_ops = [op for op in func.ops if op.name == "conv1d"]
        assert len(conv1d_ops) >= 1
        attrs = conv1d_ops[0].attributes
        assert attrs.get("stride") == [2]
        assert attrs.get("padding") == [0]
        assert attrs.get("dilation") == [1]
        assert attrs.get("groups") == 4


@pytest.mark.integration
class TestSplitExpansionIntegration:
    """aten.split_with_sizes should be expanded to slice ops."""

    def _run_compile(self, model, example_input):
        from compiler.fx_to_ir import fx_graph_to_ir

        program = torch.export.export(model, (example_input,))
        module = fx_graph_to_ir(program)
        return module.main

    def test_split_with_sizes_expanded(self):
        class SplitModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                a, b = torch.split(x, [2, 2], dim=-1)
                return a + b

        model = SplitModel()
        example = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        func = self._run_compile(model, example)

        split_ops = [op for op in func.ops if op.name == "split"]
        slice_ops = [op for op in func.ops if op.name == "slice"]
        assert len(split_ops) == 0, "split should be expanded to slices"
        assert len(slice_ops) >= 2, "should have >= 2 slice ops"

    def test_getitem_resolved(self):
        class SplitModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                a, b = torch.split(x, [3, 3], dim=-1)
                return a + b

        model = SplitModel()
        example = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        func = self._run_compile(model, example)

        getitem_ops = [op for op in func.ops if op.name == "getitem"]
        sym_size_ops = [op for op in func.ops if op.name == "sym_size"]
        assert len(getitem_ops) == 0
        assert len(sym_size_ops) == 0


@pytest.mark.integration
class TestChunkExpansionIntegration:
    """aten.chunk should be expanded to slice ops."""

    def _run_compile(self, model, example_input):
        from compiler.fx_to_ir import fx_graph_to_ir

        program = torch.export.export(model, (example_input,))
        module = fx_graph_to_ir(program)
        return module.main

    def test_chunk_expanded(self):
        class ChunkModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                a, b = torch.chunk(x, 2, dim=-1)
                return torch.cat([a, b], dim=-1)

        model = ChunkModel()
        example = torch.randn(1, 4)
        func = self._run_compile(model, example)

        chunk_ops = [op for op in func.ops if op.name == "chunk"]
        assert len(chunk_ops) == 0, "chunk should be expanded to slices"

        getitem_ops = [op for op in func.ops if op.name == "getitem"]
        assert len(getitem_ops) == 0


@pytest.mark.integration
class TestFullPipelineIntegration:
    """End-to-end tests: compile + execute models with multi-output ops."""

    def test_split_model_forward(self):
        class SplitModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                a, b = torch.split(x, [2, 2], dim=-1)
                return a + b

        from compiler.pipeline import default_pipeline
        from engine.executor import Executor
        from hal.pytorch_backend import PyTorchBackend

        model = SplitModel().eval()
        example = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        ir_mod = default_pipeline().compile(model, (example,), emit_mlir=False)

        backend = PyTorchBackend("cpu")
        executor = Executor(ir_mod, backend)
        result = executor.forward(example)
        assert torch.allclose(result, torch.tensor([[4.0, 6.0]]), atol=1e-5)

    def test_chunk_model_forward(self):
        class ChunkModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                a, b, c = torch.chunk(x, 3, dim=-1)
                return a + b + c

        from compiler.pipeline import default_pipeline
        from engine.executor import Executor
        from hal.pytorch_backend import PyTorchBackend

        model = ChunkModel().eval()
        example = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        ir_mod = default_pipeline().compile(model, (example,), emit_mlir=False)

        backend = PyTorchBackend("cpu")
        executor = Executor(ir_mod, backend)
        result = executor.forward(example)
        assert torch.allclose(result, torch.tensor([[9.0, 12.0]]), atol=1e-5)

    def test_bf16_model_forward(self):
        class BFloat16Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(8, 4, dtype=torch.bfloat16)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)

        from compiler.pipeline import default_pipeline
        from engine.executor import Executor
        from hal.pytorch_backend import PyTorchBackend

        model = BFloat16Model().eval()
        example = torch.randn(1, 8, dtype=torch.float32)
        ir_mod = default_pipeline().compile(model, (example,), emit_mlir=False)

        backend = PyTorchBackend("cpu")
        executor = Executor(ir_mod, backend)
        logits = executor.forward(example)
        assert logits.shape == (1, 4)

    def test_shape_query_forward(self):
        class ShapeQueryModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                s = x.shape[1]
                return x * s

        from compiler.pipeline import default_pipeline
        from engine.executor import Executor
        from hal.pytorch_backend import PyTorchBackend

        model = ShapeQueryModel().eval()
        example = torch.ones(1, 2, 3)
        ir_mod = default_pipeline().compile(model, (example,), emit_mlir=False)

        backend = PyTorchBackend("cpu")
        executor = Executor(ir_mod, backend)
        result = executor.forward(example)
        assert result.shape == (1, 2, 3)
        assert torch.allclose(result, torch.ones(1, 2, 3) * 2.0, atol=1e-5)
