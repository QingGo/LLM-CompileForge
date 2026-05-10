"""Tests for MlirExecutor — MLIR-native execution path.

Mirrors test_executor.py patterns but uses MlirModule/MlirOp instead of
IrModule/IrOp.  KV cache tests are shared via the _KVCacheMixin;
this file focuses on MLIR-specific behaviour (SSA parsing, operand
lookup, weight constants).
"""

from __future__ import annotations

import pytest
import torch

from compiler.mlir_artifact import MlirFunction, MlirModule, MlirOp
from engine.mlir_executor import MlirExecutor
from hal.pytorch_backend import PyTorchBackend


def _make_mlir_module() -> MlirModule:
    """Minimal MlirModule: %input → sf.gelu → %output."""
    return MlirModule(
        functions=[
            MlirFunction(
                name="main",
                inputs=[("%input", "tensor<?xi32>")],
                outputs=[("%output", "tensor<?xf32>")],
                ops=[
                    MlirOp(
                        name="sf.gelu",
                        dialect="sf",
                        op_name="gelu",
                        operands=["%input"],
                        results=["%output"],
                    ),
                ],
            )
        ]
    )


def _make_linear_module() -> MlirModule:
    """MlirModule: %x,%w → sf.linear → %z."""
    return MlirModule(
        functions=[
            MlirFunction(
                name="main",
                inputs=[("%x", "tensor<?x4xf32>"), ("%w", "tensor<4x4xf32>")],
                outputs=[("%z", "tensor<?x4xf32>")],
                ops=[
                    MlirOp(
                        name="sf.linear",
                        dialect="sf",
                        op_name="linear",
                        operands=["%x", "%w"],
                        results=["%z"],
                    ),
                ],
                weights={"%w": torch.randn(4, 4)},
            )
        ]
    )


@pytest.mark.unit
class TestMlirExecutorKV:
    """KV cache tests for MlirExecutor — verifies shared _KVCacheMixin."""

    @staticmethod
    def _make_executor():
        mod = _make_mlir_module()
        backend = PyTorchBackend("cpu")
        return MlirExecutor(mod, backend)

    def test_prepare_kv_blocks_shape(self):
        ex = self._make_executor()
        kv = ex.prepare_kv_blocks(num_layers=32, num_kv_heads=8, head_dim=128, block_size=16, num_blocks=10)
        assert kv.shape == (10, 32, 2, 16, 8, 128)

    def test_write_and_gather_kv(self):
        ex = self._make_executor()
        kv = ex.prepare_kv_blocks(
            num_layers=4, num_kv_heads=2, head_dim=4, block_size=4, num_blocks=8, dtype=torch.float32
        )
        ex.set_kv_cache(kv, block_tables={"r0": [0, 1]}, num_kv_heads=2, head_dim=4, block_size=4)

        k_data = torch.randn(4, 2, 4)
        v_data = torch.randn(4, 2, 4)
        ex.write_kv_to_cache(k_data, v_data, torch.tensor([0, 1, 2, 3]), {"r0": [0, 1]}, layer_idx=0)

        k_gathered, v_gathered = ex.gather_kv_from_cache({"r0": [0, 1]}, max_seq_len=4, layer_idx=0)
        assert k_gathered.shape == (1, 4, 2, 4)
        assert torch.allclose(k_gathered, k_data.unsqueeze(0))
        assert torch.allclose(v_gathered, v_data.unsqueeze(0))

    def test_write_layer_isolation(self):
        ex = self._make_executor()
        kv = ex.prepare_kv_blocks(
            num_layers=4, num_kv_heads=2, head_dim=4, block_size=4, num_blocks=8, dtype=torch.float32
        )
        ex.set_kv_cache(kv, block_tables={"r0": [0]}, num_kv_heads=2, head_dim=4, block_size=4)

        k0 = torch.ones(4, 2, 4)
        k1 = torch.ones(4, 2, 4) * 2.0
        v0 = torch.zeros(4, 2, 4)
        ex.write_kv_to_cache(k0, v0, torch.tensor([0, 1, 2, 3]), {"r0": [0]}, layer_idx=0)
        ex.write_kv_to_cache(k1, v0, torch.tensor([0, 1, 2, 3]), {"r0": [0]}, layer_idx=1)

        kg0, _ = ex.gather_kv_from_cache({"r0": [0]}, max_seq_len=4, layer_idx=0)
        kg1, _ = ex.gather_kv_from_cache({"r0": [0]}, max_seq_len=4, layer_idx=1)
        assert torch.allclose(kg0, torch.ones(1, 4, 2, 4))
        assert torch.allclose(kg1, torch.ones(1, 4, 2, 4) * 2.0)


@pytest.mark.unit
class TestMlirExecutorForward:
    """Forward pass tests — verify MlirExecutor dispatches through HAL."""

    def test_forward_gelu(self):
        mod = _make_mlir_module()
        ex = MlirExecutor(mod, PyTorchBackend("cpu"))
        inp = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
        result = ex.forward(inp)
        assert result.shape == inp.shape
        assert result[1] == 0.0  # GELU(0) = 0
        assert result[2] > 0.5  # GELU(1) ≈ 0.84

    def test_forward_linear(self):
        mod = _make_linear_module()
        ex = MlirExecutor(mod, PyTorchBackend("cpu"))
        x = torch.randn(1, 4)
        result = ex.forward(x)
        assert result.shape == (1, 4)

    def test_forward_weight_constant(self):
        """Weight constant op returns tensor, downstream op uses it."""
        w = torch.randn(3, 3)
        mod = MlirModule(
            functions=[
                MlirFunction(
                    name="main",
                    inputs=[("%x", "tensor<?x3xf32>")],
                    outputs=[("%out", "tensor<?x3xf32>")],
                    ops=[
                        MlirOp(
                            name="sf.weight",
                            dialect="sf",
                            op_name="weight",
                            operands=["w"],
                            results=["%w_val"],
                            attributes={"name": "w"},
                        ),
                        MlirOp(
                            name="sf.matmul",
                            dialect="sf",
                            op_name="matmul",
                            operands=["%x", "%w_val"],
                            results=["%out"],
                        ),
                    ],
                    weights={"w": w},
                )
            ]
        )
        ex = MlirExecutor(mod, PyTorchBackend("cpu"))
        inp = torch.randn(1, 3)
        result = ex.forward(inp)
        assert result.shape == (1, 3)

    def test_forward_with_kv_returns_tuple(self):
        mod = _make_mlir_module()
        ex = MlirExecutor(mod, PyTorchBackend("cpu"))
        inp = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
        logits, kv_tensors = ex.forward_with_kv(inp)
        assert logits is not None
        assert isinstance(kv_tensors, list)


@pytest.mark.unit
class TestMlirExecutorInputHandling:
    """SSA operand resolution tests — MLIR-specific behaviour."""

    def test_ssa_with_percent_prefix(self):
        w = torch.ones(3, 3)
        mod = MlirModule(
            functions=[
                MlirFunction(
                    name="main",
                    inputs=[("%x", "tensor<?x3xf32>")],
                    outputs=[("%out", "tensor<?x3xf32>")],
                    ops=[
                        MlirOp(
                            name="sf.matmul",
                            dialect="sf",
                            op_name="matmul",
                            operands=["%x", "weight_mat"],
                            results=["%out"],
                        ),
                    ],
                    weights={"weight_mat": w},
                )
            ]
        )
        ex = MlirExecutor(mod, PyTorchBackend("cpu"))
        x = torch.tensor([[1.0, 2.0, 3.0]])
        result = ex.forward(x)
        assert result.shape == (1, 3)

    def test_input_lookup_without_percent(self):
        """Verify input IDs stored without % can be found when operand has %."""
        mod = MlirModule(
            functions=[
                MlirFunction(
                    name="main",
                    inputs=[("x", "tensor<?xf32>"), ("y", "tensor<?xf32>")],
                    outputs=[("%z", "tensor<?xf32>")],
                    ops=[
                        MlirOp(
                            name="sf.add",
                            dialect="sf",
                            op_name="add",
                            operands=["%x", "%y"],
                            results=["%z"],
                        ),
                    ],
                )
            ]
        )
        ex = MlirExecutor(mod, PyTorchBackend("cpu"))
        x = torch.tensor([1.0, 2.0])
        result = ex.forward(x, y=torch.tensor([3.0, 4.0]))
        assert torch.allclose(result, torch.tensor([4.0, 6.0]))

    def test_unknown_operand_raises(self):
        mod = MlirModule(
            functions=[
                MlirFunction(
                    name="main",
                    inputs=[("%x", "tensor<?xf32>")],
                    outputs=[("%out", "tensor<?xf32>")],
                    ops=[
                        MlirOp(
                            name="sf.missing_op",
                            dialect="sf",
                            op_name="missing_op",
                            operands=["%x"],
                            results=["%out"],
                        ),
                    ],
                )
            ]
        )
        ex = MlirExecutor(mod, PyTorchBackend("cpu"))
        with pytest.raises(ValueError, match="Unknown op"):
            ex.forward(torch.tensor([1.0]))


@pytest.mark.unit
class TestMlirExecutorOutputHandling:
    """Output resolution — fallback when explicit output SSA is missing."""

    def test_explicit_output_used(self):
        mod = _make_mlir_module()
        ex = MlirExecutor(mod, PyTorchBackend("cpu"))
        inp = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
        result = ex.forward(inp)
        assert result.shape == (3,)

    def test_last_op_output_as_fallback(self):
        """When no explicit output list, use last op's last result."""
        mod = MlirModule(
            functions=[
                MlirFunction(
                    name="main",
                    inputs=[("%x", "tensor<?xf32>")],
                    outputs=[],
                    ops=[
                        MlirOp(
                            name="sf.gelu",
                            dialect="sf",
                            op_name="gelu",
                            operands=["%x"],
                            results=["%out1", "%out2"],
                        ),
                    ],
                )
            ]
        )
        ex = MlirExecutor(mod, PyTorchBackend("cpu"))
        inp = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
        result = ex.forward(inp)
        assert result.shape == (3,)

    def test_empty_module_returns_empty_tensor(self):
        mod = MlirModule(functions=[])
        with pytest.raises(ValueError, match="no functions"):
            _ = mod.main

    def test_forward_with_kv_fallback_no_outputs(self):
        """P0-7: forward_with_kv should fallback to last op result when no outputs declared."""
        mod = MlirModule(
            functions=[MlirFunction(
                name="main", inputs=[("%x", "tensor<?xf32>")], outputs=[],
                ops=[MlirOp(name="sf.gelu", dialect="sf", op_name="gelu",
                             operands=["%x"], results=["%out"])],
            )]
        )
        ex = MlirExecutor(mod, PyTorchBackend("cpu"))
        logits, kv = ex.forward_with_kv(torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32))
        assert logits.numel() > 0, "logits should not be empty (fallback worked)"


@pytest.mark.unit
class TestStaticShapeDetection:
    """Verify static vs dynamic shape detection for cache strategy."""

    def test_dynamic_shape_detected(self) -> None:
        mod = MlirModule(functions=[MlirFunction(
            name="main", inputs=[("%x", "tensor<?x16xi64>")], outputs=[], ops=[]
        )])
        ex = MlirExecutor(mod, PyTorchBackend("cpu"))
        assert not ex._uses_static_shape

    def test_static_shape_detected(self) -> None:
        mod = MlirModule(functions=[MlirFunction(
            name="main", inputs=[("%x", "tensor<1x64xi64>")], outputs=[], ops=[]
        )])
        ex = MlirExecutor(mod, PyTorchBackend("cpu"))
        assert ex._uses_static_shape

    def test_cache_manager_disabled_for_static(self) -> None:
        from compiler.cache_policy import CachePolicy
        mod = MlirModule(functions=[MlirFunction(
            name="main", inputs=[("%x", "tensor<1x64xi64>")], outputs=[], ops=[]
        )], metadata={"cache_policy": CachePolicy.for_llama(4, 8, 64).to_dict()})
        ex = MlirExecutor(mod, PyTorchBackend("cpu"))
        assert ex._uses_static_shape
        assert not ex._uses_cache_manager, "cache manager should be disabled for static shapes"
