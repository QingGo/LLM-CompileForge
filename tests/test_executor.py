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


# ═══════════════════════════════════════════════════════════
# PagedAttention KV Cache
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
class TestPagedAttentionKV:
    """Tests for PagedAttention KV cache write + gather during forward."""

    @staticmethod
    def _make_executor():
        mod = _make_test_module()
        backend = PyTorchBackend("cpu")
        return Executor(mod, backend)

    @staticmethod
    def _alloc_kv_cache(executor, num_layers=4, num_kv_heads=2, head_dim=4, num_blocks=8, block_size=4):
        return executor.prepare_kv_blocks(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            num_blocks=num_blocks,
            dtype=torch.float32,
        )

    def test_prepare_kv_blocks_shape(self):
        ex = self._make_executor()
        kv = self._alloc_kv_cache(ex)
        assert kv.shape == (8, 4, 2, 4, 2, 4)

    def test_write_and_gather_kv(self):
        """Write K/V to cache then gather — gathered values should match."""
        ex = self._make_executor()
        kv = self._alloc_kv_cache(ex, num_kv_heads=2, head_dim=4, block_size=4)
        ex.set_kv_cache(kv, block_tables={"r0": [0, 1]}, num_kv_heads=2, head_dim=4, block_size=4)

        # Write 4 tokens at positions 0-3
        k_data = torch.randn(4, 2, 4)  # [seq, heads, hd]
        v_data = torch.randn(4, 2, 4)
        ex.write_kv_to_cache(k_data, v_data, torch.tensor([0, 1, 2, 3]), {"r0": [0, 1]}, layer_idx=0)

        # Gather all 4 tokens
        k_gathered, v_gathered = ex.gather_kv_from_cache({"r0": [0, 1]}, max_seq_len=4, layer_idx=0)
        assert k_gathered.shape == (1, 4, 2, 4)
        assert torch.allclose(k_gathered, k_data.unsqueeze(0))
        assert torch.allclose(v_gathered, v_data.unsqueeze(0))

    def test_write_multiple_tokens_across_blocks(self):
        """Write 8 tokens across 2 blocks — first 4 in block 0, next 4 in block 1."""
        ex = self._make_executor()
        kv = self._alloc_kv_cache(ex, num_kv_heads=2, head_dim=4, block_size=4)
        ex.set_kv_cache(kv, block_tables={"r0": [0, 1]}, num_kv_heads=2, head_dim=4, block_size=4)

        k_data = torch.randn(8, 2, 4)
        v_data = torch.randn(8, 2, 4)
        ex.write_kv_to_cache(k_data, v_data, torch.arange(8), {"r0": [0, 1]}, layer_idx=0)

        k_gathered, v_gathered = ex.gather_kv_from_cache({"r0": [0, 1]}, max_seq_len=8, layer_idx=0)
        assert k_gathered.shape == (1, 8, 2, 4)
        assert torch.allclose(k_gathered, k_data.unsqueeze(0))

    def test_gather_partial_seq(self):
        """Gather only first 6 tokens from 8 written — should be a prefix slice."""
        ex = self._make_executor()
        kv = self._alloc_kv_cache(ex, num_kv_heads=2, head_dim=4, block_size=4)
        ex.set_kv_cache(kv, block_tables={"r0": [0, 1]}, num_kv_heads=2, head_dim=4, block_size=4)

        k_data = torch.randn(8, 2, 4)
        v_data = torch.randn(8, 2, 4)
        ex.write_kv_to_cache(k_data, v_data, torch.arange(8), {"r0": [0, 1]}, layer_idx=0)

        k_gathered, v_gathered = ex.gather_kv_from_cache({"r0": [0, 1]}, max_seq_len=6, layer_idx=0)
        assert k_gathered.shape == (1, 6, 2, 4)
        assert torch.allclose(k_gathered, k_data[:6].unsqueeze(0))

    def test_write_layer_isolation(self):
        """Different layers should have independent K/V storage."""
        ex = self._make_executor()
        kv = self._alloc_kv_cache(ex, num_layers=4, num_kv_heads=2, head_dim=4, block_size=4)
        ex.set_kv_cache(kv, block_tables={"r0": [0]}, num_kv_heads=2, head_dim=4, block_size=4)

        k0 = torch.ones(4, 2, 4)
        k1 = torch.ones(4, 2, 4) * 2.0
        v0 = torch.zeros(4, 2, 4)
        v1 = torch.zeros(4, 2, 4)

        ex.write_kv_to_cache(k0, v0, torch.tensor([0, 1, 2, 3]), {"r0": [0]}, layer_idx=0)
        ex.write_kv_to_cache(k1, v1, torch.tensor([0, 1, 2, 3]), {"r0": [0]}, layer_idx=1)

        kg0, _ = ex.gather_kv_from_cache({"r0": [0]}, max_seq_len=4, layer_idx=0)
        kg1, _ = ex.gather_kv_from_cache({"r0": [0]}, max_seq_len=4, layer_idx=1)

        assert torch.allclose(kg0, torch.ones(1, 4, 2, 4))
        assert torch.allclose(kg1, torch.ones(1, 4, 2, 4) * 2.0)

    def test_forward_with_kv_cache_does_not_crash(self):
        """Forward with KV cache active should not crash (no SDPA in test module)."""
        ex = self._make_executor()
        kv = self._alloc_kv_cache(ex, num_kv_heads=2, head_dim=3)
        ex.set_kv_cache(kv, block_tables={"r0": [0]}, num_kv_heads=2, head_dim=3, block_size=4)
        input_ids = torch.randn(1, 4)
        logits, kv_tensors = ex.forward_with_kv(input_ids, positions=torch.tensor([0, 1, 2, 3]))
        assert logits is not None

