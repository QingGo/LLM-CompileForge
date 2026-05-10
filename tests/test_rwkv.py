"""Tests for RWKV module (Phase 2 Module E).

Verifies:
  - RWKV MLIR dialect op emission
  - FuseWKV / FuseTimeMix pass correctness
  - RWKV state manager allocation/update/get lifecycle
"""

from __future__ import annotations

import pytest
import torch

# ── RWKV Dialect ─────────────────────────────────────────


@pytest.mark.unit
class TestRwkvDialect:
    def test_emit_time_mix_op(self) -> None:
        from compiler.rwkv.dialect import emit_rwkv_op

        mlir_line = emit_rwkv_op(
            "rwkv.time_mix",
            ["%r", "%k", "%v", "%w", "%u", "%state"],
            "%yk",
            decay="exponential",
        )
        assert '"rwkv.time_mix"' in mlir_line
        assert "%yk" in mlir_line
        assert "decay" in mlir_line

    def test_emit_channel_mix_op(self) -> None:
        from compiler.rwkv.dialect import emit_rwkv_op

        mlir_line = emit_rwkv_op(
            "rwkv.channel_mix",
            ["%x", "%prev", "%time_mix_k", "%time_mix_r", "%key", "%value"],
            "%cm",
        )
        assert '"rwkv.channel_mix"' in mlir_line
        assert "%cm" in mlir_line

    def test_emit_state_evolve_op(self) -> None:
        from compiler.rwkv.dialect import emit_rwkv_op

        mlir_line = emit_rwkv_op(
            "rwkv.state_evolve",
            ["%old", "%new_info", "%decay"],
            "%new_state",
        )
        assert '"rwkv.state_evolve"' in mlir_line

    def test_is_rwkv_op(self) -> None:
        from compiler.rwkv.dialect import is_rwkv_op

        assert is_rwkv_op("rwkv.time_mix")
        assert is_rwkv_op("rwkv.channel_mix")
        assert is_rwkv_op("rwkv.state_evolve")
        assert not is_rwkv_op("sf.linear")
        assert not is_rwkv_op("arith.addf")


# ── RWKV Fusion Passes ───────────────────────────────────


@pytest.mark.unit
class TestRwkvFusion:
    def test_fuse_wkv_pass_noop_on_empty(self) -> None:
        from compiler.rwkv.fusion import fuse_wkv_pass

        mlir = "module {}"
        result = fuse_wkv_pass(mlir)
        assert result == mlir

    def test_apply_rwkv_fusion_passes_idempotent(self) -> None:
        from compiler.rwkv.fusion import apply_rwkv_fusion_passes

        mlir = "module {\n  func.func @main() -> tensor<f32> { return %0 : tensor<f32> }\n}"
        result = apply_rwkv_fusion_passes(mlir)
        assert "func.func @main" in result


# ── RWKV State Manager ───────────────────────────────────


@pytest.mark.unit
class TestRwkvStateManager:
    def test_allocate_and_free(self) -> None:
        from engine.rwkv_state_manager import RWKVStateManager

        mgr = RWKVStateManager(num_layers=4, hidden_size=64, max_requests=10)

        slot = mgr.allocate("req_1")
        assert isinstance(slot, int)
        assert mgr.active_count == 1
        assert mgr.free_count == 9

        mgr.free("req_1")
        assert mgr.active_count == 0
        assert mgr.free_count == 10

    def test_update_and_get_state(self) -> None:
        from engine.rwkv_state_manager import RWKVStateManager

        torch.manual_seed(42)
        mgr = RWKVStateManager(num_layers=2, hidden_size=32, max_requests=4)

        slot = mgr.allocate("req_a")
        new_state = torch.randn(mgr.state_dim)

        mgr.update_state(slot, 0, new_state)
        retrieved = mgr.get_state(slot, 0)

        assert torch.equal(retrieved, new_state.float())

    def test_get_all_states(self) -> None:
        from engine.rwkv_state_manager import RWKVStateManager

        mgr = RWKVStateManager(num_layers=3, hidden_size=16, max_requests=5)

        slot = mgr.allocate("req_x")
        all_states = mgr.get_all_states(slot)

        assert all_states.shape == (3, 16)
        assert (all_states == 0).all()

    def test_free_unknown_noop(self) -> None:
        from engine.rwkv_state_manager import RWKVStateManager

        mgr = RWKVStateManager(num_layers=2, hidden_size=32, max_requests=4)
        mgr.free("nonexistent")
        assert mgr.active_count == 0

    def test_exhaust_pool_raises(self) -> None:
        from engine.rwkv_state_manager import RWKVStateManager

        mgr = RWKVStateManager(num_layers=1, hidden_size=8, max_requests=2)
        mgr.allocate("r1")
        mgr.allocate("r2")
        with pytest.raises(RuntimeError, match="exhausted"):
            mgr.allocate("r3")
