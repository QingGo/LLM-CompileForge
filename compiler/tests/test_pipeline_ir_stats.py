"""Tests for per-stage IR dialect op counting and DEBUG snapshots.

Verifies that:
  1. StageResult.context contains dialect_counts_pre and post
  2. Per-stage log output contains IR dialect op count info
  3. DEBUG mode saves IR snapshot files to outputs/logs/pipeline/stages/
  4. INFO mode does NOT save snapshot files

All tests skip gracefully when MLIR Python bindings are unavailable.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pytest

from tests.helpers import has_mlir_bindings


def _make_test_module(ctx):
    import mlir.ir as ir

    mlir_text = (
        "module {\n"
        "  func.func @main() -> i32 {\n"
        "    %c42 = arith.constant 42 : i32\n"
        "    %c58 = arith.constant 58 : i32\n"
        "    %sum = arith.addi %c42, %c58 : i32\n"
        "    return %sum : i32\n"
        "  }\n"
        "}\n"
    )
    return ir.Module.parse(mlir_text, ctx)


def _noop_action(_module):
    pass


def _make_matmul_test_module(ctx):
    """Create a simple module with matmul ops for tiling tests."""
    import mlir.ir as ir

    mlir_text = """
module {
  func.func @test_matmul(%arg0: tensor<32x128xf32>,
                         %arg1: tensor<128x256xf32>)
      -> tensor<32x256xf32> {
    %c0 = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<32x256xf32>
    %fill = linalg.fill ins(%c0 : f32) outs(%init : tensor<32x256xf32>)
        -> tensor<32x256xf32>
    %result = linalg.matmul
        ins(%arg0, %arg1 : tensor<32x128xf32>, tensor<128x256xf32>)
        outs(%fill : tensor<32x256xf32>)
        -> tensor<32x256xf32>
    return %result : tensor<32x256xf32>
  }
}
"""
    return ir.Module.parse(mlir_text, ctx)


# ── Helpers for snapshot directory path ─────────────────────────────────


def _stages_dir() -> Path:
    return Path("logs") / "pipeline" / "stages"


def _clean_stages_dir() -> None:
    shutil.rmtree(_stages_dir(), ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════
# 1. StageResult context has dialect counts
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestStageResultDialectCounts:
    """StageResult.context contains dialect_counts_pre and post."""

    def test_stage_result_has_dialect_counts(self, mlir_context) -> None:
        if not has_mlir_bindings():
            pytest.skip("MLIR bindings not available")

        from compiler.mlir_dialect.pipeline.pipeline_stages import Stage

        module = _make_test_module(mlir_context)
        stage = Stage(name="test_counts", action=_noop_action, timeout=5.0)
        result = stage.run(module, mlir_context)

        assert result.success
        assert "dialect_counts_pre" in result.context
        assert "dialect_counts_post" in result.context
        assert isinstance(result.context["dialect_counts_pre"], dict)
        assert isinstance(result.context["dialect_counts_post"], dict)
        # Test module has at least arith + func ops
        assert len(result.context["dialect_counts_pre"]) >= 2


# ═══════════════════════════════════════════════════════════════════════
# 2. IR stats log output per stage
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestIRStatsLogging:
    """Per-stage log output contains dialect op count info."""

    def test_ir_stats_logged_per_stage(self, mlir_context, caplog) -> None:
        if not has_mlir_bindings():
            pytest.skip("MLIR bindings not available")

        caplog.set_level(logging.INFO)

        from compiler.mlir_dialect.pipeline.pipeline_stages import Stage

        module = _make_test_module(mlir_context)
        stage = Stage(name="ir_stats_test", action=_noop_action, timeout=5.0)
        stage.run(module, mlir_context)

        assert "IR stats" in caplog.text
        assert "total:" in caplog.text


# ═══════════════════════════════════════════════════════════════════════
# 3. DEBUG mode saves snapshots
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestDebugSnapshots:
    """DEBUG mode creates IR snapshot files."""

    def test_debug_mode_saves_snapshots(self, mlir_context) -> None:
        if not has_mlir_bindings():
            pytest.skip("MLIR bindings not available")

        from compiler.mlir_dialect.pipeline.pipeline_stages import Stage, run_stages

        module = _make_test_module(mlir_context)
        stages = [Stage(name="debug_stage_1", action=_noop_action, timeout=5.0)]

        root_logger = logging.getLogger()
        old_level = root_logger.level
        root_logger.setLevel(logging.DEBUG)

        _clean_stages_dir()
        try:
            run_stages(module, mlir_context, stages)
            assert _stages_dir().is_dir()
            mlir_files = list(_stages_dir().glob("*.mlir"))
            assert len(mlir_files) > 0, f"Expected snapshots in DEBUG, got {len(mlir_files)}"
        finally:
            root_logger.setLevel(old_level)
            _clean_stages_dir()


# ═══════════════════════════════════════════════════════════════════════
# 4. INFO mode does NOT save snapshots
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestInfoModeNoSnapshots:
    """INFO mode does NOT create IR snapshot files."""

    def test_info_mode_no_snapshots(self, mlir_context) -> None:
        if not has_mlir_bindings():
            pytest.skip("MLIR bindings not available")

        from compiler.mlir_dialect.pipeline.pipeline_stages import Stage, run_stages

        module = _make_test_module(mlir_context)
        stages = [Stage(name="info_stage_1", action=_noop_action, timeout=5.0)]

        root_logger = logging.getLogger()
        old_level = root_logger.level
        root_logger.setLevel(logging.INFO)

        _clean_stages_dir()
        try:
            run_stages(module, mlir_context, stages)
            # run_stages always creates dir, but should NOT write .mlir in INFO mode
            if _stages_dir().is_dir():
                mlir_files = list(_stages_dir().glob("*.mlir"))
                assert len(mlir_files) == 0, f"Expected 0 snapshots in INFO, got {len(mlir_files)}"
        finally:
            root_logger.setLevel(old_level)
            _clean_stages_dir()


# ═══════════════════════════════════════════════════════════════════════
# 5. Stage contract test: tiling produces scf.for
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestStageContract:
    """Verify pipeline stages produce expected IR changes."""

    def test_tiling_produces_scf_for(self, mlir_context) -> None:
        """Verify _make_tile_stage() produces scf.for loops.

        If tiling silently fails, this test catches it.
        """
        if not has_mlir_bindings():
            pytest.skip("MLIR bindings not available")

        from compiler.mlir_dialect.pipeline.pipeline_stages import _make_tile_stage

        mod = _make_matmul_test_module(mlir_context)
        stage = _make_tile_stage()
        result = stage.run(mod, mlir_context)

        module_str = str(mod)
        scf_count = module_str.count("scf.for")

        assert result.success, f"Tiling stage failed: {result.error}"
        assert scf_count > 0, (
            f"Tiling produced zero scf.for loops -- tiling may be silently failing.\n"
            f"Stage result: success={result.success}, error={result.error}\n"
            f"Module IR:\n{module_str[:2000]}"
        )

    def test_stage_changes_ir(self, mlir_context) -> None:
        """Verify canonicalize changes IR (not a no-op).

        Uses a module with redundant ops that canonicalize
        will constant-fold. If the stage produces identical
        IR before and after, it may be dead code.
        """
        if not has_mlir_bindings():
            pytest.skip("MLIR bindings not available")

        import mlir.ir as ir

        from compiler.mlir_dialect.pipeline.pipeline_stages import Stage

        # Module with redundant ops that canonicalize will fold
        noncanonical_text = """
module {
  func.func @test_fold() -> i32 {
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %sum = arith.addi %c1, %c2 : i32
    return %sum : i32
  }
}
"""
        mod = ir.Module.parse(noncanonical_text, mlir_context)
        before = str(mod)

        stage = Stage("canonicalize", "canonicalize", timeout=5.0)
        result = stage.run(mod, mlir_context)
        after = str(mod)

        assert result.success
        assert before != after, (
            f"Stage '{stage.name}' produced identical IR -- possible dead stage"
        )


# ── Pass-through for manual invocation ──────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
