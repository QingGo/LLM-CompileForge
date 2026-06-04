"""Tests for per-stage IR snapshot stats and stage-level verification.

Verifies that:
  1. ``_save_ir_stats`` writes a stats file with op counts and top-10
  2. ``_verify_stage_output`` detects func.func and func.return count changes
  3. ``Stage.run()`` captures IR before/after and emits verification warnings
  4. The module-verify stage (appended to BUILTIN_STAGES) runs and reports

All tests skip gracefully when MLIR Python bindings are unavailable.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pytest

from tests.helpers import has_mlir_bindings


def _clean_pipeline_logs() -> None:
    log_dir = Path("logs") / "pipeline"
    for p in log_dir.glob("stats_test_*.txt"):
        p.unlink(missing_ok=True)
    for p in log_dir.glob("snapshot_test_*.mlir"):
        p.unlink(missing_ok=True)


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


# ═══════════════════════════════════════════════════════════════════════
# 1. _save_ir_stats writes stats file with op names and counts
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestSaveIrStats:
    def test_stats_file_created(self, mlir_context) -> None:
        if not has_mlir_bindings():
            pytest.skip("MLIR bindings not available")

        from compiler.mlir_dialect.pipeline.pipeline_stages_utils import _save_ir_stats

        module = _make_test_module(mlir_context)
        _clean_pipeline_logs()
        try:
            counts = _save_ir_stats(module, "test_stats")

            assert isinstance(counts, dict)
            assert len(counts) >= 4, f"Expected at least 4 op types, got {len(counts)}"
            assert "func.func" in counts

            log_dir = Path("logs") / "pipeline"
            stats_files = list(log_dir.glob("stats_test_stats_*.txt"))
            assert len(stats_files) >= 1, "No stats file created"
            content = stats_files[0].read_text()
            assert "func.func" in content
            assert "Top 10" in content
            assert "Total ops:" in content
        finally:
            _clean_pipeline_logs()

    def test_stats_file_contains_top10(self, mlir_context) -> None:
        if not has_mlir_bindings():
            pytest.skip("MLIR bindings not available")

        from compiler.mlir_dialect.pipeline.pipeline_stages_utils import _save_ir_stats

        module = _make_test_module(mlir_context)
        _clean_pipeline_logs()
        try:
            _save_ir_stats(module, "test_top10")
            log_dir = Path("logs") / "pipeline"
            stats_files = list(log_dir.glob("stats_test_top10_*.txt"))
            assert len(stats_files) >= 1
            content = stats_files[0].read_text()
            assert "Top 10 op names:" in content
            assert "All op name counts:" in content
            assert "arith.constant: 2" in content
        finally:
            _clean_pipeline_logs()


# ═══════════════════════════════════════════════════════════════════════
# 2. _verify_stage_output detects func.func / func.return mismatches
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestVerifyStageOutput:
    def test_no_mismatch_returns_empty(self) -> None:
        from compiler.mlir_dialect.pipeline.pipeline_stages_utils import _verify_stage_output

        ir_text = (
            "module {\n"
            "  func.func @foo() -> i32 {\n"
            "    %c = arith.constant 1 : i32\n"
            "    return %c : i32\n"
            "  }\n"
            "  func.func @bar() -> i32 {\n"
            "    %c = arith.constant 2 : i32\n"
            "    return %c : i32\n"
            "  }\n"
            "}\n"
        )
        warnings = _verify_stage_output(ir_text, ir_text, "test")
        assert warnings == [], f"Expected no warnings, got {warnings}"

    def test_detects_func_func_change(self) -> None:
        from compiler.mlir_dialect.pipeline.pipeline_stages_utils import _verify_stage_output

        before = (
            "module {\n"
            "  func.func @foo() -> i32 {\n"
            "    %c = arith.constant 1 : i32\n"
            "    return %c : i32\n"
            "  }\n"
            "}\n"
        )
        after = (
            "module {\n"
            "  func.func @foo() -> i32 {\n"
            "    %c = arith.constant 1 : i32\n"
            "    return %c : i32\n"
            "  }\n"
            "  func.func @bar() -> i32 {\n"
            "    %c = arith.constant 2 : i32\n"
            "    return %c : i32\n"
            "  }\n"
            "}\n"
        )
        warnings = _verify_stage_output(before, after, "test_func_add")
        func_warnings = [w for w in warnings if "func.func" in w]
        assert len(func_warnings) >= 1, (
            f"Expected func.func count change warning, got {warnings}"
        )

    def test_detects_func_return_change(self) -> None:
        from compiler.mlir_dialect.pipeline.pipeline_stages_utils import _verify_stage_output

        before = (
            "module {\n"
            "  func.func @foo() -> i32 {\n"
            "    %c = arith.constant 1 : i32\n"
            "    return %c : i32\n"
            "  }\n"
            "}\n"
        )
        after = (
            "module {\n"
            "  func.func @foo() -> i32 {\n"
            "    %c = arith.constant 1 : i32\n"
            "    return %c : i32\n"
            "    return %c : i32\n"
            "  }\n"
            "}\n"
        )
        warnings = _verify_stage_output(before, after, "test_ret_add")
        ret_warnings = [w for w in warnings if "func.return" in w]
        assert len(ret_warnings) >= 1, (
            f"Expected func.return count change warning, got {warnings}"
        )


# ═══════════════════════════════════════════════════════════════════════
# 3. Stage.run() with verification — snapshot + stats on failure
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestStageRunVerification:
    def test_snapshot_and_stats_on_failure(self, mlir_context) -> None:
        if not has_mlir_bindings():
            pytest.skip("MLIR bindings not available")

        from compiler.mlir_dialect.pipeline.pipeline_stages import Stage

        module = _make_test_module(mlir_context)
        _clean_pipeline_logs()
        try:
            def _fail_action(_m):
                raise RuntimeError("deliberate failure for test")

            stage = Stage(
                name="test_verify_fail",
                action=_fail_action,
                timeout=5.0,
                save_snapshot=True,
            )
            result = stage.run(module, mlir_context)

            assert not result.success
            assert result.ir_snapshot_path is not None

            snapshot_path = Path(result.ir_snapshot_path)
            assert snapshot_path.exists(), f"Snapshot not found: {snapshot_path}"

            stats_files = list(
                Path("logs").glob("pipeline/stats_test_verify_fail_snapshot_*.txt")
            )
            assert len(stats_files) >= 1, "No companion stats file created"
        finally:
            _clean_pipeline_logs()

    def test_no_warnings_on_noop_stage(self, mlir_context, caplog) -> None:
        if not has_mlir_bindings():
            pytest.skip("MLIR bindings not available")

        from compiler.mlir_dialect.pipeline.pipeline_stages import Stage

        caplog.set_level(logging.WARNING)

        module = _make_test_module(mlir_context)
        stage = Stage(name="test_verify_noop", action=_noop_action, timeout=5.0)
        result = stage.run(module, mlir_context)

        assert result.success
        # No invariant warnings expected for a noop action
        assert "Stage invariant" not in caplog.text, (
            f"Unexpected warnings: {caplog.text}"
        )


# ═══════════════════════════════════════════════════════════════════════
# 4. module-verify stage is appended to BUILTIN_STAGES and runs
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestModuleVerifyStage:
    def test_verify_stage_in_builtin_stages(self) -> None:
        from compiler.mlir_dialect.pipeline.pipeline_stages import BUILTIN_STAGES

        names = [s.name for s in BUILTIN_STAGES]
        assert "module-verify" in names, (
            f"module-verify not found in BUILTIN_STAGES ({names})"
        )
        assert names[-1] == "module-verify", (
            f"module-verify should be last stage, got {names[-1]}"
        )

    def test_verify_stage_pass_on_valid_module(self, mlir_context, caplog) -> None:
        if not has_mlir_bindings():
            pytest.skip("MLIR bindings not available")

        from compiler.mlir_dialect.pipeline.pipeline_stages import _make_verify_stage

        caplog.set_level(logging.INFO)

        module = _make_test_module(mlir_context)
        stage = _make_verify_stage()
        result = stage.run(module, mlir_context)

        assert result.success
        assert "PASS" in caplog.text or "0 issues" in caplog.text, (
            f"Expected PASS message in log: {caplog.text}"
        )

    def test_verify_stage_in_no_fma_list(self) -> None:
        from compiler.mlir_dialect.pipeline.pipeline_stages import BUILTIN_STAGES_NO_FMA

        names = [s.name for s in BUILTIN_STAGES_NO_FMA]
        assert "module-verify" in names, (
            f"module-verify not in BUILTIN_STAGES_NO_FMA ({names})"
        )


# ═══════════════════════════════════════════════════════════════════════
# 5. End-to-end: run_stages produces snapshot files with DEBUG logging
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestEndToEndSnapshots:
    def test_run_stages_creates_snapshots_in_debug(self, mlir_context) -> None:
        if not has_mlir_bindings():
            pytest.skip("MLIR bindings not available")

        from compiler.mlir_dialect.pipeline.pipeline_stages import Stage, run_stages

        module = _make_test_module(mlir_context)
        stages = [
            Stage(name="e2e_stage_1", action=_noop_action, timeout=5.0),
            Stage(name="e2e_stage_2", action=_noop_action, timeout=5.0),
        ]

        root_logger = logging.getLogger()
        old_level = root_logger.level
        root_logger.setLevel(logging.DEBUG)

        stages_dir = Path("logs") / "pipeline" / "stages"
        shutil.rmtree(stages_dir, ignore_errors=True)
        try:
            results = run_stages(module, mlir_context, stages)
            assert len(results) == 2
            assert all(r.success for r in results)

            assert stages_dir.is_dir()
            mlir_files = list(stages_dir.glob("*.mlir"))
            assert len(mlir_files) >= 2, (
                f"Expected ≥2 DEBUG snapshots, got {len(mlir_files)}"
            )

            for f in mlir_files:
                assert f.stat().st_size > 0, f"Empty snapshot: {f}"
        finally:
            root_logger.setLevel(old_level)
            shutil.rmtree(stages_dir, ignore_errors=True)

    def test_count_module_ops_works_on_full_pipeline(self, mlir_context) -> None:
        """Verify _count_module_ops can analyze after a stage run."""
        if not has_mlir_bindings():
            pytest.skip("MLIR bindings not available")

        from compiler.mlir_dialect.pipeline.pipeline_stages import (
            Stage,
            _count_module_ops,
            run_stages,
        )

        module = _make_test_module(mlir_context)
        stages = [Stage(name="e2e_counts", action=_noop_action, timeout=5.0)]

        root_logger = logging.getLogger()
        old_level = root_logger.level
        root_logger.setLevel(logging.DEBUG)

        stages_dir = Path("logs") / "pipeline" / "stages"
        shutil.rmtree(stages_dir, ignore_errors=True)
        try:
            run_stages(module, mlir_context, stages)

            total, dialect_counts = _count_module_ops(str(module))
            assert total > 0
            assert "arith" in dialect_counts, (
                f"Expected arith dialect, got {dialect_counts}"
            )
            assert "func" in dialect_counts, (
                f"Expected func dialect, got {dialect_counts}"
            )
        finally:
            root_logger.setLevel(old_level)
            shutil.rmtree(stages_dir, ignore_errors=True)


# ── Pass-through for manual invocation ────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
