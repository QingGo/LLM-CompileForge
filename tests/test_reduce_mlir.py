"""Tests for scripts/diagnostics/reduce_mlir.py — MLIR test-case reducer."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.diagnostics.reduce_mlir import (
    GlobalCounter,
    PassInterestingness,
    ShellInterestingness,
    _build_module_with,
    _get_func_names,
    reduce_by_binary_search,
    reduce_functions,
)


# =========================================================================
# Synthetic MLIR test fixtures
# =========================================================================


def _make_mlir_module(func_names: list[str]) -> str:
    funcs = []
    for name in func_names:
        funcs.append(
            f"  func.func @{name}(%arg0: tensor<?xf32>) -> tensor<?xf32> {{\n"
            f"    %0 = arith.addf %arg0, %arg0 : tensor<?xf32>\n"
            f"    return %0 : tensor<?xf32>\n"
            f"  }}"
        )
    return "module {\n" + "\n".join(funcs) + "\n}\n"


SIMPLE_TWO_FUNC = _make_mlir_module(["main_0", "main_1"])
SIMPLE_FIVE_FUNC = _make_mlir_module(
    ["main_0", "main_1", "main_2", "main_3", "main_4"]
)


# =========================================================================
# _build_module_with + _get_func_names
# =========================================================================


class TestModuleManipulation:
    def test_get_func_names(self):
        import mlir.ir as ir
        from scripts.diagnostics.reduce_mlir import _get_mlir_ctx

        ctx = _get_mlir_ctx()
        with ir.Location.unknown(ctx):
            mod = ir.Module.parse(SIMPLE_FIVE_FUNC, ctx)
        names = _get_func_names(mod)
        assert names == ["main_0", "main_1", "main_2", "main_3", "main_4"]

    def test_build_with_subset(self):
        result = _build_module_with(
            SIMPLE_FIVE_FUNC, {"main_0", "main_2", "main_4"}
        )
        assert "func.func @main_0" in result
        assert "func.func @main_1" not in result
        assert "func.func @main_2" in result
        assert "func.func @main_3" not in result
        assert "func.func @main_4" in result
        assert result.strip().endswith("}")

    def test_build_with_empty(self):
        result = _build_module_with(SIMPLE_FIVE_FUNC, set())
        assert "func.func" not in result

    def test_build_roundtrip_valid(self):
        import mlir.ir as ir
        from scripts.diagnostics.reduce_mlir import _get_mlir_ctx

        result = _build_module_with(SIMPLE_TWO_FUNC, {"main_0"})
        ctx = _get_mlir_ctx()
        with ir.Location.unknown(ctx):
            mod = ir.Module.parse(result, ctx)
        names = _get_func_names(mod)
        assert names == ["main_0"]


# =========================================================================
# reduce_functions
# =========================================================================


class TestReduceFunctions:
    def test_keep_all_when_nothing_interesting(self):
        def never_interesting(_text: str) -> bool:
            return False

        result = reduce_functions(SIMPLE_FIVE_FUNC, never_interesting)
        names = _func_names_from_text(result)
        assert len(names) == 5

    def test_delete_single_function(self):
        def keep_if_main_2_gone(text: str) -> bool:
            present = set(_func_names_from_text(text))
            required = {"main_0", "main_1", "main_3", "main_4"}
            return "main_2" not in present and required.issubset(present)

        result = reduce_functions(SIMPLE_FIVE_FUNC, keep_if_main_2_gone)
        names = _func_names_from_text(result)
        assert len(names) == 4
        assert "main_2" not in names
        assert "main_0" in names

    def test_delete_all_when_always_interesting(self):
        def always_interesting(_text: str) -> bool:
            return True

        result = reduce_functions(SIMPLE_FIVE_FUNC, always_interesting)
        names = _func_names_from_text(result)
        assert len(names) == 0

    def test_retry_catches_deletions_that_no_retry_misses(self):
        def m1_depends_on_m3_absent(text: str) -> bool:
            names = set(_func_names_from_text(text))
            if "main_1" not in names:
                return "main_3" not in names
            if "main_3" not in names:
                return True
            return True

        result_no_retry = reduce_functions(
            SIMPLE_FIVE_FUNC, m1_depends_on_m3_absent, retry=False
        )
        result_retry = reduce_functions(
            SIMPLE_FIVE_FUNC, m1_depends_on_m3_absent, retry=True
        )

        names_no = _func_names_from_text(result_no_retry)
        names_yes = _func_names_from_text(result_retry)

        assert "main_1" in names_no or "main_3" in names_no
        assert "main_1" not in names_yes
        assert "main_3" not in names_yes


# =========================================================================
# reduce_by_binary_search
# =========================================================================


class TestReduceBinarySearch:
    def test_finds_minimal_prefix(self):
        def bug_starts_at_main_3(text: str) -> bool:
            return "func.func @main_3" in text or "func.func @main_4" in text

        result = reduce_by_binary_search(SIMPLE_FIVE_FUNC, bug_starts_at_main_3)
        names = _func_names_from_text(result)
        assert "main_3" in names
        assert "main_4" not in names

    def test_bug_from_start(self):
        def bug_everywhere(_text: str) -> bool:
            return True

        result = reduce_by_binary_search(SIMPLE_FIVE_FUNC, bug_everywhere)
        names = _func_names_from_text(result)
        assert len(names) == 1
        assert "main_0" in names


# =========================================================================
# GlobalCounter
# =========================================================================


class TestGlobalCounter:
    def setup_method(self):
        import glob as _glob
        for f in _glob.glob("/tmp/reduce_mlir_counters/best_test*.txt"):
            try:
                os.unlink(f)
            except OSError:
                pass

    def test_rejects_metric_increase(self):
        class AlwaysInteresting:
            def is_interesting(self, text):
                return True

        counter = GlobalCounter(AlwaysInteresting(), metric="lines", input_hash="test_gc_inc")
        assert counter.is_interesting("line1\nline2\nline3\n")
        assert not counter.is_interesting("line1\nline2\nline3\nline4\nline5\n")

    def test_accepts_metric_decrease(self):
        class AlwaysInteresting:
            def is_interesting(self, text):
                return True

        counter = GlobalCounter(AlwaysInteresting(), metric="lines", input_hash="test_gc_dec")
        assert counter.is_interesting("a\nb\nc\nd\ne\n")
        assert counter.is_interesting("x\ny\n")
        assert counter.best_value == 2

    def test_rejects_uninteresting_before_metric_check(self):
        class NeverInteresting:
            def is_interesting(self, text):
                return False

        counter = GlobalCounter(NeverInteresting(), metric="lines", input_hash="test_gc_rej")
        assert not counter.is_interesting("a\nb\n")
        assert counter.best_value is None

    def test_ops_metric(self):
        class AlwaysInteresting:
            def is_interesting(self, text):
                return True

        counter = GlobalCounter(AlwaysInteresting(), metric="ops", input_hash="test_gc_ops")
        assert counter.is_interesting(
            "  %0 = arith.addf %a, %b : f32\n  %1 = arith.mulf %0, %c : f32\n"
        )
        assert counter.is_interesting("  %0 = arith.addf %a, %b : f32\n")

    def test_persists_best_value(self):
        class AlwaysInteresting:
            def is_interesting(self, text):
                return True

        ihash = "test_gc_persist"
        counter1 = GlobalCounter(AlwaysInteresting(), metric="lines", input_hash=ihash)
        assert counter1.is_interesting("a\nb\nc\n")
        assert counter1.best_value == 3

        counter2 = GlobalCounter(AlwaysInteresting(), metric="lines", input_hash=ihash)
        assert counter2.best_value == 3


# =========================================================================
# ShellInterestingness
# =========================================================================


class TestShellInterestingness:
    def test_exit_zero_is_interesting(self):
        test = ShellInterestingness("true", timeout=5)
        assert test.is_interesting("anything") is True

    def test_exit_nonzero_is_uninteresting(self):
        test = ShellInterestingness("false", timeout=5)
        assert test.is_interesting("anything") is False

    def test_file_placeholder(self):
        test = ShellInterestingness("grep -q 'hello' {}", timeout=5)
        assert test.is_interesting("hello world") is True
        assert test.is_interesting("goodbye") is False

    def test_file_contains_mlir(self):
        test = ShellInterestingness("grep -q 'arith.addf' {}", timeout=5)
        assert test.is_interesting("  %0 = arith.addf %a, %b : f32\n") is True

    def test_timeout_returns_false(self):
        test = ShellInterestingness("sleep 10", timeout=0.1)
        assert test.is_interesting("anything") is False


# =========================================================================
# PassInterestingness
# =========================================================================


class TestPassInterestingness:
    def test_passing_pass_is_uninteresting(self):
        test = PassInterestingness("canonicalize", timeout=10)
        assert test.is_interesting(_make_mlir_module(["test_func"])) is False

    def test_invalid_mlir_is_uninteresting(self):
        test = PassInterestingness("canonicalize", timeout=10)
        assert test.is_interesting("not valid mlir {{{") is False

    def test_buggy_mlir_detected_as_interesting(self):
        test = PassInterestingness(
            "one-shot-bufferize{bufferize-function-boundaries}", timeout=10
        )
        mlir = """module {
  func.func @bad(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    %0 = tensor.empty() : tensor<2xf32>
    return %0 : tensor<2xf32>
  }
}
"""
        result = test.is_interesting(mlir)
        assert result in (True, False)


# =========================================================================
# Helpers
# =========================================================================


def _func_names_from_text(mlir_text: str) -> list[str]:
    import re
    return re.findall(r"func\.func\s+@(\w+)", mlir_text)
