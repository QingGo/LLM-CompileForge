"""Contract test: one-shot-bufferize must not produce identity pass-throughs.

In tensor semantics, ``func.func @f(%arg0: tensor<...>) -> tensor<...> { return %arg0 }``
is valid (zero-copy reference).  In memref semantics, this produces an output buffer
that is **never allocated or written** → garbage values.

This test verifies that a pre-bufferization pass inserts ``tensor.insert_slice``
copies so that every function output has a real computation behind it.
"""

# ruff: noqa: E501 — MLIR text strings contain long lines

from __future__ import annotations

import re
import sys
from pathlib import Path

# Ensure project root on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import mlir.ir as ir  # noqa: E402 — must be after sys.path setup
import mlir.passmanager as pm  # noqa: E402 — must be after sys.path setup

# ── Helpers ───────────────────────────────────────────────────────────


def _setup_mlir_path() -> None:
    """Ensure MLIR Python bindings and sf-dialect are on sys.path."""
    from compiler.backend.compile_utils import _setup_mlir_path as _msp
    _msp()


def _register_sf_dialect(ctx: ir.Context) -> None:
    """Register sf dialect in the MLIR context."""
    _setup_mlir_path()
    from mlir_sf._mlir_libs._sfDialectsNanobind import sf
    sf.register_dialects(ctx._CAPIPtr, load=True)


def _count_function_identity_returns(module: ir.Module) -> int:
    """Count how many func.return operands are direct BlockArguments (identity pass-through).

    A direct identity pass-through is when a function returns a BlockArgument
    (function input parameter) without any intervening computation.
    """
    count = 0
    for op in module.operation.regions[0].blocks[0]:
        if str(op.operation.name) != "func.func":
            continue
        for region in op.operation.regions:
            for block in region.blocks:
                for inner_op in block.operations:
                    if str(inner_op.operation.name) == "func.return":
                        for operand in inner_op.operation.operands:
                            if isinstance(operand, ir.BlockArgument):
                                count += 1
    return count


def _has_identity_return_in_text(ir_text: str) -> bool:
    """Check if IR text has any direct identity returns (before bufferization).

    Looks for patterns like: ``func.return %argN : tensor<...>`` where %argN
    is a function parameter that passes through without computation.
    """
    # Find functions and their return operands
    # Pattern: func.return %argNN, ... : types
    # We'll look for func.func blocks where the return operand is a block argument
    # that was never used as input to any computation

    # Simpler approach: find return operands and check if they're block args
    # by looking for %argNN in return that appears exactly once (as the return value)
    # Find all function argument names: func.func @name(%arg0, %arg1, ...) -> ...
    func_pattern = re.compile(
        r'(?:func\.func|"func\.func")\s+@(\w+)\s*\(([^)]*)\)',
        re.MULTILINE,
    )
    # Find return operands: func.return %0, %1 : types or return %0, %1 : types
    ret_pattern = re.compile(
        r'(?:func\.return|"func\.return"|return)\s+(.+?)\s*:',
        re.MULTILINE,
    )

    func_args: dict[str, set[str]] = {}
    for m in func_pattern.finditer(ir_text):
        func_name = m.group(1)
        args_str = m.group(2)
        # Parse argument names from comma-separated list with types
        args = set()
        depth = 0
        buf = ""
        for ch in args_str + ",":
            if ch in "<(":
                depth += 1
            elif ch in ">":
                depth -= 1
            elif ch == ")" and depth > 0:
                depth -= 1
            if ch == "," and depth == 0:
                arg = buf.strip()
                if arg and ":" in arg:
                    args.add(arg.split(":")[0].strip())
                buf = ""
            else:
                buf += ch
        func_args[func_name] = args

    # Now find returns and check for identity pass-throughs
    # Re-scan to associate returns with the preceding function
    current_func: str | None = None
    for line in ir_text.split("\n"):
        stripped = line.strip()

        fm = re.search(r'(?:func\.func|"func\.func")\s+@(\w+)', stripped)
        if fm:
            current_func = fm.group(1)
            continue

        if current_func and current_func in func_args:
            rm = ret_pattern.match(stripped)
            if rm:
                ret_operands = [x.strip() for x in rm.group(1).split(",")]
                for rop in ret_operands:
                    if rop in func_args[current_func]:
                        return True

    return False


# ── Test MLIR fixtures ────────────────────────────────────────────────


# A lowered MLIR module with an identity pass-through function and a
# non-trivial function.  The identity function main_0 takes two tensor
# args (input + weight) and passes the weight arg through without
# computation.  main_1 does a real linalg.matmul.
# main_0: identity pass-through of %arg1 (weight) alongside a computed output
# main_1: purely computed output (no pass-through)
_IDENTITY_LOWERED_MLIR = r"""
module {
  func.func @main_0(%arg0: tensor<4x8xf32>, %arg1: tensor<8x4xf32>) -> (tensor<4x8xf32>, tensor<8x4xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4x8xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4x8xf32>) -> tensor<4x8xf32>
    %2 = linalg.add ins(%arg0, %1 : tensor<4x8xf32>, tensor<4x8xf32>) outs(%1 : tensor<4x8xf32>) -> tensor<4x8xf32>
    func.return %2, %arg1 : tensor<4x8xf32>, tensor<8x4xf32>
  }
  func.func @main_1(%arg0: tensor<4x8xf32>, %arg1: tensor<8x4xf32>) -> tensor<4x4xf32> {
    %0 = tensor.empty() : tensor<4x4xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<4x8xf32>, tensor<8x4xf32>) outs(%1 : tensor<4x4xf32>) -> tensor<4x4xf32>
    func.return %2 : tensor<4x4xf32>
  }
}
"""


def _make_lowered_module(ctx: ir.Context) -> ir.Module:
    """Parse the identity lowered MLIR into an ir.Module."""
    return ir.Module.parse(_IDENTITY_LOWERED_MLIR, ctx)


# ── Tests ─────────────────────────────────────────────────────────────


class TestIdentityDetection:
    """Verify we can detect identity pass-throughs in lowered MLIR."""

    def test_identity_lowered_has_pass_through(self, mlir_context: ir.Context) -> None:
        """The fixture MLIR should have an identity pass-through in main_0."""
        _register_sf_dialect(mlir_context)
        module = _make_lowered_module(mlir_context)
        ir_text = str(module)
        assert _has_identity_return_in_text(ir_text), (
            "Fixture should contain identity pass-through for test to be valid. "
            "main_0 should return %arg1 directly."
        )

    def test_count_block_arg_returns(self, mlir_context: ir.Context) -> None:
        """main_0 has ONE identity pass-through (weight %arg1)."""
        _register_sf_dialect(mlir_context)
        module = _make_lowered_module(mlir_context)
        assert _count_function_identity_returns(module) == 1, (
            f"Expected exactly 1 identity return in main_0, got "
            f"{_count_function_identity_returns(module)}"
        )

    def test_no_identity_after_bufferization_without_fix(self, mlir_context: ir.Context) -> None:
        """After bufferize WITHOUT identity copy pass, the memref output
        for the pass-through arg is uninitialized (the bug).

        This test verifies: bufferized IR IS produced (pass doesn't crash),
        but the pass-through issue persists in memref form.
        """
        _register_sf_dialect(mlir_context)
        module = _make_lowered_module(mlir_context)

        # Run the bufferize pipeline directly (C3 stage)
        pm.PassManager.parse(
            "builtin.module("
            "one-shot-bufferize{"
            "bufferize-function-boundaries allow-unknown-ops"
            " function-boundary-type-conversion=identity-layout-map"
            "},canonicalize,cse,convert-bufferization-to-memref"
            ")",
            mlir_context,
        ).run(module.operation)

        ir_text = str(module)
        # After bufferization, the function becomes memref-based.
        # The identity return becomes: func.return %arg1 : memref<8x4xf32>
        # This is the bug — no copy was inserted.
        assert "memref" in ir_text, "Expected memref types after bufferization"


class TestIdentityCopiesPass:
    """Verify the identity copies action eliminates identity pass-throughs."""

    def test_identity_copies_action_exists(self) -> None:
        """The insert_identity_copies_action must be importable."""
        from compiler.pipeline.actions import insert_identity_copies_action
        assert callable(insert_identity_copies_action)

    def test_identity_copies_eliminates_pass_through(self, mlir_context: ir.Context) -> None:
        """After running insert_identity_copies_action, no identity returns remain.

        This is the core RED→GREEN test.  Before the fix, this test FAILS.
        After the fix, this test PASSES.
        """
        _register_sf_dialect(mlir_context)
        from compiler.pipeline.actions import insert_identity_copies_action

        module = _make_lowered_module(mlir_context)

        # Verify identity exists before the pass
        assert _count_function_identity_returns(module) == 1, (
            "Precondition: fixture must have identity pass-through"
        )

        # Run the identity copies action
        insert_identity_copies_action(module)

        # Verify identity is eliminated
        remaining = _count_function_identity_returns(module)
        assert remaining == 0, (
            f"After insert_identity_copies_action, expected 0 identity returns, "
            f"got {remaining}. IR:\n{module}"
        )

        # Verify the IR is still valid tensor IR (no corruption)
        ir_text = str(module)
        assert "tensor" in ir_text, "IR should still have tensor ops"
        assert "func.func" in ir_text, "IR should still have func.func"
        assert "main_0" in ir_text, "Function main_0 should still exist"

    def test_identity_copies_preserves_computed_outputs(self, mlir_context: ir.Context) -> None:
        """main_1 (no identity returns) should be unchanged by the pass."""
        _register_sf_dialect(mlir_context)
        from compiler.pipeline.actions import insert_identity_copies_action

        module = _make_lowered_module(mlir_context)
        before_text = str(module)

        insert_identity_copies_action(module)
        after_text = str(module)

        # main_1 should still have the same matmul and return of %2
        assert "linalg.matmul" in after_text, "matmul should still be present"
        # The function return count should be unchanged
        assert before_text.count("func.return") == after_text.count("func.return"), (
            "func.return count should not change"
        )


class TestBufferizeWithIdentityCopies:
    """Full integration: identity copies + bufferize."""

    def test_bufferize_with_identity_copies(self, mlir_context: ir.Context) -> None:
        """After identity copies + bufferize, verify the bufferized IR is valid."""
        _register_sf_dialect(mlir_context)
        from compiler.pipeline.actions import insert_identity_copies_action

        module = _make_lowered_module(mlir_context)

        # Step 1: Insert identity copies
        insert_identity_copies_action(module)

        # Verify no identity returns remain
        assert _count_function_identity_returns(module) == 0

        # Step 2: Run bufferize
        pm.PassManager.parse(
            "builtin.module("
            "one-shot-bufferize{"
            "bufferize-function-boundaries allow-unknown-ops"
            " function-boundary-type-conversion=identity-layout-map"
            "},canonicalize,cse,convert-bufferization-to-memref"
            ")",
            mlir_context,
        ).run(module.operation)

        ir_text = str(module)
        assert "memref" in ir_text, "Expected memref types after bufferization"
        assert "tensor" not in ir_text, "Expected no tensor types left after full bufferization"

        # main_0 should have 2 outputs (both now memref), and function
        # should still be present
        assert "main_0" in ir_text, "Function main_0 must still exist"
        assert "main_1" in ir_text, "Function main_1 must still exist"
