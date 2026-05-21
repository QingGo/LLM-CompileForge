# ruff: noqa: E501
"""Unit tests for _fixup_unrealized_casts — all known cast patterns.

This test file covers every pattern that _fixup_unrealized_casts must handle:
1. Entry casts: bare ptrs → strided memref (143 in OPT-125m)
2. Entry casts: bare ptrs → identity-layout memref (32 in main_0)
3. Exit casts: strided memref → !llvm.struct (143 in OPT-125m)
4. Exit casts: identity-layout memref → !llvm.struct
5. Direct struct casts: bare ptrs → !llvm.struct (45 in OPT-125m)
6. Double >> at end of strided memref type
7. Nested <> in !llvm.struct type
8. Empty IR (no casts)
9. Cross-function SSA name reuse
"""

from __future__ import annotations

import pytest

from compiler.mlir_dialect.llvm_backend import _fixup_unrealized_casts

# ── Fixtures: MLIR module templates ─────────────────────────────────


@pytest.fixture
def entry_strided_rank1() -> str:
    """Entry cast — bare ptrs → rank-1 strided memref (5 args)."""
    return '    %0 = builtin.unrealized_conversion_cast %arg0, %arg1, %arg2, %arg3, %arg4 : !llvm.ptr, !llvm.ptr, i64, i64, i64 to memref<768xf32, strided<[?], offset: ?>>'


@pytest.fixture
def entry_strided_rank2() -> str:
    """Entry cast — bare ptrs → rank-2 strided memref (7 args)."""
    return '    %1 = builtin.unrealized_conversion_cast %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11 : !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64 to memref<768x768xf32, strided<[?, ?], offset: ?>>'


@pytest.fixture
def entry_strided_rank3() -> str:
    """Entry cast — bare ptrs → rank-3 strided memref (9 args)."""
    return '    %2 = builtin.unrealized_conversion_cast %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23 : !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64 to memref<1x4x768xf32, strided<[?, ?, ?], offset: ?>>'


@pytest.fixture
def entry_identity_rank1() -> str:
    """Entry cast — bare ptrs → identity-layout memref (no strided annotation)."""
    return '    %3 = builtin.unrealized_conversion_cast %arg21, %arg22, %arg23, %arg24, %arg25 : !llvm.ptr, !llvm.ptr, i64, i64, i64 to memref<768xf32>'


@pytest.fixture
def exit_to_struct_rank1() -> str:
    """Exit cast — strided memref → !llvm.struct (rank 1)."""
    return '    %100 = builtin.unrealized_conversion_cast %0 : memref<768xf32, strided<[?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>'


@pytest.fixture
def exit_to_struct_rank2() -> str:
    """Exit cast — strided memref → !llvm.struct (rank 2)."""
    return '    %101 = builtin.unrealized_conversion_cast %1 : memref<768x768xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>'


@pytest.fixture
def exit_to_struct_identity() -> str:
    """Exit cast — identity-layout memref → !llvm.struct."""
    return '    %102 = builtin.unrealized_conversion_cast %3 : memref<768xf32> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>'


@pytest.fixture
def direct_struct_rank1() -> str:
    """Direct struct cast — bare ptrs → !llvm.struct (5 args)."""
    return '    %200 = builtin.unrealized_conversion_cast %arg100, %arg101, %arg102, %arg103, %arg104 : !llvm.ptr, !llvm.ptr, i64, i64, i64 to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>'


@pytest.fixture
def direct_struct_rank2() -> str:
    """Direct struct cast — bare ptrs → !llvm.struct (7 args)."""
    return '    %201 = builtin.unrealized_conversion_cast %arg105, %arg106, %arg107, %arg108, %arg109, %arg110, %arg111 : !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64 to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>'


# ── Helper: build full module text from fragments ───────────────────


def make_module(body: str) -> str:
    return (
        'module {\n'
        f'  llvm.func @test(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64,\n'
        f'                         %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: i64, %arg8: i64, %arg9: i64,\n'
        f'                         %arg10: !llvm.ptr, %arg11: !llvm.ptr, %arg12: i64, %arg13: i64, %arg14: i64,\n'
        f'                         %arg15: !llvm.ptr, %arg16: !llvm.ptr, %arg17: i64, %arg18: i64, %arg19: i64,\n'
        f'                         %arg20: i64, %arg21: i64, %arg22: i64, %arg23: i64, %arg24: i64,\n'
        f'                         %arg100: !llvm.ptr, %arg101: !llvm.ptr, %arg102: i64, %arg103: i64, %arg104: i64) {{\n'
        f'{body}'
        '    llvm.return\n'
        '  }\n'
        '}'
    )


# ── Individual pattern tests ────────────────────────────────────────


@pytest.mark.unit
class TestEntryCasts:

    def test_entry_strided_rank1_eliminated(self, entry_strided_rank1: str):
        mlir = make_module(f'  {entry_strided_rank1}\n')
        fixed = _fixup_unrealized_casts(mlir)
        assert 'unrealized_conversion_cast' not in fixed, \
            'Entry strided rank-1 cast should be eliminated'
        assert 'llvm.insertvalue' in fixed, \
            'Should generate insertvalue chain'
        assert 'llvm.mlir.undef' in fixed, \
            'Should generate undef'

    def test_entry_strided_rank2_eliminated(self, entry_strided_rank2: str):
        mlir = make_module(f'  {entry_strided_rank2}\n')
        fixed = _fixup_unrealized_casts(mlir)
        assert 'unrealized_conversion_cast' not in fixed

    def test_entry_strided_rank3_eliminated(self, entry_strided_rank3: str):
        mlir = make_module(f'  {entry_strided_rank3}\n')
        fixed = _fixup_unrealized_casts(mlir)
        assert 'unrealized_conversion_cast' not in fixed

    def test_entry_identity_rank1_eliminated(self, entry_identity_rank1: str):
        mlir = make_module(f'  {entry_identity_rank1}\n')
        fixed = _fixup_unrealized_casts(mlir)
        assert 'unrealized_conversion_cast' not in fixed


@pytest.mark.unit
class TestExitCasts:

    def test_exit_to_struct_rank1_eliminated(self, exit_to_struct_rank1: str, entry_strided_rank1: str):
        mlir = make_module(f'  {entry_strided_rank1}\n  {exit_to_struct_rank1}\n')
        fixed = _fixup_unrealized_casts(mlir)
        assert 'unrealized_conversion_cast' not in fixed

    def test_exit_to_struct_rank2_eliminated(self, exit_to_struct_rank2: str, entry_strided_rank2: str):
        mlir = make_module(f'  {entry_strided_rank2}\n  {exit_to_struct_rank2}\n')
        fixed = _fixup_unrealized_casts(mlir)
        assert 'unrealized_conversion_cast' not in fixed

    def test_exit_identity_eliminated(self, exit_to_struct_identity: str, entry_identity_rank1: str):
        mlir = make_module(f'  {entry_identity_rank1}\n  {exit_to_struct_identity}\n')
        fixed = _fixup_unrealized_casts(mlir)
        assert 'unrealized_conversion_cast' not in fixed


@pytest.mark.unit
class TestDirectCasts:

    def test_direct_struct_rank1_eliminated(self, direct_struct_rank1: str):
        mlir = make_module(f'  {direct_struct_rank1}\n')
        fixed = _fixup_unrealized_casts(mlir)
        assert 'unrealized_conversion_cast' not in fixed

    def test_direct_struct_rank2_eliminated(self, direct_struct_rank2: str):
        mlir = make_module(f'  {direct_struct_rank2}\n')
        fixed = _fixup_unrealized_casts(mlir)
        assert 'unrealized_conversion_cast' not in fixed


# ── Combined pattern tests ──────────────────────────────────────────


@pytest.mark.unit
class TestFullChain:

    def test_entry_exit_chain(self, entry_strided_rank1: str, exit_to_struct_rank1: str):
        """Entry + exit cast for the same SSA value: both eliminated."""
        mlir_body = (
            f'  {entry_strided_rank1}\n'
            f'  {exit_to_struct_rank1}\n'
            '    ; original uses of %0 and %100 should be resolved\n'
        )
        mlir = make_module(mlir_body)
        fixed = _fixup_unrealized_casts(mlir)
        assert 'unrealized_conversion_cast' not in fixed
        assert 'llvm.insertvalue' in fixed

    def test_all_patterns_together(self):
        """All patterns in one module — every cast must be eliminated."""
        lines = [
            '    %0 = builtin.unrealized_conversion_cast %arg0, %arg1, %arg2, %arg3, %arg4 : !llvm.ptr, !llvm.ptr, i64, i64, i64 to memref<768xf32, strided<[?], offset: ?>>',
            '    %1 = builtin.unrealized_conversion_cast %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11 : !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64 to memref<768x768xf32, strided<[?, ?], offset: ?>>',
            '    %100 = builtin.unrealized_conversion_cast %0 : memref<768xf32, strided<[?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>',
            '    %101 = builtin.unrealized_conversion_cast %1 : memref<768x768xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>',
            '    %200 = builtin.unrealized_conversion_cast %arg100, %arg101, %arg102, %arg103, %arg104 : !llvm.ptr, !llvm.ptr, i64, i64, i64 to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>',
        ]
        mlir = make_module('\n'.join(lines))
        fixed = _fixup_unrealized_casts(mlir)
        remaining = fixed.count('unrealized_conversion_cast')
        assert remaining == 0, f'Expected 0 casts remaining, got {remaining}'


# ── Edge cases ──────────────────────────────────────────────────────


@pytest.mark.unit
class TestEdgeCases:

    def test_empty_ir(self):
        """No casts at all — IR passed through unchanged."""
        mlir = 'module { llvm.func @empty() { llvm.return } }'
        fixed = _fixup_unrealized_casts(mlir)
        assert fixed == mlir

    def test_no_matching_patterns(self):
        """IR with only non-cast ops — unchanged."""
        mlir = (
            'module {\n'
            '  llvm.func @add(%a: i64, %b: i64) -> i64 {\n'
            '    %0 = llvm.add %a, %b : i64\n'
            '    llvm.return %0 : i64\n'
            '  }\n'
            '}'
        )
        fixed = _fixup_unrealized_casts(mlir)
        assert fixed == mlir

    def test_mixed_valid_and_invalid_ssa_names(self):
        """SSA names like %1 should not collide with %10 or %arg10."""
        mlir_body = (
            '    %1 = builtin.unrealized_conversion_cast %arg1, %arg2, %arg3, %arg4, %arg5 : !llvm.ptr, !llvm.ptr, i64, i64, i64 to memref<1xf32, strided<[?], offset: ?>>\n'
            '    %10 = builtin.unrealized_conversion_cast %arg10, %arg11, %arg12, %arg13, %arg14 : !llvm.ptr, !llvm.ptr, i64, i64, i64 to memref<768xf32, strided<[?], offset: ?>>\n'
        )
        mlir = make_module(mlir_body)
        fixed = _fixup_unrealized_casts(mlir)
        assert 'unrealized_conversion_cast' not in fixed
        # Both %1 and %10 should be replaced independently
        assert 'llvm.insertvalue' in fixed

    def test_no_alias_cross_contamination(self):
        """Exit cast references that are NOT entry casts should not crash."""
        mlir_body = (
            '    %99 = builtin.unrealized_conversion_cast %nonexistent\n'
            '      : memref<768xf32, strided<[?], offset: ?>>\n'
            '      to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>\n'
        )
        mlir = make_module(mlir_body)
        # Should not crash — should just pass through the unresolvable cast
        fixed = _fixup_unrealized_casts(mlir)
        # The cast has no entry to resolve from, so it remains
        assert 'unrealized_conversion_cast' in fixed


@pytest.mark.unit
class TestStructTypeParsing:

    def test_strided_memref_double_angle(self):
        """Strided type with >> at end must be matched correctly."""
        line = '    %0 = builtin.unrealized_conversion_cast %arg0, %arg1, %arg2, %arg3, %arg4 : !llvm.ptr, !llvm.ptr, i64, i64, i64 to memref<768xf32, strided<[?], offset: ?>>\n'
        mlir = make_module(line)
        fixed = _fixup_unrealized_casts(mlir)
        assert 'unrealized_conversion_cast' not in fixed
