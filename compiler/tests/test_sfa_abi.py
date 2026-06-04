"""Seam tests for sfa_abi regex patterns against LLVM IR snippets.

Tests ``_CIFACE_DEF_RE`` and ``_RANK_RE`` from ``compiler.sfa_abi``
— pure regex patterns that parse LLVM IR text to extract ciface wrappers
and descriptor struct ranks. No MLIR, no compiled dylib, no sf-dialect.
"""

from compiler.sfa_abi import _CIFACE_DEF_RE, _RANK_RE

# ── Minimal LLVM IR snippets ────────────────────────────────────────

LLVM_CIFACE_SIMPLE = r"""
define void @_mlir_ciface_main_0(ptr %0, ptr %1, ptr %2) {
  %4 = call { ptr, ptr, i64, [2 x i64], [2 x i64] } @main_0(ptr %1, ptr %2)
  ret void
}
"""

LLVM_CIFACE_TWO_FUNCS = r"""
define void @_mlir_ciface_proj(ptr %0, ptr %1, ptr %2, ptr %3) {
  %5 = call { ptr, ptr, i64, [3 x i64], [3 x i64] } @proj(ptr %1, ptr %2, ptr %3)
  ret void
}

define void @_mlir_ciface_attn(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4) {
  %6 = call { ptr, ptr, i64, [4 x i64], [4 x i64] } @attn(ptr %1, ptr %2, ptr %3, ptr %4)
  ret void
}
"""

LLVM_NO_CIFACE = r"""
define i32 @add(i32 %a, i32 %b) {
  %c = add i32 %a, %b
  ret i32 %c
}

define void @malloc(i64 %size) {
  ret void
}
"""


class TestCifaceDefRegex:
    """Verify _CIFACE_DEF_RE matches ciface wrapper definitions in LLVM IR."""

    def test_matches_single_ciface_wrapper(self) -> None:
        """Positive: match a ciface function definition."""
        matches = _CIFACE_DEF_RE.findall(LLVM_CIFACE_SIMPLE)
        assert len(matches) == 1
        name, params = matches[0]
        assert name == "_mlir_ciface_main_0"
        assert "ptr" in params

    def test_matches_multiple_ciface_wrappers(self) -> None:
        """Positive: match multiple ciface functions in one IR file."""
        matches = _CIFACE_DEF_RE.findall(LLVM_CIFACE_TWO_FUNCS)
        assert len(matches) == 2
        names = [m[0] for m in matches]
        assert "_mlir_ciface_proj" in names
        assert "_mlir_ciface_attn" in names

    def test_no_match_on_non_ciface_functions(self) -> None:
        """Negative: regex does NOT match regular (non-ciface) functions."""
        matches = _CIFACE_DEF_RE.findall(LLVM_NO_CIFACE)
        assert len(matches) == 0


class TestRankRegex:
    """Verify _RANK_RE extracts rank values from descriptor struct arrays."""

    def test_extract_rank_2(self) -> None:
        """Extract rank=2 from [2 x i64]."""
        ranks = _RANK_RE.findall("{ ptr, ptr, i64, [2 x i64], [2 x i64] }")
        assert ranks == ["2", "2"]

    def test_extract_rank_3(self) -> None:
        """Extract rank=3 from [3 x i64]."""
        ranks = _RANK_RE.findall("{ ptr, ptr, i64, [3 x i64], [3 x i64] }")
        assert ranks == ["3", "3"]

    def test_extract_rank_4(self) -> None:
        """Extract rank=4 from [4 x i64]."""
        ranks = _RANK_RE.findall("{ ptr, ptr, i64, [4 x i64], [4 x i64] }")
        assert ranks == ["4", "4"]
