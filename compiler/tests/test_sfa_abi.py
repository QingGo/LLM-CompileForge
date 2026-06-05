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


class TestConsumedInternallyContract:
    """Verify consumed_internally is correctly propagated through the proto pipeline.

    Contract (include/sfa_abi.proto):
      OutputDescriptor.consumed_internally = true means the output is
      consumed by the KV cache system and should NOT be exposed as an
      SSA value to downstream functions.
    """

    def test_kv_split_outputs_marked_consumed(self) -> None:
        """Contract: KV split function (_a suffix) outputs marked consumed.

        When a function produces K/V cache outputs (identified by the
        _a suffix naming convention), the serialized proto MUST have
        consumed_internally=true on its OutputDescriptor.
        """
        from compiler.sfa_abi import merge_with_semantics, serialize_abi
        from gen.proto.python.sfa_abi_pb2 import SfaAbiHeader

        # Signatures use ciface wrapper names
        sigs = {
            "_mlir_ciface_main_1a": (4, 3),
            "_mlir_ciface_main_1b": (5, 1),
            "_mlir_ciface_main_2": (3, 1),
        }
        pre_lowering = {
            "functions": [
                {"name": "main_1a", "inputs": [], "outputs": [
                    ("k_cache", "tensor<?x12x?x64xf32>", True),
                    ("v_cache", "tensor<?x12x?x64xf32>", True),
                    ("q_out", "tensor<?x12x?x64xf32>", False),
                ]},
                {"name": "main_1b", "inputs": [], "outputs": [
                    ("out", "tensor<?x?x768xf32>", False),
                ]},
                {"name": "main_2", "inputs": [], "outputs": [
                    ("out", "tensor<?x?x768xf32>", False),
                ]},
            ]
        }
        metas = merge_with_semantics(sigs, pre_lowering, lowered_arg_types={}, lowered_output_types={})
        abi_bytes = serialize_abi(metas)
        abi = SfaAbiHeader()
        abi.ParseFromString(abi_bytes)

        assert len(abi.funcs) == 3, f"expected 3 funcs, got {len(abi.funcs)}"
        assert abi.funcs[0].outputs[0].consumed_internally is True, (
            "Contract violation: KV split func 'main_1a' must have consumed_internally=true"
        )
        assert abi.funcs[1].outputs[0].consumed_internally is False, (
            "Contract: 'main_1b' must have consumed_internally=false"
        )
        assert abi.funcs[2].outputs[0].consumed_internally is False, (
            "Contract: 'main_2' must have consumed_internally=false"
        )

    def test_no_outputs_defaults_to_false(self) -> None:
        """Contract: functions without consumed outputs default to false."""
        from compiler.sfa_abi import merge_with_semantics, serialize_abi
        from gen.proto.python.sfa_abi_pb2 import SfaAbiHeader

        sigs = {"_mlir_ciface_main_0": (2, 1)}
        pre_lowering = {"functions": [
            {"name": "main_0", "inputs": [], "outputs": [
                ("out", "tensor<?x?x768xf32>", False),
            ]},
        ]}
        metas = merge_with_semantics(sigs, pre_lowering, lowered_arg_types={}, lowered_output_types={})
        abi_bytes = serialize_abi(metas)
        abi = SfaAbiHeader()
        abi.ParseFromString(abi_bytes)

        assert abi.funcs[0].outputs[0].consumed_internally is False, (
            "Contract: non-KV output must default to consumed_internally=false"
        )
