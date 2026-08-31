"""SSA result-name uniqueness regression tests.

LLaMA-3.2-1B preflight exposed duplicate ``%attn_mask`` results: two
``aten.expand`` nodes in the same layer (GQA K and V repeat-kv) were both
assigned the fixed semantic name ``%attn_mask``.  Duplicate SSA names in a
function are invalid and break producer lookup during split/wiring.
"""

from __future__ import annotations

from types import SimpleNamespace

from compiler.fx.converter import _semantic_ssa_name


def _node(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name, args=())


class TestSemanticSsaNames:
    def test_expand_results_are_unique(self) -> None:
        first = _semantic_ssa_name("expand", _node("expand"), 10, {}, {})
        second = _semantic_ssa_name("expand", _node("expand_1"), 11, {}, {})
        assert first != second
        assert first.startswith("%attn_mask")
        assert second.startswith("%attn_mask")

    def test_non_expand_names_stay_stable(self) -> None:
        assert _semantic_ssa_name("add", _node("add"), 7, {}, {}) == "%add"
        assert _semantic_ssa_name("view", _node("view_3"), 9, {}, {}) == "%reshape_9"
