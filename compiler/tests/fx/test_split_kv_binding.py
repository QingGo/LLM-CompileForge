"""TDD tests: ``_make_multi_functions`` returns the KV cache binding table.

Contract (see .omo/plans/contract-hardening-next.md Phase 1):

  ``_make_multi_functions(..., cache_policy=policy)`` returns, as its fifth
  element, ``bindings: list[(func_index, output_index, slab_id)]`` — one
  entry per cache-consumed a-block output.

    * ``func_index``   — index of the a-block function in the final function
      list (== Rust ``SfaAbiHeader.funcs`` index).
    * ``output_index`` — position of the K/V output within that function's
      output list (post [Q, K, V] ordering, so K=1, V=2 in the normal case).
    * ``slab_id``      — taken from the CachePolicy intercept whose
      ``source`` operand (``operand[1]`` = K, ``operand[2]`` = V) matches.

  Without a cache_policy (or without SDPA intercepts) the table is empty.
"""

from __future__ import annotations

from compiler.cache_policy import CachePolicy
from compiler.dialect.mlir_op_types import MlirOp
from compiler.fx.split import _make_multi_functions


def _op(
    name: str,
    op_name: str,
    operands: list[str],
    results: list[str],
    output_types: list[str],
) -> MlirOp:
    return MlirOp(
        name=name,
        dialect="sf",
        op_name=op_name,
        operands=operands,
        results=results,
        attributes={},
        input_types=[],
        output_types=output_types,
    )


def _boundary() -> MlirOp:
    return MlirOp(
        name="_sentinel",
        dialect="_sentinel",
        op_name="_func_boundary",
        operands=[],
        results=[],
        attributes={},
        input_types=[],
        output_types=[],
    )


KV_TYPE = "tensor<?x12x?x64xf32>"
HIDDEN_TYPE = "tensor<?x?x768xf32>"


def _sdpa_layer_ops(tag: str) -> list[MlirOp]:
    """One split layer: a-block producing Q/K/V, b-block consuming them."""
    return [
        _op(f"sf.linear_{tag}", "linear", [f"%x_{tag}"], [f"%x_{tag}_out"], [HIDDEN_TYPE]),
        _op(f"sf.transpose_q_{tag}", "transpose", [f"%x_{tag}_out"], [f"%tq_{tag}"], [KV_TYPE]),
        _op(f"sf.transpose_k_{tag}", "transpose", [f"%x_{tag}_out"], [f"%tk_{tag}"], [KV_TYPE]),
        _op(f"sf.transpose_v_{tag}", "transpose", [f"%x_{tag}_out"], [f"%tv_{tag}"], [KV_TYPE]),
        _boundary(),
        _op(
            f"sf.scaled_dot_product_attention_{tag}",
            "scaled_dot_product_attention",
            [f"%tq_{tag}", f"%tk_{tag}", f"%tv_{tag}", "%mask"],
            [f"%sdpa_{tag}"],
            [HIDDEN_TYPE],
        ),
        _op(f"sf.linear_out_{tag}", "linear", [f"%sdpa_{tag}"], [f"%layer_{tag}_out"], [HIDDEN_TYPE]),
    ]


def _split(
    cache_policy: CachePolicy | None,
) -> tuple[list, list[str], list[tuple[int, int, str]]]:
    """Split a two-layer graph and return (funcs, names, bindings)."""
    ops = _sdpa_layer_ops("l0") + [_boundary()] + _sdpa_layer_ops("l1")
    funcs, names, _, _, bindings = _make_multi_functions(
        ops,
        global_inputs=[],
        global_outputs=[("%layer_l1_out", HIDDEN_TYPE, False)],
        weights={},
        param_names=set(),
        const_names=set(),
        base_name="main",
        cache_policy=cache_policy,
    )
    return funcs, names, bindings


def _llama_policy() -> CachePolicy:
    return CachePolicy.for_llama(num_layers=2, num_kv_heads=12, head_dim=64)


class TestSplitKvBinding:
    """split.py must emit the (func_index, output_index, slab_id) table."""

    def test_bindings_map_a_blocks_to_k_and_v_slabs(self) -> None:
        """Each a-block contributes (fi, 1, "k") and (fi, 2, "v")."""
        funcs, names, bindings = _split(_llama_policy())

        assert names == ["main_0a", "main_0b", "main_1a", "main_1b"], names
        # Blocks become functions in order: a0=0, b0=1, a1=2, b1=3.
        assert bindings == [
            (0, 1, "k"),
            (0, 2, "v"),
            (2, 1, "k"),
            (2, 2, "v"),
        ], f"unexpected bindings: {bindings}"

        # The a-block outputs are [Q, K, V]: binding indices must point at
        # the consumed K/V slots, whose SSA names match the table's claim.
        a0_outputs = [name for name, _, consumed in funcs[0].outputs]
        assert a0_outputs[1] == "%tk_l0", a0_outputs
        assert a0_outputs[2] == "%tv_l0", a0_outputs
        assert [c for _, _, c in funcs[0].outputs] == [False, True, True]

    def test_bindings_empty_without_cache_policy(self) -> None:
        """No cache_policy → empty binding table (legacy behavior intact)."""
        _funcs, names, bindings = _split(None)
        assert names == ["main_0a", "main_0b", "main_1a", "main_1b"]
        assert bindings == []

    def test_bindings_empty_for_empty_policy(self) -> None:
        """An empty CachePolicy (no SDPA intercepts) → empty table."""
        _funcs, _names, bindings = _split(CachePolicy())
        assert bindings == []

    def test_bindings_empty_for_non_sdpa_policy(self) -> None:
        """Policies without scaled_dot_product_attention intercepts bind nothing."""
        policy = CachePolicy.for_rwkv(num_layers=2, state_dim=64)
        _funcs, _names, bindings = _split(policy)
        assert bindings == []
