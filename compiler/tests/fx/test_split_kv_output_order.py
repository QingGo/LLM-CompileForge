"""Regression tests: SD-PA split 'a' block output ordering contract.

Bug (2026-08-14): ``_make_multi_functions`` ordered exported outputs with
``sorted(exported)`` (lexicographic SSA names).  When a layer's Q/K/V
transpose SSA names cross the tens boundary (e.g. ``%transpose_8``,
``%transpose_9``, ``%transpose_10``), the sorted order becomes ``[V, Q, K]``
with consumed flags ``[True, False, True]`` instead of the contract order
``[Q, K, V]`` with ``[False, True, True]``.

The Rust runtime assumes the first consumed sub-output is K and the second
is V (``intercept.rs`` / ``compute_graph_runner.rs``).  With the broken
order, layer 3 of the opt-125m KV dylib wrote Q/V into the wrong cache
slots and rotated the SDPA operands — corrupting multistep decode logits
(generated ``[5, 5, 5, 5, 5, 6]`` instead of ``[5, 812, 9, 5, 1515]``).

Contract under test:
  * Every ``main_Xa`` function returns its outputs in semantic order
    [Q, K, V].
  * consumed_internally flags are [False, True, True] — Q is exported,
    K and V are cache-consumed.
  * This holds regardless of the lexicographic order of SSA names.
"""

from __future__ import annotations

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


def _make_ops() -> list[MlirOp]:
    """Build a two-block graph whose a-block produces Q/K/V transposes.

    The SSA names are chosen so lexicographic sorting disagrees with
    semantic order: transpose_8 (Q), transpose_9 (K), transpose_10 (V)
    sort as [transpose_10, transpose_8, transpose_9].
    """
    return [
        # ── a-block: QKV projection ──
        _op("sf.linear", "linear", ["%x_in"], ["%x"], [HIDDEN_TYPE]),
        _op("sf.transpose", "transpose", ["%x"], ["%transpose_8"], [KV_TYPE]),  # Q
        _op("sf.transpose", "transpose", ["%x"], ["%transpose_9"], [KV_TYPE]),  # K
        _op("sf.transpose", "transpose", ["%x"], ["%transpose_10"], [KV_TYPE]),  # V
        _boundary(),
        # ── b-block: SDPA consumes (Q, K, V) ──
        _op(
            "sf.scaled_dot_product_attention",
            "scaled_dot_product_attention",
            ["%transpose_8", "%transpose_9", "%transpose_10", "%mask"],
            ["%sdpa_out"],
            [HIDDEN_TYPE],
        ),
        _op("sf.linear", "linear", ["%sdpa_out"], ["%layer_out"], [HIDDEN_TYPE]),
    ]


def _split() -> tuple[list, list]:
    ops = _make_ops()
    funcs, names, _, _, _ = _make_multi_functions(
        ops,
        global_inputs=[],
        global_outputs=[("%layer_out", HIDDEN_TYPE, False)],
        weights={},
        param_names=set(),
        const_names=set(),
        base_name="main",
    )
    return funcs, names


class TestSplitKvOutputOrder:
    """The a-block must emit [Q, K, V] with flags [False, True, True]."""

    def test_a_block_outputs_are_qkv_order(self) -> None:
        """Outputs follow semantic [Q, K, V] order, not SSA-name sort."""
        funcs, names = _split()
        assert names == ["main_0a", "main_0b"], f"unexpected names: {names}"
        a_func = funcs[0]

        out_names = [name for name, _, _ in a_func.outputs]
        assert out_names == ["%transpose_8", "%transpose_9", "%transpose_10"], (
            f"a-block outputs must be [Q, K, V], got {out_names}"
        )

    def test_a_block_consumed_flags_are_false_true_true(self) -> None:
        """Only K and V are cache-consumed; Q is exported to the graph."""
        funcs, _ = _split()
        a_func = funcs[0]

        consumed = [flag for _, _, flag in a_func.outputs]
        assert consumed == [False, True, True], (
            f"a-block consumed flags must be [False, True, True], got {consumed}"
        )

    def test_b_block_keeps_single_output(self) -> None:
        funcs, _ = _split()
        b_func = funcs[1]
        assert len(b_func.outputs) == 1
        assert b_func.outputs[0][2] is False
