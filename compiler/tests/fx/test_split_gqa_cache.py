"""GQA repeat-kv cache split contract (S3-pre LLaMA preflight finding).

LLaMA-3.2-1B exports ``scaled_dot_product_attention`` on K/V **after**
``unsqueeze -> expand -> view`` has expanded 8 KV heads to 32 query heads.
Caching those expanded tensors loses GQA and, worse, makes the ABI head
count disagree with the config-driven CachePolicy (32 != 8).

Contract under test:
  * boundary insertion walks the repeat-kv chain and splits **before** the
    first repeat op, not at the SDPA op;
  * the a-block consumes the pre-repeat K/V (8 heads);
  * the b-block performs repeat-kv itself from cache-read K/V;
  * standard dense SDPA (OPT) still splits at SDPA unchanged.
"""

from __future__ import annotations

from compiler.cache_policy import CachePolicy
from compiler.dialect.mlir_op_types import MlirOp
from compiler.fx.split import _make_multi_functions, sdpa_cache_boundary_indices


def _op(
    name: str,
    op_name: str,
    operands: list[str],
    results: list[str],
    output_types: list[str],
) -> MlirOp:
    return MlirOp(
        name=f"sf.{name}",
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


KV8_TYPE = "tensor<1x8x?x64xbf16>"
KV32_TYPE = "tensor<1x32x?x64xf32>"
Q_TYPE = "tensor<1x1x?x?x64xbf16>"
HIDDEN_TYPE = "tensor<1x?x2048xbf16>"


def _gqa_layer(tag: str) -> list[MlirOp]:
    """One GQA layer: K/V repeat-kv precedes SDPA in the same block."""
    return [
        _op(f"linear_q_{tag}", "linear", [f"%x_{tag}"], [f"%q_proj_{tag}"], [HIDDEN_TYPE]),
        _op(f"view_q_{tag}", "view", [f"%q_proj_{tag}"], [f"%q_4d_{tag}"], [Q_TYPE]),
        _op(f"linear_k_{tag}", "linear", [f"%x_{tag}"], [f"%k_proj_{tag}"], [HIDDEN_TYPE]),
        _op(f"view_k_{tag}", "view", [f"%k_proj_{tag}"], [f"%k_4d_{tag}"], [KV8_TYPE]),
        _op(f"add_rope_k_{tag}", "add", [f"%k_4d_{tag}", f"%rope_k_{tag}"], [f"%k_rope_{tag}"], [KV8_TYPE]),
        _op(f"linear_v_{tag}", "linear", [f"%x_{tag}"], [f"%v_proj_{tag}"], [HIDDEN_TYPE]),
        _op(f"view_v_{tag}", "view", [f"%v_proj_{tag}"], [f"%v_4d_{tag}"], [KV8_TYPE]),
        _op(f"unsqueeze_k_{tag}", "unsqueeze", [f"%k_rope_{tag}"], [f"%k_u_{tag}"], ["tensor<1x8x1x?x64xbf16>"]),
        _op(f"expand_k_{tag}", "expand", [f"%k_u_{tag}"], [f"%k_e_{tag}"], ["tensor<1x8x4x?x64xbf16>"]),
        _op(f"view_k_exp_{tag}", "view", [f"%k_e_{tag}"], [f"%k_exp_{tag}"], [KV32_TYPE]),
        _op(f"unsqueeze_v_{tag}", "unsqueeze", [f"%v_4d_{tag}"], [f"%v_u_{tag}"], ["tensor<1x8x1x?x64xbf16>"]),
        _op(f"expand_v_{tag}", "expand", [f"%v_u_{tag}"], [f"%v_e_{tag}"], ["tensor<1x8x4x?x64xbf16>"]),
        _op(f"view_v_exp_{tag}", "view", [f"%v_e_{tag}"], [f"%v_exp_{tag}"], [KV32_TYPE]),
        _op(
            f"sdpa_{tag}",
            "scaled_dot_product_attention",
            [f"%q_4d_{tag}", f"%k_exp_{tag}", f"%v_exp_{tag}", "%mask"],
            [f"%sdpa_{tag}"],
            [HIDDEN_TYPE],
        ),
        _op(f"linear_out_{tag}", "linear", [f"%sdpa_{tag}"], [f"%layer_{tag}_out"], [HIDDEN_TYPE]),
    ]


def _dense_layer(tag: str) -> list[MlirOp]:
    """OPT-style layer: SDPA directly consumes the a-block K/V transposes."""
    return [
        _op(f"linear_{tag}", "linear", [f"%x_{tag}"], [f"%x_{tag}_out"], [HIDDEN_TYPE]),
        _op(f"transpose_q_{tag}", "transpose", [f"%x_{tag}_out"], [f"%tq_{tag}"], [KV8_TYPE]),
        _op(f"transpose_k_{tag}", "transpose", [f"%x_{tag}_out"], [f"%tk_{tag}"], [KV8_TYPE]),
        _op(f"transpose_v_{tag}", "transpose", [f"%x_{tag}_out"], [f"%tv_{tag}"], [KV8_TYPE]),
        _op(
            f"sdpa_{tag}",
            "scaled_dot_product_attention",
            [f"%tq_{tag}", f"%tk_{tag}", f"%tv_{tag}", "%mask"],
            [f"%sdpa_{tag}"],
            [HIDDEN_TYPE],
        ),
        _op(f"linear_out_{tag}", "linear", [f"%sdpa_{tag}"], [f"%layer_{tag}_out"], [HIDDEN_TYPE]),
    ]


class TestSdpaCacheBoundaryIndices:
    def test_gqa_boundary_is_before_first_repeat_op(self) -> None:
        ops = _gqa_layer("l0")
        boundaries = sdpa_cache_boundary_indices(ops)
        k_u = next(i for i, op in enumerate(ops) if op.results == ["%k_u_l0"])
        assert boundaries == [k_u]

    def test_dense_boundary_is_at_sdpa_op(self) -> None:
        ops = _dense_layer("l0")
        sdpa_idx = next(i for i, op in enumerate(ops) if op.op_name == "scaled_dot_product_attention")
        assert sdpa_cache_boundary_indices(ops) == [sdpa_idx]

    def test_two_gqa_layers_get_two_boundaries(self) -> None:
        ops = _gqa_layer("l0") + _gqa_layer("l1")
        boundaries = sdpa_cache_boundary_indices(ops)
        k_u0 = next(i for i, op in enumerate(ops) if op.results == ["%k_u_l0"])
        k_u1 = next(i for i, op in enumerate(ops) if op.results == ["%k_u_l1"])
        assert boundaries == [k_u0, k_u1]


class TestSplitGqaCache:
    def test_a_block_caches_pre_repeat_kv(self) -> None:
        """a-block outputs [Q, pre-repeat K, pre-repeat V], flags [F,T,T]."""
        ops = _gqa_layer("l0")
        k_u = next(i for i, op in enumerate(ops) if op.results == ["%k_u_l0"])
        sentinel_ops = ops[:k_u] + [_boundary()] + ops[k_u:]
        funcs, names, _, _, bindings = _make_multi_functions(
            sentinel_ops,
            global_inputs=[],
            global_outputs=[("%layer_l0_out", HIDDEN_TYPE, False)],
            weights={},
            param_names=set(),
            const_names=set(),
            base_name="main",
            cache_policy=CachePolicy.for_llama(num_layers=1, num_kv_heads=8, head_dim=64),
        )
        assert names == ["main_0a", "main_0b"], names
        a_func, b_func = funcs
        out_names = [name for name, _, _ in a_func.outputs]
        assert out_names == ["%q_4d_l0", "%k_rope_l0", "%v_4d_l0"], out_names
        assert [c for _, _, c in a_func.outputs] == [False, True, True]
        assert bindings == [(0, 1, "k"), (0, 2, "v")]

        # The b-block still performs repeat-kv itself before SDPA.
        assert b_func.ops[0].op_name == "unsqueeze"
        assert any(op.op_name == "scaled_dot_product_attention" for op in b_func.ops)

    def test_dense_split_keeps_legacy_contract(self) -> None:
        """Dense SDPA a-block still caches direct K/V operands."""
        ops = _dense_layer("l0")
        sdpa_idx = next(i for i, op in enumerate(ops) if op.op_name == "scaled_dot_product_attention")
        sentinel_ops = ops[:sdpa_idx] + [_boundary()] + ops[sdpa_idx:]
        funcs, names, _, _, bindings = _make_multi_functions(
            sentinel_ops,
            global_inputs=[],
            global_outputs=[("%layer_l0_out", HIDDEN_TYPE, False)],
            weights={},
            param_names=set(),
            const_names=set(),
            base_name="main",
            cache_policy=CachePolicy.for_llama(num_layers=1, num_kv_heads=8, head_dim=64),
        )
        assert names == ["main_0a", "main_0b"], names
        a_func = funcs[0]
        assert [name for name, _, _ in a_func.outputs] == ["%tq_l0", "%tk_l0", "%tv_l0"]
        assert [c for _, _, c in a_func.outputs] == [False, True, True]
        assert bindings == [(0, 1, "k"), (0, 2, "v")]
