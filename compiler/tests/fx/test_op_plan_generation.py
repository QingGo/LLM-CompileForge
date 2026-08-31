"""Seam tests for compiler/op_plan.py (Phase 5 M1).

The generator must work purely from the parsed sf-dialect MlirModule and
metadata, with no dependency on the Rust runtime.  The synthetic module
here mirrors the OPT KV decoder-layer split contract one-to-one.
"""

from __future__ import annotations

import json
import os

import pytest

from compiler.dialect.mlir_op_types import MlirFunction, MlirModule, MlirOp
from compiler.op_plan import PLAN_OP_NAMES, generate_op_plan, validate_op_plan
from gen.proto.python import sfa_abi_pb2


def _op(name: str, dialect: str, operands: list[str], result: str, attrs: dict, out_type: str) -> MlirOp:
    return MlirOp(
        name=f"{dialect}.{name}",
        dialect=dialect,
        op_name=name,
        operands=operands,
        results=[result],
        attributes=attrs,
        input_types=[],
        output_types=[out_type],
    )


def _layer_module() -> tuple[MlirModule, dict]:
    # main_0: provides hidden, mask, and a scalar sym-size.
    main0 = MlirFunction(
        name="main_0",
        inputs=[("%input_ids", "tensor<1x8xi64>")],
        outputs=[
            ("%add_5", "tensor<?x?x4xf32>", False),
            ("%attn_mask", "tensor<?x1x?x?xf32>", False),
            ("%input_ids_dim_1", "tensor<1xf32>", False),
            ("%position_ids_dim_0", "tensor<1xf32>", False),
        ],
        ops=[],
    )

    # main_1a: attn ln + QKV + layout, K/V consumed.
    a_ops = [
        _op(
            "layer_norm", "sf", ["%add_5", "%ln_w", "%ln_b"], "%ln_out", {"normalized_shape": [4]}, "tensor<?x?x4xf32>"
        ),
        _op("linear", "sf", ["%ln_out", "%q_w", "%q_b"], "%q_lin", {}, "tensor<?x?x4xf32>"),
        _op("mul", "sf", ["%q_lin", "%scale"], "%q_scaled", {}, "tensor<?x?x4xf32>"),
        _op(
            "view",
            "sf",
            ["%q_scaled", "%position_ids_dim_0"],
            "%q_view",
            {"shape": [-2, -1, 2, 2]},
            "tensor<?x?x2x2xf32>",
        ),
        _op("transpose", "sf", ["%q_view"], "%q_bnsd", {"dim0": 1, "dim1": 2}, "tensor<?x2x?x2xf32>"),
        _op("linear", "sf", ["%ln_out", "%k_w", "%k_b"], "%k_lin", {}, "tensor<?x?x4xf32>"),
        _op(
            "view", "sf", ["%k_lin", "%position_ids_dim_0"], "%k_view", {"shape": [-2, -1, 2, 2]}, "tensor<?x?x2x2xf32>"
        ),
        _op("transpose", "sf", ["%k_view"], "%k_bnsd", {"dim0": 1, "dim1": 2}, "tensor<?x2x?x2xf32>"),
        _op("linear", "sf", ["%ln_out", "%v_w", "%v_b"], "%v_lin", {}, "tensor<?x?x4xf32>"),
        _op(
            "view", "sf", ["%v_lin", "%position_ids_dim_0"], "%v_view", {"shape": [-2, -1, 2, 2]}, "tensor<?x?x2x2xf32>"
        ),
        _op("transpose", "sf", ["%v_view"], "%v_bnsd", {"dim0": 1, "dim1": 2}, "tensor<?x2x?x2xf32>"),
    ]
    main1a = MlirFunction(
        name="main_1a",
        inputs=[
            ("%position_ids_dim_0", "tensor<1xf32>"),
            ("%add_5", "tensor<?x?x4xf32>"),
            ("%scale", "tensor<1xf32>"),
            ("%k_b", "tensor<4xf32>"),
            ("%k_w", "tensor<4x4xf32>"),
            ("%ln_b", "tensor<4xf32>"),
            ("%ln_w", "tensor<4xf32>"),
            ("%q_b", "tensor<4xf32>"),
            ("%q_w", "tensor<4x4xf32>"),
            ("%v_b", "tensor<4xf32>"),
            ("%v_w", "tensor<4x4xf32>"),
        ],
        outputs=[
            ("%q_bnsd", "tensor<?x2x?x2xf32>", False),
            ("%k_bnsd", "tensor<?x2x?x2xf32>", True),
            ("%v_bnsd", "tensor<?x2x?x2xf32>", True),
        ],
        ops=a_ops,
        param_weight_names={"k_b", "k_w", "ln_b", "ln_w", "q_b", "q_w", "v_b", "v_w"},
        const_weight_names={"scale"},
    )

    # main_1b: SDPA + out proj + residual + MLP.
    b_ops = [
        _op(
            "scaled_dot_product_attention",
            "sf",
            ["%q_bnsd", "%k_bnsd", "%v_bnsd", "%attn_mask"],
            "%attn",
            {"scale": 1.0},
            "tensor<?x2x?x2xf32>",
        ),
        _op("transpose", "sf", ["%attn"], "%attn_t", {"dim0": 1, "dim1": 2}, "tensor<?x?x2x2xf32>"),
        _op(
            "view",
            "sf",
            ["%attn_t", "%position_ids_dim_0", "%input_ids_dim_1"],
            "%attn_flat",
            {"shape": [-2, -3, -1]},
            "tensor<?x?x?xf32>",
        ),
        _op("linear", "sf", ["%attn_flat", "%o_w", "%o_b"], "%out", {}, "tensor<?x?x4xf32>"),
        _op("identity", "sf", ["%out"], "%dropout", {"dtype": 0.1}, "tensor<?x?x4xf32>"),
        _op("add", "sf", ["%add_5", "%dropout"], "%res", {}, "tensor<?x?x4xf32>"),
        _op("view", "sf", ["%res"], "%res2d", {"shape": [-1, 4]}, "tensor<?x4xf32>"),
        _op("layer_norm", "sf", ["%res2d", "%ln2_w", "%ln2_b"], "%ln2", {"normalized_shape": [4]}, "tensor<?x4xf32>"),
        _op("linear", "sf", ["%ln2", "%fc1_w", "%fc1_b"], "%fc1", {}, "tensor<?x8xf32>"),
        _op("relu", "sf", ["%fc1"], "%act", {}, "tensor<?x8xf32>"),
        _op("linear", "sf", ["%act", "%fc2_w", "%fc2_b"], "%fc2", {}, "tensor<?x4xf32>"),
        _op("identity", "sf", ["%fc2"], "%dropout2", {"dtype": 0.1}, "tensor<?x4xf32>"),
        _op("add", "sf", ["%res2d", "%dropout2"], "%mlp_res", {}, "tensor<?x4xf32>"),
        _op(
            "view",
            "sf",
            ["%mlp_res", "%position_ids_dim_0", "%input_ids_dim_1"],
            "%hidden",
            {"shape": [-2, -3, 4]},
            "tensor<?x?x4xf32>",
        ),
    ]
    main1b = MlirFunction(
        name="main_1b",
        inputs=[
            ("%input_ids_dim_1", "tensor<1xf32>"),
            ("%q_bnsd", "tensor<?x2x?x2xf32>"),
            ("%k_bnsd", "tensor<?x2x?x2xf32>"),
            ("%v_bnsd", "tensor<?x2x?x2xf32>"),
            ("%attn_mask", "tensor<?x1x?x?xf32>"),
            ("%add_5", "tensor<?x?x4xf32>"),
            ("%fc1_b", "tensor<8xf32>"),
            ("%fc1_w", "tensor<8x4xf32>"),
            ("%fc2_b", "tensor<4xf32>"),
            ("%fc2_w", "tensor<4x8xf32>"),
            ("%ln2_b", "tensor<4xf32>"),
            ("%ln2_w", "tensor<4xf32>"),
            ("%o_b", "tensor<4xf32>"),
            ("%o_w", "tensor<4x4xf32>"),
            ("%position_ids_dim_0", "tensor<1xf32>"),
        ],
        outputs=[("%hidden", "tensor<?x?x4xf32>", False)],
        ops=b_ops,
        param_weight_names={"fc1_b", "fc1_w", "fc2_b", "fc2_w", "ln2_b", "ln2_w", "o_b", "o_w"},
    )

    module = MlirModule(
        functions=[main0, main1a, main1b],
        metadata={},
        chain_order=["main_0", "main_1a", "main_1b"],
    )
    metadata = {
        "cache_bindings": [(1, 1, "k"), (1, 2, "v")],
    }
    return module, metadata


def test_generate_op_plan_synthetic_layer() -> None:
    module, metadata = _layer_module()
    raw = generate_op_plan(module, metadata)
    assert raw is not None
    plan = sfa_abi_pb2.OpPlan()
    plan.ParseFromString(raw)

    op_names = [n.op_name for n in plan.nodes]
    assert set(op_names) <= PLAN_OP_NAMES
    assert "layer_norm" in op_names
    assert "linear_transb" in op_names
    assert "attention_causal" in op_names

    # Every node output/input is typed float32 and has a source function.
    for node in plan.nodes:
        assert node.source_func_indices
        assert node.outputs[0].spec.dtype == "float32"

    # Boundary inputs must project through FUNC_OUTPUT.
    boundary = [i for n in plan.nodes for i in n.inputs if i.source == sfa_abi_pb2.OpPlanInput.FUNC_OUTPUT]
    assert boundary, "main_0 boundary projections are required"
    assert all(i.HasField("func_producer") for i in boundary)

    # Cache projection is bijective with metadata cache_bindings.
    cache_proj = {
        (o.cache.source_func_index, o.cache.source_output_index): o.cache.slab_id
        for n in plan.nodes
        for o in n.outputs
        if o.HasField("cache")
    }
    assert cache_proj == {(1, 1): "k", (1, 2): "v"}

    # Func outputs: [Q, K, V] for main_1a and [hidden] for main_1b.
    assert [(fo.func_index, fo.output_index, fo.consumed_internally) for fo in plan.func_outputs] == [
        (1, 0, False),
        (1, 1, True),
        (1, 2, True),
        (2, 0, False),
    ]

    # The last visible hidden output is the plan global output.
    last_fo = plan.func_outputs[-1]
    assert plan.global_output.node_index == last_fo.value.node_index


def test_validate_rejects_bad_cache_projection() -> None:
    module, metadata = _layer_module()
    raw = generate_op_plan(module, metadata)
    plan = sfa_abi_pb2.OpPlan()
    plan.ParseFromString(raw)

    bad_metadata = dict(metadata, cache_bindings=[(1, 1, "k")])
    with pytest.raises(ValueError, match="cache projection|cache binding|not projected"):
        validate_op_plan(plan, module, bad_metadata)


@pytest.mark.skipif(
    not os.path.exists("outputs/compiled/opt_125m_kv/model.mlir"),
    reason="compiled opt_125m_kv artifact not present",
)
def test_generate_op_plan_real_opt_artifact() -> None:
    from compiler.artifact import _parse_mlir_text

    module = _parse_mlir_text(open("outputs/compiled/opt_125m_kv/model.mlir").read())
    metadata = json.load(open("outputs/compiled/opt_125m_kv/metadata.json"))
    wc = metadata.get("weight_classification", {})
    for func in module.functions:
        fwc = wc.get(func.name, {})
        func.param_weight_names = set(fwc.get("params", []))
        func.const_weight_names = set(fwc.get("constants", []))

    raw = generate_op_plan(module, metadata)
    assert raw is not None
    plan = sfa_abi_pb2.OpPlan()
    plan.ParseFromString(raw)
    assert len(plan.nodes) > 200
    assert len(plan.func_outputs) == 48
    assert sum(1 for n in plan.nodes for o in n.outputs if o.HasField("cache")) == 24
