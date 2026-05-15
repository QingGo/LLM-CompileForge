"""Tests for compiler.mlir_dialect — sf dialect definition and shape inference.

Each test creates its own MLIR context via ``with ir.Context() as ctx:``
— the documented and safe pattern for context lifecycle management.
"""

from __future__ import annotations

import mlir.ir as ir
import pytest


def _t(shape, elt_str="f32"):
    elt_map = {
        "f32": ir.F32Type.get(), "f16": ir.F16Type.get(),
        "bf16": ir.BF16Type.get(), "f64": ir.F64Type.get(),
        "i32": ir.IntegerType.get_signless(32),
        "i64": ir.IntegerType.get_signless(64),
    }
    elt = elt_map.get(elt_str, ir.F32Type.get())
    return ir.RankedTensorType.get(list(shape), elt)


@pytest.mark.unit
class TestShapeInference:

    def test_elementwise(self):
        with ir.Context() as ctx:
            ctx.allow_unregistered_dialects = True
            with ir.Location.unknown(ctx):
                from compiler.mlir_dialect.shape_inference import infer_output_type
                r = infer_output_type("add", [_t((2, 64, 128))])
                assert tuple(r[0].shape) == (2, 64, 128)

    def test_matmul_2d(self):
        with ir.Context() as ctx:
            ctx.allow_unregistered_dialects = True
            with ir.Location.unknown(ctx):
                from compiler.mlir_dialect.shape_inference import infer_output_type
                r = infer_output_type("matmul", [_t((32, 64)), _t((64, 128))])
                assert tuple(r[0].shape) == (32, 128)

    def test_matmul_3d(self):
        with ir.Context() as ctx:
            ctx.allow_unregistered_dialects = True
            with ir.Location.unknown(ctx):
                from compiler.mlir_dialect.shape_inference import infer_output_type
                r = infer_output_type("matmul", [_t((4, 32, 64)), _t((64, 128))])
                assert tuple(r[0].shape) == (4, 32, 128)

    def test_linear(self):
        with ir.Context() as ctx:
            ctx.allow_unregistered_dialects = True
            with ir.Location.unknown(ctx):
                from compiler.mlir_dialect.shape_inference import infer_output_type
                r = infer_output_type("linear", [_t((1, 64, 1024)), _t((4096, 1024))])
                assert tuple(r[0].shape) == (1, 64, 4096)

    def test_view(self):
        with ir.Context() as ctx:
            ctx.allow_unregistered_dialects = True
            with ir.Location.unknown(ctx):
                from compiler.mlir_dialect.shape_inference import infer_output_type
                r = infer_output_type("view", [_t((2, 3, 4))], shape=(6, 4))
                assert tuple(r[0].shape) == (6, 4)

    def test_unsqueeze(self):
        with ir.Context() as ctx:
            ctx.allow_unregistered_dialects = True
            with ir.Location.unknown(ctx):
                from compiler.mlir_dialect.shape_inference import infer_output_type
                r = infer_output_type("unsqueeze", [_t((2, 64))], dim=0)
                assert tuple(r[0].shape) == (1, 2, 64)

    def test_transpose(self):
        with ir.Context() as ctx:
            ctx.allow_unregistered_dialects = True
            with ir.Location.unknown(ctx):
                from compiler.mlir_dialect.shape_inference import infer_output_type
                r = infer_output_type("transpose", [_t((1, 2, 3, 4))], dim0=1, dim1=2)
                assert tuple(r[0].shape) == (1, 3, 2, 4)

    def test_slice(self):
        with ir.Context() as ctx:
            ctx.allow_unregistered_dialects = True
            with ir.Location.unknown(ctx):
                from compiler.mlir_dialect.shape_inference import infer_output_type
                r = infer_output_type("slice", [_t((2, 100, 256))], dim=1, start=1, end=5)
                assert tuple(r[0].shape) == (2, 4, 256)

    def test_select(self):
        with ir.Context() as ctx:
            ctx.allow_unregistered_dialects = True
            with ir.Location.unknown(ctx):
                from compiler.mlir_dialect.shape_inference import infer_output_type
                r = infer_output_type("select", [_t((2, 100, 256))], dim=1, index=5)
                assert tuple(r[0].shape) == (2, 256)

    def test_cat(self):
        with ir.Context() as ctx:
            ctx.allow_unregistered_dialects = True
            with ir.Location.unknown(ctx):
                from compiler.mlir_dialect.shape_inference import infer_output_type
                r = infer_output_type("cat", [_t((2, 32, 64)), _t((2, 48, 64))], dim=1)
                assert tuple(r[0].shape) == (2, 80, 64)

    def test_mean(self):
        with ir.Context() as ctx:
            ctx.allow_unregistered_dialects = True
            with ir.Location.unknown(ctx):
                from compiler.mlir_dialect.shape_inference import infer_output_type
                r = infer_output_type("mean", [_t((4, 32, 64))], dim=1, keepdim=False)
                assert tuple(r[0].shape) == (4, 64)

    def test_embedding(self):
        with ir.Context() as ctx:
            ctx.allow_unregistered_dialects = True
            with ir.Location.unknown(ctx):
                from compiler.mlir_dialect.shape_inference import infer_output_type
                # Arguments: [weight, indices] (weight first, then indices)
                r = infer_output_type("embedding", [_t((32000, 1024)), _t((1, 64), "i64")])
                assert tuple(r[0].shape) == (1, 64, 1024)

    def test_sdpa(self):
        with ir.Context() as ctx:
            ctx.allow_unregistered_dialects = True
            with ir.Location.unknown(ctx):
                from compiler.mlir_dialect.shape_inference import infer_output_type
                q = _t((1, 8, 64, 128))
                r = infer_output_type("scaled_dot_product_attention", [q, q, q])
                assert tuple(r[0].shape) == (1, 8, 64, 128)


@pytest.mark.unit
class TestSfModule:

    def test_basic(self):
        with ir.Context() as ctx:
            ctx.allow_unregistered_dialects = True
            with ir.Location.unknown(ctx):
                from compiler.mlir_dialect.builder import SfModule
                mod = SfModule("main", input_types=[_t((1, 64))])
                r = mod.add_op("add", [mod.inputs[0], mod.inputs[0]])
                r = mod.add_op("relu", [r])
                mod.set_outputs([r])
                mlir = str(mod)
                assert "sf.add" in mlir and "sf.relu" in mlir
                assert "tensor<1x64xf32>" in mlir
                ir.Module.parse(mlir)

    def test_matmul_chain(self):
        with ir.Context() as ctx:
            ctx.allow_unregistered_dialects = True
            with ir.Location.unknown(ctx):
                from compiler.mlir_dialect.builder import SfModule
                mod = SfModule("main", input_types=[_t((1, 10, 256))])
                w1 = mod.add_weight_op("w1")
                w2 = mod.add_weight_op("w2")
                r = mod.add_op("matmul", [mod.inputs[0], w1])
                r = mod.add_op("silu", [r])
                r = mod.add_op("matmul", [r, w2])
                mod.set_outputs([r])
                mlir = str(mod)
                assert "w1" in mlir
                ir.Module.parse(mlir)

    def test_multiple_outputs(self):
        with ir.Context() as ctx:
            ctx.allow_unregistered_dialects = True
            with ir.Location.unknown(ctx):
                from compiler.mlir_dialect.builder import SfModule
                mod = SfModule("main", input_types=[_t((1, 64))])
                r = mod.add_op("add", [mod.inputs[0], mod.inputs[0]])
                mod.set_outputs([r, mod.inputs[0]])
                ir.Module.parse(str(mod))


@pytest.mark.unit
class TestSfOpRegistry:

    def test_all_ops_registered(self):
        from compiler.mlir_dialect.sf import _ALL_OPS, get_op_class
        assert len(_ALL_OPS) >= 45
        for name in ["add", "matmul", "silu", "rms_norm", "softmax",
                      "scaled_dot_product_attention", "view", "transpose",
                      "slice", "cat", "embedding", "fused_silu_mul",
                      "fused_rms_norm_matmul", "weight"]:
            assert get_op_class(f"sf.{name}") is not None

    def test_op_names_match_kwargs(self):
        from compiler.mlir_dialect.sf import _ALL_OPS
        from compiler.mlir_dialect.shape_inference import _INFERENCE_TABLE
        for op_name in _ALL_OPS:
            short = op_name[len("sf."):]
            assert short in _INFERENCE_TABLE, \
                f"op {op_name} has no shape inference (missing infer_{short})"
