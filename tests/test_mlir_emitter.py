"""Tests for compiler/mlir_emitter.py — real MLIR text generation."""

import pytest

from compiler.ir import IrFunction, IrModule, IrOp, IrType
from compiler.mlir_emitter import (
    _attr_to_mlir,
    _dtype_to_mlir,
    _map_op_to_mlir,
    _tensor_type_str,
    ir_module_to_mlir,
)

# ═══════════════════════════════════════════════════════════
# Utils
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
class TestDtypeMapping:
    def test_float32(self):
        assert _dtype_to_mlir("float32") == "f32"

    def test_float16(self):
        assert _dtype_to_mlir("float16") == "f16"

    def test_bfloat16(self):
        assert _dtype_to_mlir("bfloat16") == "bf16"

    def test_int64(self):
        assert _dtype_to_mlir("int64") == "i64"

    def test_unknown_dtype_falls_back_to_f32(self):
        assert _dtype_to_mlir("quant8") == "f32"


@pytest.mark.unit
class TestTensorType:
    def test_static_shape(self):
        assert _tensor_type_str("float32", (1, 4)) == "tensor<1x4xf32>"

    def test_dynamic_shape(self):
        assert _tensor_type_str("float32", (None, 4)) == "tensor<?x4xf32>"

    def test_scalar(self):
        assert _tensor_type_str("float32", ()) == "tensor<f32>"

    def test_bfloat16_dynamic(self):
        assert _tensor_type_str("bfloat16", (None, None)) == "tensor<?x?xbf16>"


@pytest.mark.unit
class TestOpMapping:
    def test_matmul_to_sf(self):
        assert _map_op_to_mlir("matmul") == ("sf", "matmul")

    def test_add_to_sf(self):
        assert _map_op_to_mlir("add") == ("sf", "add")

    def test_silu_to_sf(self):
        assert _map_op_to_mlir("silu") == ("sf", "silu")

    def test_rms_norm_to_sf(self):
        assert _map_op_to_mlir("rms_norm") == ("sf", "rms_norm")

    def test_fused_op_to_sf(self):
        assert _map_op_to_mlir("fused_rms_norm_matmul") == ("sf", "fused_rms_norm_matmul")

    def test_unknown_op_to_sf(self):
        assert _map_op_to_mlir("nonexistent_op") == ("sf", "nonexistent_op")


@pytest.mark.unit
class TestAttrConversion:
    def test_bool(self):
        assert _attr_to_mlir(True) == "true"
        assert _attr_to_mlir(False) == "false"

    def test_int(self):
        assert _attr_to_mlir(42) == "42"

    def test_float(self):
        assert "0.000000e+00" in _attr_to_mlir(0.0)

    def test_string(self):
        assert _attr_to_mlir("hello") == '"hello"'

    def test_list(self):
        assert _attr_to_mlir([1, 2, 3]) == "[1, 2, 3]"


# ═══════════════════════════════════════════════════════════
# MLIR emission
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
class TestMLIREmitter:
    def test_emits_valid_module_wrapper(self):
        func = IrFunction(name="main", inputs=[], outputs=[])
        mod = IrModule(functions=[func])
        mlir = ir_module_to_mlir(mod)
        assert mlir.startswith("module {")
        assert mlir.rstrip().endswith("}")

    def test_emits_func_with_input_and_output(self):
        func = IrFunction(
            name="main",
            inputs=[("input_ids", IrType(dtype="int64", shape=(None, 4)))],
            outputs=[("logits", IrType(dtype="float32", shape=(None, 4, 50272)))],
        )
        mod = IrModule(functions=[func])
        mlir = ir_module_to_mlir(mod)
        assert "func.func @main" in mlir
        assert "tensor<?x4xi64>" in mlir
        assert "func.return" in mlir

    def test_emits_linear_op(self):
        func = IrFunction(
            name="main",
            inputs=[("x", IrType(dtype="float32", shape=(None, 4)))],
            outputs=[("y", IrType(dtype="float32", shape=(None, 4)))],
            ops=[IrOp(name="matmul", inputs=["x", "w"], outputs=["y1"])],
        )
        mod = IrModule(functions=[func])
        mlir = ir_module_to_mlir(mod)
        assert '"sf.matmul"' in mlir

    def test_emits_add_op(self):
        func = IrFunction(
            name="main",
            inputs=[("a", IrType(dtype="float32", shape=(4,)))],
            outputs=[("c", IrType(dtype="float32", shape=(4,)))],
            ops=[IrOp(name="add", inputs=["a", "b"], outputs=["c"])],
        )
        mod = IrModule(functions=[func])
        mlir = ir_module_to_mlir(mod)
        assert '"sf.add"' in mlir

    def test_emits_gelu_op(self):
        func = IrFunction(
            name="main",
            inputs=[("x", IrType(dtype="float32", shape=(4,)))],
            outputs=[("y", IrType(dtype="float32", shape=(4,)))],
            ops=[IrOp(name="gelu", inputs=["x"], outputs=["y"])],
        )
        mod = IrModule(functions=[func])
        mlir = ir_module_to_mlir(mod)
        assert '"sf.gelu"' in mlir

    def test_emits_softmax_op(self):
        func = IrFunction(
            name="main",
            inputs=[("x", IrType(dtype="float32", shape=(None, 8)))],
            outputs=[("y", IrType(dtype="float32", shape=(None, 8)))],
            ops=[IrOp(name="softmax", inputs=["x"], outputs=["y"], attributes={"dim": -1})],
        )
        mod = IrModule(functions=[func])
        mlir = ir_module_to_mlir(mod)
        assert '"sf.softmax"' in mlir
        assert "dim" in mlir

    def test_emits_fused_rms_norm_matmul(self):
        func = IrFunction(
            name="main",
            inputs=[("x", IrType(dtype="float32", shape=(None, 16)))],
            outputs=[("y", IrType(dtype="float32", shape=(None, 16)))],
            ops=[
                IrOp(
                    name="fused_rms_norm_matmul",
                    inputs=["x", "w"],
                    outputs=["y"],
                    attributes={"eps": 1e-5},
                )
            ],
        )
        mod = IrModule(functions=[func])
        mlir = ir_module_to_mlir(mod)
        assert '"sf.fused_rms_norm_matmul"' in mlir

    def test_emits_weights_as_sf_weight_constants(self):
        import torch

        func = IrFunction(
            name="main",
            inputs=[("x", IrType(dtype="float32", shape=(4,)))],
            outputs=[("y", IrType(dtype="float32", shape=(4,)))],
            ops=[IrOp(name="add", inputs=["x", "w"], outputs=["y"])],
            weights={"w": torch.tensor([1.0, 2.0, 3.0, 4.0])},
        )
        mod = IrModule(functions=[func])
        mlir = ir_module_to_mlir(mod)
        assert '"sf.weight"' in mlir
        assert "w" in mlir

    def test_emits_multiple_ops_in_order(self):
        func = IrFunction(
            name="main",
            inputs=[("x", IrType(dtype="float32", shape=(4,)))],
            outputs=[("z", IrType(dtype="float32", shape=(4,)))],
            ops=[
                IrOp(name="gelu", inputs=["x"], outputs=["y"]),
                IrOp(name="add", inputs=["y", "b"], outputs=["z"]),
            ],
        )
        mod = IrModule(functions=[func])
        mlir = ir_module_to_mlir(mod)
        gelu_pos = mlir.find('"sf.gelu"')
        add_pos = mlir.find('"sf.add"')
        assert gelu_pos > 0
        assert add_pos > 0
        assert gelu_pos < add_pos

    def test_emits_ssa_chaining(self):
        func = IrFunction(
            name="main",
            inputs=[("x", IrType(dtype="float32", shape=(4,)))],
            outputs=[("z", IrType(dtype="float32", shape=(4,)))],
            ops=[
                IrOp(name="add", inputs=["x", "b"], outputs=["y"]),
                IrOp(name="mul", inputs=["y", "c"], outputs=["z"]),
            ],
        )
        mod = IrModule(functions=[func])
        mlir = ir_module_to_mlir(mod)
        for ssa in ["%0", "%1", "%2"]:
            assert ssa in mlir

    def test_emits_sdpa_op(self):
        func = IrFunction(
            name="main",
            inputs=[("q", IrType(dtype="float32", shape=(1, 4, 8)))],
            outputs=[("out", IrType(dtype="float32", shape=(1, 4, 8)))],
            ops=[
                IrOp(
                    name="scaled_dot_product_attention",
                    inputs=["q", "k", "v"],
                    outputs=["out"],
                    attributes={"is_causal": True},
                )
            ],
        )
        mod = IrModule(functions=[func])
        mlir = ir_module_to_mlir(mod)
        assert '"sf.scaled_dot_product_attention"' in mlir
        assert "is_causal" in mlir
