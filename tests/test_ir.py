import pytest
import torch

from compiler.ir import IrFunction, IrModule, IrOp, IrType, module_from_json, module_to_json, pack_weights

# ═══════════════════════════════════════════════════════════
# IrType
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
class TestIrType:
    def test_creation(self):
        t = IrType(dtype="float32", shape=(1, 2, 3))
        assert t.dtype == "float32"
        assert t.shape == (1, 2, 3)

    def test_defaults(self):
        t = IrType(dtype="int64")
        assert t.shape == ()

    def test_dynamic_shape(self):
        t = IrType(dtype="float16", shape=(None, 128))
        assert t.shape == (None, 128)
        assert "?" in str(t)

    def test_to_dict_roundtrip(self):
        t = IrType(dtype="float32", shape=(3, None, 4))
        t2 = IrType.from_dict(t.to_dict())
        assert t == t2


# ═══════════════════════════════════════════════════════════
# IrOp
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
class TestIrOp:
    def test_creation(self):
        op = IrOp(name="matmul", inputs=["a", "b"], outputs=["c"])
        assert op.name == "matmul"
        assert op.inputs == ["a", "b"]
        assert op.outputs == ["c"]

    def test_defaults(self):
        op = IrOp(name="add")
        assert op.inputs == []
        assert op.outputs == []

    def test_attributes(self):
        op = IrOp(name="softmax", attributes={"dim": -1})
        assert op.attributes["dim"] == -1

    def test_to_dict_roundtrip(self):
        op = IrOp(name="gelu", inputs=["x"], outputs=["y"], attributes={})
        op2 = IrOp.from_dict(op.to_dict())
        assert op2.name == op.name
        assert op2.inputs == op.inputs
        assert op2.outputs == op.outputs


# ═══════════════════════════════════════════════════════════
# IrFunction
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
class TestIrFunction:
    def test_creation(self):
        func = IrFunction(name="main", inputs=[("x", IrType("float32", (1,)))])
        assert func.name == "main"
        assert len(func.inputs) == 1

    def test_add_and_find_op(self):
        func = IrFunction(name="test")
        op = IrOp(name="add", inputs=["a", "b"], outputs=["c"])
        func.ops.append(op)
        found = func.find_op_by_output("c")
        assert found is op

    def test_find_missing_returns_none(self):
        func = IrFunction(name="test")
        assert func.find_op_by_output("missing") is None

    def test_weights_storage(self):
        w = torch.tensor([1.0, 2.0])
        func = IrFunction(name="test", weights={"w1": w})
        assert torch.equal(func.weights["w1"], w)

    def test_to_dict_includes_weight_names(self):
        w = torch.tensor([1.0])
        func = IrFunction(name="test", weights={"w1": w})
        d = func.to_dict()
        assert "weight_names" in d
        assert "w1" in d["weight_names"]
        # Weights themselves are not in the dict
        assert "weights" not in [str(k) for k in d.keys()]


# ═══════════════════════════════════════════════════════════
# IrModule
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
class TestIrModule:
    def test_empty_module(self):
        mod = IrModule()
        with pytest.raises(ValueError, match="no functions"):
            _ = mod.main

    def test_add_function(self):
        mod = IrModule()
        func = IrFunction(name="main")
        mod.add_function(func)
        assert mod.main is func

    def test_metadata(self):
        mod = IrModule(metadata={"source": "test"})
        assert mod.metadata["source"] == "test"

    def test_roundtrip_without_weights(self):
        func = IrFunction(
            name="main",
            inputs=[("x", IrType("float32", (1, 2)))],
            outputs=[("y", IrType("float32", (1, 2)))],
            ops=[IrOp(name="gelu", inputs=["x"], outputs=["y"])],
        )
        mod = IrModule(functions=[func], metadata={"key": "val"})
        json_str = module_to_json(mod)
        mod2 = module_from_json(json_str)
        assert mod2.metadata["key"] == "val"
        assert len(mod2.functions) == 1
        assert mod2.functions[0].name == "main"

    def test_roundtrip_with_weights(self):
        w = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        func = IrFunction(
            name="main",
            ops=[IrOp(name="matmul", inputs=["x", "w1"], outputs=["y"])],
            weights={"w1": w},
        )
        mod = IrModule(functions=[func])
        json_str = module_to_json(mod)
        packed = pack_weights(mod)
        mod2 = module_from_json(json_str, weights=packed)
        assert torch.equal(mod2.functions[0].weights["w1"], w)


# ═══════════════════════════════════════════════════════════
# pack_weights
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
class TestPackWeights:
    def test_empty_module(self):
        mod = IrModule()
        assert pack_weights(mod) == {}

    def test_single_function(self):
        w = torch.randn(4, 4)
        func = IrFunction(name="main", weights={"w": w})
        mod = IrModule(functions=[func])
        packed = pack_weights(mod)
        assert "main" in packed
        assert torch.equal(packed["main"]["w"], w)

    def test_multiple_functions(self):
        w1 = torch.tensor([1.0])
        w2 = torch.tensor([2.0])
        mod = IrModule(
            functions=[
                IrFunction(name="fn1", weights={"a": w1}),
                IrFunction(name="fn2", weights={"b": w2}),
            ]
        )
        packed = pack_weights(mod)
        assert len(packed) == 2
