import pytest
import torch

from python_runtime.hal.interface import Buffer, Device, OpExecutor


class _ConcreteDevice(Device):
    def synchronize(self) -> None:
        pass


class _ConcreteBuffer(Buffer):
    @property
    def data_ptr(self) -> int:
        return 0

    def copy_from(self, src: torch.Tensor) -> None:
        pass

    def copy_to(self, dst: torch.Tensor) -> None:
        pass

    def create_tensor(self, shape, dtype, device) -> torch.Tensor:
        return torch.empty(shape, dtype=dtype, device=device)


class _ConcreteOpExecutor(OpExecutor):
    def execute(self, op_name: str, inputs: list, **kwargs):
        return inputs[0]


class _PartialDevice(Device):
    """缺少 synchronize 实现的子类。"""

    pass  # type: ignore[abstract]


class _PartialBuffer(Buffer):
    """缺少 data_ptr 属性的子类。"""

    pass  # type: ignore[abstract]


class _PartialOpExecutor(OpExecutor):
    """缺少 execute 方法的子类。"""

    pass  # type: ignore[abstract]


# ── Device 测试 ──────────────────────────────────────────────


@pytest.mark.unit
class TestDevice:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError, match="abstract"):
            Device()  # type: ignore[abstract]

    def test_concrete_instantiable(self):
        device = _ConcreteDevice()
        assert isinstance(device, Device)

    def test_concrete_synchronize_does_not_raise(self):
        device = _ConcreteDevice()
        device.synchronize()

    def test_partial_missing_method_raises_on_instantiate(self):
        with pytest.raises(TypeError, match="abstract"):
            _PartialDevice()  # type: ignore[abstract]


# ── Buffer 测试 ──────────────────────────────────────────────


@pytest.mark.unit
class TestBuffer:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError, match="abstract"):
            Buffer()  # type: ignore[abstract]

    def test_concrete_instantiable(self):
        buf = _ConcreteBuffer()
        assert isinstance(buf, Buffer)

    def test_concrete_data_ptr_accessible(self):
        buf = _ConcreteBuffer()
        assert isinstance(buf.data_ptr, int)

    def test_concrete_create_tensor_returns_tensor(self):
        buf = _ConcreteBuffer()
        t = buf.create_tensor((2, 3), torch.float32, "cpu")
        assert isinstance(t, torch.Tensor)
        assert t.shape == (2, 3)

    def test_partial_missing_property_raises_on_instantiate(self):
        with pytest.raises(TypeError, match="abstract"):
            _PartialBuffer()  # type: ignore[abstract]


# ── OpExecutor 测试 ──────────────────────────────────────────


@pytest.mark.unit
class TestOpExecutor:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError, match="abstract"):
            OpExecutor()  # type: ignore[abstract]

    def test_concrete_instantiable(self):
        executor = _ConcreteOpExecutor()
        assert isinstance(executor, OpExecutor)

    def test_concrete_execute_returns_result(self):
        executor = _ConcreteOpExecutor()
        t = torch.tensor([1.0, 2.0])
        result = executor.execute("identity", [t])
        assert torch.equal(result, t)

    def test_partial_missing_method_raises_on_instantiate(self):
        with pytest.raises(TypeError, match="abstract"):
            _PartialOpExecutor()  # type: ignore[abstract]
