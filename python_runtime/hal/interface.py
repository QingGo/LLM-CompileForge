from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


class Device(ABC):
    """抽象设备。

    HAL 设备抽象层的最小接口——任何硬件平台只需实现此接口
    即可接入 LLM-ServeForge 推理引擎。
    """

    @abstractmethod
    def synchronize(self) -> None:
        """等待所有已提交到设备上的操作完成。"""
        ...


class Buffer(ABC):
    """抽象缓冲区。

    包装设备上的内存块，提供统一的数据传输和张量创建接口。
    """

    @property
    @abstractmethod
    def data_ptr(self) -> int:
        """返回底层内存地址（指针）。"""
        ...

    @abstractmethod
    def copy_from(self, src: torch.Tensor) -> None:
        """从 src 张量复制数据到本缓冲区。"""
        ...

    @abstractmethod
    def copy_to(self, dst: torch.Tensor) -> None:
        """从本缓冲区复制数据到 dst 张量。"""
        ...

    @abstractmethod
    def create_tensor(self, shape: Any, dtype: Any, device: Any) -> torch.Tensor:
        """创建共享本缓冲区底层内存的张量视图。"""
        ...


class OpExecutor(ABC):
    """算子执行器抽象。

    所有算子调用通过此接口与硬件交互。第一版调用 PyTorch 后端，
    后续版本逐步替换为自定义 CUDA / ACL 内核。
    """

    @abstractmethod
    def execute(self, op_name: str, inputs: list[Any], **kwargs: Any) -> torch.Tensor:
        """执行指定算子，返回结果张量。"""
        ...

    # Phase 3 extensions — concrete backends override these when supported.

    def async_copy(self, src: Any, dst: Any, byte_count: int) -> None:  # noqa: B027
        raise NotImplementedError("async_copy not supported")

    def memory_retrieve(self, key: Any, table: Any) -> torch.Tensor:  # noqa: B027
        raise NotImplementedError("memory_retrieve not supported")

    def send_kv_cache(self, blocks: Any, target_device: str) -> None:  # noqa: B027
        raise NotImplementedError("send_kv_cache not supported")
