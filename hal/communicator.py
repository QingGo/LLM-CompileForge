"""HAL Communicator — abstract collective communication interface.

Defines the minimal set of collective operations needed for tensor
parallelism and distributed inference.  Backends implement concrete
transport layers (NCCL / Gloo / custom SDKs).

Reference: design-phase2.md §2.4.3
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Communicator(ABC):
    """Abstract collective communication interface.

    Each TP group has one Communicator instance.  Operation
    semantics match torch.distributed conventions.
    """

    @abstractmethod
    def all_reduce(self, tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
        """Reduce tensor across all ranks and broadcast the result.

        Args:
            tensor: Local tensor to reduce.
            op: Reduction operation ("sum", "avg", "min", "max").

        Returns:
            Reduced tensor (same shape as input).
        """
        ...

    @abstractmethod
    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather tensors from all ranks along the last dimension.

        Args:
            tensor: Local tensor to gather.

        Returns:
            Concatenated tensor (last dim × world_size).
        """
        ...

    @abstractmethod
    def broadcast(self, tensor: torch.Tensor, src: int) -> torch.Tensor:
        """Broadcast tensor from source rank to all ranks.

        Args:
            tensor: Local tensor (only meaningful on src rank).
            src: Source rank index.

        Returns:
            Broadcasted tensor.
        """
        ...

    @property
    @abstractmethod
    def rank(self) -> int:
        """Current rank within this communication group."""
        ...

    @property
    @abstractmethod
    def world_size(self) -> int:
        """Number of ranks in this communication group."""
        ...
