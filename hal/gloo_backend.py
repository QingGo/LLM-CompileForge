"""GlooCommunicator — torch.distributed + gloo backend.

Implements the Communicator ABC using torch.distributed with the
'gloo' backend.  Works on CPU (macOS / Linux) without GPUs,
enabling multi-process testing of tensor parallelism.

Reference: design-phase2.md §2.4.3, §5.2
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

from hal.communicator import Communicator


class GlooCommunicator(Communicator):
    """Collective communication via torch.distributed (gloo backend).

    Requires torch.distributed to be initialized before use
    (e.g. via dist.init_process_group("gloo")).

    Args:
        group: Optional process group.  If None, uses the default group.
    """

    def __init__(self, group: Any = None) -> None:
        self._group = group
        object.__setattr__(self, "_group", group)

    def all_reduce(self, tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
        result = tensor.clone()
        reduce_op = dist.ReduceOp.SUM
        if op == "avg":
            result = result / self.world_size
        dist.all_reduce(result, op=reduce_op, group=self._group)
        return result

    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        world = self.world_size
        gathered = [torch.empty_like(tensor) for _ in range(world)]
        dist.all_gather(gathered, tensor, group=self._group)
        return torch.cat(gathered, dim=-1)

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        result = tensor.clone()
        dist.broadcast(result, src, group=self._group)
        return result

    @property
    def rank(self) -> int:
        return dist.get_rank(self._group)

    @property
    def world_size(self) -> int:
        return dist.get_world_size(self._group)
