"""Protocol for TP communication — owned by compiler/tp/, implemented by python_runtime/hal/.

This avoids a dependency inversion violation: compiler/tp/ defines the
interface it needs; python_runtime/hal/ provides the implementation.
Through structural subtyping (Protocol), no explicit inheritance or
runtime import is required.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class TPCommunicator(Protocol):
    """Protocol for collective communication needed by tensor-parallel layers.

    Implementations must provide:
      - world_size: total number of ranks in the TP group
      - rank: local rank index (0-based)
      - all_gather: gather tensors from all ranks along a new first dim
      - all_reduce: sum tensors across all ranks

    ``python_runtime.hal.communicator.Communicator`` satisfies this protocol
    through structural subtyping — no explicit import needed.
    """

    @property
    def world_size(self) -> int: ...

    @property
    def rank(self) -> int: ...

    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor: ...

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor: ...
