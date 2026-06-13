"""RWKV state manager — fixed-size state management.

Unlike Transformer's variable-length KV cache, RWKV maintains a
fixed-size state matrix per layer that is completely updated at
each inference step.  This enables predictable memory allocation
and high concurrency.

Reference: design-phase2.md §2.5.4
"""

from __future__ import annotations

import torch


class RWKVStateManager:
    """Manages fixed-size RWKV states for multiple concurrent requests.

    Each request gets a slot in a pre-allocated state pool.  The state
    size per layer is constant (hidden_size × num_heads), independent
    of sequence length.

    Args:
        num_layers: Number of RWKV layers.
        hidden_size: Model hidden dimension.
        num_heads: Number of attention heads.
        max_requests: Maximum concurrent requests (pool capacity).
    """

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_heads: int = 1,
        max_requests: int = 256,
    ) -> None:
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.max_requests = max_requests
        self.state_dim = hidden_size * num_heads

        self.state_pool = torch.zeros(max_requests, num_layers, self.state_dim)
        self.free_slots: list[int] = list(range(max_requests))
        self.active_slots: dict[str, int] = {}

    def allocate(self, request_id: str) -> int:
        """Allocate a state slot for a new request.

        Args:
            request_id: Unique request identifier.

        Returns:
            Slot index.

        Raises:
            RuntimeError: If no slots are available.
        """
        if not self.free_slots:
            raise RuntimeError(f"RWKV state pool exhausted (max_requests={self.max_requests})")
        slot = self.free_slots.pop()
        self.state_pool[slot].zero_()
        self.active_slots[request_id] = slot
        return slot

    def free(self, request_id: str) -> None:
        """Release a request's state slot back to the pool.

        Args:
            request_id: Request to free.
        """
        slot = self.active_slots.pop(request_id, None)
        if slot is not None:
            self.free_slots.append(slot)

    def update_state(self, slot_or_id: int | str, layer_idx: int, new_state: torch.Tensor) -> None:
        """Update the state for a specific layer.

        Args:
            slot_or_id: Slot index or request_id.
            layer_idx: Layer index (0-based).
            new_state: New state tensor [state_dim].
        """
        slot = slot_or_id if isinstance(slot_or_id, int) else self.active_slots[slot_or_id]
        self.state_pool[slot, layer_idx] = new_state.float()

    def get_state(self, slot_or_id: int | str, layer_idx: int) -> torch.Tensor:
        """Retrieve the state for a specific layer.

        Args:
            slot_or_id: Slot index or request_id.
            layer_idx: Layer index (0-based).

        Returns:
            State tensor [state_dim].
        """
        slot = slot_or_id if isinstance(slot_or_id, int) else self.active_slots[slot_or_id]
        return self.state_pool[slot, layer_idx]

    def get_all_states(self, slot_or_id: int | str) -> torch.Tensor:
        """Retrieve all layer states for a request.

        Args:
            slot_or_id: Slot index or request_id.

        Returns:
            State tensor [num_layers, state_dim].
        """
        slot = slot_or_id if isinstance(slot_or_id, int) else self.active_slots[slot_or_id]
        return self.state_pool[slot]

    @property
    def active_count(self) -> int:
        return len(self.active_slots)

    @property
    def free_count(self) -> int:
        return len(self.free_slots)
