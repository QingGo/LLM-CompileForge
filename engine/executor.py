"""Model executor — bridges the compiled IR to HAL.

The Executor is responsible for:
  1. Loading a compiled IrModule (with weights).
  2. Running forward passes by iterating through IrOps and dispatching
     each operation to the HAL OpExecutor.
  3. Managing KV cache tensor and PagedAttention block-aware computation.

PagedAttention (MVP):
  During each forward pass, K and V tensors produced by the model are
  written into the paged KV cache. The attention op then reads K/V from
  the cache using block_tables, ensuring that historical KV states are
  visible even during single-token decode steps.
"""

from __future__ import annotations

from typing import Any

import torch

from compiler.ir import IrFunction, IrModule, IrOp
from hal.interface import OpExecutor


class Executor:
    """Executes a compiled model through the HAL.

    The executor walks the IR graph in order, feeding tensors through
    the HAL's OpExecutor. Weight tensors are looked up from the IrFunction.

    PagedAttention: when kv_cache and block_tables are provided, the
    executor intercepts scaled_dot_product_attention ops to:
      1. Write newly-computed K/V into the paged cache.
      2. Gather full K/V (historical + new) from cache for attention.
    """

    def __init__(
        self,
        module: IrModule,
        hal_backend: OpExecutor,
    ) -> None:
        self._module = module
        self._hal = hal_backend
        self._function = module.main

        self._weights: dict[str, torch.Tensor] = {}
        for name, tensor in self._function.weights.items():
            self._weights[name] = tensor

        # PagedAttention state (set per step via set_kv_cache)
        self._kv_cache: torch.Tensor | None = None
        self._block_tables: dict[str, list[int]] = {}
        self._block_size: int = 16
        self._num_kv_heads: int = 0
        self._head_dim: int = 0

    @property
    def function(self) -> IrFunction:
        return self._function

    def set_kv_cache(
        self,
        kv_cache: torch.Tensor | None,
        block_tables: dict[str, list[int]] | None = None,
        block_size: int = 16,
        num_kv_heads: int = 0,
        head_dim: int = 0,
    ) -> None:
        """Configure PagedAttention for the current step.

        Args:
            kv_cache: [num_blocks, num_layers, 2, block_size, num_kv_heads, head_dim]
            block_tables: request_id → list[physical_block_id]
            block_size: tokens per KV cache block.
            num_kv_heads: number of KV attention heads.
            head_dim: dimension per attention head.
        """
        self._kv_cache = kv_cache
        self._block_tables = block_tables or {}
        self._block_size = block_size
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Run a forward pass through the compiled graph.

        Args:
            input_ids: Input token IDs tensor.
            **kwargs: Additional inputs (positions, attention_mask, etc.).

        Returns:
            Logits tensor from the final output.
        """
        ssa_values: dict[str, torch.Tensor] = {}

        input_names = [name for name, _ in self._function.inputs]
        if input_names:
            ssa_values[input_names[0]] = input_ids
            for named_input in input_names[1:]:
                if named_input in kwargs:
                    ssa_values[named_input] = kwargs[named_input]

        for op in self._function.ops:
            result = self._execute_op(op, ssa_values)
            if result is not None and op.outputs:
                ssa_values[op.outputs[0]] = result

        if self._function.outputs:
            last_output_name = self._function.outputs[-1][0]
            if last_output_name in ssa_values:
                return ssa_values[last_output_name]

        if ssa_values:
            return list(ssa_values.values())[-1]

        return torch.tensor([])

    def _execute_op(self, op: IrOp, ssa_values: dict[str, torch.Tensor]) -> torch.Tensor | None:
        if op.name == "constant":
            if op.inputs and op.inputs[0] in self._weights:
                return self._weights[op.inputs[0]]
            return None

        # Collect runtime inputs
        tensor_inputs: list[torch.Tensor] = []
        for inp_name in op.inputs:
            if inp_name in ssa_values:
                tensor_inputs.append(ssa_values[inp_name])
            elif inp_name in self._weights:
                tensor_inputs.append(self._weights[inp_name])
            else:
                raise KeyError(f"Unknown input '{inp_name}' for op '{op.name}'")

        if not tensor_inputs:
            return self._hal.execute(op.name, [], **op.attributes)

        return self._hal.execute(op.name, tensor_inputs, **op.attributes)

    # ── KV Cache utilities ──────────────────────────────────

    def prepare_kv_blocks(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        num_blocks: int,
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """Allocate a contiguous KV cache tensor.

        Shape: [num_blocks, num_layers, 2, block_size, num_kv_heads, head_dim]
        The 2 dimension stores K (index 0) and V (index 1).
        """
        shape = (num_blocks, num_layers, 2, block_size, num_kv_heads, head_dim)
        return torch.zeros(shape, dtype=dtype)

    def write_kv_to_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        block_tables: dict[str, list[int]],
        layer_idx: int = 0,
    ) -> None:
        """Write K and V tensors into the paged KV cache at given positions.

        Args:
            key: [total_tokens, num_kv_heads, head_dim]
            value: [total_tokens, num_kv_heads, head_dim]
            positions: [total_tokens] — position of each token within its sequence.
            block_tables: request_id → list[physical_block_id].
            layer_idx: which transformer layer (index into cache dim 1).
        """
        if self._kv_cache is None:
            return

        bs = self._block_size
        for i, pos in enumerate(positions.tolist()):
            # Determine which block and offset
            block_idx = pos // bs
            offset = pos % bs

            # Find the physical block — walk through all block tables
            written = False
            for blocks in block_tables.values():
                if block_idx < len(blocks):
                    phys_id = blocks[block_idx]
                    self._kv_cache[phys_id, layer_idx, 0, offset] = key[i]
                    self._kv_cache[phys_id, layer_idx, 1, offset] = value[i]
                    written = True
                    break
            if not written:
                # Fallback: use first available block table
                if block_tables:
                    first_blocks = next(iter(block_tables.values()))
                    phys_id = first_blocks[min(block_idx, len(first_blocks) - 1)]
                    self._kv_cache[phys_id, layer_idx, 0, offset] = key[i]
                    self._kv_cache[phys_id, layer_idx, 1, offset] = value[i]

    def gather_kv_from_cache(
        self,
        block_tables: dict[str, list[int]],
        max_seq_len: int,
        layer_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather all K and V from the paged cache for current requests.

        Args:
            block_tables: request_id → list[physical_block_id].
            max_seq_len: maximum sequence length to gather.
            layer_idx: which transformer layer.

        Returns:
            (key, value) where each has shape
            [num_requests, max_seq_len, num_kv_heads, head_dim]
        """
        if self._kv_cache is None or not block_tables:
            raise RuntimeError("KV cache not initialized")

        bs = self._block_size
        num_requests = len(block_tables)
        nkh = self._num_kv_heads
        hd = self._head_dim

        key = torch.zeros(num_requests, max_seq_len, nkh, hd, dtype=self._kv_cache.dtype)
        value = torch.zeros(num_requests, max_seq_len, nkh, hd, dtype=self._kv_cache.dtype)

        for req_idx, (_, blocks) in enumerate(sorted(block_tables.items())):
            for blk_offset, phys_id in enumerate(blocks):
                start = blk_offset * bs
                end = min(start + bs, max_seq_len)
                if start >= max_seq_len:
                    break
                key[req_idx, start:end] = self._kv_cache[phys_id, layer_idx, 0, :end - start]
                value[req_idx, start:end] = self._kv_cache[phys_id, layer_idx, 1, :end - start]

        return key, value
