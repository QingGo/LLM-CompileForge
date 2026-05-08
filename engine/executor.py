"""Model executor — bridges the compiled IR to HAL.

The Executor is responsible for:
  1. Loading a compiled IrModule (with weights).
  2. Running forward passes by iterating through IrOps and dispatching
     each operation to the HAL OpExecutor.
  3. Managing KV cache tensor and PagedAttention block-aware computation.

PagedAttention (MVP):
  During each forward pass, SDPA ops are intercepted:
    1. Newly-computed K/V are written into the paged KV cache.
    2. In decode mode (single token K/V), full K/V (historical + new)
       are gathered from cache and substituted into the SDPA inputs.
  This ensures historical KV states are visible during decode steps.
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

        # Per-forward-pass state
        self._current_positions: torch.Tensor | None = None
        self._sda_layer_count: int = 0

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

        self._reset_forward_state(kwargs)

        for op in self._function.ops:
            result = self._execute_op(op, ssa_values)
            if result is not None and op.outputs:
                ssa_values[op.outputs[0]] = result

        if self._function.outputs:
            first_output_name = self._function.outputs[0][0]
            if first_output_name in ssa_values:
                return ssa_values[first_output_name]

        if ssa_values:
            return list(ssa_values.values())[-1]

        return torch.tensor([])

    def forward_with_kv(
        self, input_ids: torch.Tensor, **kwargs: Any
    ) -> tuple[torch.Tensor, list[tuple[str, torch.Tensor]]]:
        """Run a forward pass and also return all KV cache tensors.

        Returns (logits, kv_tensors) where kv_tensors is a list of
        (output_name, tensor) pairs for non-logits model outputs.
        These are the K/V tensors produced by each attention layer.
        """
        ssa_values: dict[str, torch.Tensor] = {}

        input_names = [name for name, _ in self._function.inputs]
        if input_names:
            ssa_values[input_names[0]] = input_ids
            for named_input in input_names[1:]:
                if named_input in kwargs:
                    ssa_values[named_input] = kwargs[named_input]

        self._reset_forward_state(kwargs)

        for op in self._function.ops:
            result = self._execute_op(op, ssa_values)
            if result is not None and op.outputs:
                ssa_values[op.outputs[0]] = result

        logits = torch.tensor([])
        kv_tensors: list[tuple[str, torch.Tensor]] = []
        for i, (out_name, _) in enumerate(self._function.outputs):
            if out_name in ssa_values:
                if i == 0:
                    logits = ssa_values[out_name]
                else:
                    kv_tensors.append((out_name, ssa_values[out_name]))

        return logits, kv_tensors

    def _reset_forward_state(self, kwargs: dict[str, Any]) -> None:
        """Reset per-forward-pass state and store positions if provided."""
        self._sda_layer_count = 0
        self._current_positions = kwargs.get("positions", None)

    def _execute_op(self, op: IrOp, ssa_values: dict[str, torch.Tensor]) -> torch.Tensor | None:
        if op.name == "constant":
            if op.inputs and op.inputs[0] in self._weights:
                return self._weights[op.inputs[0]]
            return None

        # ── PagedAttention SDPA interception ─────────────────
        if op.name == "scaled_dot_product_attention" and self._kv_cache is not None and self._block_tables:
            self._intercept_sdpa(op, ssa_values)

        # Collect runtime inputs
        tensor_inputs: list[torch.Tensor] = []
        for inp_name in op.inputs:
            if inp_name in ssa_values:
                tensor_inputs.append(ssa_values[inp_name])
            elif inp_name in self._weights:
                tensor_inputs.append(self._weights[inp_name])
            elif inp_name in self._function.weights:
                tensor_inputs.append(self._function.weights[inp_name])
            else:
                available = sorted(ssa_values.keys())[:20]
                raise KeyError(
                    f"Op '{op.name}' needs input '{inp_name}' which was never produced. "
                    f"Available SSA values (first 20): {available}"
                )

        if not tensor_inputs:
            result = self._hal.execute(op.name, [], **op.attributes)
        else:
            result = self._hal.execute(op.name, tensor_inputs, **op.attributes)

        # For in-place ops (copy_), propagate the modification to the
        # destination tensor's SSA name so downstream ops see the update.
        if op.name == "copy_" and op.inputs and op.outputs:
            dst_name = op.inputs[0]
            if dst_name in ssa_values and result is not None:
                ssa_values[dst_name] = result

        return result

    # ── PagedAttention interception ──────────────────────────

    def _intercept_sdpa(self, op: IrOp, ssa_values: dict[str, torch.Tensor]) -> None:
        """Intercept SDPA op: write K/V to cache, gather full K/V for decode.

        The SDPA op expects inputs: Q, K, V, [attn_mask].
        During decode (K/V have seq_len == 1), we replace the single-token
        K/V with the full historical K/V gathered from the paged cache.
        """
        layer_idx = self._sda_layer_count
        self._sda_layer_count += 1

        if len(op.inputs) < 3:
            return

        k_name = op.inputs[1]
        v_name = op.inputs[2]
        k_new = ssa_values.get(k_name)
        v_new = ssa_values.get(v_name)

        if k_new is None or v_new is None:
            return

        # Extract K/V per-token: from [batch, heads, seq, head_dim] → [seq, heads, head_dim]
        if k_new.dim() >= 4 and k_new.shape[0] == 1:
            k_new_sq = k_new.squeeze(0)
            v_new_sq = v_new.squeeze(0)
        else:
            k_new_sq = k_new
            v_new_sq = v_new

        # Handle various K/V shapes:
        #   [heads, seq, head_dim] → [seq, heads, head_dim]
        #   [seq, heads, head_dim]   → leave as-is
        #   [seq, hidden] where hidden = num_kv_heads * head_dim → [seq, heads, head_dim]
        nkh = self._num_kv_heads
        hd = self._head_dim
        if k_new_sq.dim() == 3 and k_new_sq.shape[0] == nkh and k_new_sq.shape[-1] == hd:
            k_new_sq = k_new_sq.permute(1, 0, 2)
            v_new_sq = v_new_sq.permute(1, 0, 2)
        elif k_new_sq.dim() == 2 and k_new_sq.shape[-1] == nkh * hd:
            k_new_sq = k_new_sq.reshape(-1, nkh, hd)
            v_new_sq = v_new_sq.reshape(-1, nkh, hd)
        elif k_new_sq.dim() == 3 and k_new_sq.shape[-1] == nkh * hd:
            k_new_sq = k_new_sq.reshape(-1, nkh, hd)
            v_new_sq = v_new_sq.reshape(-1, nkh, hd)

        # Write to cache
        positions = self._current_positions
        if positions is not None:
            flat_pos = positions.squeeze(0) if positions.dim() >= 2 else positions
            self._write_kv_flat(k_new_sq, v_new_sq, flat_pos, self._block_tables, layer_idx)

        # If single-token (decode), gather full K/V from cache
        token_count = k_new_sq.shape[0] if k_new_sq.dim() >= 1 else 1
        if token_count == 1 and self._block_tables:
            max_seq = self._max_seq_from_tables(self._block_tables)
            k_full, v_full = self._gather_kv_flat(self._block_tables, max_seq, layer_idx)
            # Gather returns [num_requests, seq, heads, hd].
            # Model expects [batch, heads, seq, hd] → permute then expand batch if needed.
            if k_full.dim() == 4:
                k_gathered = k_full.permute(0, 2, 1, 3)
                v_gathered = v_full.permute(0, 2, 1, 3)
            else:
                k_gathered = k_full
                v_gathered = v_full
            ssa_values[k_name] = k_gathered.to(k_new.dtype)
            ssa_values[v_name] = v_gathered.to(v_new.dtype)

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

    def _max_seq_from_tables(self, block_tables: dict[str, list[int]]) -> int:
        total = 0
        for blocks in block_tables.values():
            total = max(total, len(blocks))
        return total * self._block_size

    def _write_kv_flat(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        block_tables: dict[str, list[int]],
        layer_idx: int,
    ) -> None:
        """Write K and V into the paged cache.

        Args:
            key: [seq, num_kv_heads, head_dim]
            value: [seq, num_kv_heads, head_dim]
            positions: [seq] — absolute positions.
            block_tables: request_id → list[physical_block_id].
            layer_idx: transformer layer index.
        """
        if self._kv_cache is None:
            return
        bs = self._block_size
        pos_list = positions.tolist() if positions.dim() >= 1 else [int(positions.item())]
        if not isinstance(pos_list, list):
            pos_list = [int(positions.item())]
        for i, pos in enumerate(pos_list):
            block_idx = pos // bs
            offset = pos % bs
            written = False
            for blocks in block_tables.values():
                if block_idx < len(blocks):
                    phys_id = blocks[block_idx]
                    self._kv_cache[phys_id, layer_idx, 0, offset] = key[i]
                    self._kv_cache[phys_id, layer_idx, 1, offset] = value[i]
                    written = True
                    break
            if not written and block_tables:
                first_blocks = next(iter(block_tables.values()))
                phys_id = first_blocks[min(block_idx, len(first_blocks) - 1)]
                self._kv_cache[phys_id, layer_idx, 0, offset] = key[i]
                self._kv_cache[phys_id, layer_idx, 1, offset] = value[i]

    def _gather_kv_flat(
        self,
        block_tables: dict[str, list[int]],
        max_seq_len: int,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather all K/V from the paged cache.

        Returns:
            (key, value): each [num_requests, max_seq_len, num_kv_heads, head_dim]
        """
        if self._kv_cache is None or not block_tables:
            raise RuntimeError("KV cache not initialized")

        bs = self._block_size
        num_requests = len(block_tables)
        nkh = self._num_kv_heads
        hd = self._head_dim
        dtype = self._kv_cache.dtype

        key = torch.zeros(num_requests, max_seq_len, nkh, hd, dtype=dtype)
        value = torch.zeros(num_requests, max_seq_len, nkh, hd, dtype=dtype)

        for req_idx, (_, blocks) in enumerate(sorted(block_tables.items())):
            for blk_offset, phys_id in enumerate(blocks):
                start = blk_offset * bs
                end = min(start + bs, max_seq_len)
                if start >= max_seq_len:
                    break
                key[req_idx, start:end] = self._kv_cache[phys_id, layer_idx, 0, :end - start]
                value[req_idx, start:end] = self._kv_cache[phys_id, layer_idx, 1, :end - start]

        return key, value

    # ── Public KV cache helpers (used by LLMEngine) ──────────

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
        self._write_kv_flat(key, value, positions, block_tables, layer_idx)

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
        return self._gather_kv_flat(block_tables, max_seq_len, layer_idx)
