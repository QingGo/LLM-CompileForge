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
from engine._kv_cache import _KVCacheMixin
from hal.interface import OpExecutor


class Executor(_KVCacheMixin):
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

        self._kv_cache: torch.Tensor | None = None
        self._block_tables: dict[str, list[int]] = {}
        self._block_size: int = 16
        self._num_kv_heads: int = 0
        self._head_dim: int = 0

        self._current_positions: torch.Tensor | None = None
        self._sda_layer_count: int = 0

    @property
    def function(self) -> IrFunction:
        return self._function

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
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
        self._sda_layer_count = 0
        self._current_positions = kwargs.get("positions", None)

    def _execute_op(self, op: IrOp, ssa_values: dict[str, torch.Tensor]) -> torch.Tensor | None:
        if op.name == "constant":
            if op.inputs and op.inputs[0] in self._weights:
                return self._weights[op.inputs[0]]
            return None

        if op.name == "scaled_dot_product_attention" and self._kv_cache is not None and self._block_tables:
            self._intercept_sdpa(op, ssa_values)

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

        if op.name == "copy_" and op.inputs and op.outputs:
            dst_name = op.inputs[0]
            if dst_name in ssa_values and result is not None:
                ssa_values[dst_name] = result

        return result

    def _intercept_sdpa(self, op: IrOp, ssa_values: dict[str, torch.Tensor]) -> None:
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

        k_new_sq, v_new_sq = _normalize_kv_for_cache(
            k_new, v_new, self._num_kv_heads, self._head_dim
        )

        positions = self._current_positions
        if positions is not None:
            flat_pos = positions.squeeze(0) if positions.dim() >= 2 else positions
            self._write_kv_flat(k_new_sq, v_new_sq, flat_pos, self._block_tables, layer_idx)

        token_count = k_new_sq.shape[0] if k_new_sq.dim() >= 1 else 1
        if token_count == 1 and self._block_tables:
            max_seq = self._max_seq_from_tables(self._block_tables)
            k_full, v_full = self._gather_kv_flat(self._block_tables, max_seq, layer_idx)
            if k_full.dim() == 4:
                k_gathered = k_full.permute(0, 2, 1, 3)
                v_gathered = v_full.permute(0, 2, 1, 3)
            else:
                k_gathered = k_full
                v_gathered = v_full
            ssa_values[k_name] = k_gathered.to(k_new.dtype)
            ssa_values[v_name] = v_gathered.to(v_new.dtype)


def _normalize_kv_for_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if k.dim() >= 4 and k.shape[0] == 1:
        k = k.squeeze(0)
        v = v.squeeze(0)

    nkh = num_kv_heads
    hd = head_dim

    if k.dim() == 3 and k.shape[0] == nkh and k.shape[-1] == hd:
        k = k.permute(1, 0, 2)
        v = v.permute(1, 0, 2)
    elif k.dim() == 2 and k.shape[-1] == nkh * hd:
        k = k.reshape(-1, nkh, hd)
        v = v.reshape(-1, nkh, hd)
    elif k.dim() == 3 and k.shape[-1] == nkh * hd:
        k = k.reshape(-1, nkh, hd)
        v = v.reshape(-1, nkh, hd)

    return k, v
