"""MLIR-based model executor — bridges compiled MLIR to HAL.

Walks the parsed MlirModule (from model.mlir) and dispatches each
operation to the HAL OpExecutor.  This is the MLIR-native counterpart
to the legacy Executor (which walks Python IrModule).

Key features:
  - Weight tensor lookup from MlirFunction.weights
  - Same HAL dispatch as the Python IR executor
  - Same PagedAttention SDPA interception for KV cache
  - Full forward() and forward_with_kv() API compatibility
"""

from __future__ import annotations

from typing import Any

import torch

from compiler.mlir_artifact import MlirFunction, MlirModule, MlirOp
from hal.interface import OpExecutor

# Ops that materialize weight constants (no HAL execution needed)
_WEIGHT_OPS = frozenset({"sf.weight", "sf.constant", "constant"})


class MlirExecutor:
    """Executes a compiled model from MLIR artifact through the HAL.

    Usage:
        module = load_mlir_artifact("./compiled/model")
        backend = PyTorchBackend("cpu")
        executor = MlirExecutor(module, backend)
        logits = executor.forward(input_ids)
    """

    def __init__(
        self,
        module: MlirModule,
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
    def function(self) -> MlirFunction:
        return self._function

    def set_kv_cache(
        self,
        kv_cache: torch.Tensor | None,
        block_tables: dict[str, list[int]] | None = None,
        block_size: int = 16,
        num_kv_heads: int = 0,
        head_dim: int = 0,
    ) -> None:
        """Configure PagedAttention for the current step."""
        self._kv_cache = kv_cache
        self._block_tables = block_tables or {}
        self._block_size = block_size
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Run a forward pass through the compiled MLIR graph."""
        ssa_values: dict[str, torch.Tensor] = {}

        # Map first input to input_ids
        if self._function.inputs:
            first_input = self._function.inputs[0][0]
            ssa_values[first_input] = input_ids
            ssa_values[first_input.lstrip("%")] = input_ids  # also store without %
            for named_input, _ in self._function.inputs[1:]:
                clean = named_input.replace("%", "")
                if clean in kwargs:
                    ssa_values[named_input] = kwargs[clean]

        self._reset_forward_state(kwargs)

        # Find the expected output SSA name
        output_ssa = None
        if self._function.outputs:
            output_ssa = self._function.outputs[0][0]
        # Also check the last op's last result as fallback
        if output_ssa is None and self._function.ops:
            last_op = self._function.ops[-1]
            if last_op.results:
                output_ssa = last_op.results[-1]

        for op in self._function.ops:
            result = self._execute_op(op, ssa_values)
            if result is not None and op.results:
                ssa_values[op.results[0]] = result

        if output_ssa and output_ssa in ssa_values:
            return ssa_values[output_ssa]

        if ssa_values:
            return list(ssa_values.values())[-1]

        return torch.tensor([])

    def forward_with_kv(
        self, input_ids: torch.Tensor, **kwargs: Any
    ) -> tuple[torch.Tensor, list[tuple[str, torch.Tensor]]]:
        """Run forward and return (logits, kv_tensors)."""
        ssa_values: dict[str, torch.Tensor] = {}

        if self._function.inputs:
            first_input = self._function.inputs[0][0]
            ssa_values[first_input] = input_ids
            ssa_values[first_input.lstrip("%")] = input_ids
            for named_input, _ in self._function.inputs[1:]:
                clean = named_input.replace("%", "")
                if clean in kwargs:
                    ssa_values[named_input] = kwargs[clean]

        self._reset_forward_state(kwargs)

        for op in self._function.ops:
            result = self._execute_op(op, ssa_values)
            if result is not None and op.results:
                ssa_values[op.results[0]] = result

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

    def _execute_op(self, op: MlirOp, ssa_values: dict[str, torch.Tensor]) -> torch.Tensor | None:
        # Weight constant: return the weight tensor
        if op.name in _WEIGHT_OPS:
            wname = op.attributes.get("name", "")
            if not wname and op.operands:
                wname = op.operands[0]
            if wname in self._weights:
                return self._weights[wname]
            if wname in self._function.weights:
                return self._function.weights[wname]
            return None

        # SDPA interception for KV cache
        if "sdpa" in op.name.lower() and self._kv_cache is not None and self._block_tables:
            self._intercept_sdpa(op, ssa_values)

        # Collect runtime inputs
        tensor_inputs: list[torch.Tensor] = []
        for inp_name in op.operands:
            if inp_name in ssa_values:
                tensor_inputs.append(ssa_values[inp_name])
            elif inp_name in self._weights:
                tensor_inputs.append(self._weights[inp_name])
            elif inp_name in self._function.weights:
                tensor_inputs.append(self._function.weights[inp_name])
            elif inp_name.startswith("%"):
                clean = inp_name.replace("%", "")
                if clean in ssa_values:
                    tensor_inputs.append(ssa_values[clean])
                elif clean in self._weights:
                    tensor_inputs.append(self._weights[clean])
                elif clean in self._function.weights:
                    tensor_inputs.append(self._function.weights[clean])
                else:
                    available = sorted(ssa_values.keys())[:20]
                    raise KeyError(
                        f"Op '{op.name}' needs input '{inp_name}' which was never produced. "
                        f"Available SSA values (first 20): {available}"
                    )
            elif inp_name.lstrip("%") in ssa_values:
                tensor_inputs.append(ssa_values[inp_name.lstrip("%")])
            elif inp_name.lstrip("%") in self._weights:
                tensor_inputs.append(self._weights[inp_name.lstrip("%")])
            else:
                available = sorted(ssa_values.keys())[:20]
                raise KeyError(
                    f"Op '{op.name}' needs input '{inp_name}' which was never produced. "
                    f"Available SSA values (first 20): {available}"
                )

        # Execute via HAL (op_name is already the unqualified name, e.g. 'fused_attention_output')
        result = self._hal.execute(op.op_name, tensor_inputs, **op.attributes)

        # For in-place ops (copy_), propagate to destination
        if op.op_name in ("copy_", "copy") and op.operands and op.results:
            dst_name = op.operands[0]
            if dst_name in ssa_values and result is not None:
                ssa_values[dst_name] = result

        return result

    # ── PagedAttention interception ──────────────────────────

    def _intercept_sdpa(self, op: MlirOp, ssa_values: dict[str, torch.Tensor]) -> None:
        """Same SDPA interception as the legacy Executor."""
        layer_idx = self._sda_layer_count
        self._sda_layer_count += 1

        if len(op.operands) < 3:
            return

        k_name = op.operands[1]
        v_name = op.operands[2]
        k_new = ssa_values.get(k_name)
        v_new = ssa_values.get(v_name)

        if k_new is None or v_new is None:
            return

        if k_new.dim() >= 4 and k_new.shape[0] == 1:
            k_new_sq = k_new.squeeze(0)
            v_new_sq = v_new.squeeze(0)
        else:
            k_new_sq = k_new
            v_new_sq = v_new

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

    def write_kv_to_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        block_tables: dict[str, list[int]],
        layer_idx: int = 0,
    ) -> None:
        self._write_kv_flat(key, value, positions, block_tables, layer_idx)

    def gather_kv_from_cache(
        self,
        block_tables: dict[str, list[int]],
        max_seq_len: int,
        layer_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._gather_kv_flat(block_tables, max_seq_len, layer_idx)
