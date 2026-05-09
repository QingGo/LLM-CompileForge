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
from engine._kv_cache import _KVCacheMixin, _normalize_kv_for_cache
from hal.interface import OpExecutor

_WEIGHT_OPS = frozenset({"sf.weight", "sf.constant", "constant"})


class MlirExecutor(_KVCacheMixin):
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

        self._kv_cache: torch.Tensor | None = None
        self._block_tables: dict[str, list[int]] = {}
        self._block_size: int = 16
        self._num_kv_heads: int = 0
        self._head_dim: int = 0

        self._current_positions: torch.Tensor | None = None
        self._sda_layer_count: int = 0

    @property
    def function(self) -> MlirFunction:
        return self._function

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
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

        output_ssa = None
        if self._function.outputs:
            output_ssa = self._function.outputs[0][0]
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
        if op.name in _WEIGHT_OPS:
            wname = op.attributes.get("name", "") or op.attributes.get('"name"', "")
            if not wname and op.operands:
                wname = op.operands[0]
            if wname in self._weights:
                return self._weights[wname]
            if wname in self._function.weights:
                return self._function.weights[wname]
            return None

        if "sdpa" in op.name.lower() and self._kv_cache is not None and self._block_tables:
            self._intercept_sdpa(op, ssa_values)

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

        result = self._hal.execute(op.op_name, tensor_inputs, **op.attributes)

        if op.op_name in ("copy_", "copy") and op.operands and op.results:
            dst_name = op.operands[0]
            if dst_name in ssa_values and result is not None:
                ssa_values[dst_name] = result

        return result

    def _intercept_sdpa(self, op: MlirOp, ssa_values: dict[str, torch.Tensor]) -> None:
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
