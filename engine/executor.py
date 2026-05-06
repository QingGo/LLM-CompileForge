"""Model executor — bridges the compiled IR to HAL.

The Executor is responsible for:
  1. Loading a compiled IrModule (with weights).
  2. Running forward passes by iterating through IrOps and dispatching
     each operation to the HAL OpExecutor.
  3. Managing KV cache tensor allocation (in coordination with BlockManager).
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

    KV cache management is delegated to the BlockManager; the executor
    only provides the tensor allocation for blocks.
    """

    def __init__(
        self,
        module: IrModule,
        hal_backend: OpExecutor,
    ) -> None:
        self._module = module
        self._hal = hal_backend
        self._function = module.main

        # Pre-load weights onto the appropriate device
        self._weights: dict[str, torch.Tensor] = {}
        for name, tensor in self._function.weights.items():
            self._weights[name] = tensor

    @property
    def function(self) -> IrFunction:
        return self._function

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Run a forward pass through the compiled graph.

        The SSA value map is the core state: it maps SSA names to computed
        torch.Tensor values as the executor walks through IrOps.

        Args:
            input_ids: Input token IDs tensor.
            **kwargs: Additional inputs (e.g. positions, attention_mask).

        Returns:
            Logits tensor from the final output.
        """
        # Initialize SSA value map with inputs
        ssa_values: dict[str, torch.Tensor] = {}

        # Map function inputs to their SSA names
        input_names = [name for name, _ in self._function.inputs]
        if input_names:
            ssa_values[input_names[0]] = input_ids
            # Map other named inputs from kwargs
            for named_input in input_names[1:]:
                if named_input in kwargs:
                    ssa_values[named_input] = kwargs[named_input]

        # Execute ops in order
        for op in self._function.ops:
            result = self._execute_op(op, ssa_values)
            if result is not None and op.outputs:
                ssa_values[op.outputs[0]] = result

        # Return the last output value
        if self._function.outputs:
            last_output_name = self._function.outputs[-1][0]
            if last_output_name in ssa_values:
                return ssa_values[last_output_name]

        # Fallback: return the last computed value
        if ssa_values:
            return list(ssa_values.values())[-1]

        return torch.tensor([])

    def _execute_op(self, op: IrOp, ssa_values: dict[str, torch.Tensor]) -> torch.Tensor | None:
        """Execute a single IrOp through the HAL.

        Constant/weight ops are extracted directly from the weight store.
        All other ops are dispatched to hal_backend.execute().
        """
        if op.name == "constant":
            # Constant op: load from weights
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
            return None

        return self._hal.execute(op.name, tensor_inputs, **op.attributes)

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
        return torch.empty(shape, dtype=dtype)
