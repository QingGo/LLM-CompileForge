"""Constant folding pass.

Identifies operations where all inputs are compile-time constants
(weight tensors) and pre-computes the result, replacing the runtime
operation with a constant weight reference.
"""

from __future__ import annotations

import torch

from compiler.ir import IrFunction, IrModule, IrOp
from compiler.passes.base import Pass

# Ops that can be constant-folded when all inputs are constants
_FOLDABLE_OPS = frozenset(
    {
        "add", "mul", "matmul", "view", "permute", "transpose", "cat",
        "sub", "neg", "arange", "expand", "slice", "unsqueeze",
        "ne", "eq", "le", "lt", "gt",
        "logical_and",
        "cumsum", "triu",
        "identity",
        # Qwen-specific
        "diff", "select",
    }
)


class ConstantFold(Pass):
    """Fold operations with all-constant inputs at compile time."""

    def apply(self, module: IrModule) -> IrModule:
        for func in module.functions:
            self._fold_function(func)
        return module

    @staticmethod
    def _fold_function(func: IrFunction) -> None:
        # Identify which SSA values are compile-time constants
        constants: set[str] = set(func.weights.keys())

        new_ops: list[IrOp] = []

        for op in func.ops:
            if op.name in _FOLDABLE_OPS and all(inp in constants for inp in op.inputs):
                # Attempt to fold
                result = ConstantFold._try_fold(op, func.weights, constants)
                if result is not None:
                    const_name = f"_folded_{len(func.weights)}"
                    func.weights[const_name] = result
                    constants.add(const_name)
                    # Mark op's output names as constants (for downstream folding)
                    for out_name in op.outputs:
                        constants.add(out_name)
                    # Create a no-op replacement that outputs a constant reference
                    new_ops.append(
                        IrOp(
                            name="constant",
                            inputs=[const_name],
                            outputs=op.outputs,
                            attributes={"folded": True, "original_op": op.name},
                        )
                    )
                    continue

            new_ops.append(op)
            # Outputs of non-folded ops are not compile-time constants
            # (their values depend on runtime inputs)

        func.ops = new_ops

    @staticmethod
    def _try_fold(
        op: IrOp,
        weights: dict[str, torch.Tensor],
        constants: set[str],
    ) -> torch.Tensor | None:
        """Try to compute the result of an op at compile time.

        Returns the folded tensor, or None if folding is not possible.
        """
        try:
            tensors = []
            for inp in op.inputs:
                if inp in weights:
                    tensors.append(weights[inp])
                else:
                    return None  # Missing constant

            if op.name == "add":
                return tensors[0] + tensors[1]
            elif op.name == "mul":
                return tensors[0] * tensors[1]
            elif op.name == "sub":
                return tensors[0] - tensors[1]
            elif op.name == "neg":
                return -tensors[0]
            elif op.name == "matmul":
                return torch.matmul(tensors[0], tensors[1])
            elif op.name == "view":
                shape = op.attributes.get("shape", None)
                if shape is not None:
                    resolved = []
                    for _i, s in enumerate(shape):
                        if isinstance(s, str) and s in weights:
                            resolved.append(int(weights[s].item()))
                        elif isinstance(s, int):
                            resolved.append(s)
                        else:
                            resolved.append(s)
                    return tensors[0].reshape(*resolved)
                return None
            elif op.name == "permute":
                dims = op.attributes.get("dims", None)
                if dims is not None:
                    return tensors[0].permute(*dims)
                return None
            elif op.name == "transpose":
                dim0 = op.attributes.get("dim0", 0)
                dim1 = op.attributes.get("dim1", 1)
                return tensors[0].transpose(dim0, dim1)
            elif op.name == "cat":
                dim = op.attributes.get("dim", 0)
                return torch.cat(tensors, dim=dim)
            elif op.name == "expand":
                shape_list = []
                for t in tensors[1:]:
                    shape_list.append(int(t.item()) if t.numel() == 1 else -1)
                return tensors[0].expand(*shape_list)
            elif op.name == "slice":
                dim = op.attributes.get("dim", 0)
                start = op.attributes.get("start", 0)
                end = op.attributes.get("end", None)
                step = op.attributes.get("step", 1)
                if end is None or end > tensors[0].shape[dim]:
                    end = tensors[0].shape[dim]
                if start >= end:
                    start = 0
                result = tensors[0].narrow(dim, start, min(end - start, tensors[0].shape[dim] - start))
                if step != 1:
                    slices = [slice(None)] * tensors[0].dim()
                    slices[dim] = slice(None, None, step)
                    result = result[tuple(slices)]
                return result.clone()
            elif op.name == "unsqueeze":
                dim = op.attributes.get("dim", 0)
                return tensors[0].unsqueeze(dim)
            elif op.name == "arange":
                end = int(tensors[0].item()) if tensors else 64
                return torch.arange(end)
            elif op.name == "ne":
                return tensors[0] != tensors[1]
            elif op.name == "eq":
                return tensors[0] == tensors[1]
            elif op.name == "le":
                return tensors[0] <= tensors[1]
            elif op.name == "lt":
                return tensors[0] < tensors[1]
            elif op.name == "gt":
                return tensors[0] > tensors[1]
            elif op.name == "logical_and":
                return torch.logical_and(tensors[0], tensors[1])
            elif op.name == "cumsum":
                dim = op.attributes.get("dim", -1)
                return torch.cumsum(tensors[0], dim=dim)
            elif op.name == "triu":
                diagonal = op.attributes.get("diagonal", 0)
                return torch.triu(tensors[0], diagonal=diagonal)
            elif op.name == "identity":
                return tensors[0]
            elif op.name == "diff":
                dim = op.attributes.get("dim", -1)
                return torch.diff(tensors[0], dim=dim)
            elif op.name == "select":
                dim = op.attributes.get("dim", 0)
                index = op.attributes.get("index", 0)
                return tensors[0].select(dim, index)
            return None
        except Exception:
            return None
