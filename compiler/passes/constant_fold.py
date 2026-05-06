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
_FOLDABLE_OPS = frozenset({"add", "mul", "matmul", "view", "permute", "transpose", "cat"})


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
            elif op.name == "matmul":
                return torch.matmul(tensors[0], tensors[1])
            elif op.name == "view":
                shape = op.attributes.get("shape", None)
                if shape is not None:
                    return tensors[0].view(*shape)
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
            return None
        except Exception:
            return None
