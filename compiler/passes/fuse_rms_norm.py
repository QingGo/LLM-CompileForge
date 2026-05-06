"""RMSNorm + MatMul fusion pass.

Detects the pattern:
    linalg.norm → arith.divf (normalize) → arith.mulf (gain) → linalg.matmul

and replaces it with a single fused.rms_norm_matmul operation,
eliminating kernel launch overhead and one HBM round-trip.

From design doc §4.2.2: estimated 5-15% latency reduction on A100.
"""

from __future__ import annotations

from compiler.ir import IrFunction, IrModule, IrOp
from compiler.passes.base import Pass


class FuseRMSNorm(Pass):
    """Fuse RMSNorm operation chains into a single fused operation."""

    def apply(self, module: IrModule) -> IrModule:
        for func in module.functions:
            self._fuse_function(func)
        return module

    @staticmethod
    def _fuse_function(func: IrFunction) -> None:
        """Find and fuse RMSNorm → Mul (weight) → MatMul chains."""
        # Build index: output_name → IrOp
        producer_map: dict[str, IrOp] = {}
        for op in func.ops:
            for out in op.outputs:
                producer_map[out] = op

        fused_ops: list[IrOp] = []
        removed_outputs: set[str] = set()

        for op in func.ops:
            # Skip ops already consumed by fusion
            if any(out in removed_outputs for out in op.outputs):
                continue

            # Pattern: rms_norm output feeds into mul then matmul
            if op.name == "rms_norm" and len(op.outputs) >= 1:
                norm_out = op.outputs[0]
                # Find mul consumer
                mul_op = FuseRMSNorm._find_consumer(norm_out, "mul", producer_map)
                if mul_op and len(mul_op.outputs) >= 1:
                    mul_out = mul_op.outputs[0]
                    matmul_op = FuseRMSNorm._find_consumer(mul_out, "matmul", producer_map)
                    if matmul_op:
                        # Fuse: replace rms_norm + mul + matmul with fused op
                        fused = IrOp(
                            name="fused_rms_norm_matmul",
                            inputs=op.inputs + matmul_op.inputs[1:],
                            outputs=matmul_op.outputs,
                            attributes={
                                "eps": op.attributes.get("eps", 1e-5),
                                **matmul_op.attributes,
                            },
                        )
                        fused_ops.append(fused)
                        removed_outputs.update(op.outputs)
                        removed_outputs.update(mul_op.outputs)
                        removed_outputs.update(matmul_op.outputs)
                        continue

            fused_ops.append(op)

        # Remap remaining op inputs
        fused_outputs: set[str] = {out for o in fused_ops for out in o.outputs}
        for op in fused_ops:
            new_inputs: list[str] = []
            for inp in op.inputs:
                # If input is from a removed op's output, find the fused replacement
                if inp in removed_outputs and inp not in fused_outputs:
                    # Look through fused_ops for a replacement output
                    found = False
                    for fo in fused_ops:
                        if inp in fo.outputs:
                            new_inputs.append(fo.outputs[0])
                            found = True
                            break
                    if found:
                        continue
                new_inputs.append(inp)
            op.inputs = new_inputs

        func.ops = fused_ops

    @staticmethod
    def _find_consumer(output_name: str, op_name: str, producer_map: dict[str, IrOp]) -> IrOp | None:
        """Find an IrOp that consumes the given output."""
        for op in producer_map.values():
            if output_name in op.inputs and op.name == op_name:
                return op
        return None
