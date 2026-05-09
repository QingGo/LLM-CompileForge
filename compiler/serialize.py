"""Model artifact serialization / deserialization.

The sole artifact format is MLIR (model.mlir) with weights (weights.pth)
and metadata (metadata.json).

Output structure:
  compiled/<model_name>/
    model.mlir       — MLIR text (standard-compliant)
    weights.pth      — PyTorch state dict
    metadata.json    — compilation metadata
"""

from __future__ import annotations

from pathlib import Path

from compiler.ir import IrFunction as IrFunc
from compiler.ir import IrModule, IrOp, IrType


def save_artifact(module: IrModule, directory: str) -> None:
    """Persist a compiled IrModule to disk as MLIR artifact."""
    from compiler.mlir_artifact import save_mlir_artifact

    save_mlir_artifact(module, directory)


def load_artifact(directory: str) -> IrModule:
    """Load a compiled IrModule from MLIR artifact.

    Parses model.mlir, loads weights from weights.pth, and reconstructs
    an IrModule for downstream consumers (executor, tests, etc.).
    """
    in_dir = Path(directory)
    if not in_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    from compiler.mlir_artifact import load_mlir_artifact

    mlir_module = load_mlir_artifact(str(in_dir))

    ir_functions = []
    for mf in mlir_module.functions:
        ir_ops = []
        for mop in mf.ops:
            if mop.name in ("sf.weight", "sf.constant"):
                ir_ops.append(IrOp(
                    name="constant",
                    inputs=[mop.attributes.get("name", mop.operands[0] if mop.operands else "")],
                    outputs=mop.results,
                    attributes=mop.attributes,
                ))
            else:
                ir_ops.append(IrOp(
                    name=mop.op_name,
                    inputs=mop.operands,
                    outputs=mop.results,
                    attributes=mop.attributes,
                ))

        inp_ir = [(name.replace("%", ""), IrType("float32")) for name, _ in mf.inputs]
        out_ir = [(name.replace("%", ""), IrType("float32")) for name, _ in mf.outputs]
        ir_func = IrFunc(
            name=mf.name,
            inputs=inp_ir,
            outputs=out_ir,
            ops=ir_ops,
            weights=mf.weights,
        )
        ir_functions.append(ir_func)

    return IrModule(functions=ir_functions, metadata=mlir_module.metadata)
