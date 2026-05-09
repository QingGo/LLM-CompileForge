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
from compiler.mlir_artifact import MlirModule


def save_artifact(module: IrModule, directory: str) -> None:
    from compiler.mlir_artifact import save_mlir_artifact

    save_mlir_artifact(module, directory)


def load_artifact(directory: str) -> MlirModule:
    """Load a compiled model as an MlirModule (canonical MLIR artifact).

    Parses model.mlir and loads weights from weights.pth.  Returns an
    MlirModule suitable for MlirExecutor or LLMEngine.
    """
    in_dir = Path(directory)
    if not in_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    from compiler.mlir_artifact import load_mlir_artifact

    return load_mlir_artifact(str(in_dir))


def load_artifact_ir(directory: str) -> IrModule:
    """Load a compiled model as an IrModule (legacy Python IR).

    Deprecated: prefer ``load_artifact()`` which returns MlirModule.
    This bridge reconstructs an IrModule from the MLIR artifact for
    code that still requires the Python IR (compiler passes, etc.).
    """
    mlir_module = load_artifact(directory)

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
