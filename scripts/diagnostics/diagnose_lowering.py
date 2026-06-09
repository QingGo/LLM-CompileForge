"""Diagnostic script: run C++ lowering and capture backtrace via lldb.
Usage: lldb -b -o "run" -o "bt all" -o "quit" -- python3 scripts/diagnose_lowering.py
"""
import mlir.ir as ir
import mlir.passmanager as pm
from mlir_sf._mlir_libs._sfDialectsNanobind import sf

from compiler.artifact import mlir_module_to_ir_module
from compiler.pipeline.lowering import SF_LOWERING_PIPELINE
from compiler.serialize import load_artifact

ctx = ir.Context()
sf.register_dialects(ctx._CAPIPtr, load=True)
orig = load_artifact("outputs/compiled/opt_125m")
ir_mod = mlir_module_to_ir_module(orig, ctx=ctx)
pman = pm.PassManager.parse(
    "builtin.module(" + SF_LOWERING_PIPELINE + ")", ctx)
pman.run(ir_mod.operation)
print("DONE")
