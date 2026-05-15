"""Diagnostic script: run C++ lowering and capture backtrace via lldb.
Usage: lldb -b -o "run" -o "bt all" -o "quit" -- python3 scripts/diagnose_lowering.py
"""
import mlir.passmanager as pm
import mlir.ir as ir
from mlir_sf._mlir_libs._sfDialectsNanobind import sf
from compiler.serialize import load_artifact
from compiler.mlir_artifact import mlir_module_to_ir_module

ctx = ir.Context()
sf.register_dialects(ctx._CAPIPtr, load=True)
orig = load_artifact("compiled/opt_125m")
ir_mod = mlir_module_to_ir_module(orig, ctx=ctx)
pman = pm.PassManager.parse(
    "builtin.module(sf-promote-weights,canonicalize,cse,sf-lower-to-linalg)", ctx)
pman.run(ir_mod.operation)
print("DONE")
