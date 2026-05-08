"""Compiler optimization passes."""

from compiler.passes.base import Pass, PassManager
from compiler.passes.constant_fold import ConstantFold
from compiler.passes.cse_pass import CommonSubexpressionElimination
from compiler.passes.dce_pass import DeadCodeElimination
from compiler.passes.fuse_rms_norm import FuseRMSNorm
from compiler.passes.fuse_silu import FuseSiLU
from compiler.passes.validate_ir import ValidateIR

__all__ = [
    "CommonSubexpressionElimination",
    "ConstantFold",
    "DeadCodeElimination",
    "FuseRMSNorm",
    "FuseSiLU",
    "Pass",
    "PassManager",
    "ValidateIR",
]
