"""Compiler optimization passes."""

from typing import Any

__all__ = [
    "CommonSubexpressionElimination",
    "ConstantFold",
    "DeadCodeElimination",
    "FuseAttentionBlock",
    "FuseAttentionPattern",
    "FuseQKVProjection",
    "FuseRMSNorm",
    "FuseSiLU",
    "Pass",
    "PassManager",
    "ValidateIR",
]

_LAZY_ATTRS = frozenset(__all__)


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        import compiler.passes.base as _base
        import compiler.passes.constant_fold as _cf
        import compiler.passes.cse_pass as _cse
        import compiler.passes.dce_pass as _dce
        import compiler.passes.fuse_attention as _fa
        import compiler.passes.fuse_attention_block as _fab
        import compiler.passes.fuse_qkv as _fqkv
        import compiler.passes.fuse_rms_norm as _frn
        import compiler.passes.fuse_silu as _fs
        import compiler.passes.validate_ir as _v

        _globals: dict[str, Any] = {
            "CommonSubexpressionElimination": _cse.CommonSubexpressionElimination,
            "ConstantFold": _cf.ConstantFold,
            "DeadCodeElimination": _dce.DeadCodeElimination,
            "FuseAttentionPattern": _fa.FuseAttentionPattern,
            "FuseAttentionBlock": _fab.FuseAttentionBlock,
            "FuseQKVProjection": _fqkv.FuseQKVProjection,
            "FuseRMSNorm": _frn.FuseRMSNorm,
            "FuseSiLU": _fs.FuseSiLU,
            "Pass": _base.Pass,
            "PassManager": _base.PassManager,
            "ValidateIR": _v.ValidateIR,
        }
        if name in _globals:
            value = _globals[name]
            globals()[name] = value
            return value
    raise AttributeError(f"module 'compiler.passes' has no attribute '{name}'")
