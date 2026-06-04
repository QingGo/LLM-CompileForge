"""sf dialect operation definitions with full tensor types.

Each op is defined as a Python class with:
  - op_name: MLIR operation name (e.g. "sf.add")
  - input_types: list of MLIR tensor types for operands
  - output_types: list of MLIR tensor types for results (if known)
  - verify(): operation-level verification
  - canonicalize(): optional canonicalization pattern

Shape inference is provided separately in shape_inference.py and
computes output tensor type from input tensor types.
"""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar

import mlir.ir as ir

from compiler.mlir_dialect.shape.shape_inference_utils import _get_elt_type_map


def _tensor_type(
    element_type: str = "f32",
    shape: tuple[int | None, ...] | None = None,
) -> ir.RankedTensorType:
    """Construct an MLIR RankedTensorType.

    Args:
        element_type: "f32", "f16", "bf16", "i32", "i64", "i8", "bool"
        shape: tuple of int dimensions (None = dynamic)
    """
    elt = _get_elt_type_map().get(element_type, ir.F32Type.get())
    if shape is None:
        return ir.UnrankedTensorType.get(elt)
    return ir.RankedTensorType.get(list(shape), elt)


def _parse_ranked_tensor(tp: ir.Type) -> tuple[tuple[int | None, ...], str] | None:
    """Parse a RankedTensorType into (shape_tuple, element_type_string)."""
    if isinstance(tp, ir.RankedTensorType):
        shape = tuple(d for d in tp.shape)
        et = str(tp.element_type)
        return shape, et
    if isinstance(tp, ir.UnrankedTensorType):
        et = str(tp.element_type)
        return (None,), et
    return None


@dataclasses.dataclass
class SfOp:
    """Base class for sf dialect operations."""

    op_name: ClassVar[str] = "sf.unknown"

    @classmethod
    def build(
        cls,
        operands: list[ir.Value],
        result_types: list[ir.Type],
        attributes: dict[str, Any] | None = None,
        ip: Any | None = None,
    ) -> ir.Operation:
        return ir.Operation.create(
            cls.op_name,
            operands=operands,
            results=result_types,
            attributes=attributes or {},
            ip=ip,
        )


# ── Element-wise arithmetic ──────────────────────────────────


class Add(SfOp):
    op_name = "sf.add"


class Mul(SfOp):
    op_name = "sf.mul"


class Sub(SfOp):
    op_name = "sf.sub"


class Neg(SfOp):
    op_name = "sf.neg"


class Pow(SfOp):
    op_name = "sf.pow"


class Max(SfOp):
    op_name = "sf.max"


class Div(SfOp):
    op_name = "sf.div"


# ── Activation functions ─────────────────────────────────────


class Relu(SfOp):
    op_name = "sf.relu"


class Gelu(SfOp):
    op_name = "sf.gelu"


class Silu(SfOp):
    op_name = "sf.silu"


class Sigmoid(SfOp):
    op_name = "sf.sigmoid"


class Softplus(SfOp):
    op_name = "sf.softplus"


class Exp(SfOp):
    op_name = "sf.exp"


class Tanh(SfOp):
    op_name = "sf.tanh"


class Sqrt(SfOp):
    op_name = "sf.sqrt"


class ClampMin(SfOp):
    op_name = "sf.clamp_min"


class Rsqrts(SfOp):
    op_name = "sf.rsqrt"


# ── Trigonometric ────────────────────────────────────────────


class Cos(SfOp):
    op_name = "sf.cos"


class Sin(SfOp):
    op_name = "sf.sin"


# ── Matmul / Linear ──────────────────────────────────────────


class Matmul(SfOp):
    op_name = "sf.matmul"


class Linear(SfOp):
    op_name = "sf.linear"


# ── Normalization ────────────────────────────────────────────


class Softmax(SfOp):
    op_name = "sf.softmax"


class LayerNorm(SfOp):
    op_name = "sf.layer_norm"


class RmsNorm(SfOp):
    op_name = "sf.rms_norm"


# ── Shape manipulation ───────────────────────────────────────


class View(SfOp):
    op_name = "sf.view"


class Unsqueeze(SfOp):
    op_name = "sf.unsqueeze"


class Expand(SfOp):
    op_name = "sf.expand"


class Permute(SfOp):
    op_name = "sf.permute"


class Transpose(SfOp):
    op_name = "sf.transpose"


class Slice(SfOp):
    op_name = "sf.slice"


class Select(SfOp):
    op_name = "sf.select"


class Cat(SfOp):
    op_name = "sf.cat"


class Split(SfOp):
    op_name = "sf.split"


class Chunk(SfOp):
    op_name = "sf.chunk"


class Pad(SfOp):
    op_name = "sf.pad"


class Index(SfOp):
    op_name = "sf.index"


class Einsum(SfOp):
    op_name = "sf.einsum"


class Stack(SfOp):
    op_name = "sf.stack"


class ViewAs(SfOp):
    op_name = "sf.view_as"


class ExpandAs(SfOp):
    op_name = "sf.expand_as"


class SymSize(SfOp):
    op_name = "sf.sym_size"


# ── Reduction ────────────────────────────────────────────────


class Mean(SfOp):
    op_name = "sf.mean"


class Sum(SfOp):
    op_name = "sf.sum"


class Cumsum(SfOp):
    op_name = "sf.cumsum"


class LinalgNorm(SfOp):
    op_name = "sf.linalg_norm"


class Var(SfOp):
    op_name = "sf.var"


# ── Comparison ───────────────────────────────────────────────


class Gt(SfOp):
    op_name = "sf.gt"


class Lt(SfOp):
    op_name = "sf.lt"


class Eq(SfOp):
    op_name = "sf.eq"


class Ne(SfOp):
    op_name = "sf.ne"


class Le(SfOp):
    op_name = "sf.le"


class LogicalAnd(SfOp):
    op_name = "sf.logical_and"


# ── Tensor creation / utility ────────────────────────────────


class Embedding(SfOp):
    op_name = "sf.embedding"


class Triu(SfOp):
    op_name = "sf.triu"


class Tril(SfOp):
    op_name = "sf.tril"


class MaskedFill(SfOp):
    op_name = "sf.masked_fill"


class CopyOp(SfOp):
    op_name = "sf.copy_"


class TypeAs(SfOp):
    op_name = "sf.type_as"


class Identity(SfOp):
    op_name = "sf.identity"


class Conv1d(SfOp):
    op_name = "sf.conv1d"


class Arange(SfOp):
    op_name = "sf.arange"


class OnesLike(SfOp):
    op_name = "sf.ones_like"


class FullLike(SfOp):
    op_name = "sf.full_like"


class Zeros(SfOp):
    op_name = "sf.zeros"


class ZerosLike(SfOp):
    op_name = "sf.zeros_like"


class NewOnes(SfOp):
    op_name = "sf.new_ones"


class Eye(SfOp):
    op_name = "sf.eye"


class Diff(SfOp):
    op_name = "sf.diff"


# ── Attention ────────────────────────────────────────────────


class ScaledDotProductAttention(SfOp):
    op_name = "sf.scaled_dot_product_attention"


# ── Fused ops (produced by fusion passes) ────────────────────


class FusedSiluMul(SfOp):
    op_name = "sf.fused_silu_mul"


class FusedRmsNormMatmul(SfOp):
    op_name = "sf.fused_rms_norm_matmul"


class FusedQKV(SfOp):
    op_name = "sf.fused_qkv"


class FusedAttentionOutput(SfOp):
    op_name = "sf.fused_attention_output"


class FusedAttentionBlock(SfOp):
    op_name = "sf.fused_attention_block"


# ── Weight constant ──────────────────────────────────────────


class Weight(SfOp):
    op_name = "sf.weight"


class Constant(SfOp):
    op_name = "sf.constant"


# ── Registry ─────────────────────────────────────────────────


_ALL_OPS: dict[str, type[SfOp]] = {}


def _collect_ops() -> dict[str, type[SfOp]]:
    result: dict[str, type[SfOp]] = {}
    for _name, obj in list(globals().items()):
        if (
            isinstance(obj, type)
            and issubclass(obj, SfOp)
            and obj is not SfOp
            and obj.op_name != "sf.unknown"
        ):
            result[obj.op_name] = obj
    return result


_ALL_OPS = _collect_ops()


def get_op_class(op_name: str) -> type[SfOp] | None:
    """Look up the SfOp class for a given MLIR operation name."""
    return _ALL_OPS.get(op_name)
