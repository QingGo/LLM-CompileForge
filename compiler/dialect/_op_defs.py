"""FX → MLIR operator definitions — single source of truth for op conversion.

``_OP_DEFS`` is the canonical registry mapping ATen operators to HAL ops.
Adding a new op requires *only* adding one entry here — all four lookup
tables (``_ATEN_TO_HAL``, ``_LIST_ARG_ATTR``, ``_SCALAR_KWARG_NAMES``,
``_SCALAR_INT_POSITIONS``) are auto-derived by ``_build_tables()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_SUPPRESS_LIST = "_SKIP_"


@dataclass
class _OpDef:
    hal_name: str
    aten_names: tuple[str, ...]
    list_arg_attr: str | None = _SUPPRESS_LIST
    scalar_kwargs: dict[int, str] = field(default_factory=dict)
    scalar_skip: tuple[int, ...] = ()


_OP_DEFS: list[_OpDef] = [
    _OpDef("add", ("aten.add.Tensor", "aten.add.Scalar", "aten.add", "aten.add_.Tensor", "add")),
    _OpDef("mul", ("aten.mul.Tensor", "aten.mul.Scalar", "aten.mul", "aten.mul_.Tensor", "aten.mul_.Scalar")),
    _OpDef("sub", ("aten.sub", "aten.sub.Tensor", "aten.rsub", "aten.rsub.Scalar")),
    _OpDef("neg", ("neg", "aten.neg.default", "aten.neg")),
    _OpDef("pow", ("pow", "aten.pow.Tensor_Scalar", "aten.pow")),
    _OpDef("max", ("aten.max", "aten.max.other")),
    _OpDef("relu", ("aten.relu", "aten.relu.default")),
    _OpDef("gelu", ("aten.gelu", "aten.gelu.default")),
    _OpDef("silu", ("aten.silu", "aten.silu.default")),
    _OpDef("sigmoid", ("aten.sigmoid.default", "aten.sigmoid")),
    _OpDef("softplus", ("aten.softplus.default",)),
    _OpDef("exp", ("aten.exp.default",)),
    _OpDef(
        "layer_norm",
        ("aten.layer_norm", "aten.layer_norm.default", "aten.native_layer_norm", "aten.native_layer_norm.default"),
        list_arg_attr="normalized_shape",
    ),
    _OpDef("rms_norm", ("aten.rms_norm", "aten.rms_norm.default"), list_arg_attr="normalized_shape"),
    _OpDef("softmax", ("aten._softmax", "aten._softmax.default", "aten.softmax.int"), scalar_kwargs={1: "dim"}),
    _OpDef("matmul", ("aten.matmul", "aten.matmul.default", "aten.mm", "aten.mm.default", "aten.bmm")),
    _OpDef("linear", ("aten.linear", "aten.linear.default")),
    _OpDef("view", ("aten.view", "aten.view.default", "aten.reshape", "aten.reshape.default"), list_arg_attr="shape"),
    _OpDef("unsqueeze", ("aten.unsqueeze", "aten.unsqueeze.default"), scalar_kwargs={1: "dim"}),
    _OpDef("expand", ("aten.expand", "aten.expand.default"), list_arg_attr=None),
    _OpDef("permute", ("aten.permute", "aten.permute.default"), list_arg_attr="dims"),
    _OpDef("transpose", ("aten.transpose", "aten.transpose.int"), scalar_kwargs={1: "dim0", 2: "dim1"}),
    _OpDef(
        "slice",
        ("aten.slice.Tensor", "aten.slice_copy.Tensor"),
        scalar_kwargs={1: "dim", 2: "start", 3: "end", 4: "step"},
    ),
    _OpDef("select", ("aten.select.int", "aten.select"), scalar_kwargs={1: "dim", 2: "index"}),
    _OpDef("cat", ("aten.cat", "aten.cat.default"), list_arg_attr=None, scalar_kwargs={1: "dim"}),
    _OpDef("split", ("aten.split_with_sizes.default",), list_arg_attr="split_sizes", scalar_kwargs={2: "dim"}),
    _OpDef("chunk", ("aten.chunk.default",), scalar_kwargs={1: "chunks", 2: "dim"}),
    _OpDef(
        "scaled_dot_product_attention",
        ("aten.scaled_dot_product_attention", "aten.scaled_dot_product_attention.default"),
    ),
    _OpDef("gt", ("gt", "aten.gt.Tensor", "aten.gt")),
    _OpDef("lt", ("aten.lt", "aten.lt.Tensor")),
    _OpDef("eq", ("aten.eq.Tensor",)),
    _OpDef("ne", ("aten.ne.Scalar", "aten.ne.Tensor")),
    _OpDef("le", ("aten.le.Tensor",)),
    _OpDef("logical_and", ("aten.__and__.Tensor",)),
    _OpDef("cos", ("aten.cos.default", "aten.cos")),
    _OpDef("sin", ("aten.sin.default", "aten.sin")),
    _OpDef("rsqrt", ("rsqrt", "aten.rsqrt.default", "aten.rsqrt")),
    _OpDef("mean", ("mean", "aten.mean.dim", "aten.mean")),
    _OpDef("triu", ("triu", "aten.triu.default", "aten.triu"), scalar_kwargs={1: "diagonal"}),
    _OpDef("tril", ("aten.tril.default", "aten.tril"), scalar_kwargs={1: "diagonal"}),
    _OpDef("cumsum", ("aten.cumsum", "aten.cumsum.default"), scalar_kwargs={1: "dim"}),
    _OpDef("sum", ("aten.sum.dim_IntList",), list_arg_attr="dim", scalar_kwargs={1: "dim", 2: "keepdim"}),
    _OpDef("diff", ("aten.diff.default",), scalar_kwargs={1: "n", 2: "dim"}),
    _OpDef("arange", ("aten.arange.start", "aten.arange", "aten.arange.default")),
    _OpDef("ones_like", ("aten.ones", "aten.ones.default"), list_arg_attr="shape"),
    _OpDef("full_like", ("aten.full", "aten.full.default"), list_arg_attr="shape", scalar_kwargs={1: "fill_value"}),
    _OpDef("zeros", ("aten.zeros.default",), list_arg_attr="shape"),
    _OpDef("zeros_like", ("aten.zeros_like.default",)),
    _OpDef("new_ones", ("aten.new_ones.default",)),
    _OpDef("eye", ("aten.eye.default",), scalar_kwargs={0: "n", 1: "m"}),
    _OpDef("embedding", ("aten.embedding", "aten.embedding.default"), scalar_skip=(2,)),
    _OpDef(
        "masked_fill", ("aten.masked_fill", "aten.masked_fill.Scalar", "aten.masked_fill_", "aten.masked_fill_.Scalar")
    ),
    _OpDef(
        "conv1d",
        ("aten.conv1d.default",),
        list_arg_attr="__conv1d__",
        scalar_kwargs={2: "bias", 3: "stride", 4: "padding", 5: "dilation", 6: "groups"},
    ),
    _OpDef("pad", ("aten.pad.default",), list_arg_attr="pad", scalar_skip=(1,)),
    _OpDef("index", ("aten.index.Tensor",), list_arg_attr=None),
    _OpDef("sym_size", ("sym_size", "aten.sym_size.int", "aten.sym_size"), scalar_kwargs={1: "dim"}),
    _OpDef("copy_", ("aten.copy_.default",)),
    _OpDef("type_as", ("aten.type_as", "aten.type_as.default")),
    _OpDef(
        "identity",
        (
            "_assert_tensor_metadata",
            "aten._assert_tensor_metadata.default",
            "aten._assert_tensor_metadata",
            "aten.to",
            "aten.to.dtype",
            "aten.to.dtype_layout",
            "aten.contiguous",
            "aten.contiguous.default",
            "aten.clone",
            "aten.clone.default",
            "aten.dropout",
            "aten.dropout.default",
            "aten.detach",
            "aten.detach.default",
            "aten.detach_",
            "aten.detach_.default",
            "aten.alias",
            "aten.alias.default",
            "aten.lift_fresh_copy",
            "aten.lift_fresh_copy.default",
        ),
        scalar_kwargs={1: "dtype"},
    ),
    _OpDef("getitem", ("getitem",)),
    _OpDef("_skip_wrap", ("wrap_with_set_grad_enabled",)),
    # ── RWKV ops ─────────────────────────────────
    _OpDef("div", ("aten.div.Tensor", "aten.div.default")),
    _OpDef("tanh", ("aten.tanh.default",)),
    _OpDef("sqrt", ("aten.sqrt.default",)),
    _OpDef("clamp_min", ("aten.clamp_min.default",)),
    _OpDef("einsum", ("aten.einsum.default",), scalar_kwargs={0: "equation"}, list_arg_attr=None),
    _OpDef("stack", ("aten.stack.default",)),
    _OpDef("linalg_norm", ("aten.linalg_vector_norm.default",)),
    _OpDef("var", ("aten.var.dim",)),
    _OpDef("view_as", ("aten.view_as.default",)),
    _OpDef("expand_as", ("aten.expand_as.default",)),
]

_ATEN_TO_HAL: dict[str, str] = {}
_LIST_ARG_ATTR: dict[str, str | None] = {}
_SCALAR_KWARG_NAMES: dict[str, dict[int, str]] = {}
_SCALAR_INT_POSITIONS: dict[str, list[int]] = {}


def _build_tables() -> None:
    for od in _OP_DEFS:
        hal = od.hal_name
        for aten_name in od.aten_names:
            if aten_name in _ATEN_TO_HAL and _ATEN_TO_HAL[aten_name] != hal:
                raise AssertionError(f"aten '{aten_name}' maps to both '{_ATEN_TO_HAL[aten_name]}' and '{hal}'")
            _ATEN_TO_HAL[aten_name] = hal
        if od.list_arg_attr != _SUPPRESS_LIST:
            _LIST_ARG_ATTR.setdefault(hal, od.list_arg_attr)
        if od.scalar_kwargs:
            _SCALAR_KWARG_NAMES.setdefault(hal, od.scalar_kwargs)
        positions = set(od.scalar_kwargs.keys()) | set(od.scalar_skip)
        if positions:
            existing = set(_SCALAR_INT_POSITIONS.get(hal, []))
            _SCALAR_INT_POSITIONS[hal] = sorted(existing | positions)


_build_tables()
