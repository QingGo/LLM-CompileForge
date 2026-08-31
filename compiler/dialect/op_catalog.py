"""SfaOpCatalog builder — populates the proto catalog from _op_defs.py.

Provides ``build_op_catalog()`` which returns a populated ``SfaOpCatalog``
proto message covering all HAL operators known to the compiler.
"""

from __future__ import annotations

from gen.proto.python.sfa_abi_pb2 import (  # type: ignore[attr-defined]
    SfaOpCatalog,
    SfaOpParam,
)

from ._op_defs import _OP_DEFS

# ── Kind taxonomy ────────────────────────────────────────────────────
# Each HAL op maps to one of these categories.  The ordering is the
# same as the proto documentation so generated catalog entries are
# grouped logically.

_OPS_BY_KIND: dict[str, list[str]] = {
    "matmul": [],
    "element_wise": [],
    "reduce": [],
    "attention": [],
    "normalization": [],
    "reshape": [],
    "compare": [],
    "gather": [],
    "fill": [],
    "shape": [],
    "slice": [],
    "softmax": [],
}

# ── Per-op kind assignments ──────────────────────────────────────────


def _classify_ops() -> None:
    """Assign each HAL op name to its kind category."""
    rules: dict[str, str] = {
        # matmul
        "linear": "matmul",
        "matmul": "matmul",
        "conv1d": "matmul",
        "einsum": "matmul",
        # element_wise
        "add": "element_wise",
        "mul": "element_wise",
        "sub": "element_wise",
        "neg": "element_wise",
        "pow": "element_wise",
        "relu": "element_wise",
        "gelu": "element_wise",
        "silu": "element_wise",
        "sigmoid": "element_wise",
        "softplus": "element_wise",
        "exp": "element_wise",
        "cos": "element_wise",
        "sin": "element_wise",
        "rsqrt": "element_wise",
        "triu": "element_wise",
        "tril": "element_wise",
        "diff": "element_wise",
        "masked_fill": "element_wise",
        "copy_": "element_wise",
        "type_as": "element_wise",
        "identity": "element_wise",
        "div": "element_wise",
        "tanh": "element_wise",
        "sqrt": "element_wise",
        "clamp_min": "element_wise",
        # reduce
        "mean": "reduce",
        "sum": "reduce",
        "cumsum": "reduce",
        "linalg_norm": "reduce",
        "var": "reduce",
        # attention
        "scaled_dot_product_attention": "attention",
        # normalization
        "layer_norm": "normalization",
        "rms_norm": "normalization",
        # reshape
        "view": "reshape",
        "unsqueeze": "reshape",
        "expand": "reshape",
        "permute": "reshape",
        "transpose": "reshape",
        "cat": "reshape",
        "split": "reshape",
        "chunk": "reshape",
        "pad": "reshape",
        "stack": "reshape",
        "view_as": "reshape",
        "expand_as": "reshape",
        # compare
        "gt": "compare",
        "lt": "compare",
        "eq": "compare",
        "ne": "compare",
        "le": "compare",
        "logical_and": "compare",
        "max": "compare",
        # gather
        "embedding": "gather",
        "index": "gather",
        # fill
        "arange": "fill",
        "ones_like": "fill",
        "full_like": "fill",
        "zeros": "fill",
        "zeros_like": "fill",
        "new_ones": "fill",
        "eye": "fill",
        # shape
        "sym_size": "shape",
        # slice
        "slice": "slice",
        "select": "slice",
        "getitem": "slice",
        # softmax
        "softmax": "softmax",
    }

    # Internal markers that should not appear in the catalog.
    _skip_ops: set[str] = {"_skip_wrap", "fused_silu_mul"}

    seen: set[str] = set()
    for od in _OP_DEFS:
        hal = od.hal_name
        if hal in seen or hal in _skip_ops:
            continue
        seen.add(hal)
        kind = rules.get(hal, "element_wise")
        _OPS_BY_KIND[kind].append(hal)

    # Sort each group for deterministic output.
    for k in _OPS_BY_KIND:
        _OPS_BY_KIND[k].sort()


_classify_ops()
del _classify_ops

# ── Param builder helpers ────────────────────────────────────────────


def _p(name: str, dtype: str = "float32", rank: int = 0) -> SfaOpParam:
    """Shorthand for building a SfaOpParam."""
    param = SfaOpParam()
    param.name = name
    param.dtype = dtype
    param.rank = rank
    return param


_PINP = _p("input", "float32", 0)
_PIN1 = _p("input", "float32", 1)
_PIN2 = _p("input", "float32", 2)
_PIN3 = _p("input", "float32", 3)
_PIN4 = _p("input", "float32", 4)
_PWGT = _p("weight", "float32", 2)
_PBIAS = _p("bias", "float32", 1)
_POTHER = _p("other", "float32", 0)
_PSCALAR = _p("scalar", "float32", 0)
_PDIM = _p("dim", "int64", 0)
_PINDEX = _p("index", "int64", 1)

# ── Per-op param definitions ─────────────────────────────────────────

_OP_PARAMS: dict[str, list[SfaOpParam]] = {
    "linear": [_PIN2, _PWGT, _PBIAS],
    "matmul": [_PIN2, _PIN2],
    "conv1d": [_PIN3, _PWGT],
    "einsum": [_PIN2, _PIN2],
    "add": [_PIN2, _PIN2],
    "mul": [_PIN2, _PIN2],
    "sub": [_PIN2, _PIN2],
    "neg": [_PIN2],
    "pow": [_PIN2, _PSCALAR],
    "relu": [_PIN2],
    "gelu": [_PIN2],
    "silu": [_PIN2],
    "sigmoid": [_PIN2],
    "softplus": [_PIN2],
    "exp": [_PIN2],
    "cos": [_PIN2],
    "sin": [_PIN2],
    "rsqrt": [_PIN2],
    "triu": [_PIN2, _PDIM],
    "tril": [_PIN2, _PDIM],
    "diff": [_PIN2, _PDIM],
    "masked_fill": [_PIN2, _PIN2, _PSCALAR],
    "copy_": [_PIN2, _PIN2],
    "type_as": [_PIN2, _PIN2],
    "identity": [_PIN2],
    "div": [_PIN2, _PIN2],
    "tanh": [_PIN2],
    "sqrt": [_PIN2],
    "clamp_min": [_PIN2, _PSCALAR],
    "mean": [_PIN2, _PDIM],
    "sum": [_PIN2, _PDIM],
    "cumsum": [_PIN2, _PDIM],
    "linalg_norm": [_PIN2, _PDIM],
    "var": [_PIN2, _PDIM],
    "scaled_dot_product_attention": [_PIN4, _PIN4, _PIN4, _PIN2],
    "layer_norm": [_PIN2, _PWGT, _PBIAS],
    "rms_norm": [_PIN2, _PWGT],
    "view": [_PIN2, _p("shape", "int64", 1)],
    "unsqueeze": [_PIN2, _PDIM],
    "expand": [_PIN2, _p("shape", "int64", 1)],
    "permute": [_PIN2, _p("dims", "int64", 1)],
    "transpose": [_PIN2, _PDIM, _PDIM],
    "cat": [_PIN2, _PIN2, _PDIM],
    "split": [_PIN2, _p("split_sizes", "int64", 1), _PDIM],
    "chunk": [_PIN2, _p("chunks", "int64", 0), _PDIM],
    "pad": [_PIN2, _p("pad", "int64", 1)],
    "stack": [_PIN2, _PIN2, _PDIM],
    "view_as": [_PIN2, _PIN2],
    "expand_as": [_PIN2, _PIN2],
    "gt": [_PIN2, _PIN2],
    "lt": [_PIN2, _PIN2],
    "eq": [_PIN2, _PIN2],
    "ne": [_PIN2, _PIN2],
    "le": [_PIN2, _PIN2],
    "logical_and": [_PIN2, _PIN2],
    "max": [_PIN2, _PIN2],
    "embedding": [_PIN2, _PWGT, _PINDEX],
    "index": [_PIN2, _PINDEX],
    "arange": [_PSCALAR, _PSCALAR, _PSCALAR],
    "ones_like": [_p("shape", "int64", 1)],
    "full_like": [_p("shape", "int64", 1), _PSCALAR],
    "zeros": [_p("shape", "int64", 1)],
    "zeros_like": [_PIN2],
    "new_ones": [_PIN2, _p("size", "int64", 1)],
    "eye": [_p("n", "int64", 0), _p("m", "int64", 0)],
    "sym_size": [_PIN2, _PDIM],
    "slice": [_PIN2, _PDIM, _PSCALAR, _PSCALAR, _PSCALAR],
    "select": [_PIN2, _PDIM, _PINDEX],
    "getitem": [_PIN2, _PINDEX],
    "softmax": [_PIN2, _PDIM],
}

# ── Output dtype defaults ────────────────────────────────────────────

_DEFAULT_OUTPUT_DTYPE = ["float32"]

_OP_OUTPUT_DTYPES: dict[str, list[str]] = {
    "gt": ["bool"],
    "lt": ["bool"],
    "eq": ["bool"],
    "ne": ["bool"],
    "le": ["bool"],
    "logical_and": ["bool"],
    "arange": ["float32"],
    "zeros": ["float32"],
    "ones_like": ["float32"],
    "full_like": ["float32"],
    "sym_size": ["int64"],
}


# ── Plan-only canonical HAL kernels ─────────────────────────────────
# These names appear in the Phase 5 OpPlan proto.  They are generated by
# the op-plan pass (compiler/op_plan.py), not by FX→MLIR conversion, so
# they live next to the catalog rather than in _OP_DEFS (which is the
# ATen mapping registry).  The runtime kernel registry is built from the
# same canonical names — no ad-hoc op string dispatch.

_PLAN_ONLY_OPS: dict[str, tuple[str, list[SfaOpParam]]] = {
    "linear_transb": ("matmul", [_PIN2, _PWGT, _PBIAS]),
    "attention_causal": ("attention", [_PIN4, _PIN4, _PIN4, _PIN2]),
}

# ── Public API ───────────────────────────────────────────────────────


def build_op_catalog() -> SfaOpCatalog:
    """Build a populated SfaOpCatalog with all HAL operators.

    Returns:
        SfaOpCatalog proto message containing every known HAL op
        grouped by kind.
    """
    catalog = SfaOpCatalog()
    emitted: set[str] = set()
    for kind in _OPS_BY_KIND:
        for name in _OPS_BY_KIND[kind]:
            _append_catalog_op(catalog, emitted, kind, name)
    for name, (kind, params) in sorted(_PLAN_ONLY_OPS.items()):
        _append_catalog_op(catalog, emitted, kind, name, params)
    return catalog


def _append_catalog_op(
    catalog: SfaOpCatalog,
    emitted: set[str],
    kind: str,
    name: str,
    params: list[SfaOpParam] | None = None,
) -> None:
    if name in emitted:
        return
    emitted.add(name)
    op_def = catalog.ops.add()
    op_def.name = name
    op_def.kind = kind
    for p in params if params is not None else _OP_PARAMS.get(name, [_PIN2]):
        param = op_def.params.add()
        param.CopyFrom(p)
    for dt in _OP_OUTPUT_DTYPES.get(name, _DEFAULT_OUTPUT_DTYPE):
        op_def.output_dtypes.append(dt)
