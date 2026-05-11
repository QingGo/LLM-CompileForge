"""sf → standard dialect lowering using PassManager Python passes."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlir.ir as ir
import mlir.passmanager as pm

_CTX: ir.Context | None = None


def _setup_mlir_path() -> None:
    _mlir_pkg = Path(__file__).resolve().parent.parent.parent / "mlir_binding" / "mlir_package"
    if _mlir_pkg.is_dir() and str(_mlir_pkg) not in sys.path:
        sys.path.insert(0, str(_mlir_pkg))


def _has_bindings() -> bool:
    _setup_mlir_path()
    try:
        import mlir.ir  # noqa: F401
        return True
    except ImportError:
        return False


def sf_to_linalg_pass(mlir_text: str) -> str:
    global _CTX
    if not _has_bindings():
        return mlir_text
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    _CTX = ctx
    with ctx, ir.Location.unknown(ctx):
        module = ir.Module.parse(mlir_text, ctx)
        pmgr = pm.PassManager.parse("builtin.module()", ctx)
        pmgr.add(_lower_sf_ops)
        pmgr.run(module.operation)
        return str(module)


def _lower_sf_ops(op: Any, pass_: Any) -> None:
    sf_ops: list[Any] = []

    def _collect(inner: Any) -> Any:
        name = inner.name if hasattr(inner, "name") else str(inner)
        if name.startswith("sf.") and name not in ("sf.weight", "sf.constant"):
            sf_ops.append(inner)
        return ir.WalkResult.ADVANCE

    op.walk(_collect)
    for sf_op in sf_ops:
        new_op = _LOWER_TABLE.get(sf_op.name, lambda _o: None)(sf_op)
        if new_op is not None and sf_op.operation.results:
            sf_op.operation.results[0].replace_all_uses_with(new_op.operation.results[0])
            sf_op.operation.erase()


def _ip(op: Any) -> ir.InsertionPoint:
    return ir.InsertionPoint(op.operation)


def _el(rt: ir.Type) -> ir.Type:
    return rt.element_type


# ── linalg.generic builders ───────────────────────────────────


def _make_generic(
    inputs: list[ir.Value],
    output: ir.Value,
    rt: ir.Type,
    ins: int,
    ip: ir.InsertionPoint,
    body_fn: Callable[[list[ir.Value]], ir.Value],
    iterator_types: list[Any] | None = None,
    indexing_maps: list[ir.AffineMap] | None = None,
) -> Any:
    import mlir.dialects.linalg as _lin

    rank = len(rt.shape)
    ident = ir.AffineMap.get_identity(rank)
    maps = ir.ArrayAttr.get([ir.AffineMapAttr.get(m) for m in (indexing_maps or [ident] * (ins + 1))])
    iters = _lin._IteratorTypeArrayAttr(
        iterator_types or ([_lin.IteratorType.parallel] * rank), context=_CTX,
    )
    gop = ir.Operation.create(
        "linalg.generic",
        operands=list(inputs) + [output],
        results=[rt],
        attributes={
            "indexing_maps": maps,
            "iterator_types": iters,
            "operandSegmentSizes": ir.DenseI32ArrayAttr.get([ins, 1]),
        },
        regions=1,
        ip=ip,
    )
    region = gop.operation.regions[0]
    elt = rt.element_type
    block = ir.Block.create_at_start(region, [elt] * ins + [elt])
    args = list(block.arguments)
    with ir.InsertionPoint(block):
        result = body_fn(args[:-1])
        ir.Operation.create("linalg.yield", operands=[result])
    return gop


def _make_matmul(
    inputs: list[ir.Value],
    output: ir.Value,
    rt: ir.Type,
    ip: ir.InsertionPoint,
    op_name: str = "linalg.matmul",
) -> Any:
    mop = ir.Operation.create(
        op_name, operands=list(inputs) + [output], results=[rt],
        attributes={"operandSegmentSizes": ir.DenseI32ArrayAttr.get([len(inputs), 1])},
        regions=1, ip=ip,
    )
    region = mop.operation.regions[0]
    elt = rt.element_type
    n_in = len(inputs)
    block = ir.Block.create_at_start(region, [elt] * n_in + [elt])
    args = list(block.arguments)
    with ir.InsertionPoint(block):
        mul = ir.Operation.create("arith.mulf", operands=[args[0], args[1]], infer_type=True)
        add = ir.Operation.create("arith.addf", operands=[mul.result, args[2]], infer_type=True)
        ir.Operation.create("linalg.yield", operands=[add.result])
    return mop


# ── Binary / Unary / Passthrough ──────────────────────────────


def _lower_binary(arith_op: str) -> Callable[[Any], Any | None]:
    def lower(op: Any) -> Any | None:
        if len(op.operation.operands) < 2:
            return None
        a, b = op.operation.operands[0], op.operation.operands[1]
        rt = op.result.type
        ip = _ip(op)
        empty = ir.Operation.create("tensor.empty", results=[rt], ip=ip)
        return _make_generic(
            [a, b], empty.result, rt, 2, ip,
            lambda ia: ir.Operation.create(arith_op, operands=[ia[0], ia[1]], infer_type=True).result,
        )
    return lower


def _lower_unary(arith_op: str) -> Callable[[Any], Any | None]:
    def lower(op: Any) -> Any | None:
        if len(op.operation.operands) < 1:
            return None
        a = op.operation.operands[0]
        rt = op.result.type
        ip = _ip(op)
        empty = ir.Operation.create("tensor.empty", results=[rt], ip=ip)
        return _make_generic(
            [a], empty.result, rt, 1, ip,
            lambda ia: ir.Operation.create(arith_op, operands=[ia[0]], infer_type=True).result,
        )
    return lower


def _lower_passthrough(op: Any) -> Any | None:
    if len(op.operation.operands) < 1:
        return None
    a = op.operation.operands[0]
    rt = op.result.type
    ip = _ip(op)
    empty = ir.Operation.create("tensor.empty", results=[rt], ip=ip)
    return _make_generic([a], empty.result, rt, 1, ip, lambda ia: ia[0])


# ── Activations ───────────────────────────────────────────────


def _lower_relu(op: Any) -> Any | None:
    if len(op.operation.operands) < 1:
        return None
    a = op.operation.operands[0]
    rt = op.result.type
    elt = _el(rt)
    ip = _ip(op)
    empty = ir.Operation.create("tensor.empty", results=[rt], ip=ip)

    def body(ia: list[ir.Value]) -> ir.Value:
        zero = ir.Operation.create("arith.constant", results=[elt], attributes={"value": ir.FloatAttr.get(elt, 0.0)})
        return ir.Operation.create("arith.maxnumf", operands=[ia[0], zero.result], infer_type=True).result

    return _make_generic([a], empty.result, rt, 1, ip, body)


def _lower_silu(op: Any) -> Any | None:
    if len(op.operation.operands) < 1:
        return None
    a = op.operation.operands[0]
    rt = op.result.type
    elt = _el(rt)
    ip = _ip(op)
    empty = ir.Operation.create("tensor.empty", results=[rt], ip=ip)

    def body(ia: list[ir.Value]) -> ir.Value:
        x = ia[0]
        neg = ir.Operation.create("arith.negf", operands=[x], infer_type=True)
        exp = ir.Operation.create("math.exp", operands=[neg.result], infer_type=True)
        one = ir.Operation.create("arith.constant", results=[elt], attributes={"value": ir.FloatAttr.get(elt, 1.0)})
        denom = ir.Operation.create("arith.addf", operands=[one.result, exp.result], infer_type=True)
        sig = ir.Operation.create("arith.divf", operands=[one.result, denom.result], infer_type=True)
        return ir.Operation.create("arith.mulf", operands=[x, sig.result], infer_type=True).result

    return _make_generic([a], empty.result, rt, 1, ip, body)


def _lower_gelu(op: Any) -> Any | None:
    if len(op.operation.operands) < 1:
        return None
    a = op.operation.operands[0]
    rt = op.result.type
    elt = _el(rt)
    ip = _ip(op)
    empty = ir.Operation.create("tensor.empty", results=[rt], ip=ip)

    def body(ia: list[ir.Value]) -> ir.Value:
        x = ia[0]
        half = ir.Operation.create("arith.constant", results=[elt], attributes={"value": ir.FloatAttr.get(elt, 0.5)})
        one = ir.Operation.create("arith.constant", results=[elt], attributes={"value": ir.FloatAttr.get(elt, 1.0)})
        c1 = ir.Operation.create(
            "arith.constant", results=[elt],
            attributes={"value": ir.FloatAttr.get(elt, 0.7978845608)},
        )
        c2 = ir.Operation.create("arith.constant", results=[elt], attributes={"value": ir.FloatAttr.get(elt, 0.044715)})
        x3 = ir.Operation.create("arith.mulf", operands=[x, x], infer_type=True)
        x3 = ir.Operation.create("arith.mulf", operands=[x3.result, x], infer_type=True)
        i1 = ir.Operation.create("arith.mulf", operands=[c2.result, x3.result], infer_type=True)
        i2 = ir.Operation.create("arith.addf", operands=[x, i1.result], infer_type=True)
        sc = ir.Operation.create("arith.mulf", operands=[c1.result, i2.result], infer_type=True)
        th = ir.Operation.create("math.tanh", operands=[sc.result], infer_type=True)
        p1 = ir.Operation.create("arith.addf", operands=[one.result, th.result], infer_type=True)
        hx = ir.Operation.create("arith.mulf", operands=[half.result, x], infer_type=True)
        return ir.Operation.create("arith.mulf", operands=[hx.result, p1.result], infer_type=True).result

    return _make_generic([a], empty.result, rt, 1, ip, body)


# ── Matmul / Linear ───────────────────────────────────────────


def _lower_matmul(op: Any) -> Any | None:
    if len(op.operation.operands) < 2:
        return None
    a, b = op.operation.operands[0], op.operation.operands[1]
    rt = op.result.type
    ip = _ip(op)
    op_name = "linalg.batch_matmul" if len(rt.shape) > 2 else "linalg.matmul"
    empty = ir.Operation.create("tensor.empty", results=[rt], ip=ip)
    return _make_matmul([a, b], empty.result, rt, ip, op_name)


def _lower_linear(op: Any) -> Any | None:
    if len(op.operation.operands) < 2:
        return None
    x_val, w_val = op.operation.operands[0], op.operation.operands[1]
    rt = op.result.type
    ip = _ip(op)
    w_type = w_val.type

    if isinstance(w_type, ir.RankedTensorType) and len(w_type.shape) == 2:
        import mlir.dialects.linalg as _lin
        w_shape = list(w_type.shape)
        t_type = ir.RankedTensorType.get([w_shape[1], w_shape[0]], w_type.element_type)
        ident = ir.AffineMap.get_identity(2)
        maps_t = ir.ArrayAttr.get([
            ir.AffineMapAttr.get(ir.AffineMap.get_permutation([1, 0])),
            ir.AffineMapAttr.get(ident),
        ])
        iters_t = _lin._IteratorTypeArrayAttr([_lin.IteratorType.parallel] * 2, context=_CTX)
        t_op = ir.Operation.create(
            "linalg.generic",
            operands=[w_val, ir.Operation.create("tensor.empty", results=[t_type], ip=ip).result],
            results=[t_type],
            attributes={
                "indexing_maps": maps_t,
                "iterator_types": iters_t,
                "operandSegmentSizes": ir.DenseI32ArrayAttr.get([1, 1]),
            },
            regions=1, ip=ip,
        )
        reg = t_op.operation.regions[0]
        blk = ir.Block.create_at_start(reg, [w_type.element_type, w_type.element_type])
        args = list(blk.arguments)
        with ir.InsertionPoint(blk):
            ir.Operation.create("linalg.yield", operands=[args[0]])
        w_transposed = t_op.result
    else:
        w_transposed = w_val

    op_name = "linalg.batch_matmul" if len(rt.shape) > 2 else "linalg.matmul"
    empty = ir.Operation.create("tensor.empty", results=[rt], ip=ip)
    return _make_matmul([x_val, w_transposed], empty.result, rt, ip, op_name)


# ── Transpose ─────────────────────────────────────────────────


def _lower_transpose(op: Any) -> Any | None:
    a = op.operation.operands[0]
    rt = op.result.type
    rank = len(rt.shape)
    attrs = op.operation.attributes
    d0 = _get_int(attrs.get("dim0")) or 0
    d1 = _get_int(attrs.get("dim1")) or 1
    perm = list(range(rank))
    perm[d0], perm[d1] = perm[d1], perm[d0]
    import mlir.dialects.linalg as _lin
    ip = _ip(op)
    output = ir.Operation.create("tensor.empty", results=[rt], ip=ip)
    perm_map = ir.AffineMap.get_permutation(perm)
    ident_map = ir.AffineMap.get_identity(rank)
    maps = ir.ArrayAttr.get([ir.AffineMapAttr.get(perm_map), ir.AffineMapAttr.get(ident_map)])
    iters = _lin._IteratorTypeArrayAttr([_lin.IteratorType.parallel] * rank, context=_CTX)
    t_op = ir.Operation.create(
        "linalg.generic", operands=[a, output.result], results=[rt],
        attributes={
            "indexing_maps": maps,
            "iterator_types": iters,
            "operandSegmentSizes": ir.DenseI32ArrayAttr.get([1, 1]),
        },
        regions=1, ip=ip,
    )
    reg = t_op.operation.regions[0]
    elt = rt.element_type
    blk = ir.Block.create_at_start(reg, [elt, elt])
    with ir.InsertionPoint(blk):
        ir.Operation.create("linalg.yield", operands=[list(blk.arguments)[0]])
    return t_op


# ── Reductions (mean, sum) ────────────────────────────────────


def _lower_reduction(reduce_op: str) -> Callable[[Any], Any | None]:
    import mlir.dialects.linalg as _lin

    def lower(op: Any) -> Any | None:
        if len(op.operation.operands) < 1:
            return None
        a = op.operation.operands[0]
        rt = op.result.type
        in_type = a.type
        if not isinstance(in_type, ir.RankedTensorType):
            return None

        in_rank = len(in_type.shape)
        dim_v = _get_int(op.operation.attributes.get("dim"))
        if dim_v < 0:
            dim_v = in_rank + dim_v

        iter_types: list[Any] = [_lin.IteratorType.parallel] * in_rank
        if 0 <= dim_v < in_rank:
            iter_types[dim_v] = _lin.IteratorType.reduction

        ip = _ip(op)
        empty = ir.Operation.create("tensor.empty", results=[rt], ip=ip)
        ident_map = ir.AffineMap.get_identity(in_rank)
        maps = ir.ArrayAttr.get([ir.AffineMapAttr.get(ident_map)] * 2)
        iters = _lin._IteratorTypeArrayAttr(iter_types, context=_CTX)
        elt = rt.element_type
        gop = ir.Operation.create(
            "linalg.generic", operands=[a, empty.result], results=[rt],
            attributes={
            "indexing_maps": maps,
            "iterator_types": iters,
            "operandSegmentSizes": ir.DenseI32ArrayAttr.get([1, 1]),
        },
            regions=1, ip=ip,
        )
        region = gop.operation.regions[0]
        blk = ir.Block.create_at_start(region, [elt, elt])
        args = list(blk.arguments)
        with ir.InsertionPoint(blk):
            r = ir.Operation.create(reduce_op, operands=[args[0], args[1]], infer_type=True)
            ir.Operation.create("linalg.yield", operands=[r.result])
        return gop
    return lower


# ── Shape ops with affine maps ────────────────────────────────


def _lower_copy(op: Any) -> Any | None:
    return _lower_passthrough(op)


def _lower_slice(op: Any) -> Any | None:
    if len(op.operation.operands) < 1:
        return None
    a = op.operation.operands[0]
    rt = op.result.type
    in_type = a.type
    if not isinstance(in_type, ir.RankedTensorType):
        return _lower_passthrough(op)

    attrs = op.operation.attributes
    dim_v = _get_int(attrs.get("dim")) or 0
    start_v = _get_int(attrs.get("start")) or 0
    in_rank = len(in_type.shape)
    out_rank = len(rt.shape)

    from mlir.ir import AffineExpr
    in_exprs: list[AffineExpr] = []
    for i in range(in_rank):
        d = AffineExpr.get_dim(i)
        in_exprs.append(d + AffineExpr.get_constant(start_v) if i == dim_v else d)
    in_map = ir.AffineMap.get(out_rank, 0, in_exprs)
    out_map = ir.AffineMap.get_identity(out_rank)

    import mlir.dialects.linalg as _lin
    ip = _ip(op)
    empty = ir.Operation.create("tensor.empty", results=[rt], ip=ip)
    return _make_generic(
        [a], empty.result, rt, 1, ip, lambda ia: ia[0],
        iterator_types=[_lin.IteratorType.parallel] * out_rank, indexing_maps=[in_map, out_map],
    )


def _lower_select(op: Any) -> Any | None:
    if len(op.operation.operands) < 1:
        return None
    a = op.operation.operands[0]
    rt = op.result.type
    in_type = a.type
    if not isinstance(in_type, ir.RankedTensorType):
        return _lower_passthrough(op)

    attrs = op.operation.attributes
    dim_v = _get_int(attrs.get("dim")) or 0
    index_v = _get_int(attrs.get("index")) or 0
    in_rank = len(in_type.shape)
    out_rank = len(rt.shape)

    from mlir.ir import AffineExpr
    in_exprs: list[AffineExpr] = []
    out_dim = 0
    for i in range(in_rank):
        if i == dim_v:
            in_exprs.append(AffineExpr.get_constant(index_v))
        else:
            in_exprs.append(AffineExpr.get_dim(out_dim))
            out_dim += 1
    in_map = ir.AffineMap.get(out_rank, 0, in_exprs)
    out_map = ir.AffineMap.get_identity(out_rank)

    import mlir.dialects.linalg as _lin
    ip = _ip(op)
    empty = ir.Operation.create("tensor.empty", results=[rt], ip=ip)
    return _make_generic(
        [a], empty.result, rt, 1, ip, lambda ia: ia[0],
        iterator_types=[_lin.IteratorType.parallel] * out_rank, indexing_maps=[in_map, out_map],
    )


def _lower_unsqueeze(op: Any) -> Any | None:
    if len(op.operation.operands) < 1:
        return None
    a = op.operation.operands[0]
    rt = op.result.type
    in_type = a.type
    if not isinstance(in_type, ir.RankedTensorType):
        return _lower_passthrough(op)

    attrs = op.operation.attributes
    dim_v = _get_int(attrs.get("dim")) or 0
    out_rank = len(rt.shape)

    from mlir.ir import AffineExpr
    in_exprs: list[AffineExpr] = []
    for i in range(out_rank):
        if i != dim_v:
            in_exprs.append(AffineExpr.get_dim(i))
    in_map = ir.AffineMap.get(out_rank, 0, in_exprs)
    out_map = ir.AffineMap.get_identity(out_rank)

    import mlir.dialects.linalg as _lin
    ip = _ip(op)
    empty = ir.Operation.create("tensor.empty", results=[rt], ip=ip)
    return _make_generic(
        [a], empty.result, rt, 1, ip, lambda ia: ia[0],
        iterator_types=[_lin.IteratorType.parallel] * out_rank, indexing_maps=[in_map, out_map],
    )


# ── Utility ───────────────────────────────────────────────────


def _get_int(attr: Any) -> int:
    if attr is None:
        return 0
    if isinstance(attr, ir.IntegerAttr):
        return int(ir.IntegerAttr(attr).value)
    try:
        return int(str(attr))
    except (ValueError, TypeError):
        return 0


# ── Dispatch table ────────────────────────────────────────────

_LOWER_TABLE: dict[str, Callable[[Any], Any | None]] = {
    "sf.add": _lower_binary("arith.addf"),
    "sf.mul": _lower_binary("arith.mulf"),
    "sf.sub": _lower_binary("arith.subf"),
    "sf.div": _lower_binary("arith.divf"),
    "sf.max": _lower_binary("arith.maxnumf"),
    "sf.pow": _lower_binary("math.powf"),
    "sf.relu": _lower_relu,
    "sf.silu": _lower_silu,
    "sf.gelu": _lower_gelu,
    "sf.sigmoid": _lower_unary("math.sigmoid"),
    "sf.exp": _lower_unary("math.exp"),
    "sf.neg": _lower_unary("arith.negf"),
    "sf.rsqrt": _lower_unary("math.rsqrt"),
    "sf.cos": _lower_unary("math.cos"),
    "sf.sin": _lower_unary("math.sin"),
    "sf.tanh": _lower_unary("math.tanh"),
    "sf.sqrt": _lower_unary("math.sqrt"),
    "sf.matmul": _lower_matmul,
    "sf.linear": _lower_linear,
    "sf.transpose": _lower_transpose,
    "sf.permute": _lower_passthrough,
    "sf.identity": _lower_passthrough,
    "sf.layer_norm": _lower_passthrough,
    "sf.rms_norm": _lower_passthrough,
    "sf.scaled_dot_product_attention": _lower_passthrough,
    "sf.softmax": _lower_passthrough,
    "sf.mean": lambda _o: None,
    "sf.sum": lambda _o: None,
    "sf.view": lambda _o: None,
    "sf.unsqueeze": lambda _o: None,
    "sf.cat": lambda _o: None,
    "sf.slice": lambda _o: None,
    "sf.select": lambda _o: None,
    "sf.copy_": _lower_copy,
    "sf.expand": lambda _o: None,
    "sf.fused_silu_mul": _lower_passthrough,
    "sf.fused_rms_norm_matmul": _lower_passthrough,
    "sf.fused_qkv": _lower_passthrough,
    "sf.fused_attention_output": _lower_passthrough,
    "sf.fused_attention_block": _lower_passthrough,
    "sf.pad": _lower_passthrough,
    "sf.cumsum": _lower_passthrough,
    "sf.masked_fill": _lower_passthrough,
    "sf.type_as": _lower_passthrough,
    "sf.arange": _lower_passthrough,
    "sf.zeros": _lower_passthrough,
    "sf.zeros_like": _lower_passthrough,
    "sf.ones_like": _lower_passthrough,
    "sf.full_like": _lower_passthrough,
    "sf.new_ones": _lower_passthrough,
    "sf.eye": _lower_passthrough,
    "sf.diff": _lower_passthrough,
    "sf.sym_size": _lower_passthrough,
    "sf.index": _lower_passthrough,
    "sf.einsum": _lower_passthrough,
    "sf.stack": _lower_passthrough,
    "sf.clamp_min": _lower_passthrough,
    "sf.softplus": _lower_passthrough,
    "sf.conv1d": _lower_passthrough,
    "sf.linalg_norm": _lower_passthrough,
    "sf.var": _lower_passthrough,
    "sf.view_as": _lower_passthrough,
    "sf.expand_as": _lower_passthrough,
    "sf.split": _lower_passthrough,
    "sf.chunk": _lower_passthrough,
    "sf.eq": _lower_passthrough,
    "sf.ne": _lower_passthrough,
    "sf.le": _lower_passthrough,
    "sf.lt": _lower_passthrough,
    "sf.gt": _lower_passthrough,
    "sf.logical_and": _lower_passthrough,
    "sf.triu": _lower_passthrough,
    "sf.tril": _lower_passthrough,
    "sf.embedding": _lower_passthrough,
}
