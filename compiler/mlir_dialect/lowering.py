"""sf → standard dialect lowering using PassManager Python passes."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Ensure standard dialects are loaded for infer_type=True
import mlir.dialects.arith  # noqa: F401
import mlir.dialects.linalg  # noqa: F401
import mlir.dialects.math  # noqa: F401
import mlir.dialects.tensor  # noqa: F401
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


def sf_to_linalg_pass_on_module(ir_module: Any) -> str:
    """Run sf→linalg lowering directly on an ir.Module object.

    Runs _lower_sf_ops directly on the module operation, without
    using PassManager (which can fail on partial lowering results).
    """
    global _CTX
    ctx = ir_module.operation.context
    _CTX = ctx
    ctx.allow_unregistered_dialects = True
    with ctx, ir.Location.unknown(ctx):
        _lower_sf_ops(ir_module.operation, None)
        return str(ir_module)


def sf_to_linalg_pass(mlir_text: str) -> str:
    """Run sf→linalg lowering by parsing MLIR text.

    Prefer sf_to_linalg_pass_on_module() for new code to avoid
    text round-trip parsing issues.
    """
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
    errors_by_op: dict[str, int] = {}
    first_error_msg: dict[str, str] = {}
    lowered = 0
    finalized = 0
    for sf_op in sf_ops:
        op_name: str = sf_op.name  # cache: op may be invalidated after erase
        try:
            new_op = _LOWER_TABLE.get(op_name, lambda _o: None)(sf_op)
        except Exception as e:
            errors_by_op[op_name] = errors_by_op.get(op_name, 0) + 1
            if op_name not in first_error_msg:
                first_error_msg[op_name] = str(e)
            continue
        if new_op is None and sf_op.operation.results:
            try:
                if sf_op.operation.operands:
                    new_op = _lower_passthrough(sf_op)
                    if new_op is not None:
                        finalized += 1
            except Exception:
                pass
        if new_op is not None and sf_op.operation.results:
            try:
                sf_op.operation.results[0].replace_all_uses_with(new_op.operation.results[0])
                sf_op.operation.erase()
                if op_name not in ("sf.weight", "sf.constant"):
                    lowered += 1
            except Exception:
                errors_by_op[op_name + " (replace)"] = errors_by_op.get(op_name + " (replace)", 0) + 1
    total_errors = sum(errors_by_op.values())
    total_collected = len(sf_ops)
    _log = logging.getLogger("compiler.lowering")
    _log.info("sf_to_linalg: %d lowered, %d finalized (passthrough), %d errors across %d ops (total: %d)",
              lowered, finalized, len(errors_by_op), total_errors, total_collected)
    if errors_by_op:
        for op_name, cnt in sorted(errors_by_op.items(), key=lambda x: -x[1]):
            msg = first_error_msg.get(op_name, "unknown")
            _log.warning("  %s: %d failures — %s", op_name, cnt, msg[:120])


def _ip(op: Any) -> ir.InsertionPoint:
    return ir.InsertionPoint(op.operation)


def _el(rt: ir.Type) -> ir.Type:
    return rt.element_type


# ── linalg.generic builders ───────────────────────────────────


def _make_empty(rt: ir.Type, inputs: list[ir.Value], ip: ir.InsertionPoint) -> ir.Value:
    """Create tensor.empty with dynamic size operands.

    For dynamic output dimensions (<0), uses tensor.dim on the first input
    to obtain runtime sizes.  This is required by one-shot-bufferize.
    """
    dynamic_operands: list[ir.Value] = []
    if isinstance(rt, ir.RankedTensorType) and inputs:
        in_val = inputs[0]
        in_type = in_val.type
        if isinstance(in_type, ir.RankedTensorType):
            for i, s in enumerate(rt.shape):
                if s < 0:
                    idx_val = ir.Operation.create(
                        "arith.constant", results=[ir.IndexType.get()],
                        attributes={"value": ir.IntegerAttr.get(ir.IndexType.get(), i)},
                        ip=ip,
                    )
                    dim_val = ir.Operation.create(
                        "tensor.dim", operands=[in_val, idx_val.result],
                        results=[ir.IndexType.get()], ip=ip,
                    )
                    dynamic_operands.append(dim_val.result)
    return ir.Operation.create(
        "tensor.empty", operands=dynamic_operands, results=[rt], ip=ip,
    ).result


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

    rank = len(rt.shape) if isinstance(rt, ir.RankedTensorType) else 1
    ident = ir.AffineMap.get_identity(rank) if isinstance(rt, ir.RankedTensorType) else ir.AffineMap.get(rank, 0, [])
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
    elt = rt.element_type if isinstance(rt, ir.ShapedType) else rt
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
        # Skip broadcast ops: output rank must match all input ranks
        a_type = a.type
        b_type = b.type
        if (isinstance(rt, ir.RankedTensorType)
                and isinstance(a_type, ir.RankedTensorType)
                and isinstance(b_type, ir.RankedTensorType)):
            if len(rt.shape) != len(a_type.shape) or len(rt.shape) != len(b_type.shape):
                return None
        ip = _ip(op)
        empty_val = _make_empty(rt, [a], ip)
        return _make_generic(
            [a, b], empty_val, rt, 2, ip,
            lambda ia: ir.Operation.create(arith_op, operands=[ia[0], ia[1]], infer_type=True).result,
        )
    return lower


def _lower_unary(arith_op: str) -> Callable[[Any], Any | None]:
    def lower(op: Any) -> Any | None:
        if len(op.operation.operands) < 1:
            return None
        a = op.operation.operands[0]
        rt = op.result.type
        # Skip if output rank differs from input rank
        a_type = a.type
        if isinstance(rt, ir.RankedTensorType) and isinstance(a_type, ir.RankedTensorType):
            if len(rt.shape) != len(a_type.shape):
                return None
        ip = _ip(op)
        empty_val = _make_empty(rt, [a], ip)
        return _make_generic(
            [a], empty_val, rt, 1, ip,
            lambda ia: ir.Operation.create(arith_op, operands=[ia[0]], infer_type=True).result,
        )
    return lower


def _lower_passthrough(op: Any) -> Any | None:
    """Passthrough → linalg.copy or tensor.reshape (or linalg.generic fallback)."""
    if len(op.operation.operands) < 1:
        return None
    a = op.operation.operands[0]
    rt = op.result.type
    if isinstance(a.type, ir.RankedTensorType) and isinstance(rt, ir.RankedTensorType):
        if len(a.type.shape) != len(rt.shape):
            return _lower_reshape(op)
        if list(a.type.shape) != list(rt.shape):
            return _lower_reshape(op)
    ip = _ip(op)
    empty_val = _make_empty(rt, [a], ip)
    if isinstance(a.type, ir.RankedTensorType) and isinstance(rt, ir.RankedTensorType) and len(rt.shape) > 0:
        import mlir.dialects.linalg as _lin
        copy_op = _lin.CopyOp(result_tensors=[rt], inputs=[a], outputs=[empty_val], ip=ip)
        _lin.fill_builtin_region(copy_op.operation)
        return copy_op
    return _make_generic([a], empty_val, rt, 1, ip, lambda ia: ia[0])


def _lower_reshape(op: Any) -> Any | None:
    """sf.identity/copy_/view with rank change → tensor.reshape (static or dynamic)."""
    if len(op.operation.operands) < 1:
        return None
    a = op.operation.operands[0]
    rt = op.result.type
    in_type = a.type
    if not isinstance(in_type, ir.RankedTensorType) or not isinstance(rt, ir.RankedTensorType):
        return None
    if in_type.element_type != rt.element_type:
        return None
    if list(in_type.shape) == list(rt.shape):
        return None  # same shape, should use linalg.copy

    ip = _ip(op)
    idx_type = ir.IndexType.get()
    has_dynamic = any(s < 0 for s in rt.shape) or any(s < 0 for s in in_type.shape)

    if not has_dynamic:
        # Static path: use DenseIntElementsAttr
        import numpy as np
        shape_dims = [int(s) for s in rt.shape]
        shape_type = ir.RankedTensorType.get([len(shape_dims)], ir.IntegerType.get_signless(64))
        shape_attr = ir.DenseIntElementsAttr.get(
            np.array(shape_dims, dtype=np.int64), type=ir.IntegerType.get_signless(64),
        )
        shape_const = ir.Operation.create(
            "arith.constant", results=[shape_type], attributes={"value": shape_attr}, ip=ip,
        )
        return ir.Operation.create("tensor.reshape", operands=[a, shape_const.result], results=[rt], ip=ip)

    # Dynamic path: build shape tensor from tensor.dim + arith
    # Parse target dims from the shape attribute; -1 means "infer from total elements"
    shape_attr = op.operation.attributes.get("shape")
    if shape_attr is not None and hasattr(shape_attr, "__iter__"):
        target_raw = [int(x) for x in shape_attr]
    else:
        target_raw = [int(s) if s >= 0 else -1 for s in rt.shape]

    in_rank = len(in_type.shape)

    # Compute total source elements
    total_val = _make_index_constant(ip, idx_type, 1).result
    for i in range(in_rank):
        if in_type.shape[i] >= 0:
            dim_val = _make_index_constant(ip, idx_type, in_type.shape[i]).result
        else:
            c = _make_index_constant(ip, idx_type, i)
            dim_val = ir.Operation.create("tensor.dim", operands=[a, c.result],
                                          results=[idx_type], ip=ip).result
        total_val = ir.Operation.create("arith.muli", operands=[total_val, dim_val],
                                         results=[idx_type], ip=ip).result

    # Compute product of known target dims and track -1 positions
    one_val = _make_index_constant(ip, idx_type, 1).result
    known_prod_val = one_val
    neg_one_idx = -1
    shape_vals: list[Any] = []
    for i, s in enumerate(target_raw):
        if s == -1:
            neg_one_idx = i
            shape_vals.append(None)
        else:
            v = _make_index_constant(ip, idx_type, s).result
            shape_vals.append(v)
            known_prod_val = ir.Operation.create("arith.muli", operands=[known_prod_val, v],
                                                  results=[idx_type], ip=ip).result

    if neg_one_idx >= 0:
        shape_vals[neg_one_idx] = ir.Operation.create(
            "arith.divui", operands=[total_val, known_prod_val],
            results=[idx_type], ip=ip,
        ).result
    if any(v is None for v in shape_vals):
        return None

    shape_rt = ir.RankedTensorType.get([len(shape_vals)], idx_type)
    from_elements = ir.Operation.create(
        "tensor.from_elements", operands=shape_vals, results=[shape_rt], ip=ip,
    )
    return ir.Operation.create("tensor.reshape", operands=[a, from_elements.result], results=[rt], ip=ip)


def _make_index_constant(ip: Any, idx_type: ir.Type, value: int) -> Any:
    return ir.Operation.create(
        "arith.constant", results=[idx_type],
        attributes={"value": ir.IntegerAttr.get(idx_type, int(value))},
        ip=ip,
    )


# ── Activations ───────────────────────────────────────────────


def _lower_relu(op: Any) -> Any | None:
    """sf.relu → linalg.generic with arith.maxnumf + 0.0 constant."""
    if len(op.operation.operands) < 1:
        return None
    a = op.operation.operands[0]
    rt = op.result.type
    elt = _el(rt)
    ip = _ip(op)
    empty_val = _make_empty(rt, [a], ip)

    def body(ia: list[ir.Value]) -> ir.Value:
        zero = ir.Operation.create("arith.constant", results=[elt], attributes={"value": ir.FloatAttr.get(elt, 0.0)})
        return ir.Operation.create("arith.maxnumf", operands=[ia[0], zero.result], infer_type=True).result

    return _make_generic([a], empty_val, rt, 1, ip, body)


def _lower_sigmoid(op: Any) -> Any | None:
    """sf.sigmoid → 1/(1+exp(-x)) — math.sigmoid not available in mlir-core 22.1.5."""
    if len(op.operation.operands) < 1:
        return None
    a = op.operation.operands[0]
    rt = op.result.type
    elt = _el(rt)
    ip = _ip(op)
    empty_val = _make_empty(rt, [a], ip)
    import mlir.dialects.linalg as _lin
    out_rank = len(rt.shape) if isinstance(rt, ir.RankedTensorType) else 1

    def body(ia: list[ir.Value]) -> ir.Value:
        neg = ir.Operation.create("arith.negf", operands=[ia[0]], infer_type=True)
        exp = ir.Operation.create("math.exp", operands=[neg.result], infer_type=True)
        one = ir.Operation.create("arith.constant", results=[elt], attributes={"value": ir.FloatAttr.get(elt, 1.0)})
        denom = ir.Operation.create("arith.addf", operands=[one.result, exp.result], infer_type=True)
        return ir.Operation.create("arith.divf", operands=[one.result, denom.result], infer_type=True).result

    return _make_generic(
        [a], empty_val, rt, 1, ip, body,
        iterator_types=[_lin.IteratorType.parallel] * out_rank,
    )


def _lower_silu(op: Any) -> Any | None:
    if len(op.operation.operands) < 1:
        return None
    a = op.operation.operands[0]
    rt = op.result.type
    elt = _el(rt)
    ip = _ip(op)
    empty_val = _make_empty(rt, [a], ip)

    def body(ia: list[ir.Value]) -> ir.Value:
        x = ia[0]
        neg = ir.Operation.create("arith.negf", operands=[x], infer_type=True)
        exp = ir.Operation.create("math.exp", operands=[neg.result], infer_type=True)
        one = ir.Operation.create("arith.constant", results=[elt], attributes={"value": ir.FloatAttr.get(elt, 1.0)})
        denom = ir.Operation.create("arith.addf", operands=[one.result, exp.result], infer_type=True)
        sig = ir.Operation.create("arith.divf", operands=[one.result, denom.result], infer_type=True)
        return ir.Operation.create("arith.mulf", operands=[x, sig.result], infer_type=True).result

    return _make_generic([a], empty_val, rt, 1, ip, body)


def _lower_gelu(op: Any) -> Any | None:
    if len(op.operation.operands) < 1:
        return None
    a = op.operation.operands[0]
    rt = op.result.type
    elt = _el(rt)
    ip = _ip(op)
    empty_val = _make_empty(rt, [a], ip)

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

    return _make_generic([a], empty_val, rt, 1, ip, body)


# ── Matmul / Linear ───────────────────────────────────────────


def _lower_matmul(op: Any) -> Any | None:
    if len(op.operation.operands) < 2:
        return None
    a, b = op.operation.operands[0], op.operation.operands[1]
    rt = op.result.type
    ip = _ip(op)
    op_name = "linalg.batch_matmul" if len(rt.shape) > 2 else "linalg.matmul"
    empty_val = _make_empty(rt, [a], ip)
    return _make_matmul([a, b], empty_val, rt, ip, op_name)


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
    empty_val = _make_empty(rt, [x_val], ip)
    return _make_matmul([x_val, w_transposed], empty_val, rt, ip, op_name)


# ── Transpose ─────────────────────────────────────────────────


def _lower_transpose(op: Any) -> Any | None:
    a = op.operation.operands[0]
    rt = op.result.type
    if not isinstance(rt, ir.RankedTensorType):
        return _lower_passthrough(op)
    rank = len(rt.shape)
    if rank < 2:
        return _lower_passthrough(op)
    attrs = op.operation.attributes
    d0 = _get_int(attrs.get("dim0")) or 0
    d1 = _get_int(attrs.get("dim1")) or 1
    perm = list(range(rank))
    if d0 >= rank or d1 >= rank:
        return _lower_passthrough(op)
    perm[d0], perm[d1] = perm[d1], perm[d0]
    import mlir.dialects.linalg as _lin
    ip = _ip(op)
    output_val = _make_empty(rt, [a], ip)
    perm_map = ir.AffineMap.get_permutation(perm)
    ident_map = ir.AffineMap.get_identity(rank)
    maps = ir.ArrayAttr.get([ir.AffineMapAttr.get(perm_map), ir.AffineMapAttr.get(ident_map)])
    iters = _lin._IteratorTypeArrayAttr([_lin.IteratorType.parallel] * rank, context=_CTX)
    t_op = ir.Operation.create(
        "linalg.generic", operands=[a, output_val], results=[rt],
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


def _lower_sum(op: Any) -> Any | None:
    """sf.sum → linalg.generic with arith.addf reduction + squeeze."""
    return _lower_reduction("arith.addf")(op)


def _lower_mean(op: Any) -> Any | None:
    """sf.mean → sum reduction → divide by dim_size."""
    sum_result = _lower_reduction("arith.addf")(op)
    if sum_result is None:
        return None

    a = op.operation.operands[0]
    rt = op.result.type
    in_type = a.type
    if not isinstance(in_type, ir.RankedTensorType) or not isinstance(rt, ir.RankedTensorType):
        return None
    dim_v = _get_int(op.operation.attributes.get("dim"))
    in_rank = len(in_type.shape)
    if dim_v < 0:
        dim_v = in_rank + dim_v
    if not (0 <= dim_v < in_rank):
        return None
    dim_size = in_type.shape[dim_v]
    if dim_size <= 0:
        return None

    elt = rt.element_type
    div_val = ir.FloatAttr.get(elt, float(dim_size))
    ip = _ip(op)
    import mlir.dialects.linalg as _lin

    div_rt = sum_result.operation.results[0].type
    empty_val = _make_empty(div_rt, [a], ip)
    rank = len(div_rt.shape) if isinstance(div_rt, ir.RankedTensorType) else 1
    return _make_generic(
        [sum_result.result], empty_val, div_rt, 1, ip,
        lambda ia: ir.Operation.create(
            "arith.divf",
            operands=[
                ia[0],
                ir.Operation.create("arith.constant", results=[elt], attributes={"value": div_val}).result,
            ],
            infer_type=True,
        ).result,
        iterator_types=[_lin.IteratorType.parallel] * rank,
    )


def _lower_reduction(reduce_op: str) -> Callable[[Any], Any | None]:
    import mlir.dialects.linalg as _lin

    def lower(op: Any) -> Any | None:
        if len(op.operation.operands) < 1:
            return None
        a = op.operation.operands[0]
        rt = op.result.type
        in_type = a.type
        if not isinstance(in_type, ir.RankedTensorType) or not isinstance(rt, ir.RankedTensorType):
            return None

        in_rank = len(in_type.shape)
        out_rank = len(rt.shape)
        dim_v = _get_int(op.operation.attributes.get("dim"))
        if dim_v < 0:
            dim_v = in_rank + dim_v
        if not (0 <= dim_v < in_rank):
            return None

        iter_types: list[Any] = [_lin.IteratorType.parallel] * in_rank
        iter_types[dim_v] = _lin.IteratorType.reduction

        ip = _ip(op)

        # Input map: identity on in_rank dimensions
        in_map = ir.AffineMap.get_identity(in_rank)

        # Output map: project in_rank iteration space to out_rank results
        from mlir.ir import AffineExpr
        out_exprs: list[AffineExpr] = []
        in_dim = 0
        for _ in range(out_rank):
            if in_dim == dim_v:
                in_dim += 1
            if in_dim < in_rank:
                out_exprs.append(AffineExpr.get_dim(in_dim))
            else:
                out_exprs.append(AffineExpr.get_constant(0))
            in_dim += 1
        out_map = ir.AffineMap.get(in_rank, 0, out_exprs)

        maps = ir.ArrayAttr.get([ir.AffineMapAttr.get(in_map), ir.AffineMapAttr.get(out_map)])
        iters = _lin._IteratorTypeArrayAttr(iter_types, context=_CTX)
        elt = rt.element_type
        empty_val = _make_empty(rt, [a], ip)
        gop = ir.Operation.create(
            "linalg.generic", operands=[a, empty_val], results=[rt],
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
    """sf.slice → linalg.copy (same shape) or tensor.extract_slice (different shape).

    Uses linalg.copy for the common case of same-shape slices with dynamic dims,
    avoiding tensor.extract_slice bufferization issues in mlir-core 22.1.5.
    """
    if len(op.operation.operands) < 1:
        return None
    a = op.operation.operands[0]
    rt = op.result.type
    in_type = a.type
    if not isinstance(in_type, ir.RankedTensorType) or not isinstance(rt, ir.RankedTensorType):
        return _lower_passthrough(op)

    # Same shape → use linalg.copy (fast, no bufferize issues)
    if list(in_type.shape) == list(rt.shape):
        return _lower_passthrough(op)

    attrs = op.operation.attributes
    dim_v = _get_int(attrs.get("dim")) or 0
    start_v = _get_int(attrs.get("start")) or 0
    end_v = _get_int(attrs.get("end")) or 0
    in_rank = len(in_type.shape)

    if dim_v < 0:
        dim_v = in_rank + dim_v

    # If the slice has dynamic size, fall back to passthrough
    size = end_v - start_v
    if size < 0:
        return _lower_passthrough(op)
    for i, s in enumerate(in_type.shape):
        if i != dim_v and s < 0:
            return _lower_passthrough(op)

    # Static slice: tensor.extract_slice is safe
    static_offsets: list[int] = [0] * in_rank
    static_sizes: list[int] = []
    static_strides: list[int] = [1] * in_rank

    for i in range(in_rank):
        if i == dim_v:
            static_offsets[i] = start_v
            static_sizes.append(size)
        else:
            static_sizes.append(in_type.shape[i])

    ip = _ip(op)
    result_op = ir.Operation.create(
        "tensor.extract_slice",
        operands=[a],
        results=[rt],
        attributes={
            "static_offsets": ir.DenseI64ArrayAttr.get(static_offsets),
            "static_sizes": ir.DenseI64ArrayAttr.get(static_sizes),
            "static_strides": ir.DenseI64ArrayAttr.get(static_strides),
            "operandSegmentSizes": ir.DenseI32ArrayAttr.get([1, 0, 0, 0]),
        },
        ip=ip,
    )
    return result_op


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
    empty_val = _make_empty(rt, [a], ip)
    return _make_generic(
        [a], empty_val, rt, 1, ip, lambda ia: ia[0],
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
    if dim_v < 0:
        dim_v = out_rank + dim_v

    from mlir.ir import AffineExpr
    in_exprs: list[AffineExpr] = []
    for i in range(out_rank):
        if i != dim_v:
            in_exprs.append(AffineExpr.get_dim(i))
    in_map = ir.AffineMap.get(out_rank, 0, in_exprs)
    out_map = ir.AffineMap.get_identity(out_rank)

    import mlir.dialects.linalg as _lin
    ip = _ip(op)
    empty_val = _make_empty(rt, [a], ip)
    return _make_generic(
        [a], empty_val, rt, 1, ip, lambda ia: ia[0],
        iterator_types=[_lin.IteratorType.parallel] * out_rank, indexing_maps=[in_map, out_map],
    )


def _lower_view(op: Any) -> Any | None:
    """sf.view → tensor.reshape (rank/shape changing)."""
    return _lower_reshape(op)


def _lower_expand(op: Any) -> Any | None:
    """sf.expand → broadcast via linalg.generic with broadcast affine map."""
    if len(op.operation.operands) < 1:
        return None
    a = op.operation.operands[0]
    rt = op.result.type
    in_type = a.type
    if not isinstance(in_type, ir.RankedTensorType) or not isinstance(rt, ir.RankedTensorType):
        return _lower_passthrough(op)

    in_shapes = [s for s in in_type.shape if s > 0]
    out_rank = len(rt.shape)
    if not in_shapes:
        return _lower_passthrough(op)

    from mlir.ir import AffineExpr
    in_idx = 0
    in_exprs: list[AffineExpr] = []
    for i in range(out_rank):
        if in_idx < len(in_shapes) and rt.shape[i] == in_shapes[in_idx]:
            in_exprs.append(AffineExpr.get_dim(i))
            in_idx += 1
        else:
            in_exprs.append(AffineExpr.get_constant(0))
    in_map = ir.AffineMap.get(out_rank, 0, in_exprs)
    out_map = ir.AffineMap.get_identity(out_rank)

    import mlir.dialects.linalg as _lin
    ip = _ip(op)
    empty_val = _make_empty(rt, [a], ip)
    return _make_generic(
        [a], empty_val, rt, 1, ip, lambda ia: ia[0],
        iterator_types=[_lin.IteratorType.parallel] * out_rank, indexing_maps=[in_map, out_map],
    )


def _lower_cat(op: Any) -> Any | None:
    """sf.cat → concatenate via linalg.generic with offset affine maps."""
    operands = op.operation.operands
    if len(operands) < 2:
        return _lower_passthrough(op) if operands else None
    rt = op.result.type
    if not isinstance(rt, ir.RankedTensorType):
        return _lower_passthrough(op)

    dim_v = _get_int(op.operation.attributes.get("dim")) or 0
    out_rank = len(rt.shape)
    if dim_v < 0:
        dim_v = out_rank + dim_v

    import mlir.dialects.linalg as _lin
    ip = _ip(op)

    # Compute offsets
    offsets: list[int] = []
    offset = 0
    for o_val in operands:
        if isinstance(o_val.type, ir.RankedTensorType):
            offsets.append(offset)
            offset += o_val.type.shape[dim_v] if 0 <= dim_v < len(o_val.type.shape) else 1

    from mlir.ir import AffineExpr
    empty_val = _make_empty(rt, [operands[0]], ip)
    generic_operands = list(operands) + [empty_val]
    generic_result_types = [rt]

    maps_list: list[ir.AffineMap] = []
    for idx, o_val in enumerate(operands):
        if isinstance(o_val.type, ir.RankedTensorType):
            in_exprs: list[AffineExpr] = []
            for i in range(out_rank):
                d = AffineExpr.get_dim(i)
                in_exprs.append(d + AffineExpr.get_constant(offsets[idx]) if i == dim_v else d)
            maps_list.append(ir.AffineMap.get(out_rank, 0, in_exprs))
        else:
            maps_list.append(ir.AffineMap.get(out_rank, 0, [AffineExpr.get_dim(i) for i in range(out_rank)]))
    maps_list.append(ir.AffineMap.get_identity(out_rank))

    n_inputs = len(operands)
    iters = _lin._IteratorTypeArrayAttr([_lin.IteratorType.parallel] * out_rank, context=_CTX)
    elt = rt.element_type
    cat_op = ir.Operation.create(
        "linalg.generic",
        operands=generic_operands,
        results=generic_result_types,
        attributes={
            "indexing_maps": ir.ArrayAttr.get([ir.AffineMapAttr.get(m) for m in maps_list]),
            "iterator_types": iters,
            "operandSegmentSizes": ir.DenseI32ArrayAttr.get([n_inputs, 1]),
        },
        regions=1, ip=ip,
    )
    region = cat_op.operation.regions[0]
    blk = ir.Block.create_at_start(region, [elt] * n_inputs + [elt])
    args = list(blk.arguments)
    with ir.InsertionPoint(blk):
        ir.Operation.create("linalg.yield", operands=[args[n_inputs - 1]])
    return cat_op


def _lower_softmax(op: Any) -> Any | None:
    """sf.softmax → exp / (reduce_sum exp) via linalg generic ops."""
    if len(op.operation.operands) < 1:
        return None
    a = op.operation.operands[0]
    rt = op.result.type
    in_type = a.type
    if not isinstance(in_type, ir.RankedTensorType) or not isinstance(rt, ir.RankedTensorType):
        return _lower_passthrough(op)

    dim_v = _get_int(op.operation.attributes.get("dim")) or -1
    in_rank = len(in_type.shape)
    if dim_v < 0:
        dim_v = in_rank + dim_v

    import mlir.dialects.linalg as _lin
    ip = _ip(op)

    # Step 1: max reduction (for numerical stability)
    iter_types: list[Any] = [_lin.IteratorType.parallel] * in_rank
    if 0 <= dim_v < in_rank:
        iter_types[dim_v] = _lin.IteratorType.reduction

    empty_max_val = _make_empty(rt, [a], ip)
    ident_map = ir.AffineMap.get_identity(in_rank)
    maps = ir.ArrayAttr.get([ir.AffineMapAttr.get(ident_map)] * 2)
    iters = _lin._IteratorTypeArrayAttr(iter_types, context=_CTX)
    elt = rt.element_type
    max_op = ir.Operation.create(
        "linalg.generic", operands=[a, empty_max_val], results=[rt],
        attributes={
            "indexing_maps": maps,
            "iterator_types": iters,
            "operandSegmentSizes": ir.DenseI32ArrayAttr.get([1, 1]),
        },
        regions=1, ip=ip,
    )
    region = max_op.operation.regions[0]
    blk = ir.Block.create_at_start(region, [elt, elt])
    args = list(blk.arguments)
    with ir.InsertionPoint(blk):
        r = ir.Operation.create("arith.maxnumf", operands=[args[0], args[1]], infer_type=True)
        ir.Operation.create("linalg.yield", operands=[r.result])

    # Step 2: subtract max (stable softmax)
    empty_sub_val = _make_empty(in_type, [a], ip)
    sub_op = _make_generic(
        [a, max_op.result], empty_sub_val, in_type, 2, ip,
        lambda ia: ir.Operation.create("arith.subf", operands=[ia[0], ia[1]], infer_type=True).result,
        iterator_types=[_lin.IteratorType.parallel] * in_rank,
    )

    # Step 3: exp
    empty_exp_val = _make_empty(in_type, [sub_op.result], ip)
    exp_op = _make_generic(
        [sub_op.result], empty_exp_val, in_type, 1, ip,
        lambda ia: ir.Operation.create("math.exp", operands=[ia[0]], infer_type=True).result,
        iterator_types=[_lin.IteratorType.parallel] * in_rank,
    )

    # Step 4: sum reduction on exp
    empty_sum_val = _make_empty(rt, [a], ip)
    sum_op = ir.Operation.create(
        "linalg.generic", operands=[exp_op.result, empty_sum_val], results=[rt],
        attributes={
            "indexing_maps": maps,
            "iterator_types": iters,
            "operandSegmentSizes": ir.DenseI32ArrayAttr.get([1, 1]),
        },
        regions=1, ip=ip,
    )
    region2 = sum_op.operation.regions[0]
    blk2 = ir.Block.create_at_start(region2, [elt, elt])
    a2 = list(blk2.arguments)
    with ir.InsertionPoint(blk2):
        r2 = ir.Operation.create("arith.addf", operands=[a2[0], a2[1]], infer_type=True)
        ir.Operation.create("linalg.yield", operands=[r2.result])

    # Step 5: divide exp by sum
    empty_div_val = _make_empty(in_type, [exp_op.result], ip)
    return _make_generic(
        [exp_op.result, sum_op.result], empty_div_val, in_type, 2, ip,
        lambda ia: ir.Operation.create("arith.divf", operands=[ia[0], ia[1]], infer_type=True).result,
        iterator_types=[_lin.IteratorType.parallel] * in_rank,
    )


def _lower_comparison(cmp_pred: int) -> Callable[[Any], Any | None]:
    def lower(op: Any) -> Any | None:
        if len(op.operation.operands) < 2:
            return None
        a, b = op.operation.operands[0], op.operation.operands[1]
        rt = op.result.type
        ip = _ip(op)
        empty_val = _make_empty(rt, [a], ip)
        import mlir.dialects.linalg as _lin
        out_rank = len(rt.shape) if isinstance(rt, ir.RankedTensorType) else 1
        ident = ir.AffineMap.get_identity(out_rank)
        maps = ir.ArrayAttr.get([ir.AffineMapAttr.get(ident)] * 3)
        iters = _lin._IteratorTypeArrayAttr([_lin.IteratorType.parallel] * out_rank, context=_CTX)
        # Block args: inputs are from operands (f32/bf16), output is i1
        in_elt_a = a.type.element_type if isinstance(a.type, ir.ShapedType) else _el(a.type)
        in_elt_b = b.type.element_type if isinstance(b.type, ir.ShapedType) else _el(b.type)
        out_elt = rt.element_type if isinstance(rt, ir.ShapedType) else rt
        gop = ir.Operation.create(
            "linalg.generic", operands=[a, b, empty_val], results=[rt],
            attributes={
                "indexing_maps": maps,
                "iterator_types": iters,
                "operandSegmentSizes": ir.DenseI32ArrayAttr.get([2, 1]),
            },
            regions=1, ip=ip,
        )
        region = gop.operation.regions[0]
        block = ir.Block.create_at_start(region, [in_elt_a, in_elt_b, out_elt])
        args = list(block.arguments)
        with ir.InsertionPoint(block):
            r = ir.Operation.create(
                "arith.cmpf",
                results=[out_elt],
                operands=[args[0], args[1]],
                attributes={"predicate": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), cmp_pred)},
                infer_type=False,
            )
            ir.Operation.create("linalg.yield", operands=[r.result])
        return gop
    return lower


def _lower_logical_and(op: Any) -> Any | None:
    """sf.logical_and → arith.andi (both inputs are i1)."""
    return _lower_binary("arith.andi")(op)


def _lower_zeros_op(op: Any) -> Any | None:
    """sf.zeros → tensor.empty + linalg.fill(0.0)."""
    rt = op.result.type
    if not isinstance(rt, ir.RankedTensorType):
        return _lower_passthrough(op)
    elt = rt.element_type
    ip = _ip(op)
    empty = ir.Operation.create("tensor.empty", results=[rt], ip=ip)
    zero_attr = ir.FloatAttr.get(elt, 0.0) if "f" in str(elt).lower() else ir.IntegerAttr.get(elt, 0)
    const = ir.Operation.create(
        "arith.constant", results=[elt], attributes={"value": zero_attr}, ip=ip,
    )
    import mlir.dialects.linalg as _lin
    fill_op = _lin.FillOp(result_tensors=[rt], inputs=[const.result], outputs=[empty.result], ip=ip)
    _lin.fill_builtin_region(fill_op.operation)
    return fill_op


def _lower_ones_like(op: Any) -> Any | None:
    """sf.ones_like → tensor.empty + linalg.fill(1.0)."""
    if len(op.operation.operands) < 1:
        return _lower_passthrough(op)
    at = op.operation.operands[0]
    rt = op.result.type
    if not isinstance(rt, ir.RankedTensorType):
        return _lower_passthrough(op)
    ip = _ip(op)
    empty_val = _make_empty(rt, [at], ip)
    elt = rt.element_type
    one_attr = ir.FloatAttr.get(elt, 1.0) if "f" in str(elt).lower() else ir.IntegerAttr.get(elt, 1)
    const = ir.Operation.create(
        "arith.constant", results=[elt], attributes={"value": one_attr}, ip=ip,
    )
    import mlir.dialects.linalg as _lin
    fill_op = _lin.FillOp(result_tensors=[rt], inputs=[const.result], outputs=[empty_val], ip=ip)
    _lin.fill_builtin_region(fill_op.operation)
    return fill_op


def _lower_softplus(op: Any) -> Any | None:
    """sf.softplus → log(1 + exp(x))."""
    if len(op.operation.operands) < 1:
        return None
    a = op.operation.operands[0]
    rt = op.result.type
    elt = rt.element_type if isinstance(rt, ir.RankedTensorType) else ir.F32Type.get()
    ip = _ip(op)
    empty_val = _make_empty(rt, [a], ip)
    one_val = ir.FloatAttr.get(elt, 1.0)
    import mlir.dialects.linalg as _lin
    out_rank = len(rt.shape) if isinstance(rt, ir.RankedTensorType) else 1

    def body(ia: list[ir.Value]) -> ir.Value:
        exp = ir.Operation.create("math.exp", operands=[ia[0]], infer_type=True)
        one = ir.Operation.create("arith.constant", results=[elt], attributes={"value": one_val})
        add = ir.Operation.create("arith.addf", operands=[one.result, exp.result], infer_type=True)
        return ir.Operation.create("math.log", operands=[add.result], infer_type=True).result

    return _make_generic(
        [a], empty_val, rt, 1, ip, body,
        iterator_types=[_lin.IteratorType.parallel] * out_rank,
    )


def _lower_clamp_min(op: Any) -> Any | None:
    """sf.clamp_min → max(x, min_val)."""
    if len(op.operation.operands) < 1:
        return None
    a = op.operation.operands[0]
    rt = op.result.type
    elt = rt.element_type if isinstance(rt, ir.RankedTensorType) else ir.F32Type.get()
    ip = _ip(op)
    min_val_attr = op.operation.attributes.get("min")
    min_val = _get_int(min_val_attr) if min_val_attr is not None else 0
    empty_val = _make_empty(rt, [a], ip)
    import mlir.dialects.linalg as _lin
    out_rank = len(rt.shape) if isinstance(rt, ir.RankedTensorType) else 1
    min_float = ir.FloatAttr.get(elt, float(min_val))

    def body(ia: list[ir.Value]) -> ir.Value:
        c = ir.Operation.create("arith.constant", results=[elt], attributes={"value": min_float})
        return ir.Operation.create("arith.maxnumf", operands=[ia[0], c.result], infer_type=True).result

    return _make_generic(
        [a], empty_val, rt, 1, ip, body,
        iterator_types=[_lin.IteratorType.parallel] * out_rank,
    )


def _lower_eye(op: Any) -> Any | None:
    """sf.eye → linalg.generic with linalg.index for diagonal check."""
    rt = op.result.type
    if not isinstance(rt, ir.RankedTensorType) or len(rt.shape) < 2:
        return _lower_passthrough(op)
    elt = rt.element_type
    ip = _ip(op)
    import mlir.dialects.linalg as _lin
    out_rank = len(rt.shape)
    empty = ir.Operation.create("tensor.empty", results=[rt], ip=ip)

    def body(ia: list[ir.Value]) -> ir.Value:
        i = ir.Operation.create(
            "linalg.index",
            results=[ir.IndexType.get()],
            attributes={"dim": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), out_rank - 2)},
        )
        j = ir.Operation.create(
            "linalg.index",
            results=[ir.IndexType.get()],
            attributes={"dim": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), out_rank - 1)},
        )
        eq_op = ir.Operation.create(
            "arith.cmpi",
            results=[ir.IntegerType.get_signless(1)],
            operands=[i.result, j.result],
            attributes={"predicate": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 0)},
            infer_type=False,
        )
        one_const = ir.Operation.create(
            "arith.constant", results=[elt],
            attributes={"value": ir.FloatAttr.get(elt, 1.0)},
        )
        zero_const = ir.Operation.create(
            "arith.constant", results=[elt],
            attributes={"value": ir.FloatAttr.get(elt, 0.0)},
        )
        return ir.Operation.create(
            "arith.select",
            operands=[eq_op.result, one_const.result, zero_const.result],
            infer_type=True,
        ).result

    return _make_generic(
        [], empty.result, rt, 0, ip, body,
        iterator_types=[_lin.IteratorType.parallel] * out_rank,
    )


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
    # Binary / unary / activation
    "sf.add": _lower_binary("arith.addf"),
    "sf.mul": _lower_binary("arith.mulf"),
    "sf.sub": _lower_binary("arith.subf"),
    "sf.div": _lower_binary("arith.divf"),
    "sf.max": _lower_binary("arith.maxnumf"),
    "sf.pow": _lower_binary("math.powf"),
    "sf.relu": _lower_relu,
    "sf.silu": _lower_silu,
    "sf.gelu": _lower_gelu,
    "sf.sigmoid": _lower_sigmoid,
    "sf.exp": _lower_unary("math.exp"),
    "sf.neg": _lower_unary("arith.negf"),
    "sf.rsqrt": _lower_unary("math.rsqrt"),
    "sf.cos": _lower_unary("math.cos"),
    "sf.sin": _lower_unary("math.sin"),
    "sf.tanh": _lower_unary("math.tanh"),
    "sf.sqrt": _lower_unary("math.sqrt"),
    # Matmul / linear
    "sf.matmul": _lower_matmul,
    "sf.linear": _lower_linear,
    # Shape / transform
    "sf.transpose": _lower_transpose,
    "sf.permute": _lower_passthrough,
    "sf.identity": _lower_passthrough,
    "sf.copy_": _lower_copy,
    "sf.view": _lower_view,
    "sf.expand": _lower_expand,
    "sf.cat": _lower_cat,
    "sf.slice": _lower_slice,
    "sf.select": _lower_select,
    "sf.unsqueeze": _lower_unsqueeze,
    "sf.split": _lower_passthrough,
    "sf.chunk": _lower_passthrough,
    "sf.pad": _lower_passthrough,
    # Reductions
    "sf.sum": _lower_sum,
    "sf.mean": _lower_mean,
    "sf.softmax": _lower_softmax,
    "sf.var": _lower_passthrough,
    "sf.linalg_norm": _lower_passthrough,
    # Comparison
    "sf.eq": _lower_comparison(0),
    "sf.ne": _lower_comparison(1),
    "sf.lt": _lower_comparison(2),
    "sf.le": _lower_comparison(3),
    "sf.gt": _lower_comparison(4),
    "sf.logical_and": _lower_logical_and,
    # Element-wise / misc
    "sf.zeros": _lower_zeros_op,
    "sf.zeros_like": _lower_zeros_op,
    "sf.ones_like": _lower_ones_like,
    "sf.full_like": _lower_passthrough,
    "sf.new_ones": _lower_passthrough,
    "sf.eye": _lower_eye,
    "sf.clamp_min": _lower_clamp_min,
    "sf.softplus": _lower_softplus,
    "sf.triu": _lower_passthrough,
    "sf.tril": _lower_passthrough,
    "sf.type_as": _lower_passthrough,
    "sf.masked_fill": _lower_passthrough,
    "sf.cumsum": _lower_passthrough,
    "sf.diff": _lower_passthrough,
    "sf.sym_size": _lower_passthrough,
    "sf.index": _lower_passthrough,
    "sf.embedding": _lower_passthrough,
    # Norms (passthrough until full decomposition)
    "sf.layer_norm": _lower_passthrough,
    "sf.rms_norm": _lower_passthrough,
    # Attention (passthrough — needs runtime handler)
    "sf.scaled_dot_product_attention": _lower_passthrough,
    # Fused ops (passthrough — needs decomposition pre-pass)
    "sf.fused_silu_mul": _lower_passthrough,
    "sf.fused_rms_norm_matmul": _lower_passthrough,
    "sf.fused_qkv": _lower_passthrough,
    "sf.fused_attention_output": _lower_passthrough,
    "sf.fused_attention_block": _lower_passthrough,
    # Misc complex ops (passthrough)
    "sf.arange": _lower_passthrough,
    "sf.einsum": _lower_passthrough,
    "sf.stack": _lower_passthrough,
    "sf.conv1d": _lower_passthrough,
    "sf.view_as": _lower_passthrough,
    "sf.expand_as": _lower_passthrough,
}
