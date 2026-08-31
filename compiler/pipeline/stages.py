# ruff: noqa: E501 — long lines in MLIR transform script f-strings

"""Pipeline stage execution engine.

See ``pipeline_stages_utils.py`` for the ``Stage`` / ``StageResult`` types
and IR utility functions.  See ``pipeline_actions.py`` for custom stage
action callables (tiling, FMA fusion, etc.).

This module contains the standard pipeline stage definitions
(``BUILTIN_STAGES``) and the main ``run_stages`` runner.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Sub-module imports ────────────────────────────────────────────────
from compiler.pipeline.stages_utils import (
    Stage,
    StageResult,
    _count_module_ops,
    _save_ir_snapshot,
    _verify_function_signatures,
)

_log = logging.getLogger(__name__)


# ── Standard pipeline stages ──────────────────────────────────────────


def _make_verify_stage() -> Stage:
    """Build a warn-only Stage that validates module structure after lowering.

    Checks all ``func.func`` ops have non-empty bodies and the module has at
    least one function.  Never blocks compilation — only logs warnings.
    """
    from compiler.passes._ops import mlir_verify_structure

    def _verify_action(m: Any) -> None:
        ctx = m.operation.context
        issues = mlir_verify_structure(m, ctx)
        if issues:
            for issue in issues:
                _log.warning("  Module verification: %s", issue)
        _log.info("  Module verification: %d issues found — %s", len(issues), "PASS" if not issues else "FAIL")

    return Stage(
        name="module-verify",
        action=_verify_action,
        timeout=10.0,
        warn_only=True,
    )


# ── Main run function ─────────────────────────────────────────────────


def _fixup_arith_tensor_constants_action(module: Any) -> int:
    """Fix arith.constant ops with scalar value + tensor result type.

    Uses MLIR Python bindings to walk the module and convert scalar-valued
    ``arith.constant`` ops with tensor result types to use ``DenseElementsAttr``
    (splat).  Replaces the regex-based fixup with proper MLIR API usage.
    """
    import mlir.ir as ir

    if not isinstance(module, ir.Module):
        return 0
    from compiler.backend.fixups import _walk_and_fix_tensor_constants

    return _walk_and_fix_tensor_constants(module)


def _emit_c_interface_action(module: Any) -> None:
    """Add ``llvm.emit_c_interface`` to each func.func."""
    import mlir.ir as ir

    main_mod = module.operation.regions[0].blocks[0]
    ctx = module.operation.context
    for op in list(main_mod):
        if str(op.operation.name) == "func.func":
            op.operation.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get(context=ctx)


def _strip_sf_attrs_action(module: Any) -> None:
    """Remove ``sf.*`` attributes from func.func ops.

    These attributes (e.g. ``sf.weight_names``) are used by the Python
    executor for weight binding but block the canonicalizer (which
    cannot handle unregistered dialect attributes at the function level).
    Stripping them before ``canonicalize,cse`` + ``bufferize`` allows
    the canonicalizer's ``InferStaticShapeOfOperands`` pattern to fix
    kDynamic dims in linalg.generic output types.
    """
    main_mod = module.operation.regions[0].blocks[0]
    keys_to_strip: list[str] = []
    for op in list(main_mod):
        if str(op.operation.name) != "func.func":
            continue
        keys_to_strip.clear()
        for attr_name in op.operation.attributes:
            if attr_name.startswith("sf."):
                keys_to_strip.append(attr_name)
        for k in keys_to_strip:
            del op.operation.attributes[k]


def _strip_sf_attrs_canon_action(module: Any) -> None:
    """Strip ``sf.*`` and ``llvm.emit_c_interface`` attrs from func.func ops.

    The ``sf.*`` function attributes (e.g. ``sf.weight_names``) block the
    canonicalizer which cannot handle unregistered dialect attributes.

    DOES NOT run canonicalize — with explicit linalg.broadcast + identity-map
    linalg.generic (S5 rewrite), canonicalize's InferStaticShapeOfOperands
    can fold broadcasts into generics incorrectly, creating shape mismatches
    that break bufferize.  (torch-mlir also skips canonicalize before bufferize.)

    ``llvm.emit_c_interface`` is stripped because it is only needed at the
    LLVM level (after ``convert-func-to-llvm``).
    """
    main_mod = module.operation.regions[0].blocks[0]
    stripped_count = 0
    for op in list(main_mod):
        if str(op.operation.name) != "func.func":
            continue
        keys = [k for k in op.operation.attributes if k.startswith("sf.")]
        if keys:
            for k in keys:
                del op.operation.attributes[k]
            stripped_count += len(keys)
    if stripped_count:
        _log.info("  Stripped %d sf.*/llvm.emit_c_interface attrs", stripped_count)


def _insert_identity_copies_action(module: Any) -> None:
    """Insert tensor.insert_slice copies for identity pass-through returns (Stage C2.5).

    Delegates to :func:`compiler.pipeline.actions.insert_identity_copies_action`.
    """
    from compiler.pipeline.actions import insert_identity_copies_action as _action

    _action(module)


def _unsqueeze_copies_action(module: Any) -> None:
    """Insert tensor.insert_slice copies before tensor.expand_shape (Stage C2.6)."""
    from compiler.pipeline.actions import insert_unsqueeze_copies_action as _action

    _action(module)


def _is_identity_region_yield(op: Any) -> bool:
    """True when a bufferized linalg.generic has the pure copy body ``yield %in``."""
    try:
        block = op.regions[0].blocks[0]
    except (IndexError, AttributeError):
        return False
    if len(block.arguments) != 2:
        return False
    body_ops = list(block.operations)
    if len(body_ops) != 1 or str(body_ops[0].operation.name) != "linalg.yield":
        return False
    yield_op = body_ops[0]
    return len(yield_op.operands) == 1 and yield_op.operands[0] == block.arguments[0]


def _generic_maps(op: Any) -> list[str]:
    maps_attr = op.operation.attributes.get("indexing_maps")
    if maps_attr is None:
        return []
    return [str(m) for m in maps_attr]


def _iterator_types_parallel(op: Any) -> bool:
    iter_attr = op.operation.attributes.get("iterator_types")
    if iter_attr is None:
        return False
    return all("parallel" in str(it) for it in iter_attr)


def _is_f32_memref(value: Any, rank: int) -> bool:
    import mlir.ir as ir

    try:
        ty = value.type
    except AttributeError:
        return False
    return (
        isinstance(ty, ir.MemRefType)
        and ty.rank == rank
        and str(ty.element_type) == "f32"
    )


def _memref_dim(value: Any, dim: int) -> int:
    ty = value.type
    return ty.get_dim_size(dim) if not ty.is_dynamic_dim(dim) else 0


def _dims_compatible(a: int, b: int) -> bool:
    return a == 0 or b == 0 or a == b


def _is_pure_transpose_generic(op: Any) -> bool:
    """Match the bufferized ``W -> W^T`` identity linalg.generic."""
    if (
        str(op.operation.name) != "linalg.generic"
        or len(op.operands) != 2
        or not _is_f32_memref(op.operands[0], 2)
        or not _is_f32_memref(op.operands[1], 2)
    ):
        return False
    if _generic_maps(op) != [
        "affine_map<(d0, d1) -> (d1, d0)>",
        "affine_map<(d0, d1) -> (d0, d1)>",
    ]:
        return False
    if not _iterator_types_parallel(op):
        return False
    if not _dims_compatible(_memref_dim(op.operands[0], 0), _memref_dim(op.operands[1], 1)):
        return False
    if not _dims_compatible(_memref_dim(op.operands[0], 1), _memref_dim(op.operands[1], 0)):
        return False
    return _is_identity_region_yield(op)


def _is_pure_broadcast_generic(op: Any) -> bool:
    """Match ``W^T[K, N] -> B[? K N]`` rank-2 to rank-3 broadcast."""
    if (
        str(op.operation.name) != "linalg.generic"
        or len(op.operands) != 2
        or not _is_f32_memref(op.operands[0], 2)
        or not _is_f32_memref(op.operands[1], 3)
    ):
        return False
    if _generic_maps(op) != [
        "affine_map<(d0, d1, d2) -> (d1, d2)>",
        "affine_map<(d0, d1, d2) -> (d0, d1, d2)>",
    ]:
        return False
    if not _iterator_types_parallel(op):
        return False
    if not _dims_compatible(_memref_dim(op.operands[0], 0), _memref_dim(op.operands[1], 1)):
        return False
    if not _dims_compatible(_memref_dim(op.operands[0], 1), _memref_dim(op.operands[1], 2)):
        return False
    return _is_identity_region_yield(op)


def _block_containing_op(op: Any) -> Any | None:
    """Return the Block object directly containing ``op``."""
    parent = getattr(op, "parent", None)
    if parent is None:
        return None

    def _search(container: Any) -> Any | None:
        for region in getattr(container, "regions", []):
            for block in region.blocks:
                if any(child == op for child in block.operations):
                    return block
                found = _search(block)
                if found is not None:
                    return found
        return None

    return _search(parent)


def _find_last_writer_before(
    block_ops: list[Any],
    target: Any,
    before_idx: int,
) -> Any | None:
    """Return the last linalg op writing ``target`` before ``before_idx``."""
    for idx in range(before_idx - 1, -1, -1):
        op = block_ops[idx]
        if len(op.operands) >= 2 and op.operands[-1] == target:
            return op
    return None


def _find_transb_rank2_chain(
    matmul_op: Any,
    block_ops: list[Any],
    matmul_idx: int,
) -> tuple[Any, Any] | None:
    """Recognize transpose(W) -> linalg.matmul and return (transpose, W)."""
    if len(matmul_op.operands) != 3:
        return None
    b_val = matmul_op.operands[1]
    if not _is_f32_memref(b_val, 2):
        return None
    transpose_op = _find_last_writer_before(block_ops, b_val, matmul_idx)
    if transpose_op is None or not _is_pure_transpose_generic(transpose_op):
        return None
    weight_val = transpose_op.operands[0]
    if not _is_f32_memref(weight_val, 2):
        return None
    if not _dims_compatible(_memref_dim(b_val, 0), _memref_dim(weight_val, 1)):
        return None
    if not _dims_compatible(_memref_dim(b_val, 1), _memref_dim(weight_val, 0)):
        return None
    return transpose_op, weight_val


def _find_transb_rank3_chain(
    matmul_op: Any,
    block_ops: list[Any],
    matmul_idx: int,
) -> tuple[Any, Any, Any] | None:
    """Recognize transpose(W) -> broadcast -> linalg.batch_matmul.

    Returns ``(transpose_op, broadcast_op, W)`` or ``None``.
    """
    if len(matmul_op.operands) != 3:
        return None
    b_val = matmul_op.operands[1]
    if not _is_f32_memref(b_val, 3):
        return None
    broadcast_op = _find_last_writer_before(block_ops, b_val, matmul_idx)
    if broadcast_op is None or not _is_pure_broadcast_generic(broadcast_op):
        return None
    broadcast_idx = block_ops.index(broadcast_op)
    transposed_val = broadcast_op.operands[0]
    if not _is_f32_memref(transposed_val, 2):
        return None
    transpose_op = _find_last_writer_before(block_ops, transposed_val, broadcast_idx)
    if transpose_op is None or not _is_pure_transpose_generic(transpose_op):
        return None
    weight_val = transpose_op.operands[0]
    if not _is_f32_memref(weight_val, 2):
        return None
    # B is [batch, K, N]; W is [N, K].
    if not _dims_compatible(_memref_dim(b_val, 1), _memref_dim(weight_val, 1)):
        return None
    if not _dims_compatible(_memref_dim(b_val, 2), _memref_dim(weight_val, 0)):
        return None
    return transpose_op, broadcast_op, weight_val


def _lower_linalg_matmul_to_sfa_blas_action(module: Any) -> None:
    """Replace bufferized linalg matmuls with calls to the SFA BLAS bridge.

    Stage C3 has already bufferized all tensor ops, so each ``linalg.matmul``
    / ``linalg.batch_matmul`` now operates on memrefs.  The default
    ``convert-linalg-to-loops`` path lowers them to scalar loops, which runs
    an order of magnitude slower than Accelerate/OpenBLAS SGEMM on CPU.

    In addition to the NoTrans bridge, this action recognizes the per-step
    weight-copy chain emitted for ``torch.nn.Linear``:

    * rank-2: ``linalg.generic(transpose W)`` -> ``linalg.matmul``
    * rank-3: ``linalg.generic(transpose W)`` ->
      ``linalg.generic(broadcast W^T)`` -> ``linalg.batch_matmul``

    Those chains are replaced by ``sfa_sgemm_transb`` /
    ``sfa_batch_sgemm_transb``, which accept the original ``[N, K]`` weight
    memref and call ``cblas_sgemm`` with ``CblasTrans``.  The transpose and
    broadcast linalg.generics are erased; a following canonicalize/cse stage
    removes the now-dead ``memref.alloc`` buffers.
    """
    import os

    import mlir.ir as ir

    if os.environ.get("SERVEFORGE_NO_SFA_BLAS") == "1":
        return

    f32 = ir.F32Type.get()
    dyn = ir.ShapedType.get_dynamic_size()
    memref2 = ir.MemRefType.get([dyn, dyn], f32)
    memref3 = ir.MemRefType.get([dyn, dyn, dyn], f32)
    callee_by_rank = {
        2: ("sfa_sgemm", memref2),
        3: ("sfa_batch_sgemm", memref3),
    }
    transb_decls = {
        "sfa_sgemm_transb": (memref2, memref2, memref2),
        "sfa_batch_sgemm_transb": (memref3, memref2, memref3),
    }

    module_body = module.operation.regions[0].blocks[0]

    def _ensure_decl(callee: str, memref_types: tuple[Any, ...]) -> None:
        for op in module_body.operations:
            if str(op.operation.name) == "func.func":
                if ir.StringAttr(op.operation.attributes.get("sym_name")).value == callee:
                    return
        ftype = ir.FunctionType.get(list(memref_types), [])
        attrs = {
            "function_type": ir.TypeAttr.get(ftype),
            "sym_name": ir.StringAttr.get(callee),
            "sym_visibility": ir.StringAttr.get("private"),
        }
        with module.operation.context, ir.Location.unknown():
            ir.Operation.create(
                "func.func",
                results=[],
                operands=[],
                attributes=attrs,
                regions=1,
                loc=ir.Location.unknown(),
                ip=ir.InsertionPoint.at_block_begin(module_body),
            )

    matmul_ops: list[tuple[str, Any, Any, str]] = []

    def _collect(op: ir.Operation) -> None:
        name = str(op.operation.name)
        if name not in ("linalg.matmul", "linalg.batch_matmul") or len(op.operands) != 3:
            pass
        else:
            # The SFA BLAS bridge is f32-only.  BF16/F16 matmuls must stay
            # on the linalg path; op-plan/runtime GEMV kernels cover them.
            element_types = [str(v.type.element_type) for v in op.operands if isinstance(v.type, ir.MemRefType)]
            if element_types and all(t == "f32" for t in element_types):
                target_ty = memref3 if name == "linalg.batch_matmul" else memref2
                callee = "sfa_batch_sgemm" if name == "linalg.batch_matmul" else "sfa_sgemm"
                matmul_ops.append((name, op, target_ty, callee))
        for region in op.regions:
            for block in region.blocks:
                for child in list(block.operations):
                    _collect(child)

    for region in module.operation.regions:
        for block in region.blocks:
            for op in list(block.operations):
                _collect(op)

    if not matmul_ops:
        return

    for _callee, _memref_ty in callee_by_rank.values():
        _ensure_decl(_callee, (_memref_ty, _memref_ty, _memref_ty))
    for _callee, _memref_tys in transb_decls.items():
        _ensure_decl(_callee, _memref_tys)

    replaced_transb = 0
    replaced_regular = 0
    for _name, op, target_ty, regular_callee in matmul_ops:
        if not all(isinstance(v.type, ir.MemRefType) for v in op.operands):
            raise RuntimeError(
                f"{_name} reached SFA BLAS rewrite before bufferization: {op}"
            )

        # The rewrite mutates the IR as it walks, so re-list the containing
        # block for each matmul to keep writer lookups and op order fresh.
        parent_block = _block_containing_op(op)
        if parent_block is None:
            raise RuntimeError(f"matmul is not inside a block: {op}")
        block_ops = list(parent_block.operations)
        matmul_idx = block_ops.index(op)

        chain: tuple[Any, ...] | None = None
        if _name == "linalg.matmul":
            found = _find_transb_rank2_chain(op, block_ops, matmul_idx)
            if found is not None:
                transpose_op, weight_val = found
                chain = ("sfa_sgemm_transb", (memref2, memref2, memref2), [transpose_op], weight_val)
        else:
            found3 = _find_transb_rank3_chain(op, block_ops, matmul_idx)
            if found3 is not None:
                transpose_op, broadcast_op, weight_val = found3
                chain = (
                    "sfa_batch_sgemm_transb",
                    (memref3, memref2, memref3),
                    [transpose_op, broadcast_op],
                    weight_val,
                )

        with module.operation.context, ir.InsertionPoint(op):
            if chain is not None:
                callee, cast_types, dead_chain_ops, weight_val = chain
                call_operands = [op.operands[0], weight_val, op.operands[2]]
            else:
                callee = regular_callee
                cast_types = (target_ty, target_ty, target_ty)
                call_operands = list(op.operands)
                dead_chain_ops = []

            cast_operands = []
            for operand, cast_ty in zip(call_operands, cast_types, strict=True):
                cast_operands.append(
                    ir.Operation.create(
                        "memref.cast",
                        results=[cast_ty],
                        operands=[operand],
                        loc=op.location,
                        ip=ir.InsertionPoint(op),
                    ).result
                )
            ir.Operation.create(
                "func.call",
                results=[],
                operands=cast_operands,
                attributes={"callee": ir.FlatSymbolRefAttr.get(callee)},
                loc=op.location,
                ip=ir.InsertionPoint(op),
            )
        op.erase()
        for dead_op in dead_chain_ops:
            dead_op.erase()
        if chain is not None:
            replaced_transb += 1
        else:
            replaced_regular += 1

    if replaced_transb:
        _log.info(
            "  Replaced %d linalg matmuls with SFA BLAS bridge calls (%d transb)",
            len(matmul_ops),
            replaced_transb,
        )
    else:
        _log.info("  Replaced %d linalg matmuls with SFA BLAS bridge calls", len(matmul_ops))


# The flattened equivalent of the LLVM lowering stages below is
# available as LINALG_TO_LLVM_PIPELINE in compile_utils.py.
# Keep the two definitions in sync.
BUILTIN_STAGES: list[Stage] = [
    # ── Phase A: sf-level cleanup ──
    Stage("A1-canonicalize,cse", "canonicalize,cse"),
    Stage("A2-eliminate-empty-tensors", "eliminate-empty-tensors"),
    Stage("A3-empty-tensor-to-alloc", "empty-tensor-to-alloc-tensor"),
    # ── Phase B: broadcast decomposition (optional, skip) ──
    # ── Phase C: bufferization ──
    Stage("C1-interface", action=_emit_c_interface_action, timeout=5.0),
    Stage("C2-strip-sf-attrs", action=_strip_sf_attrs_canon_action, timeout=30.0),
    Stage("C2.5-insert-identity-copies", action=_insert_identity_copies_action, timeout=30.0),
    Stage("C2.6-unsqueeze-copies", action=_unsqueeze_copies_action, timeout=30.0),
    Stage(
        "C3-bufferize",
        (
            "one-shot-bufferize{bufferize-function-boundaries allow-unknown-ops"
            " function-boundary-type-conversion=identity-layout-map},"
            "canonicalize,cse,convert-bufferization-to-memref"
        ),
        timeout=60.0,
    ),
    Stage("C4-linalg-matmul→sfa-blas", action=_lower_linalg_matmul_to_sfa_blas_action, timeout=60.0),
    Stage(
        "C4.1-canonicalize-dead-weight-copies",
        "canonicalize,cse",
        timeout=60.0,
    ),
    # ── Phase D: loops → control flow ──
    Stage("D1-linalg→loops", "convert-linalg-to-loops"),
    Stage("D2-lower-affine", "lower-affine"),
    Stage("D3-scf→cf", "convert-scf-to-cf"),
    Stage("D4-expand-strided", "expand-strided-metadata"),
    # ── Phase E: LLVM conversion ──
    Stage("E1-finalize-memref", "finalize-memref-to-llvm{use-generic-functions=false}"),
    Stage("E2-math→llvm", "convert-math-to-llvm"),
    Stage("E3-vector→llvm", "convert-vector-to-llvm", timeout=120.0),
    Stage("E4-arith→llvm", "convert-arith-to-llvm"),
    Stage("E5-func→llvm", "convert-func-to-llvm"),
    Stage("E6-cf→llvm", "convert-cf-to-llvm"),
    Stage("E7-bufferization→memref", "convert-bufferization-to-memref"),
    Stage("E8-finalize-memref-2", "finalize-memref-to-llvm{use-generic-functions=false}", timeout=60.0),
    Stage("E8.5-lower-affine-late", "lower-affine", timeout=30.0),
    Stage("E8.6-arith-to-llvm-late", "convert-arith-to-llvm", timeout=30.0),
    Stage("E9-reconcile-casts", "reconcile-unrealized-casts"),
    Stage("E10-strip-gep-nuw", "sf-strip-gep-nuw", timeout=10.0, warn_only=False),
    _make_verify_stage(),
]

# Variant without FMA (fused multiply-add) lowering — omitted for CPUs
# without FMA support. Currently aliased to BUILTIN_STAGES; a proper
# no-FMA variant should exclude the math-to-llvm stage or use a
# no-FMA-conversion pass.
BUILTIN_STAGES_NO_FMA: list[Stage] = BUILTIN_STAGES


def get_pipeline_specs() -> list[tuple[str, str, float, bool]]:
    """Return BUILTIN_STAGES as (name, pipeline, timeout, warn_only) tuples.

    Convenience for scripts that need the raw pipeline strings for
    custom execution (debugging, timing, etc.) without depending on
    the Stage/StageResult types.
    """
    return [
        (s.name, s.pipeline, s.timeout, s.warn_only)
        for s in BUILTIN_STAGES
        if s.pipeline  # skip action-only stages (tiling, FMA, etc.)
    ]


def run_stages(
    module: Any,
    ctx: Any,
    stages: list[Stage],
    log_dir: str = "",
) -> list[StageResult]:
    """Run a sequence of pipeline stages.

    Each stage is executed in order.  If a non-warn_only stage fails,
    execution stops and subsequent stages are not run.

    Tracks IR line count across stages and warns if growth exceeds 5x
    (indicating possible IR explosion).

    When ``LLM_SERVEFORGE_LOG=DEBUG``, full IR snapshots are saved to
    ``outputs/logs/pipeline/stages/`` after each stage.

    When ``LLM_SERVEFORGE_LOG_FORMAT=json``, structured events with
    ``event_type="pipeline_stage"`` and ``event_type="pipeline_complete"``
    are emitted via the JSON log formatter.

    Returns a list of StageResult objects, one per stage.
    """
    results: list[StageResult] = []
    prev_line_count: int = len(str(module).splitlines())

    # ── Structured event / DEBUG snapshot setup ──
    _emit_json = os.environ.get("LLM_SERVEFORGE_LOG_FORMAT", "text").lower() == "json"
    _is_debug = logging.getLogger().isEnabledFor(logging.DEBUG)

    # Clean stages directory before starting
    stages_dir = Path("outputs/logs") / "pipeline" / "stages"
    shutil.rmtree(stages_dir, ignore_errors=True)
    stages_dir.mkdir(parents=True, exist_ok=True)

    # Capture initial state for summary
    initial_module_str = str(module)
    initial_line_count = prev_line_count
    initial_op_count, _ = _count_module_ops(initial_module_str)

    for stage in stages:
        result = stage.run(module, ctx, log_dir)
        results.append(result)

        # ── Per-stage analysis ──
        pre_counts = result.context.get("dialect_counts_pre", {})
        post_counts = result.context.get("dialect_counts_post", {})
        op_count_pre = sum(pre_counts.values()) if pre_counts else 0
        op_count_post = sum(post_counts.values()) if post_counts else 0
        all_dialects = set(pre_counts.keys()) | set(post_counts.keys())
        dialect_deltas = {d: post_counts.get(d, 0) - pre_counts.get(d, 0) for d in sorted(all_dialects)}

        # DEBUG: save per-stage snapshot
        if _is_debug:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", stage.name)
            snapshot_path = stages_dir / f"{safe_name}_{timestamp}.mlir"
            snapshot_path.write_text(str(module))
            _log.debug("  Stage '%s' IR saved to %s", stage.name, snapshot_path)

        # JSON structured event: per-stage
        if _emit_json:
            _log.info(
                "",
                extra={
                    "event_type": "pipeline_stage",
                    "event_data": {
                        "stage_name": stage.name,
                        "elapsed_secs": round(result.elapsed, 3),
                        "ir_lines": result.ir_lines,
                        "op_count_pre": op_count_pre,
                        "op_count_post": op_count_post,
                        "dialect_deltas": dialect_deltas,
                        "warn_only": stage.warn_only,
                        "success": result.success,
                    },
                },
            )

        # ── Growth check (existing) ──
        if result.ir_lines > 0 and prev_line_count > 0:
            growth_ratio = result.ir_lines / prev_line_count
            if growth_ratio > 5.0:
                _log.warning(
                    "  ⚠️ Stage '%s' caused %dx IR growth (%d → %d lines) — possible explosion",
                    stage.name,
                    int(growth_ratio),
                    prev_line_count,
                    result.ir_lines,
                )
                if result.ir_snapshot_path:
                    _log.warning(
                        "  ⚠️ IR snapshot saved to %s (growth may cause downstream hangs)",
                        result.ir_snapshot_path,
                    )
        if result.ir_lines > 0:
            prev_line_count = result.ir_lines

        if not result.success and not stage.warn_only:
            raise RuntimeError(
                f"Pipeline stage '{stage.name}' failed: {result.error}. IR snapshot saved to {result.ir_snapshot_path}"
            )

    # ── Pipeline summary ──
    _log.info("=" * 60)
    total_elapsed = sum(r.elapsed for r in results)
    final_line_count = results[-1].ir_lines if results else initial_line_count
    _log.info("  Pipeline summary: %d stages in %.2f s", len(results), total_elapsed)
    _log.info(
        "  Initial IR: %d lines → Final: %d lines (%+.1f%%)",
        initial_line_count,
        final_line_count,
        ((final_line_count - initial_line_count) / max(initial_line_count, 1)) * 100,
    )

    # ── Warn-only failure report ──
    warn_failures = []
    for stage, result in zip(stages, results, strict=True):
        if stage.warn_only and not result.success:
            warn_failures.append(stage.name)
    if warn_failures:
        _log.warning(
            "  ⚠️ %d warn_only stage(s) failed: %s — outputs may be degraded",
            len(warn_failures),
            warn_failures,
        )

    _log.info("=" * 60)

    # ── Post-pipeline: function signature verification ──
    final_text = str(module)
    sig_errors = _verify_function_signatures(final_text)
    if sig_errors:
        for err in sig_errors:
            _log.warning("Post-pipeline function signature mismatch: %s", err)
        sig_snapshot_path = _save_ir_snapshot(module, "signature_mismatch")
        _log.warning(
            "  %d function(s) with signature/return mismatch — IR saved to %s "
            "(this may cause uninitialized output buffers at runtime, see Issue #45)",
            len(sig_errors),
            sig_snapshot_path,
        )

    # JSON structured event: pipeline complete
    if _emit_json and results:
        final_module_str = str(module)
        final_op_count, _ = _count_module_ops(final_module_str)
        _log.info(
            "",
            extra={
                "event_type": "pipeline_complete",
                "event_data": {
                    "num_stages": len(results),
                    "total_elapsed_secs": round(total_elapsed, 3),
                    "initial_ir_lines": initial_line_count,
                    "final_ir_lines": final_line_count,
                    "initial_op_count": initial_op_count,
                    "final_op_count": final_op_count,
                    "all_success": all(r.success for r in results),
                    "stages_completed": sum(1 for r in results if r.success),
                },
            },
        )

    return results
