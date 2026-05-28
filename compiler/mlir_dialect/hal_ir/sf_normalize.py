"""SF Normalize pass — decompose complex SF ops into primitive SF ops.

Decomposition rules (all ops stay in the SF dialect):

  sf.scaled_dot_product_attention(query, key, value, [attn_mask]) →
    key_t    = sf.transpose(key)   {dim0=2, dim1=3}
    scores   = sf.matmul(query, key_t)
    [masked] = sf.add(scores, attn_mask)   [if attn_mask present]
    weights  = sf.softmax(scores)
    output   = sf.matmul(weights, value)

  sf.linear(input, weight, [bias]) →
    result = sf.matmul(input, weight)
    result = sf.add(result, bias)          [if bias present]

  sf.layer_norm(input, weight, bias) →
    mean      = sf.mean(input)    {dim=-1}
    centered  = sf.sub(input, mean)
    sq        = sf.pow(centered, 2)
    var       = sf.mean(sq)       {dim=-1}
    rstd      = sf.rsqrt(var)
    normed    = sf.mul(centered, rstd)
    scaled    = sf.mul(normed, weight)
    output    = sf.add(scaled, bias)

Usage:
    from compiler.mlir_dialect.hal_ir.sf_normalize import normalize_sf_mlir
    normalized = normalize_sf_mlir(mlir_text)
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

from compiler.mlir_dialect.compile_utils import _setup_mlir_path

_log = logging.getLogger(__name__)


# ── Public API ──────────────────────────────────────────────────────


def normalize_sf_mlir(mlir_text: str) -> str:
    """Run SF Normalize pass on MLIR text.

    Parses MLIR text, decomposes complex SF ops (SDPA, linear, layer_norm)
    into primitive SF ops, and returns the normalized MLIR text.

    All decomposed ops stay in the SF dialect.
    """
    _setup_mlir_path()
    import mlir.ir as ir

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    try:
        from mlir_sf._mlir_libs._sfDialectsNanobind import sf
        sf.register_dialects(ctx._CAPIPtr, load=True)
    except ImportError:
        _log.warning("SF dialect C++ bindings not available; "
                      "proceeding without sf dialect registration")

    with ctx, ir.Location.unknown(ctx):
        module = ir.Module.parse(mlir_text, ctx)
        stats = _normalize_module(module)
        output = str(module)

    _log.info(
        "Normalized %d ops: %d SDPA → decomposed, %d linear → decomposed, "
        "%d layer_norm → decomposed",
        sum(stats.values()), stats["sdpa"], stats["linear"], stats["layer_norm"],
    )
    return output


def normalize_sf_module(module: ir.Module) -> dict[str, int]:
    """In-place normalize an ``ir.Module``.  Returns stats dict."""
    return _normalize_module(module)


# ── Core pass logic ────────────────────────────────────────────────


def _normalize_module(module: ir.Module) -> dict[str, int]:
    """Walk all functions/ops in the module and decompose complex SF ops.

    Returns a stats dict with keys ``sdpa``, ``linear``, ``layer_norm``.
    """
    stats: dict[str, int] = {"sdpa": 0, "linear": 0, "layer_norm": 0}

    # Collect all candidate ops first to avoid iteration-during-mutation issues.
    candidates: list[Any] = []
    _collect_candidates(module.operation, candidates)

    # Process each candidate in reverse order so that SSA value replacements
    # from earlier ops transparently update operands of later-candidate ops.
    for op, name in reversed(candidates):
        if name == "sf.scaled_dot_product_attention":
            _decompose_sdpa(op)
            stats["sdpa"] += 1
        elif name == "sf.linear":
            _decompose_linear(op)
            stats["linear"] += 1
        elif name == "sf.layer_norm":
            _decompose_layer_norm(op)
            stats["layer_norm"] += 1

    return stats


def _collect_candidates(container: Any, candidates: list[tuple[Any, str]]) -> None:
    """Recursively collect ops to decompose from a container (module/func/region).

    Each candidate is stored as ``(op_view, op_name_string)``.
    """
    if not hasattr(container, "regions"):
        return
    for region in container.regions:
        for block in region.blocks:
            for op in block:
                name = _op_name(op)
                if name in ("sf.scaled_dot_product_attention",
                            "sf.linear", "sf.layer_norm"):
                    candidates.append((op, name))
                # Recurse into nested regions (e.g. scf.while, etc.)
                _collect_candidates(op, candidates)


def _op_name(op: Any) -> str:
    """Get the MLIR op name string from an OpView or Operation."""
    if hasattr(op, "operation"):
        return str(op.operation.name)
    return str(op.name) if hasattr(op, "name") else ""


# ── Helpers ─────────────────────────────────────────────────────────


def _sdpa_score_type_from(query_type: Any, key_t_type: Any) -> Any:
    """Compute the type for SDPA scores = Q @ K^T.

    For Q: <?xHxSxD> and K^T: <?xHxDxS_k>, the score is <?xHxSxS_k>.
    The last dim is **not** head_dim (64) but the sequence length of K.
    """
    import mlir.ir as ir

    if isinstance(query_type, ir.RankedTensorType) and isinstance(key_t_type, ir.RankedTensorType):
        q_shape = list(query_type.shape)
        kt_shape = list(key_t_type.shape)
        # scores: [batch, heads, seq_q, seq_k]
        scores_shape = q_shape[:-1] + [kt_shape[-1]]  # take last dim from K^T
        return ir.RankedTensorType.get(scores_shape, query_type.element_type)
    return query_type  # fallback


def _ranked_tensor_type_from(
    existing_type: Any,
    new_shape: list[int | None] | None = None,
) -> Any:
    """Create a RankedTensorType, optionally with a new shape.

    If *new_shape* is ``None``, uses the shape from *existing_type*.
    Element type is preserved from *existing_type*.
    """
    import mlir.ir as ir

    if isinstance(existing_type, ir.RankedTensorType):
        et = existing_type.element_type
        shape = new_shape if new_shape is not None else list(existing_type.shape)
        return ir.RankedTensorType.get(shape, et)
    if isinstance(existing_type, ir.UnrankedTensorType):
        et = existing_type.element_type
        shape = new_shape if new_shape is not None else []
        return ir.RankedTensorType.get(shape, et)
    # Fallback: treat as opaque f32 type
    return ir.RankedTensorType.get(
        new_shape or [],
        ir.F32Type.get(),
    )


def _i64_attr(value: int) -> Any:
    """Create an MLIR 64-bit integer attribute."""
    import mlir.ir as ir
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(64), value)


# ── SDPA decomposition ─────────────────────────────────────────────


def _decompose_sdpa(sdpa_op: Any) -> None:
    """Decompose ``sf.scaled_dot_product_attention`` into primitives.

    sf.scaled_dot_product_attention(query, key, value, [attn_mask]) →
      1. sf.transpose(key) → key_t
      2. sf.matmul(query, key_t) → scores
      3. sf.add(scores, attn_mask)    [if attn_mask present]
      4. sf.softmax(scores) → weights
      5. sf.matmul(weights, value) → output
    """
    import mlir.ir as ir

    operands = list(sdpa_op.operands)
    if len(operands) < 3:
        _log.warning("SDPA op has fewer than 3 operands: %d", len(operands))
        return

    query, key, value = operands[0], operands[1], operands[2]
    attn_mask = operands[3] if len(operands) > 3 else None
    result_type = sdpa_op.operation.results[0].type

    ctx = sdpa_op.operation.context

    with ir.Location.unknown(ctx), ir.InsertionPoint(sdpa_op.operation):
        # 1. Transpose key: swap last two dims (<?xHx?xD> → <?xHxDx?>)
        key_type = key.type
        if isinstance(key_type, ir.RankedTensorType):
            key_shape = list(key_type.shape)
            # Swap last two dims: [B, H, S, D] → [B, H, D, S]
            if len(key_shape) >= 2:
                key_shape[-1], key_shape[-2] = key_shape[-2], key_shape[-1]
            key_t_type = _ranked_tensor_type_from(key_type, key_shape)
        else:
            key_t_type = key_type

        key_t = ir.Operation.create(
            "sf.transpose",
            results=[key_t_type],
            operands=[key],
            attributes={
                "dim0": _i64_attr(2),
                "dim1": _i64_attr(3),
            },
        )

        # 2. Matmul: scores = Q @ K^T  →  <?xHx?x?>
        #    The score type has the last dim = seq_k (dynamic), not head_dim (64).
        scores_type = _sdpa_score_type_from(query.type, key_t.operation.results[0].type)
        scores = ir.Operation.create(
            "sf.matmul",
            results=[scores_type],
            operands=[query, key_t.operation.results[0]],
        )

        current = scores.operation.results[0]

        # 3. Add attention mask if present
        if attn_mask is not None:
            masked = ir.Operation.create(
                "sf.add",
                results=[current.type],
                operands=[current, attn_mask],
            )
            current = masked.operation.results[0]

        # 4. Softmax over last dim
        weights = ir.Operation.create(
            "sf.softmax",
            results=[current.type],
            operands=[current],
        )

        # 5. Matmul: output = weights @ V
        output = ir.Operation.create(
            "sf.matmul",
            results=[result_type],
            operands=[weights.operation.results[0], value],
        )

    # Replace all uses of the SDPA result with the decomposed output
    new_result = output.operation.results[0]
    sdpa_op.operation.results[0].replace_all_uses_with(new_result)
    sdpa_op.operation.erase()


# ── Linear decomposition ───────────────────────────────────────────


def _decompose_linear(linear_op: Any) -> None:
    """Decompose ``sf.linear`` into ``sf.matmul`` [+ ``sf.add`` if bias].

    sf.linear(input, weight, [bias]) →
      sf.matmul(input, weight) → result
      sf.add(result, bias)           [if bias present]
    """
    import mlir.ir as ir

    operands = list(linear_op.operands)
    if len(operands) < 2:
        _log.warning("Linear op has fewer than 2 operands: %d", len(operands))
        return

    input_val, weight = operands[0], operands[1]
    bias = operands[2] if len(operands) > 2 else None
    result_type = linear_op.operation.results[0].type

    ctx = linear_op.operation.context

    with ir.Location.unknown(ctx), ir.InsertionPoint(linear_op.operation):
        # Matmul: output = input @ weight
        matmul_op = ir.Operation.create(
            "sf.matmul",
            results=[result_type],
            operands=[input_val, weight],
        )

        current = matmul_op.operation.results[0]

        # Add bias if present
        if bias is not None:
            add_op = ir.Operation.create(
                "sf.add",
                results=[result_type],
                operands=[current, bias],
            )
            current = add_op.operation.results[0]

    # Replace uses
    linear_op.operation.results[0].replace_all_uses_with(current)
    linear_op.operation.erase()


# ── LayerNorm decomposition ────────────────────────────────────────


def _decompose_layer_norm(ln_op: Any) -> None:
    """Decompose ``sf.layer_norm`` into element-wise primitive SF ops.

    sf.layer_norm(input, weight, bias) →
      sf.mean(input) → mean
      sf.sub(input, mean) → centered
      sf.pow(centered, 2) → sq
      sf.mean(sq) → var
      sf.rsqrt(var) → rstd
      sf.mul(centered, rstd) → normed
      sf.mul(normed, weight) → scaled
      sf.add(scaled, bias) → output
    """
    import mlir.ir as ir

    operands = list(ln_op.operands)
    if len(operands) < 3:
        _log.warning("LayerNorm op has fewer than 3 operands: %d", len(operands))
        return

    input_val, weight, bias = operands[0], operands[1], operands[2]
    result_type = ln_op.operation.results[0].type

    # Input shape determines reduction dimensions
    input_type = input_val.type
    if isinstance(input_type, ir.RankedTensorType):
        input_shape = list(input_type.shape)
        ndims = len(input_shape)
    else:
        ndims = 1

    ctx = ln_op.operation.context

    with ir.Location.unknown(ctx), ir.InsertionPoint(ln_op.operation):
        # 1. mean = sf.mean(input, dim=-1)
        mean_type = _mean_type_from(input_type, ndims)
        mean_op = ir.Operation.create(
            "sf.mean",
            results=[mean_type],
            operands=[input_val],
        )
        mean_result = mean_op.operation.results[0]

        # 2. centered = sf.sub(input, mean)
        #    mean broadcasts from [B, 1] or [B, T, 1] to [B, T, D]
        centered_op = ir.Operation.create(
            "sf.sub",
            results=[result_type],  # same element type, same shape as input
            operands=[input_val, mean_result],
        )
        centered = centered_op.operation.results[0]

        # 3. sq = sf.pow(centered, 2) — centered^2
        #    sf.pow is a binary op; we need a scalar-like constant for ^2.
        #    Use a sf.mul(centered, centered) instead since sf.pow is binary.
        #    Actually sf.pow(centered, centered) would give element-wise pow.
        #    For centered^2, we can use sf.mul(centered, centered).
        sq_op = ir.Operation.create(
            "sf.mul",
            results=[result_type],
            operands=[centered, centered],
        )
        sq = sq_op.operation.results[0]

        # 4. var = sf.mean(sq, dim=-1)  — compute variance of sq
        var_type = _mean_type_from(input_type, ndims)
        var_op = ir.Operation.create(
            "sf.mean",
            results=[var_type],
            operands=[sq],
        )
        var_result = var_op.operation.results[0]

        # 5. rstd = sf.rsqrt(var) — reciprocal sqrt
        rstd_op = ir.Operation.create(
            "sf.rsqrt",
            results=[var_type],
            operands=[var_result],
        )
        rstd = rstd_op.operation.results[0]

        # 6. normed = sf.mul(centered, rstd)
        #    rstd broadcasts from mean shape to input shape.
        normed_op = ir.Operation.create(
            "sf.mul",
            results=[result_type],
            operands=[centered, rstd],
        )
        normed = normed_op.operation.results[0]

        # 7. scaled = sf.mul(normed, weight)
        #    weight has shape [D] vs normed [B,T,D] → broadcasts
        scaled_op = ir.Operation.create(
            "sf.mul",
            results=[result_type],
            operands=[normed, weight],
        )
        scaled = scaled_op.operation.results[0]

        # 8. output = sf.add(scaled, bias)
        #    bias has shape [D] vs scaled [B,T,D] → broadcasts
        output_op = ir.Operation.create(
            "sf.add",
            results=[result_type],
            operands=[scaled, bias],
        )
        output = output_op.operation.results[0]

    # Replace uses
    ln_op.operation.results[0].replace_all_uses_with(output)
    ln_op.operation.erase()


def _mean_type_from(input_type: Any, ndims: int) -> Any:
    """Construct a type for the mean/variance output.

    For a given input type and number of dims, mean/variance over the last
    dim produces a tensor with the same rank but last dim = 1 (due to
    keepdim semantics).

    Example: tensor<?x?x768xf32> → tensor<?x?x1xf32>
    """
    import mlir.ir as ir

    if isinstance(input_type, ir.RankedTensorType):
        shape = list(input_type.shape)
        if ndims >= 1:
            shape[-1] = 1  # keepdim — last dim reduced to 1
        return ir.RankedTensorType.get(shape, input_type.element_type)
    if isinstance(input_type, ir.UnrankedTensorType):
        return ir.RankedTensorType.get([1], input_type.element_type)
    return ir.RankedTensorType.get([1], ir.F32Type.get())


# ── CLI / __main__ ──────────────────────────────────────────────────


def main() -> None:
    """Entry point: loads model.mlir, normalizes, saves to model.normalized.mlir."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="SF Normalize — decompose complex SF ops into primitives",
    )
    parser.add_argument(
        "--input", "-i",
        default="compiled/opt_125m_kv/model.mlir",
        help="Input model.mlir path",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path (default: input path with .normalized.mlir suffix)",
    )
    args = parser.parse_args()

    input_path = args.input
    if not os.path.isfile(input_path):
        _log.error("Input file not found: %s", input_path)
        sys.exit(1)

    output_path = args.output or input_path.replace(".mlir", ".normalized.mlir")

    with open(input_path) as f:
        mlir_text = f.read()

    _log.info("Loaded %s (%d bytes)", input_path, len(mlir_text))
    normalized = normalize_sf_mlir(mlir_text)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(normalized)

    # Count decomposed ops
    import re
    sdpa_count = mlir_text.count('"sf.scaled_dot_product_attention"')
    linear_count = mlir_text.count('"sf.linear"')
    ln_count = mlir_text.count('"sf.layer_norm"')
    sdpa_after = normalized.count('"sf.scaled_dot_product_attention"')
    linear_after = normalized.count('"sf.linear"')
    ln_after = normalized.count('"sf.layer_norm"')

    print(f"Saved normalized to {output_path} ({len(normalized)} bytes)")
    print(f"Decomposition summary:")
    print(f"  sf.scaled_dot_product_attention: {sdpa_count} → {sdpa_after} (expect 0)")
    print(f"  sf.linear:                       {linear_count} → {linear_after} (expect 0)")
    print(f"  sf.layer_norm:                   {ln_count} → {ln_after} (expect 0)")
    print(f"  Total ops decomposed: {sdpa_count + linear_count + ln_count}")
    print()
    print("New primitive ops introduced:")
    for op_name in ("sf.transpose", "sf.matmul", "sf.add", "sf.softmax",
                    "sf.mean", "sf.sub", "sf.mul", "sf.rsqrt", "sf.pow"):
        count = normalized.count(f'"{op_name}"')
        if count > 0:
            print(f"  {op_name}: {count}")


if __name__ == "__main__":
    main()
