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
    from compiler.hal.sf_normalize import normalize_sf_mlir
    normalized = normalize_sf_mlir(mlir_text)
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

from compiler.hal.sf_decompose import (
    _decompose_layer_norm,
    _decompose_linear,
    _decompose_sdpa,
)
from compiler.mlir_dialect.lowering.compile_utils import _setup_mlir_path

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


def normalize_sf_module(module: Any) -> dict[str, int]:
    """In-place normalize an ``ir.Module``.  Returns stats dict."""
    return _normalize_module(module)


# ── Core pass logic ────────────────────────────────────────────────


def _normalize_module(module: Any) -> dict[str, int]:
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
        default="outputs/compiled/opt_125m_kv/model.mlir",
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
    sdpa_count = mlir_text.count('"sf.scaled_dot_product_attention"')
    linear_count = mlir_text.count('"sf.linear"')
    ln_count = mlir_text.count('"sf.layer_norm"')
    sdpa_after = normalized.count('"sf.scaled_dot_product_attention"')
    linear_after = normalized.count('"sf.linear"')
    ln_after = normalized.count('"sf.layer_norm"')

    print(f"Saved normalized to {output_path} ({len(normalized)} bytes)")
    print("Decomposition summary:")
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
