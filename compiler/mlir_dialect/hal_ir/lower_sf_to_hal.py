"""Lower SF dialect ops to HAL IR JSON.

Reads normalized MLIR (SF primitives only), maps each ``sf.*`` op to a
``hal.execute @op_name`` entry, and produces ``hal_ir.json``.

The pass is **read-only** on the IR — it does not modify the MLIR module.

Op mapping table (all sf.* → hal.execute entries):

  ==========================  =================================
  ``sf.matmul``               ``hal.execute @matmul``
  ``sf.rms_norm``             ``hal.execute @rms_norm``
  ``sf.layer_norm``           ``hal.execute @layer_norm``
  ``sf.softmax``              ``hal.execute @softmax``
  ``sf.add/sub/mul/div/pow``  ``hal.execute @element_wise {kind}``
  ``sf.relu/gelu/silu/...``   ``hal.execute @element_wise {kind}``
  ``sf.tanh/exp/neg/...``     ``hal.execute @element_wise {kind}``
  ``sf.sqrt/rsqrt/cos/sin``   ``hal.execute @element_wise {kind}``
  ``sf.eq/ne/gt/lt/le``       ``hal.execute @compare {kind}``
  ``sf.logical_and``          ``hal.execute @compare {kind}``
  ``sf.view/expand``          ``hal.execute @reshape``
  ``sf.unsqueeze``            ``hal.execute @unsqueeze``
  ``sf.transpose/permute``    ``hal.execute @transpose``
  ``sf.slice``                ``hal.execute @slice``
  ``sf.cat``                  ``hal.execute @concat``
  ``sf.sum/mean``             ``hal.execute @reduce {kind}``
  ``sf.cumsum``               ``hal.execute @scan {kind}``
  ``sf.identity``             *skipped (SSA rename)*
  ``sf.ones_like/new_ones``   ``hal.execute @fill {value=1.0}``
  ``sf.arange``               ``hal.execute @fill {kind=arange}``
  ``sf.embedding``            ``hal.execute @gather``
  ``sf.index``                ``hal.execute @gather (indexed load)``
  ``sf.sym_size``             ``hal.execute @shape_of``
  ``sf.constant``             *inlined as JSON constant*
  ``sf.weight``               *recorded as weight reference*
  ``cache_write``             ``hal.execute @cache_write``  *(inserted)*
  ``cache_read``              ``hal.execute @cache_read``   *(inserted)*
  ==========================  =================================
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from compiler.mlir_dialect.hal_ir.hal_ir_builder import HalIRBuilder

_log = logging.getLogger(__name__)


# ── Top-level entry point ────────────────────────────────────────────


def lower_sf_to_hal(
    mlir_text: str,
    model_name: str = "model",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Lower normalized SF MLIR to HAL IR JSON structure.

    Args:
        mlir_text: The normalized MLIR text (SF primitives only).
        model_name: Name of the model (used in JSON output).
        metadata: Optional metadata dict from metadata.json for weight info.

    Returns:
        A dict representing the ``hal_ir.json`` structure.
    """
    builder = HalIRBuilder()
    builder.load_mlir(mlir_text)
    builder.set_model_name(model_name)
    if metadata:
        builder.load_metadata(metadata)
    return builder.build()


def lower_sf_to_hal_file(
    mlir_path: str | Path,
    output_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> str:
    """Lower a normalized MLIR file to a ``hal_ir.json`` file.

    Args:
        mlir_path: Path to the normalized MLIR file.
        output_path: Path for the output JSON. If None, derived from mlir_path.
        metadata_path: Optional path to metadata.json.

    Returns:
        The path to the written JSON file.
    """
    mlir_path = Path(mlir_path)

    # Determine model name from directory
    model_dir = mlir_path.parent
    model_name = model_dir.name

    # Read MLIR
    mlir_text = mlir_path.read_text()

    # Read metadata if available
    metadata: dict[str, Any] = {}
    if metadata_path:
        metadata = json.loads(Path(metadata_path).read_text())
    else:
        # Try sibling metadata.json
        meta_candidate = model_dir / "metadata.json"
        if meta_candidate.is_file():
            metadata = json.loads(meta_candidate.read_text())

    # Determine output path
    if output_path is None:
        output_path = model_dir / "generated" / "hal_ir.json"
    output_path = Path(output_path)

    # Build HAL IR
    result = lower_sf_to_hal(mlir_text, model_name, metadata)

    # Write JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))

    # Summary
    total_ops = sum(len(f["ops"]) for f in result["functions"])
    _log.info(
        "Lowered %d ops to hal.execute across %d functions → %s",
        total_ops,
        result["num_functions"],
        output_path,
    )
    print(
        f"[HAL IR] Lowered {total_ops} ops across {result['num_functions']} functions "
        f"→ {output_path}"
    )

    return str(output_path)


# ── Main entry point ─────────────────────────────────────────────────


def main() -> None:
    """CLI entry point: lower normalized MLIR to hal_ir.json."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Lower normalized SF MLIR to HAL IR JSON",
    )
    parser.add_argument(
        "mlir_path",
        nargs="?",
        default="compiled/opt_125m_kv/model.normalized.mlir",
        help="Path to normalized MLIR file",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON path (default: <model_dir>/generated/hal_ir.json)",
    )
    parser.add_argument(
        "--metadata", "-m",
        default=None,
        help="Path to metadata.json (default: auto from model_dir)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    result_path = lower_sf_to_hal_file(
        args.mlir_path,
        output_path=args.output,
        metadata_path=args.metadata,
    )
    print(f"Done: {result_path}")


if __name__ == "__main__":
    main()
