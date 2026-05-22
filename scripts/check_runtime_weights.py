#!/usr/bin/env python3
"""Compare weights at the RUNTIME level between Python and Rust executors.

The existing check_weights.py compares artifact weights vs safetensors.
This script goes further: it compares the ACTUAL loaded weights used
by each executor during forward pass.

Key insight: the dump_weights binary loads weights through the EXACT same
path as the Rust runtime (SFCF blob → name_mapping → safetensors → f16→f32),
so its output represents what the Rust runtime ACTUALLY uses.

For Python side: MlirExecutor._weights contains the actual weight tensors
used during execution.

Usage:
    python scripts/check_runtime_weights.py --model compiled/opt_125m_fresh
    python scripts/check_runtime_weights.py --model compiled/opt_125m_fresh --rust-dump /tmp/dump_opt125m
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

_log = logging.getLogger(__name__)


from compiler.mlir_dialect.compile_utils import _short_shape
from scripts._cos import cosine_similarity

# ── Python runtime weight loader ──────────────────────────────


def get_python_runtime_weights(model_dir: str) -> dict[str, np.ndarray]:
    """Extract weights used by Python executor during runtime.

    Creates an MlirExecutor from the compiled artifact and returns
    the actual weight tensors it loaded, keyed by their original names
    (HF key format with dots, e.g. 'model.decoder.embed_tokens.weight').
    """
    _log.info("Loading Python runtime weights from %s", model_dir)
    # Lazy imports to avoid dependency at module load time
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from compiler.serialize import load_artifact
    from engine.mlir_executor import MlirExecutor
    from hal.pytorch_backend import PyTorchBackend

    artifact = load_artifact(model_dir)
    backend = PyTorchBackend("cpu")
    executor = MlirExecutor(artifact, backend)

    weights: dict[str, np.ndarray] = {}
    for name, tensor in executor._weights.items():
        # Avoid duplicates with compiled-name aliases — prefer HF-key name
        # But keep both so reconciliation can choose the right key
        weights[name] = tensor.cpu().numpy().astype(np.float32)

    _log.info("Loaded %d Python runtime weight entries", len(weights))
    return weights


# ── Rust runtime weight loader ────────────────────────────────


def get_rust_runtime_weights(rust_dump_dir: str) -> dict[str, np.ndarray]:
    """Load Rust runtime weights from dump_weights output.

    The dump_weights binary loads weights through the exact same path
    as the Rust runtime (SFCF blob → name_mapping → safetensors → f16→f32),
    so its output represents what the Rust runtime ACTUALLY uses.

    Returns dict keyed by compiled name (underscores, e.g. 'lm_head_weight').
    """
    dump = Path(rust_dump_dir)
    if not dump.is_dir():
        _log.warning("Rust dump directory not found: %s", rust_dump_dir)
        return {}

    index_path = dump / "weights_index.json"
    if index_path.exists():
        with open(index_path) as f:
            index: dict[str, dict] = json.load(f)
    else:
        _log.warning("weights_index.json not found, scanning .npy files")
        index = {p.stem: {"file": p.name} for p in dump.glob("*.npy")}

    result: dict[str, np.ndarray] = {}
    for name, info in index.items():
        file_name = info.get("file", f"{name}.npy")
        file_path = dump / file_name
        if file_path.exists():
            try:
                arr = np.load(str(file_path))
                result[name] = arr.astype(np.float32)
            except Exception as e:
                _log.warning("Failed to load %s: %s", file_path, e)
        else:
            _log.warning("Rust weight file not found: %s", file_path)

    _log.info("Loaded %d Rust runtime weights", len(result))
    return result


# ── Naming reconciliation ──────────────────────────────────────


def _build_python_compiled_view(
    py_weights: dict[str, np.ndarray],
    hf_key_map: dict[str, str],
    tied_weights: dict[str, str],
) -> dict[str, np.ndarray]:
    """Build a view of Python weights keyed by compiled name.

    Python executor._weights has keys in HF-key format (dots).
    Rust dump uses compiled names (underscores).

    This function adds the necessary aliases so that every Rust
    compiled name can be looked up directly in the result.
    """
    view = dict(py_weights)

    # 1. Add compiled_name → tensor from hf_key_map
    #    hf_key_map: compiled_name (underscores) → hf_key (dots)
    for compiled_name, hf_key in hf_key_map.items():
        if hf_key in view and compiled_name not in view:
            view[compiled_name] = view[hf_key]

    # 2. Add tied weight aliases (both directions)
    #    tied_weights: alias → primary (alias uses primary's tensor)
    for alias, primary in tied_weights.items():
        if alias in view and primary not in view:
            view[primary] = view[alias]
        elif primary in view and alias not in view:
            view[alias] = view[primary]

    # 3. Add bare constant names (strip function prefix like 'main_0.')
    for key in list(view.keys()):
        if "." in key:
            bare = key.split(".", 1)[1]
            if bare not in view:
                view[bare] = view[key]

    return view


# ── Shape validation ──────────────────────────────────────────


def _check_shapes(
    rs_name: str,
    py_t: np.ndarray,
    rs_t: np.ndarray,
) -> str:
    """Compare shapes and return a label: 'OK', 'MISMATCH', or 'BROADCAST'."""
    if py_t.shape == rs_t.shape:
        return "OK"
    # Check if shapes are broadcast-compatible (same num elements)
    if py_t.size == rs_t.size:
        return "BROADCAST"
    return "MISMATCH"


# ── Main ──────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare runtime weights between Python and Rust executors"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to compiled model directory (e.g. compiled/opt_125m_fresh)",
    )
    parser.add_argument(
        "--rust-dump",
        default=None,
        help="Path to dump_weights output directory with .npy and weights_index.json",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.999,
        help="Cosine similarity threshold (default: 0.999)",
    )
    args = parser.parse_args()

    model_dir = Path(args.model)
    if not model_dir.is_dir():
        print(f"Model directory not found: {model_dir}")
        print("Skipping runtime weight check (exit 0)")
        return 0

    # Load metadata
    meta_path = model_dir / "metadata.json"
    if not meta_path.exists():
        print(f"metadata.json not found: {meta_path}")
        print("Skipping runtime weight check (exit 0)")
        return 0

    with open(meta_path) as f:
        meta = json.load(f)

    hf_key_map: dict[str, str] = meta.get("hf_key_map", {})
    tied_weights: dict[str, str] = {}
    tw = meta.get("tied_weights", {})
    if tw:
        tied_weights = tw
    else:
        ws = meta.get("weight_source", {})
        tied_weights = ws.get("tied_weights", {})

    print("=" * 60)
    print("RUNTIME WEIGHT COMPARISON")
    print("=" * 60)
    print(f"Model:        {model_dir}")
    print(f"Threshold:    {args.threshold}")
    print(f"hf_key_map:   {len(hf_key_map)} entries")
    print(f"tied_weights: {len(tied_weights)} entries")
    print()

    # ── Step 1: Load Python runtime weights ──────────────────
    print("─── [1/3] Python Runtime Weights ───────────────────")
    try:
        py_raw = get_python_runtime_weights(str(model_dir))
        print(f"  Loaded {len(py_raw)} weight entries from MlirExecutor._weights")
    except Exception as e:
        print(f"  ❌ Error loading Python runtime: {e}")
        return 1

    # Build unified compiled-name view
    py_view = _build_python_compiled_view(py_raw, hf_key_map, tied_weights)

    # Count what we have
    py_compiled = {k for k in py_view if "_const_" not in k}
    py_consts = {k for k in py_view if "_const_" in k}
    print(f"  Compiled-name view: {len(py_compiled)} weights + {len(py_consts)} constants")
    print()

    # ── Step 2: Load Rust runtime weights ────────────────────
    print("─── [2/3] Rust Runtime Weights ─────────────────────")
    has_rust = bool(args.rust_dump)
    rs_w: dict[str, np.ndarray] = {}
    if has_rust:
        rs_w = get_rust_runtime_weights(args.rust_dump)
        rs_weights = {k for k in rs_w if "_const_" not in k}
        rs_consts = {k for k in rs_w if "_const_" in k}
        print(f"  Loaded {len(rs_w)} weights ({len(rs_weights)} weights + {len(rs_consts)} constants)")
    else:
        print("  No --rust-dump provided; showing Python weights only")
        print("  Re-run with --rust-dump for comparison.")
        print()
        # Still print Python weight summary
        _print_python_summary(py_view)

        # Also test cos between function-prefixed and bare constants
        _print_constant_self_check(py_raw)
        return 0

    print()

    # ── Step 3: Compare ──────────────────────────────────────
    print("─── [3/3] Comparison ────────────────────────────────")

    # Find common weights by compiled name
    rs_names = set(rs_w.keys())
    py_compiled_names = set(py_view.keys())
    common = sorted(rs_names & py_compiled_names)
    missing_in_py = sorted(rs_names - py_compiled_names)
    extra_in_py = sorted(py_compiled_names - rs_names)

    print(f"  Rust weights:         {len(rs_names)}")
    print(f"  Python (compiled):    {len(py_compiled_names)}")
    print(f"  Common:               {len(common)}")
    if missing_in_py:
        print(f"  ⚠ Missing in Python:  {len(missing_in_py)} ({', '.join(missing_in_py[:5])}...)")
    if extra_in_py:
        print(f"  Python-only (no Rust): {len(extra_in_py)}")
    print()

    if not common:
        print("No common weights to compare")
        return 0

    # ── Report table ─────────────────────────────────────────
    name_width = max(max(len(n) for n in common), 36) + 2
    sep = "─" * (name_width + 4 + 14 + 14 + 10 + 6)

    header = (
        f"{'Weight':<{name_width}} "
        f"{'Shape':>14}  "
        f"{'cos(Py,Rs)':>10}  "
        f"Status"
    )
    print(header)
    print(sep)

    failed = 0
    for name in common:
        py_t = py_view[name]
        rs_t = rs_w[name]
        sim = cosine_similarity(py_t, rs_t)
        shape_label = _check_shapes(name, py_t, rs_t)

        if py_t.shape == rs_t.shape:
            shape_str = _short_shape(py_t.shape)
        else:
            shape_str = f"{_short_shape(py_t.shape)}→{_short_shape(rs_t.shape)} ({shape_label})"

        status = "OK" if sim >= args.threshold else "FAIL"
        if sim < args.threshold:
            failed += 1

        print(
            f"{name:<{name_width}} "
            f"{shape_str:>14}  "
            f"{sim:>10.8f}  "
            f"{status}"
        )

    print(sep)

    # Summary
    total = len(common)
    passed = total - failed
    if failed == 0:
        print(f"  ✅ All {total} runtime weights match (cos >= {args.threshold})")
    else:
        print(f"  ❌ {passed}/{total} passed, {failed} FAILED (threshold >= {args.threshold})")

    # Also print constant self-check
    _print_constant_self_check(py_raw)

    return 1 if failed > 0 else 0


# ── Helper: Python-only summary ──────────────────────────────


def _print_python_summary(py_view: dict[str, np.ndarray]) -> None:
    """Print summary of Python runtime weights (when no Rust dump)."""
    weights = {k: v for k, v in py_view.items() if "_const_" not in k}
    consts = {k: v for k, v in py_view.items() if "_const_" in k}

    print("Python runtime weight summary:")
    print(f"  Total entries: {len(py_view)}")
    print(f"  Named weights: {len(weights)}")
    print(f"  Constants:     {len(consts)}")

    # Show min/max/mean for a few weights
    n_show = min(5, len(weights))
    for _i, (name, tensor) in enumerate(sorted(weights.items())[:n_show]):
        arr = tensor.ravel()
        print(
            f"  {name}: shape={list(tensor.shape)}, "
            f"first={float(arr[0]):.6f}, mean={float(arr.mean()):.4f}, "
            f"std={float(arr.std()):.4f}"
        )


def _print_constant_self_check(py_raw: dict[str, np.ndarray]) -> None:
    """Check that function-prefixed constants match bare constants."""
    bare_by_name: dict[str, np.ndarray] = {}
    prefixed: dict[str, list[tuple[str, np.ndarray]]] = {}

    for name, tensor in py_raw.items():
        if "_const_" in name:
            if "." in name:
                bare = name.split(".", 1)[1]
                prefixed.setdefault(bare, []).append((name, tensor))
            else:
                bare_by_name[name] = tensor

    if not prefixed:
        return

    mismatches = 0
    for bare_name, variants in sorted(prefixed.items()):
        if bare_name in bare_by_name:
            ref = bare_by_name[bare_name]
            for vname, vtensor in variants:
                sim = cosine_similarity(ref, vtensor)
                if sim < 0.999:
                    _log.warning(
                        "Constant self-check: %s vs %s cos=%f",
                        vname, bare_name, sim,
                    )
                    mismatches += 1
    if mismatches == 0:
        _log.info("Constant self-check: all function-prefixed constants match bare versions")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    sys.exit(main())
