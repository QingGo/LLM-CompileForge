#!/usr/bin/env python3
"""Three-way weight consistency verification.

Compares weights from three sources for a compiled model:
  1. Ground truth (original saved model file: safetensors or PyTorch bin)
  2. Python artifact (compiled MLIR artifact loaded via load_artifact)
  3. Rust-dumped weights (.npy files from Rust forward pass)

Usage:
    python scripts/check_weights.py --model compiled/opt_125m_fresh
    python scripts/check_weights.py --model compiled/opt_125m_fresh --rust-dump-dir /tmp/dump_opt125m
    python scripts/check_weights.py --model compiled/opt_125m_fresh --threshold 0.99
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


def _load_artifact_lazy(model_dir: str):
    """Lazy import to avoid torch dependency at module load time."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from compiler.serialize import load_artifact
    return load_artifact(model_dir)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two flattened arrays."""
    a_f = a.flatten().astype(np.float64)
    b_f = b.flatten().astype(np.float64)
    dot = np.dot(a_f, b_f)
    norm_a = np.linalg.norm(a_f)
    norm_b = np.linalg.norm(b_f)
    if norm_a == 0.0 and norm_b == 0.0:
        return 1.0
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _short_shape(shape: tuple[int, ...]) -> str:
    return " [" + ", ".join(str(s) for s in shape) + "]"


# ── Source loaders ──────────────────────────────────────────────


def load_ground_truth(
    meta: dict,
    hf_key_map: dict[str, str],
) -> dict[str, np.ndarray]:
    """Load ground truth weights from the original saved model file.

    Maps compiled names -> tensors via hf_key_map.
    Supports safetensors and pytorch_bin formats.
    """
    ws = meta.get("weight_source", {})
    ws_path = ws.get("path", "")
    fmt = ws.get("format", "")

    if not ws_path or not os.path.isfile(ws_path):
        _log.warning("Ground truth source file not found: %s", ws_path)
        return {}

    _log.info("Loading ground truth from %s (format=%s)", ws_path, fmt)

    result: dict[str, np.ndarray] = {}

    if fmt == "safetensors":
        import safetensors
        with safetensors.safe_open(ws_path, framework="np") as st:
            for compiled_name, hf_key in hf_key_map.items():
                if hf_key in st.keys():
                    result[compiled_name] = st.get_tensor(hf_key).astype(np.float32)
                else:
                    _log.warning("HF key %r not found in safetensors", hf_key)
    elif fmt == "pytorch_bin":
        import torch
        raw: dict[str, torch.Tensor] = torch.load(
            ws_path, map_location="cpu", weights_only=True
        )
        # raw keys are HF keys (e.g. "model.decoder.embed_tokens.weight")
        for compiled_name, hf_key in hf_key_map.items():
            if hf_key in raw:
                result[compiled_name] = raw[hf_key].cpu().numpy().astype(np.float32)
            else:
                _log.warning("HF key %r not found in pytorch_bin", hf_key)
    elif fmt == "safetensors_sharded":
        _log.warning("Sharded safetensors not yet supported")
        return {}
    else:
        _log.warning("Unknown weight_source format %r", fmt)
        return {}

    return result


def load_python_artifact(model_dir: str) -> dict[str, np.ndarray]:
    """Load weights from compiled Python artifact.

    Returns a dict of HF-key-name -> tensor.
    The Python artifact stores weights with their original HF key names
    (e.g. "model.decoder.embed_tokens.weight"), not compiled names.
    """
    _log.info("Loading Python artifact from %s", model_dir)
    artifact = _load_artifact_lazy(str(model_dir))

    all_weights: dict[str, np.ndarray] = {}
    for func in artifact.functions:
        for name, tensor in func.weights.items():
            if name not in all_weights:
                all_weights[name] = tensor.cpu().numpy().astype(np.float32)

    _log.info("Loaded %d unique weights from Python artifact", len(all_weights))
    return all_weights


def load_rust_weights(rust_dump_dir: str) -> dict[str, np.ndarray]:
    """Load Rust-dumped weights from .npy files.

    Rust uses compiled names (underscores, no dots).
    Uses weights_index.json for metadata.
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

    _log.info("Loaded %d weights from Rust dump", len(result))
    return result


def _resolve_constant_name(
    py_name: str,
    rs_by_compiled: dict[str, np.ndarray],
) -> str | None:
    """Try to match a Python constant weight name to a Rust name.

    Python stores constants with function prefix (e.g. 'main_0._const_7'),
    while Rust stores them bare (e.g. '_const_7').  Tries stripping
    leading 'main_N.' prefixes to find a match.
    """
    if py_name in rs_by_compiled:
        return py_name
    # Strip function prefix: main_0._const_7 -> _const_7
    if "." in py_name:
        bare = py_name.split(".", 1)[1]
        if bare in rs_by_compiled:
            return bare
    return None


def _build_unified_weights(
    py_w: dict[str, np.ndarray],
    gt_w: dict[str, np.ndarray],
    rs_w: dict[str, np.ndarray],
    hf_key_map: dict[str, str],
    name_mapping: dict[str, str],
) -> tuple[
    dict[str, np.ndarray],  # py_by_compiled
    dict[str, np.ndarray],  # gt_by_compiled
    dict[str, np.ndarray],  # rs_by_compiled
    list[str],               # sorted weight names (compiled)
]:
    """Reconcile naming across three sources into a unified compiled-name view.

    Naming conventions by source:
      - Ground truth: compiled names (via hf_key_map lookup) — already unified
      - Python artifact: HF key names with dots (e.g. 'model.decoder...')
      - Rust dump: compiled names (underscores, e.g. 'model_decoder_...')

    Builds reverse mapping: HF key -> compiled name, then maps each source
    into the compiled-name namespace.
    """
    # Build reverse mapping: HF key -> compiled name
    hf_key_to_compiled: dict[str, str] = {}
    # Primary: hf_key_map (compiled_name -> hf_key)
    for compiled_name, hf_key in hf_key_map.items():
        hf_key_to_compiled[hf_key] = compiled_name
    # Also try name_mapping (same direction)
    for compiled_name, hf_key in name_mapping.items():
        hf_key_to_compiled.setdefault(hf_key, compiled_name)

    # Map Python artifact (HF key names) -> compiled names
    py_by_compiled: dict[str, np.ndarray] = {}
    for py_name, tensor in py_w.items():
        if py_name in hf_key_to_compiled:
            # HF key name -> compiled name
            py_by_compiled[hf_key_to_compiled[py_name]] = tensor
        else:
            # Not in hf key map (e.g. constants with function prefix)
            rs_match = _resolve_constant_name(py_name, rs_w)
            if rs_match:
                py_by_compiled[rs_match] = tensor
            else:
                # Keep original name but strip dots for display
                display_name = py_name.replace(".", "_")
                py_by_compiled[display_name] = tensor

    # Ground truth is already compiled-name-keyed
    gt_by_compiled: dict[str, np.ndarray] = dict(gt_w)

    # Rust is already compiled-name-keyed
    rs_by_compiled: dict[str, np.ndarray] = dict(rs_w)

    # Build unified name list
    all_names: set[str] = set()
    all_names |= set(py_by_compiled.keys())
    all_names |= set(gt_by_compiled.keys())
    all_names |= set(rs_by_compiled.keys())
    sorted_names = sorted(all_names)

    return py_by_compiled, gt_by_compiled, rs_by_compiled, sorted_names


def print_report(
    weight_names: list[str],
    py_w: dict[str, np.ndarray],
    gt_w: dict[str, np.ndarray],
    rs_w: dict[str, np.ndarray],
    threshold: float,
    has_gt: bool,
    has_rs: bool,
) -> int:
    """Print weight consistency report table.

    Returns the number of weights that FAIL the threshold check.
    """
    name_width = max(len(n) for n in weight_names) if weight_names else 40
    name_width = max(name_width, 36) + 2

    sep = "─" * (name_width + 4 + 14 * 3 + 12)

    if has_gt and has_rs:
        hdr = f"{'Weight':<{name_width}} {'Shape':>14}  "
        hdr += f"{'cos(Py,GT)':>10} {'cos(Rs,GT)':>10} {'cos(Py,Rs)':>10}  Status"
        print(hdr)
    elif has_gt:
        print(f"{'Weight':<{name_width}} {'Shape':>14}  "
              f"{'cos(Py,GT)':>10}  Status")
    else:
        print(f"{'Weight':<{name_width}} {'Shape':>14}  "
              f"{'cos(Py,Rs)':>10}  Status")
    print(sep)

    failed = 0
    for name in weight_names:
        py_t = py_w.get(name)
        gt_t = gt_w.get(name)
        rs_t = rs_w.get(name)
        shape = _short_shape(py_t.shape) if py_t is not None else " N/A"

        all_ok = True

        if has_gt and py_t is not None and gt_t is not None:
            c_py_gt = cosine_similarity(py_t, gt_t)
            if c_py_gt < threshold:
                all_ok = False
        else:
            c_py_gt = float("nan")

        if has_rs and gt_t is not None and rs_t is not None:
            c_rs_gt = cosine_similarity(rs_t, gt_t)
            if c_rs_gt < threshold:
                all_ok = False
        else:
            c_rs_gt = float("nan")

        if py_t is not None and rs_t is not None:
            c_py_rs = cosine_similarity(py_t, rs_t)
            if c_py_rs < threshold:
                all_ok = False
        else:
            c_py_rs = float("nan")

        status = "OK " if all_ok else "FAIL"
        if not all_ok:
            failed += 1

        if has_gt and has_rs:
            print(
                f"{name:<{name_width}} {shape:>14}  "
                f"{c_py_gt:>10.6f} {c_rs_gt:>10.6f} {c_py_rs:>10.6f}  {status}"
            )
        elif has_gt:
            print(
                f"{name:<{name_width}} {shape:>14}  "
                f"{c_py_gt:>10.6f}  {status}"
            )
        else:
            print(
                f"{name:<{name_width}} {shape:>14}  "
                f"{c_py_rs:>10.6f}  {status}"
            )

    print(sep)

    total = len(weight_names)
    passed = total - failed
    if failed == 0:
        print(f"All {total} weights passed (threshold >= {threshold})")
    else:
        print(
            f"{passed}/{total} passed, {failed} FAILED (threshold >= {threshold})"
        )

    missing_py = [n for n in weight_names if n not in py_w]
    missing_gt = [n for n in weight_names if n not in gt_w and has_gt]
    missing_rs = [n for n in weight_names if n not in rs_w and has_rs]

    if missing_py:
        _log.warning("Weights missing from Python artifact: %d", len(missing_py))
    if missing_gt:
        _log.warning("Weights missing from ground truth: %d", len(missing_gt))
    if missing_rs:
        _log.warning("Weights missing from Rust dump: %d", len(missing_rs))

    return failed


def check_shape_consistency(
    weight_names: list[str],
    py_w: dict[str, np.ndarray],
    gt_w: dict[str, np.ndarray],
    rs_w: dict[str, np.ndarray],
    has_gt: bool,
    has_rs: bool,
) -> int:
    """Check that shapes match across sources. Returns mismatch count."""
    mismatches = 0
    for name in weight_names:
        shapes: dict[str, tuple[int, ...]] = {}
        if name in py_w:
            shapes["Py"] = py_w[name].shape
        if has_gt and name in gt_w:
            shapes["GT"] = gt_w[name].shape
        if has_rs and name in rs_w:
            shapes["Rs"] = rs_w[name].shape

        if len(shapes) < 2:
            continue
        ref_shape = next(iter(shapes.values()))
        ref_label = next(iter(shapes.keys()))
        for label, shape in shapes.items():
            if shape != ref_shape:
                _log.warning(
                    "Shape mismatch for %r: %s=%s vs %s=%s",
                    name, label, shape, ref_label, ref_shape,
                )
                mismatches += 1
    return mismatches


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Three-way weight consistency verification"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to compiled model directory (e.g. compiled/opt_125m_fresh)",
    )
    parser.add_argument(
        "--rust-dump-dir",
        default=None,
        help="Path to Rust weight dump directory with .npy files",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.999,
        help="Cosine similarity threshold (default: 0.999)",
    )
    args = parser.parse_args()

    model_dir = Path(args.model)

    # Graceful skip if model directory doesn't exist
    if not model_dir.is_dir():
        print(f"Model directory not found: {model_dir}")
        print("Skipping weight check (exit 0)")
        return 0

    # Load metadata
    meta_path = model_dir / "metadata.json"
    if not meta_path.exists():
        print(f"metadata.json not found: {meta_path}")
        print("Skipping weight check (exit 0)")
        return 0

    with open(meta_path) as f:
        meta = json.load(f)

    hf_key_map: dict[str, str] = meta.get("hf_key_map", {})
    name_mapping: dict[str, str] = meta.get("weight_source", {}).get("name_mapping", {})

    print("=== Weight Consistency Report ===")
    print(f"Model: {model_dir}")
    print(f"Threshold: {args.threshold}")

    sources: list[str] = []
    gt_w: dict[str, np.ndarray] = {}
    rs_w: dict[str, np.ndarray] = {}
    has_gt = False
    has_rs = False

    # Ground truth
    gt_raw = load_ground_truth(meta, hf_key_map)
    if not gt_raw and name_mapping:
        _log.info("Retrying ground truth with name_mapping")
        gt_raw = load_ground_truth(meta, name_mapping)
    if gt_raw:
        sources.append("ground_truth")
        has_gt = True
        gt_w = gt_raw
    else:
        sources.append("(no ground truth)")

    # Python artifact
    try:
        py_raw = load_python_artifact(str(model_dir))
        sources.append("python_artifact")
    except FileNotFoundError as e:
        print(f"Python artifact not found: {e}")
        print("Skipping weight check (exit 0)")
        return 0
    except Exception as e:
        print(f"Error loading Python artifact: {e}")
        return 1

    # Rust dump
    has_rs = bool(args.rust_dump_dir)
    if has_rs:
        rs_raw = load_rust_weights(args.rust_dump_dir)
        if rs_raw:
            sources.append("rust_dump")
            rs_w = rs_raw
        else:
            _log.warning("No Rust weights loaded; skipping Rust comparison")
            has_rs = False
            sources.append("(rust dump empty)")

    print(f"Sources: {', '.join(sources)}")
    print()

    # Reconcile naming across sources into compiled-name namespace
    py_by_compiled, gt_by_compiled, rs_by_compiled, weight_names = (
        _build_unified_weights(py_raw, gt_w, rs_w, hf_key_map, name_mapping)
    )

    if not weight_names:
        print("No weights found for comparison")
        return 0

    # Print report
    failed = print_report(
        weight_names, py_by_compiled, gt_by_compiled, rs_by_compiled,
        args.threshold, has_gt, has_rs,
    )

    # Check shape consistency
    mismatches = check_shape_consistency(
        weight_names, py_by_compiled, gt_by_compiled, rs_by_compiled,
        has_gt, has_rs,
    )
    if mismatches:
        _log.warning("Total shape mismatches: %d", mismatches)

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s"
    )
    sys.exit(main())
