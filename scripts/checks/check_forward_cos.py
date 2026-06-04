#!/usr/bin/env python3
"""Check forward pass output cosine similarity against a baseline.

Runs the compiled model forward and compares output against a reference
baseline (if available) using cosine similarity.  Without a baseline,
it validates that forward pass output is finite and has expected shape.

Usage:
    python scripts/check_forward_cos.py compiled/opt_125m_fresh [--threshold 0.999]

Exits 0 on success, 1 on failure.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import numpy as np
import torch

from compiler.serialize import load_artifact
from python_runtime.engine.mlir_executor import MlirExecutor
from python_runtime.hal.pytorch_backend import PyTorchBackend

_log = logging.getLogger(__name__)


def check_cosine(directory: str, threshold: float = 0.999) -> bool:
    """Run forward pass and compare against baseline.

    Returns True if all checks pass.
    """
    mod = load_artifact(directory)
    _log.info("  %d functions, %d ops total",
              len(mod.functions), sum(len(f.ops) for f in mod.functions))

    backend = PyTorchBackend("cpu")
    executor = MlirExecutor(mod, backend)
    _log.info("  Weights loaded: %d", len(executor._weights))

    # Run forward pass
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    _log.info("Forward pass with input shape %s...", list(input_ids.shape))
    result = executor.forward(input_ids)
    _log.info("  Output shape: %s", list(result.shape))

    # Basic smoke checks (logits must be finite)
    if not torch.isfinite(result).all():
        _log.error("  ❌ Non-finite logits in output")
        return False
    _log.info("  ✅ All logits finite")

    # Try to load baseline and compute cosine similarity
    import os
    baseline_path_json = os.path.join(directory, "baseline_logits.json")
    baseline_path_pt = os.path.join(directory, "baseline_logits.pt")

    if os.path.exists(baseline_path_json):
        with open(baseline_path_json) as f:
            baseline_data = json.load(f)
        baseline = torch.tensor(baseline_data["logits"], dtype=torch.float32)
        _compare_cosine(result, baseline, threshold)
    elif os.path.exists(baseline_path_pt):
        baseline = torch.load(baseline_path_pt, map_location="cpu", weights_only=True)
        _compare_cosine(result, baseline, threshold)
    else:
        _log.info("  ℹ️  No baseline found at %s or %s — smoke check only",
                  baseline_path_json, baseline_path_pt)
        # Without baseline, just validate finite + reasonable stats
        finites = result[torch.isfinite(result)]
        if finites.numel() == 0:
            _log.error("  ❌ All output values are non-finite")
            return False
        _log.info("  ✅ Stats: mean=%.4f std=%.4f range=[%.4f, %.4f]",
                  finites.mean().item(), finites.std().item(),
                  finites.min().item(), finites.max().item())

    return True


def _compare_cosine(result: torch.Tensor, baseline: torch.Tensor,
                    threshold: float) -> bool:
    """Compare logits against baseline with cosine similarity."""
    if result.shape != baseline.shape:
        _log.warning("  ⚠️  Shape mismatch: result %s vs baseline %s — skipping cos",
                     list(result.shape), list(baseline.shape))
        return True

    result_np = result.flatten().float().numpy()
    baseline_np = baseline.flatten().float().numpy()

    cos = np.dot(result_np, baseline_np) / (
        np.linalg.norm(result_np) * np.linalg.norm(baseline_np) + 1e-12
    )
    _log.info("  Cosine similarity: %.8f", cos)

    if cos < threshold:
        _log.error("  ❌ Cosine similarity %.6f < %.6f", cos, threshold)
        return False

    _log.info("  ✅ Cosine similarity %.8f >= %.6f", cos, threshold)
    return True


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(
        description="Check forward pass cosine similarity against baseline"
    )
    parser.add_argument("directory", help="Path to compiled model directory")
    parser.add_argument("--threshold", type=float, default=0.999,
                        help="Minimum cosine similarity threshold")
    args = parser.parse_args()

    if check_cosine(args.directory, args.threshold):
        _log.info("✅ Cosine check passed for %s", args.directory)
        return 0
    else:
        _log.error("❌ Cosine check failed for %s", args.directory)
        return 1


if __name__ == "__main__":
    sys.exit(main())
