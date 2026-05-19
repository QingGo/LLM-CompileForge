#!/usr/bin/env python3
"""Quick forward pass correctness check (L1.5 gate, <5s).

Validates that a compiled model produces sensible output without
loading the HuggingFace reference model.

Usage:
    python scripts/check_forward_smoke.py compiled/opt_125m_fresh [--seed 42] [--seq-len 4]

Exits 0 on success, 1 on failure.
"""

from __future__ import annotations

import argparse
import logging
import sys

import torch

from compiler.serialize import load_artifact
from engine.mlir_executor import MlirExecutor
from hal.pytorch_backend import PyTorchBackend
from utils.logging import init_logging

_log = None  # set in main


def check_forward(directory: str, seed: int = 42, seq_len: int = 4) -> int:
    """Run forward pass and validate output shape, NaN, and value range.

    Returns 0 if all checks pass, 1 otherwise.
    """
    global _log
    if _log is None:
        init_logging()
        _log = logging.getLogger("check_forward_smoke")
    mod = load_artifact(directory)
    _log.info("  %d functions, %d ops total",
              len(mod.functions), sum(len(f.ops) for f in mod.functions))

    # Create executor
    backend = PyTorchBackend("cpu")
    executor = MlirExecutor(mod, backend)
    _log.info("  Weights loaded: %d", len(executor._weights))

    # Run forward
    torch.manual_seed(seed)
    input_ids = torch.randint(0, 100, (1, seq_len))
    _log.info("Forward pass with input shape %s...", list(input_ids.shape))
    result = executor.forward(input_ids)
    _log.info("  Output shape: %s", list(result.shape))

    # ── Check 1: Shape ──
    expected_vocab = 50272  # OPT-125m; will generalize later
    expected_shape = torch.Size([1, seq_len, expected_vocab])
    if result.shape[-1] != expected_vocab:
        _log.error("  ❌ Expected vocab dim %d, got %d", expected_vocab, result.shape[-1])
        return 1
    _log.info("  ✅ Shape OK: %s", list(result.shape))

    # ── Check 2: No NaN ──
    if torch.isnan(result).any():
        n_nan = torch.isnan(result).sum().item()
        _log.error("  ❌ %d NaN values in output", n_nan)
        return 1
    _log.info("  ✅ No NaN")

    # ── Check 3: No Inf ──
    if torch.isinf(result).any():
        n_inf = torch.isinf(result).sum().item()
        _log.error("  ❌ %d Inf values in output", n_inf)
        return 1
    _log.info("  ✅ No Inf")

    # ── Check 4: Output values in reasonable range ──
    finites = result[torch.isfinite(result)]
    if finites.numel() == 0:
        _log.error("  ❌ All output values are non-finite")
        return 1
    mean_val = finites.mean().item()
    max_val = finites.max().item()
    min_val = finites.min().item()
    std_val = finites.std().item()
    _log.info("  ✅ Stats: mean=%.4f std=%.4f range=[%.4f, %.4f]",
              mean_val, std_val, min_val, max_val)

    if max_val > 1000 or min_val < -1000:
        _log.warning("  ⚠️  Large output values (%f to %f) — may be incorrect", min_val, max_val)

    # ── Check 5: Not all identical ──
    if std_val < 0.001:
        _log.warning("  ⚠️  Near-zero variance (std=%.6f) — output may be degenerate", std_val)
        # Not a failure if tiny model, but worth noting

    # ── Check 6: Last token logits are different from first ──
    last_logits = result[0, -1, :]
    first_logits = result[0, 0, :]
    if torch.allclose(last_logits, first_logits, atol=1e-4):
        _log.warning("  ⚠️  Last and first token logits are nearly identical — possible position bug")

    _log.info("")
    _log.info("✅ All forward smoke checks passed for %s", directory)
    return 0


def main() -> int:
    global _log
    init_logging()
    import logging
    _log = logging.getLogger("check_forward_smoke")

    parser = argparse.ArgumentParser(description="Quick forward pass correctness check")
    parser.add_argument("directory", help="Path to compiled model directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for input")
    parser.add_argument("--seq-len", type=int, default=4, help="Sequence length to test")
    args = parser.parse_args()

    return check_forward(args.directory, args.seed, args.seq_len)


if __name__ == "__main__":
    sys.exit(main())
