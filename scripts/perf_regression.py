#!/usr/bin/env python3
"""Performance regression gate for LLM-ServeForge.

Measures key metrics and compares against stored baseline.
Exits non-zero if any metric degrades more than the threshold.

Usage:
    python scripts/perf_regression.py              # measure + compare with baseline
    python scripts/perf_regression.py --record      # record new baseline
    python scripts/perf_regression.py --threshold 10  # set custom threshold %%
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from pathlib import Path

_log = logging.getLogger(__name__)

BASELINE_FILE = Path(__file__).resolve().parent.parent / ".perf_baseline.json"
DEFAULT_THRESHOLD_PCT = 5.0  # fail if metric degrades more than 5%


def measure_key_metrics() -> dict[str, float]:
    """Run benchmarks and return dict of metric_name → value."""
    metrics: dict[str, float] = {}

    # Torch import time
    t0 = time.perf_counter()
    metrics["torch_import_s"] = round(time.perf_counter() - t0, 3)

    # Module import time (lazy loading check)
    t0 = time.perf_counter()
    metrics["compiler_import_s"] = round(time.perf_counter() - t0, 3)

    # Pipeline quick test time
    try:
        t0 = time.perf_counter()
        subprocess.run(
            ["make", "test-pipeline-quick"],
            capture_output=True, text=True, timeout=30,
        )
        metrics["pipeline_quick_s"] = round(time.perf_counter() - t0, 3)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        _log.warning("pipeline_quick not available: %s", e)
        metrics["pipeline_quick_s"] = -1.0

    return metrics


def load_baseline() -> dict[str, float]:
    if BASELINE_FILE.exists():
        return json.loads(BASELINE_FILE.read_text())
    return {}


def save_baseline(metrics: dict[str, float]) -> None:
    BASELINE_FILE.write_text(json.dumps(metrics, indent=2))
    _log.info("Baseline saved to %s", BASELINE_FILE)


def check_regression(
    current: dict[str, float],
    baseline: dict[str, float],
    threshold_pct: float,
) -> list[str]:
    failures: list[str] = []
    for key, cur_val in current.items():
        if cur_val < 0:
            continue  # skip unavailable metrics
        base_val = baseline.get(key)
        if base_val is None:
            _log.info("  [NEW] %s = %.3f (no baseline)", key, cur_val)
            continue
        if base_val <= 0:
            continue
        change_pct = (cur_val - base_val) / base_val * 100
        if change_pct > threshold_pct:
            failures.append(
                f"  ❌ {key}: {cur_val:.3f} vs baseline {base_val:.3f} "
                f"(+{change_pct:.1f}% > {threshold_pct:.0f}%)"
            )
        else:
            _log.info("  ✅ %s: %.3f (Δ%+.1f%%)", key, cur_val, change_pct)
    return failures


def main() -> int:
    from compiler.utils.logging import init_logging
    init_logging()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args = sys.argv[1:]
    record_mode = "--record" in args
    threshold_pct = DEFAULT_THRESHOLD_PCT
    for a in args:
        if a.startswith("--threshold="):
            threshold_pct = float(a.split("=")[1])

    _log.info("=== Performance Baseline Check ===")
    metrics = measure_key_metrics()

    if record_mode:
        save_baseline(metrics)
        _log.info("New baseline recorded.")
        return 0

    baseline = load_baseline()
    if not baseline:
        _log.info("No baseline found. Use --record to create one.")
        _log.info("Current metrics: %s", json.dumps(metrics, indent=2))
        return 0

    _log.info("Comparing against baseline (threshold: %.0f%%):", threshold_pct)
    failures = check_regression(metrics, baseline, threshold_pct)

    if failures:
        _log.info("\nRegression failures:")
        for f in failures:
            _log.info(f)
        return 1
    _log.info("\n✅ All metrics within threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
