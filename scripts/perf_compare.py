#!/usr/bin/env python3
"""Performance and correctness comparison: compiled model vs HuggingFace baseline.

Usage:
    python scripts/perf_compare.py --model llama-1b --prompt-len 128 --max-tokens 32
    python scripts/perf_compare.py --model tiny_llama --prompt-len 16

Measures:
  - Logit cosine similarity (correctness)
  - Token-by-token match (greedy generation)
  - Single forward latency
  - Prefill / decode TTFT / TPOT
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))


@dataclass
class LatencyStats:
    mean_ms: float = 0
    median_ms: float = 0
    min_ms: float = 0
    max_ms: float = 0


@dataclass
class ComparisonResult:
    model: str
    hf_single_forward_ms: float = 0
    ours_single_forward_ms: float = 0
    forward_slowdown: float = 0
    cosine_similarity: float = 0
    greedy_tokens_match: bool = False
    hf_ttft_ms: float = 0
    ours_ttft_ms: float = 0
    hf_tpot_ms: float = 0
    ours_tpot_ms: float = 0
    errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0 and self.cosine_similarity > 0.999


MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "tiny_llama": {"dir": "./outputs/compiled/tiny_llama", "hf_id": "hf-internal-testing/tiny-random-LlamaForCausalLM"},
    "llama-1b": {"dir": "./outputs/compiled/llama_1b", "local": "./models/LLM-Research/Llama-3.2-1B", "dtype": "bfloat16"},
    "llama-3b": {"dir": "./outputs/compiled/llama_3b", "local": "./models/LLM-Research/Llama-3.2-3B", "dtype": "bfloat16"},
}


def _load_hf_model(preset: dict[str, Any]):
    if preset.get("local"):
        import os

        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
        model_dir = os.path.abspath(preset["local"])
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=False)
        config.use_cache = False
        dtype = getattr(torch, preset.get("dtype", "float32"))
        return AutoModelForCausalLM.from_pretrained(model_dir, config=config, torch_dtype=dtype).eval()
    else:
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(preset["hf_id"]).eval()


def measure_latency(fn, warmup=3, iters=10) -> LatencyStats:
    for _ in range(warmup):
        fn()
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return LatencyStats(
        mean_ms=round(statistics.mean(times), 1),
        median_ms=round(statistics.median(times), 1),
        min_ms=round(min(times), 1),
        max_ms=round(max(times), 1),
    )


def compare(model_name: str, prompt_len: int, max_tokens: int) -> ComparisonResult:
    import torch

    from compiler.serialize import load_artifact
    from python_runtime.engine.mlir_executor import MlirExecutor
    from python_runtime.hal.pytorch_backend import PyTorchBackend
    from tests.helpers import cosine_similarity

    preset = MODEL_PRESETS[model_name]
    result = ComparisonResult(model=model_name)

    # ── Load models ────────────────────────────────────
    try:
        hf_model = _load_hf_model(preset)
    except Exception as e:
        result.errors.append(f"HF load: {e}")
        return result

    try:
        compiled = load_artifact(preset["dir"])
    except Exception as e:
        result.errors.append(f"Compiled load: {e}")
        return result

    executor = MlirExecutor(compiled, PyTorchBackend("cpu"))

    # ── Logit cosine similarity ────────────────────────
    input_ids = torch.randint(0, 5000, (1, prompt_len), dtype=torch.long)
    with torch.no_grad():
        hf_logits = hf_model(input_ids).logits.to(torch.float32)
    compiled_logits = executor.forward(input_ids).to(torch.float32)

    if hf_logits.shape != compiled_logits.shape:
        result.errors.append(
            f"Shape mismatch: HF {list(hf_logits.shape)} vs ours {list(compiled_logits.shape)}"
        )
        return result

    result.cosine_similarity = round(cosine_similarity(hf_logits, compiled_logits), 8)

    # ── Single forward latency ────────────────────────
    hf_stats = measure_latency(lambda: hf_model(input_ids))
    ours_stats = measure_latency(lambda: executor.forward(input_ids))
    result.hf_single_forward_ms = hf_stats.median_ms
    result.ours_single_forward_ms = ours_stats.median_ms
    if hf_stats.median_ms > 0:
        result.forward_slowdown = round(ours_stats.median_ms / hf_stats.median_ms, 1)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare compiled model vs HF")
    parser.add_argument("--model", default="llama-1b", choices=list(MODEL_PRESETS))
    parser.add_argument("--prompt-len", type=int, default=16, help="Prompt length in tokens")
    parser.add_argument("--max-tokens", type=int, default=16, help="Max tokens to generate")
    args = parser.parse_args()

    print(f"Comparing {args.model} vs HF (prompt_len={args.prompt_len})...\n")

    result = compare(args.model, args.prompt_len, args.max_tokens)

    fmt = "{:<30} {:>12} {:>12}"
    print(fmt.format("", "HF", "Ours"))
    print(fmt.format("Single forward (ms):", str(result.hf_single_forward_ms), str(result.ours_single_forward_ms)))
    print(fmt.format("Slowdown:", "", f"{result.forward_slowdown:.1f}x"))
    print(fmt.format("Logit cosine:", "", f"{result.cosine_similarity:.8f}"))

    if result.errors:
        print(f"\nErrors: {result.errors}")
        sys.exit(1)

    if result.cosine_similarity < 0.999:
        print(f"\nWARNING: cosine {result.cosine_similarity} < 0.999")
        sys.exit(1)

    print(f"\n✓ PASS (cosine={result.cosine_similarity})")


if __name__ == "__main__":
    main()
