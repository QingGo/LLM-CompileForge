"""Decode baseline — KV vs legacy generate() decode throughput.

Measures the Phase P1 fix (LLMEngine.generate() KV-cache decode) against
the legacy full-sequence recompute path on the same compiled artifact:

  * prefill_ms      — time of the prefill step(s), before the first output token
  * decode_tokens_s — output tokens per second across the decode phase
  * wall_ms         — end-to-end wall clock (prefill + decode)

Paths compared per (prompt_len, max_tokens) config:
  * "python_kv"     — LLMEngine with cache-manager executor (KV decode)
  * "python_legacy" — LLMEngine with KV disabled (full-sequence recompute)
  * "rust_serveforge"— serveforge CLI (optional; skipped if binary/tokenizer missing)
  * "hf_reference"  — HuggingFace transformers on CPU (optional; token sanity only)

Usage:
    python scripts/benchmarks/decode_baseline.py \
        --model outputs/compiled/opt_125m_kv \
        --prompts 8 32 128 --max-tokens 16 32 \
        --out outputs/benchmark_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from typing import Any

import torch

PROMPT_IDS: list[int] = [2, 31414, 6, 232, 328, 7181, 87, 9]

HF_SNAPSHOT = os.path.expanduser("~/.cache/huggingface/hub/models--facebook--opt-125m/snapshots")


def _prompt_ids(n: int) -> list[int]:
    """A deterministic n-token prompt (repeats the base window)."""
    out: list[int] = []
    while len(out) < n:
        out.extend(PROMPT_IDS)
    return out[:n]


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _build_engine(artifact_dir: str, use_cache_manager: bool) -> Any:
    from compiler.serialize import load_artifact
    from python_runtime.engine.llm_engine import LLMEngine
    from python_runtime.engine.mlir_executor import MlirExecutor
    from python_runtime.hal.pytorch_backend import PyTorchBackend

    module = load_artifact(artifact_dir)
    backend = PyTorchBackend("cpu")
    executor = MlirExecutor(module, backend)
    if not use_cache_manager:
        executor._uses_cache_manager = False
        executor._cache_mgr = None
    return LLMEngine(module, backend, executor=executor, num_blocks=1024, max_batch_size=2)


def run_python_engine(engine: Any, prompt: list[int], max_tokens: int) -> dict[str, float]:
    """Drive add_request + step() and time prefill vs decode separately."""
    rid = engine.add_request(prompt, max_tokens=max_tokens, temperature=0.0)

    prefill_t0 = time.perf_counter()
    first_token_seen = False
    n_prefill_steps = 0
    while not first_token_seen:
        results = engine.step()
        n_prefill_steps += 1
        for r in results:
            if r.request_id == rid:
                first_token_seen = True
        if n_prefill_steps > 1000:
            raise RuntimeError("prefill never produced a token")
    prefill_ms = (time.perf_counter() - prefill_t0) * 1000.0

    decode_t0 = time.perf_counter()
    n_decode_tokens = 0
    while True:
        results = engine.step()
        finished = False
        for r in results:
            if r.request_id == rid:
                n_decode_tokens += len(r.new_tokens)
                finished = finished or r.is_finished
        if finished or engine.is_idle:
            break
    decode_ms = (time.perf_counter() - decode_t0) * 1000.0

    return {
        "prefill_ms": round(prefill_ms, 2),
        "decode_ms": round(decode_ms, 2),
        "decode_tokens": n_decode_tokens,
        "decode_tokens_s": round(n_decode_tokens / (decode_ms / 1000.0), 2) if decode_ms > 0 else 0.0,
        "wall_ms": round(prefill_ms + decode_ms, 2),
    }


def run_rust_serveforge(
    artifact_dir: str, prompt: list[int], max_tokens: int, tokenizer_path: str
) -> dict[str, Any] | None:
    binary = "runtime/target/release/serveforge"
    if not os.path.exists(binary) or not os.path.exists(tokenizer_path):
        return None
    model_name = os.path.basename(artifact_dir)
    compiled_dir = os.path.dirname(artifact_dir) or "outputs/compiled"
    safetensors = None
    if os.path.isdir(HF_SNAPSHOT):
        for snap in sorted(os.listdir(HF_SNAPSHOT)):
            cand = os.path.join(HF_SNAPSHOT, snap, "model.safetensors")
            if os.path.exists(cand):
                safetensors = cand
                break
    prompt_text = " ".join(str(t) for t in prompt)
    cmd = [
        binary,
        "run",
        model_name,
        "--compiled-dir",
        compiled_dir,
        "--prompt",
        prompt_text,
        "--max-tokens",
        str(max_tokens),
        "--temperature",
        "0.0",
        "--no-chat-template",
        "--prompt-ids",
        "--bench",
        "--tokenizer",
        tokenizer_path,
    ]
    if safetensors:
        cmd += ["--safetensors", safetensors]
    try:
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, check=True, capture_output=True, timeout=600)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        try:
            raw = json.loads(proc.stdout.decode().strip())
        except (json.JSONDecodeError, UnicodeDecodeError):
            raw = {}
        if not isinstance(raw, dict):
            raw = {}
        bench: dict[str, Any] = raw
        bench["wall_ms"] = round(wall_ms, 2)
        bench["decode_tokens_s"] = float(bench.get("decode_tokens_s") or 0.0)
        return bench
    except (subprocess.SubprocessError, subprocess.TimeoutExpired) as exc:  # pragma: no cover
        stderr = getattr(exc, "stderr", None)
        detail = (stderr or b"").decode(errors="replace").strip().splitlines()
        return {"error": (detail[-1] if detail else str(exc))[:200]}


def run_hf_reference(prompt: list[int], max_tokens: int) -> dict[str, Any] | None:
    """HuggingFace opt-125m on CPU — sanity reference (optional)."""
    if not os.path.isdir(HF_SNAPSHOT):
        return None
    try:
        from transformers import OPTForCausalLM

        snap = os.listdir(HF_SNAPSHOT)[0]
        # transformers' stubs are incomplete for from_pretrained/generate;
        # treat the model as Any (runtime-checked by the try/except).
        model: Any = OPTForCausalLM.from_pretrained(
            os.path.join(HF_SNAPSHOT, snap),
            torch_dtype=torch.float32,
            local_files_only=True,
        )
        input_ids = torch.tensor([prompt], dtype=torch.long)
        t0 = time.perf_counter()
        with torch.no_grad():
            out: Any = model.generate(input_ids, max_new_tokens=max_tokens, do_sample=False)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        new_ids = out[0, len(prompt) :].tolist()
        return {
            "wall_ms": round(wall_ms, 2),
            "tokens": new_ids,
            "decode_tokens_s": round(len(new_ids) / (wall_ms / 1000.0), 2),
        }
    except Exception as exc:  # pragma: no cover
        return {"error": str(exc)[:200]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="outputs/compiled/opt_125m_kv")
    parser.add_argument("--prompts", type=int, nargs="+", default=[8, 32, 128])
    parser.add_argument("--max-tokens", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--tokenizer", default="outputs/compiled/opt_125m_fresh/tokenizer.json")
    parser.add_argument("--out", default="outputs/benchmark_results.json")
    parser.add_argument("--skip-hf", action="store_true", help="skip HuggingFace reference")
    parser.add_argument(
        "--pairs",
        type=int,
        default=3,
        help="measured Python KV generations per config (median reported)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="discarded Python KV generations before the measured runs",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.model):
        raise SystemExit(f"model artifact {args.model} missing — run `make rebuild-kv` first")

    configs = [(p, m) for p in args.prompts for m in args.max_tokens]
    metrics: dict[str, Any] = {}
    errors: list[str] = []

    for p_len, m_tokens in configs:
        prompt = _prompt_ids(p_len)
        key = f"p{p_len}_m{m_tokens}"
        row: dict[str, Any] = {}

        engine_kv = _build_engine(args.model, use_cache_manager=True)
        python_warm_rows: list[dict[str, float]] = []
        for _ in range(args.warmup):
            python_warm_rows.append(run_python_engine(engine_kv, prompt, m_tokens))
        python_kv_runs = [run_python_engine(engine_kv, prompt, m_tokens) for _ in range(args.pairs)]
        python_kv: dict[str, Any] = dict(python_kv_runs[0])
        python_kv["runs"] = python_kv_runs
        python_kv["cold_prefill_ms"] = (
            python_warm_rows[0]["prefill_ms"] if python_warm_rows else python_kv_runs[0]["prefill_ms"]
        )
        python_kv["median_prefill_ms"] = round(_median([float(r["prefill_ms"]) for r in python_kv_runs]), 2)
        python_kv["median_decode_ms"] = round(_median([float(r["decode_ms"]) for r in python_kv_runs]), 2)
        python_kv["median_decode_tokens_s"] = round(_median([float(r["decode_tokens_s"]) for r in python_kv_runs]), 2)
        row["python_kv"] = python_kv

        engine_legacy = _build_engine(args.model, use_cache_manager=False)
        row["python_legacy"] = run_python_engine(engine_legacy, prompt, m_tokens)

        row["rust_serveforge"] = run_rust_serveforge(args.model, prompt, m_tokens, args.tokenizer)
        if not args.skip_hf:
            row["hf_reference"] = run_hf_reference(prompt, m_tokens)

        kv_ds = row["python_kv"].get("median_decode_tokens_s") or row["python_kv"]["decode_tokens_s"]
        legacy_ds = row["python_legacy"]["decode_tokens_s"]
        speedup = round(kv_ds / legacy_ds, 2) if legacy_ds > 0 else None
        row["kv_vs_legacy_decode_speedup"] = speedup
        rust = row.get("rust_serveforge") or {}
        rust_before = 0.08  # 2026-08-15 baseline: p8_m16 = 188.96s wall
        if "decode_tokens_s" in rust:
            row["rust_vs_before_speedup"] = round(rust["decode_tokens_s"] / rust_before, 1)
        metrics[key] = row

        print(
            f"p{p_len:>3} m{m_tokens:>2}: kv decode {kv_ds:>8.2f} tok/s | "
            f"legacy {legacy_ds:>8.2f} tok/s | speedup {speedup} | "
            f"prefill kv median {row['python_kv']['median_prefill_ms']:.1f}ms"
        )
        if row.get("rust_serveforge") and "error" in row["rust_serveforge"]:
            errors.append(f"{key}: rust: {row['rust_serveforge']['error']}")

    entry = {
        "name": "decode_kv_baseline",
        "description": (
            "Python generate() decode throughput: KV cache vs legacy full-sequence "
            "recompute (opt_125m_kv, temperature 0). KV decode is O(1) per step "
            "(seq=1, shape-asserted in test_generate_kv.py); legacy recomputes the "
            "full sequence every step. S2-pre hardening: python_kv is now "
            "warm-up separated and measured with the configured paired median. "
            "rust_serveforge auto-enables the compiled cache policy and uses the "
            "O(1) KV decode path; wall_ms includes model+weight loading. "
            "rust_vs_before_speedup is relative to the pre-fix 2026-08-15 "
            "baseline of 0.08 tok/s (p8_m16 wall=188.96s). hf_reference is "
            "transformers opt-125m fp32 CPU whole-generate wall. The separate "
            "paired Rust decode entry is owned by record_decode_only_rust.py."
        ),
        "rust_before": {
            "decode_tokens_s": 0.08,
            "wall_ms_p8_m16": 188962.58,
            "note": "pre-BLAS-bridge / pre-KV / pre-O3 relink baseline",
        },
        "metrics": metrics,
        "passed": all(
            (row["python_kv"].get("median_decode_tokens_s") or row["python_kv"]["decode_tokens_s"]) > 0
            for row in metrics.values()
        ),
        "error": "; ".join(errors) or None,
    }

    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        with open(out_path) as fh:
            data = json.load(fh)
        data = [e for e in data if e.get("name") != entry["name"]]
    except (OSError, json.JSONDecodeError):
        data = []
    data.append(entry)
    with open(out_path, "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
