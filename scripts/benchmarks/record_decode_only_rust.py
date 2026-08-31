#!/usr/bin/env python3
"""Paired decode benchmark for the Rust serveforge CLI (S2-pre).

Replaces the single-shot decode-only entry with the hardened protocol from
`.omo/plans/p1-post-d3-replan.md`:

* cold-start and warm-prefill are separate numbers;
* every pair runs Python KV first, then a fresh Rust process (which performs
  its own in-process warm-up before the timed run), repeated ``--pairs``
  times;
* the gate number is the median of the per-pair decode throughput, not a
  single sample;
* token ids must match Python KV on every pair.

The written JSON entry keeps the historical name ``decode_only_rust`` (or the
``_op``/``_func``/``_fused`` variant) so downstream tooling does not need to
track a second name.

Usage:
    python scripts/benchmarks/record_decode_only_rust.py \
        --model outputs/compiled/opt_125m_kv --prompts 8 --max-tokens 16
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from typing import Any

PROMPT_IDS: list[int] = [2, 31414, 6, 232, 328, 7181, 87, 9]
DEFAULT_TOKENIZER = "outputs/compiled/opt_125m_fresh/tokenizer.json"
HF_SNAPSHOT = os.path.expanduser("~/.cache/huggingface/hub/models--facebook--opt-125m/snapshots")


def _prompt_ids(n: int) -> list[int]:
    out: list[int] = []
    while len(out) < n:
        out.extend(PROMPT_IDS)
    return out[:n]


def _find_hf_safetensors() -> str | None:
    if not os.path.isdir(HF_SNAPSHOT):
        return None
    candidates: list[str] = []
    for snap in sorted(os.listdir(HF_SNAPSHOT)):
        cand = os.path.join(HF_SNAPSHOT, snap, "model.safetensors")
        if os.path.exists(cand):
            candidates.append(cand)
    return candidates[-1] if candidates else None


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _build_python_engine(artifact_dir: str) -> Any:
    from compiler.serialize import load_artifact
    from python_runtime.engine.llm_engine import LLMEngine
    from python_runtime.engine.mlir_executor import MlirExecutor
    from python_runtime.hal.pytorch_backend import PyTorchBackend

    module = load_artifact(artifact_dir)
    backend = PyTorchBackend("cpu")
    executor = MlirExecutor(module, backend)
    return LLMEngine(module, backend, executor=executor, num_blocks=1024, max_batch_size=2)


def run_python_generation(engine: Any, prompt: list[int], max_tokens: int) -> dict[str, Any]:
    """One Python KV generation with prefill/decode split and token ids."""
    rid = engine.add_request(prompt, max_tokens=max_tokens, temperature=0.0)

    prefill_t0 = time.perf_counter()
    token_ids: list[int] = []
    n_prefill_steps = 0
    while not token_ids:
        results = engine.step()
        n_prefill_steps += 1
        for r in results:
            if r.request_id == rid:
                # The prefill step already samples the first output token;
                # keep it so token_ids is directly comparable to the Rust
                # result.tokens list.
                token_ids.extend(r.new_tokens)
        if n_prefill_steps > 1000:
            raise RuntimeError("python prefill never produced a token")
    prefill_ms = (time.perf_counter() - prefill_t0) * 1000.0
    tokens_from_prefill = len(token_ids)

    decode_t0 = time.perf_counter()
    while True:
        results = engine.step()
        finished = False
        for r in results:
            if r.request_id == rid:
                token_ids.extend(r.new_tokens)
                finished = finished or r.is_finished
        if finished or engine.is_idle:
            break
    decode_ms = (time.perf_counter() - decode_t0) * 1000.0
    # The decode timer starts after the first output token; throughput must
    # count only tokens produced inside the timed decode loop.
    decode_tokens = len(token_ids) - tokens_from_prefill
    return {
        "prefill_ms": round(prefill_ms, 2),
        "decode_ms": round(decode_ms, 2),
        "decode_tokens": decode_tokens,
        "decode_tokens_s": round(decode_tokens / (decode_ms / 1000.0), 2) if decode_ms > 0 else 0.0,
        "token_ids": token_ids,
    }


def run_rust_generation(
    binary: str,
    model_dir: str,
    prompt: list[int],
    max_tokens: int,
    fused: bool,
    exec_plan: str | None,
    weight_dtype: str | None,
    account: bool,
) -> dict[str, Any]:
    """One fresh Rust process: one in-process warm-up + one timed run."""
    model_name = os.path.basename(model_dir.rstrip("/"))
    compiled_dir = os.path.dirname(model_dir.rstrip("/")) or "outputs/compiled"
    tokenizer = os.path.join(model_dir, "tokenizer.json")
    if not os.path.exists(tokenizer):
        tokenizer = DEFAULT_TOKENIZER
    safetensors = _find_hf_safetensors()

    cmd = [
        binary,
        "run",
        model_name,
        "--compiled-dir",
        compiled_dir,
        "--prompt",
        " ".join(str(t) for t in prompt),
        "--max-tokens",
        str(max_tokens),
        "--temperature",
        "0.0",
        "--no-chat-template",
        "--prompt-ids",
        "--bench",
        "--bench-runs",
        "1",
        "--bench-warmup-runs",
        "1",
        "--tokenizer",
        tokenizer,
    ]
    if fused:
        cmd += ["--opt-fused-fastpath"]
    if exec_plan:
        cmd += ["--exec-plan", exec_plan]
    if weight_dtype:
        cmd += ["--weight-dtype", weight_dtype]
    if safetensors:
        cmd += ["--safetensors", safetensors]

    env = os.environ.copy()
    if account:
        env["SERVEFORGE_ACCOUNT"] = "1"

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, check=True, capture_output=True, timeout=600, env=env)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    raw = json.loads(proc.stdout.decode().strip())
    if not isinstance(raw, dict):
        raise SystemExit("serveforge --bench did not emit a JSON object")
    bench: dict[str, Any] = raw
    bench["wall_ms"] = round(wall_ms, 2)

    warmup = (bench.get("warmup_runs") or [{}])[0]
    bench["process_cold_prefill_ms"] = warmup.get("prefill_ms")
    bench["account"] = bench.get("account")
    return bench


def _check_tokens(key: str, expected: list[int], row: dict[str, Any]) -> None:
    actual = row.get("token_ids") or []
    if actual != expected:
        raise SystemExit(f"{key} token mismatch: expected {expected}, got {actual}")


def _summarize_runs(runs: list[dict[str, Any]], expected_tokens: list[int], label: str) -> dict[str, Any]:
    for run in runs:
        _check_tokens(label, expected_tokens, run)
    decode_values = [float(r["decode_tokens_s"]) for r in runs]
    prefill_values = [float(r["prefill_ms"]) for r in runs]
    decode_ms_values = [float(r["decode_ms"]) for r in runs]
    return {
        "runs": runs,
        "median_decode_tokens_s": round(_median(decode_values), 2),
        "median_decode_ms": round(_median(decode_ms_values), 2),
        "median_prefill_ms": round(_median(prefill_values), 2),
        "decode_tokens_s_min": round(min(decode_values), 2),
        "decode_tokens_s_max": round(max(decode_values), 2),
        "token_exact": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="outputs/compiled/opt_125m_kv")
    parser.add_argument("--binary", default="runtime/target/release/serveforge")
    parser.add_argument("--prompts", type=int, nargs="+", default=[8])
    parser.add_argument("--max-tokens", type=int, nargs="+", default=[16])
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--out", default="outputs/benchmark_results.json")
    parser.add_argument("--skip-python", action="store_true")
    parser.add_argument(
        "--opt-fused-fastpath",
        action="store_true",
        help="pass --opt-fused-fastpath to serveforge",
    )
    parser.add_argument(
        "--exec-plan",
        choices=[None, "auto", "func", "op"],
        default=None,
        help="pass --exec-plan to serveforge; op/func record under decode_only_rust_op / decode_only_rust_func",
    )
    parser.add_argument(
        "--weight-dtype",
        choices=[None, "auto", "f32", "f16", "bf16"],
        default=None,
        help="pass --weight-dtype to serveforge (f32 records under decode_only_rust_f32)",
    )
    parser.add_argument(
        "--account",
        action="store_true",
        help="set SERVEFORGE_ACCOUNT=1 and attach the step ledger to each Rust run",
    )
    args = parser.parse_args()

    if args.pairs <= 0:
        raise SystemExit("--pairs must be >= 1")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")
    if not os.path.exists(args.binary):
        raise SystemExit(f"missing {args.binary} - run `make build` first")
    if not os.path.isdir(args.model):
        raise SystemExit(f"missing model artifact {args.model}")

    metrics: dict[str, Any] = {}
    for p_len in args.prompts:
        for m_tokens in args.max_tokens:
            key = f"p{p_len}_m{m_tokens}"
            prompt = _prompt_ids(p_len)

            py_runs: list[dict[str, Any]] = []
            py_cold_prefill_ms: float | None = None
            expected_tokens: list[int] | None = None
            if not args.skip_python:
                engine = _build_python_engine(args.model)
                for _ in range(args.warmup):
                    warm = run_python_generation(engine, prompt, m_tokens)
                    if py_cold_prefill_ms is None:
                        py_cold_prefill_ms = warm["prefill_ms"]
                    expected_tokens = warm["token_ids"]
                for _ in range(args.pairs):
                    row = run_python_generation(engine, prompt, m_tokens)
                    expected_tokens = row["token_ids"]
                    py_runs.append(row)

            rust_runs: list[dict[str, Any]] = []
            rust_cold_prefill_values: list[float] = []
            for _ in range(args.pairs):
                row = run_rust_generation(
                    args.binary,
                    args.model,
                    prompt,
                    m_tokens,
                    args.opt_fused_fastpath,
                    args.exec_plan,
                    args.weight_dtype,
                    args.account,
                )
                if expected_tokens is None:
                    expected_tokens = row.get("token_ids") or []
                _check_tokens("rust", expected_tokens, row)
                rust_runs.append(row)
                cold = row.get("process_cold_prefill_ms")
                if cold is not None:
                    rust_cold_prefill_values.append(float(cold))

            assert expected_tokens is not None
            rust_summary = _summarize_runs(rust_runs, expected_tokens, "rust")
            rust_summary["cold_prefill_ms"] = (
                round(_median(rust_cold_prefill_values), 2) if rust_cold_prefill_values else None
            )
            if args.account:
                rust_summary["account"] = rust_runs[-1].get("account")

            row_metric: dict[str, Any] = {"rust": rust_summary}
            if not args.skip_python:
                py_summary = _summarize_runs(py_runs, expected_tokens, "python_kv")
                py_summary["cold_prefill_ms"] = py_cold_prefill_ms
                row_metric["python_kv"] = py_summary
                row_metric["speedup_vs_python_kv"] = round(
                    rust_summary["median_decode_tokens_s"] / py_summary["median_decode_tokens_s"],
                    2,
                )
                row_metric["token_exact"] = True
            else:
                row_metric["token_exact"] = all(r.get("token_ids") for r in rust_runs)
            metrics[key] = row_metric

            rust_ds = rust_summary["median_decode_tokens_s"]
            if not args.skip_python:
                py_ds = row_metric["python_kv"]["median_decode_tokens_s"]
                speedup = row_metric["speedup_vs_python_kv"]
                print(
                    f"{key}: python_kv median {py_ds:.2f} tok/s | "
                    f"rust median {rust_ds:.2f} tok/s | speedup {speedup:.2f}x | "
                    f"warm prefill {rust_summary['median_prefill_ms']:.1f}ms | "
                    f"cold prefill {rust_summary['cold_prefill_ms']:.1f}ms"
                )
            else:
                print(f"{key}: rust median {rust_ds:.2f} tok/s")

    if args.opt_fused_fastpath:
        name = "decode_only_rust_fused"
        extra = "  Uses --opt-fused-fastpath (Phase 4 Path C prototype)."
    elif args.weight_dtype == "f32":
        name = "decode_only_rust_f32"
        extra = "  Uses --weight-dtype f32 (f32 A/B control)."
    elif args.exec_plan == "op":
        name = "decode_only_rust_op"
        extra = "  Uses --exec-plan op (Phase 5 HAL kernel graph)."
    elif args.exec_plan == "func":
        name = "decode_only_rust_func"
        extra = "  Uses --exec-plan func (func-level dylib path)."
    else:
        name = "decode_only_rust"
        extra = (
            "  Default path (--exec-plan auto, op plan when present; "
            "--weight-dtype auto, source-preserving F16 for the OPT F16 checkpoint)."
        )

    entry = {
        "name": name,
        "description": (
            "S2-pre paired decode benchmark (opt_125m_kv, temperature 0). "
            "Python KV and Rust serveforge are measured alternately, "
            f"{args.pairs} pairs per config; the gate number is the median "
            "decode-only throughput of the per-pair runs.  Each Rust process "
            "performs one in-process warm-up before its timed run, so "
            "cold_prefill_ms (first warm-up prefill, contains f16 weight "
            "conversion) and median_prefill_ms (warm prefill) are separate. "
            "token ids must match python_kv on every pair." + extra
        ),
        "protocol": {
            "pairs": args.pairs,
            "python_warmup_runs": args.warmup,
            "rust_warmup_runs_per_process": 1,
            "median": "upper median",
            "prompt_ids": " ".join(str(t) for t in _prompt_ids(max(args.prompts))),
        },
        "metrics": metrics,
        "passed": all(
            (row.get("rust") or {}).get("token_exact")
            and float((row.get("rust") or {}).get("median_decode_tokens_s", 0.0)) > 0
            for row in metrics.values()
        ),
        "error": None,
    }

    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        with open(out_path) as fh:
            data = json.load(fh)
        data = [e for e in data if e.get("name") != name]
    except (OSError, json.JSONDecodeError):
        data = []
    data.append(entry)
    with open(out_path, "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
