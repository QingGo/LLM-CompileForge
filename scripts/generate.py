#!/usr/bin/env python3
"""Run token generation with a compiled model.

Usage:
    python scripts/generate.py                         # defaults to llama_1b
    python scripts/generate.py --model llama_1b        # explicit model
    python scripts/generate.py --prompt "Hello, world!" --max-tokens 50
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))


from compiler.backend.compile_utils import _patch_transformers_torch

MODEL_PRESETS = {
    "llama_1b": {
        "dir": "./outputs/compiled/llama_1b",
        "tokenizer_dir": "./models/LLM-Research/Llama-3.2-1B",
    },
    "llama_3b": {
        "dir": "./outputs/compiled/llama_3b",
        "tokenizer_dir": "./models/LLM-Research/Llama-3.2-3B",
    },
    "qwen": {
        "dir": "./outputs/compiled/qwen3_0.8b",
        "tokenizer_dir": "./models/Qwen/Qwen3.5-0.8B",
        "pad_to": 64,
    },
    "tiny_llama": {
        "dir": "./outputs/compiled/tiny_llama",
        "tokenizer_dir": None,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with a compiled model")
    parser.add_argument("--model", default="llama_1b",
                        choices=list(MODEL_PRESETS),
                        help="Compiled model name")
    parser.add_argument("--prompt", default="Hello, world!",
                        help="Input prompt text")
    parser.add_argument("--max-tokens", type=int, default=50,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Nucleus sampling threshold")
    parser.add_argument("--top-k", type=int, default=0,
                        help="Top-K sampling (0 = disabled)")
    parser.add_argument("--no-kv-cache", action="store_true",
                        help="Disable KV cache (full recompute each step)")
    args = parser.parse_args()

    preset = MODEL_PRESETS[args.model]
    compiled_dir = preset["dir"]
    tokenizer_dir = preset["tokenizer_dir"]

    # ── Load compiled model ─────────────────────────────
    from compiler.serialize import load_artifact
    from python_runtime.engine.llm_engine import LLMEngine
    from python_runtime.hal.pytorch_backend import PyTorchBackend

    print(f"Loading compiled model from: {compiled_dir}")
    module = load_artifact(compiled_dir)
    backend = PyTorchBackend("cpu")

    engine = LLMEngine(
        module, backend,
        num_blocks=2000,
        block_size=16,
        max_batch_size=1,
    )

    # ── Load tokenizer ──────────────────────────────────
    if tokenizer_dir and os.path.isdir(tokenizer_dir):
        _patch_transformers_torch()
        from transformers import AutoTokenizer  # type: ignore[import-untyped]
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        engine.set_tokenizer(tokenizer, eos_token_id=tokenizer.eos_token_id)
        print(f"Tokenizer loaded from: {tokenizer_dir}")
        print(f"  vocab_size={tokenizer.vocab_size}, eos_token_id={tokenizer.eos_token_id}")
    else:
        class _FallbackTokenizer:
            def encode(self, text):
                return [ord(c) % 256 for c in text]
            def decode(self, tokens):
                return " ".join(str(t) for t in tokens)
        engine.set_tokenizer(_FallbackTokenizer())
        print("Using fallback tokenizer (ord mod 256)")

    # ── Display policy info ─────────────────────────────
    policy = module.metadata.get("cache_policy")
    if policy:
        strategy = None
        for slab in policy.get("slabs", []):
            strategy = slab.get("storage", "?")
            dims = slab.get("dims", {})
            print(f"Cache policy: {strategy} slab, "
                  f"layers={dims.get('layers','?')}, "
                  f"heads={dims.get('heads','?')}, dim={dims.get('dim','?')}")
            break

    # ── Generate ────────────────────────────────────────
    prompt_text = args.prompt
    pad_to = preset.get("pad_to", 0)
    print(f"\nPrompt: {prompt_text}")

    if pad_to > 0:
        tokens = engine._tokenizer.encode(prompt_text)
        if len(tokens) < pad_to:
            pad_id = getattr(engine._tokenizer, "pad_token_id", None) or 0
            tokens = tokens + [pad_id] * (pad_to - len(tokens))
        prompt_text = tokens  # pass as token list

    print(f"Generating (max_tokens={args.max_tokens}, "
          f"temperature={args.temperature}, top_p={args.top_p})...\n")

    result = engine.generate(
        prompt_text,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    print(f"Output: {result}")


if __name__ == "__main__":
    main()
