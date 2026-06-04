"""E2E SmoothQuant W8A8 quantization pipeline.

Calibrates a compiled HuggingFace model, applies SmoothQuant quantization,
and validates output quality via cosine similarity vs the original model.

Usage:
    python scripts/quantize_smoothquant.py --model llama-1b
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))


from compiler.mlir_dialect.compile_utils import _patch_transformers_torch


def quantize_llama_1b() -> dict:
    """Calibrate and quantize Llama-3.2-1B with SmoothQuant W8A8.

    Returns a dict with calibration metrics.
    """
    import torch
    _patch_transformers_torch()

    from transformers import AutoConfig, AutoModelForCausalLM

    from compiler.quantize.smoothquant import SmoothQuantCalibrator

    model_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "LLM-Research", "Llama-3.2-1B",
    )
    model_dir = os.path.abspath(model_dir)

    print(f"Loading Llama-3.2-1B from: {model_dir}")
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=False)
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, config=config, torch_dtype=torch.float32,
    )
    model.eval()

    # Generate calibration data
    torch.manual_seed(42)
    num_samples = 64
    batch_size = 1
    seq_len = 32
    vocab_size = config.vocab_size
    calibration_data = [
        (torch.randint(0, vocab_size, (batch_size, seq_len)),)
        for _ in range(num_samples)
    ]

    print(f"Calibrating with {num_samples} samples...")
    calibrator = SmoothQuantCalibrator(model, alpha=0.5)
    calibrator.calibrate(calibration_data, num_samples=num_samples)

    num_layers = calibrator.num_layers_processed
    print(f"Smoothed {num_layers} layers")

    # Run reference forward pass
    test_input = torch.randint(0, vocab_size, (1, 8))
    with torch.no_grad():
        ref_logits = model(test_input).logits

    print("Quantizing weights...")
    calibrator.quantize()

    with torch.no_grad():
        quant_logits = model(test_input).logits

    cosine = torch.nn.functional.cosine_similarity(
        ref_logits.float().flatten(),
        quant_logits.float().flatten(),
        dim=0,
    ).item()

    print("\n┌─ SmoothQuant W8A8 Results ──────────────────")
    print(f"│ Layers smoothed:   {num_layers}")
    print(f"│ Cosine similarity: {cosine:.6f}")
    print(f"│ Threshold (>0.99): {'PASS' if cosine > 0.99 else 'FAIL'}")
    print("└──────────────────────────────────────────────")

    return {"num_layers": num_layers, "cosine": cosine}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmoothQuant W8A8 calibration")
    parser.add_argument("--model", choices=["llama-1b", "llama-3b"], default="llama-1b")
    args = parser.parse_args()

    if args.model == "llama-1b":
        quantize_llama_1b()
    else:
        quantize_llama_1b()
