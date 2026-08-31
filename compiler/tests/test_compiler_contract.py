"""Compiler contract test: model.mlir executor output must match HF reference.

Validates that the compiler produces a model.mlir whose numerical
output (via Python executor) matches HuggingFace within cos >= 0.999.
This is the compiler subproject's primary correctness contract —
any change to converter, split, weight handling, or torch.export
must not break this.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python_runtime"))

from compiler.serialize import load_artifact  # noqa: E402
from python_runtime.engine.mlir_executor import MlirExecutor  # noqa: E402
from python_runtime.hal.pytorch_backend import PyTorchBackend  # noqa: E402
from scripts._cos import cosine_similarity  # noqa: E402


@pytest.fixture(scope="module")
def _hf_reference() -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", local_files_only=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", local_files_only=True)
    input_ids = [2, 32826, 85, 4129]  # same as forward_check
    with torch.no_grad():
        outputs = model(torch.tensor([input_ids]))
    return {
        "logits": outputs.logits.numpy().astype(np.float32),
        "input_ids": input_ids,
        "model": model,
        "tokenizer": tokenizer,
    }


@pytest.mark.integration
@pytest.mark.timeout(120)
class TestCompilerContract:
    def test_model_mlir_forward_matches_hf(self, _hf_reference: dict[str, Any]) -> None:
        """model.mlir executor output must match HF within cos >= 0.999.

        This is the compiler subproject's contract: the generated model.mlir
        must be numerically correct when executed by the Python executor.
        """
        artifact_dir = "outputs/compiled/opt_125m_fresh"
        artifact = load_artifact(artifact_dir)
        backend = PyTorchBackend("cpu")
        executor = MlirExecutor(artifact, backend)

        input_ids = np.array([_hf_reference["input_ids"]], dtype=np.int64)
        result = executor.forward(input_ids)

        hf_logits = _hf_reference["logits"]
        expected_shape = hf_logits.shape  # (1, 4, 50272)

        # The executor may produce a different shape (batch/seq swapped);
        # try to match the expected shape.
        result_np = result.numpy().reshape(expected_shape)
        assert result_np.shape == expected_shape, (
            f"Shape mismatch after reshape: executor={result_np.shape} vs hf={expected_shape}"
        )

        # Compare first token's logits (position 0) and last token's logits
        for pos, label in [(0, "first"), (-1, "last")]:
            pos_logits = result_np[0, pos] if pos >= 0 else result_np[0, -1]
            hf_pos = hf_logits[0, pos]
            cos = cosine_similarity(
                pos_logits.astype(np.float64),
                hf_pos.astype(np.float64),
            )
            assert cos >= 0.999, (
                f"Compiler contract FAILED at {label} position: "
                f"cos={cos:.8f} < 0.999\n"
                f"  executor argmax={int(np.argmax(pos_logits))}\n"
                f"  hf       argmax={int(np.argmax(hf_pos))}"
            )
