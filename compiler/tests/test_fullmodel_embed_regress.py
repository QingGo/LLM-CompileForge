"""TDD regression test: dylib main_0 output[0] (embedding hidden states)
must match numpy reference from the embedding weight tensor.

Compares the dylib's raw embedding lookup result against a direct
numpy embedding computation from the loaded weight tensor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.ravel().astype(np.float64)
    b_f = b.ravel().astype(np.float64)
    denom = np.linalg.norm(a_f) * np.linalg.norm(b_f) + 1e-12
    return float(np.dot(a_f, b_f) / denom)


@pytest.mark.integration
@pytest.mark.timeout(180)
class TestFullModelEmbeddingRegression:
    @pytest.mark.xfail(reason="requires compiled artifacts — run make build-all")
    def test_embedding_output_matches_python_executor(self) -> None:
        from scripts.ctypes_forward import run_ctypes

        artifact_dir = "outputs/compiled/opt_125m"

        # Load embedding weight from artifact
        from compiler.serialize import load_artifact

        artifact = load_artifact(artifact_dir)
        emb_weight: np.ndarray | None = None
        for func in artifact.functions:
            for wname, wtensor in func.weights.items():
                if "embed" in wname.lower() or "wte" in wname.lower():
                    emb_weight = np.ascontiguousarray(wtensor.numpy())
                    break
            if emb_weight is not None:
                break
        assert emb_weight is not None, "Embedding weight not found in artifact"

        # Run dylib forward pass
        input_ids = np.array([[2, 32826, 85, 4129], [0, 0, 0, 0]], dtype=np.int64)
        ctypes_result = run_ctypes(
            artifact_dir,
            dylib_path=f"{artifact_dir}/libopt_125m.dylib",
            input_ids=input_ids,
        )

        # Find main_0's hidden state output (rank-3 tensor)
        out0 = ctypes_result.func_outputs[0]
        dylib_emb = None
        for arr in out0:
            if arr.ndim == 3:
                dylib_emb = arr
                break
        assert dylib_emb is not None, "No rank-3 output found in main_0"
        # Reference: direct numpy embedding lookup
        py_emb = emb_weight[input_ids % emb_weight.shape[0]]

        cos = _cosine_similarity(dylib_emb.ravel(), py_emb.ravel())
        assert cos >= 0.9999, (
            f"Embedding output regression: cos={cos:.8f} < 0.9999\n"
            f"Dylib mean={dylib_emb.mean():.6f}, py mean={py_emb.mean():.6f}\n"
            f"Dylib[:5]={dylib_emb.ravel()[:5].tolist()}\n"
            f"Py[:5]={py_emb.ravel()[:5].tolist()}"
        )
