"""TDD RED test: full model dylib main_0 output[12] (embedding hidden states)
has cos < 0.9999 vs Python executor reference.

This test documents the known pipeline regression. When fixed, remove xfail
and assert cos ≥ 0.9999.
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

    def test_embedding_output_matches_python_executor(self) -> None:
        from scripts.ctypes_forward import run_ctypes, run_python_executor

        ARTIFACT_DIR = "outputs/compiled/opt_125m_fresh"

        py_result = run_python_executor(ARTIFACT_DIR)
        ctypes_result = run_ctypes(
            ARTIFACT_DIR, dylib_path=f"{ARTIFACT_DIR}/libmodel.dylib"
        )

        dylib_emb = ctypes_result.func_outputs[0][12]
        py_layer0 = py_result.func_outputs[1][0]

        cos = _cosine_similarity(dylib_emb.ravel(), py_layer0.ravel())
        assert cos >= 0.9999, (
            f"Embedding output regression: cos={cos:.8f} < 0.9999\n"
            f"Dylib mean={dylib_emb.mean():.6f}, py mean={py_layer0.mean():.6f}\n"
            f"Dylib[:5]={dylib_emb.ravel()[:5].tolist()}\n"
            f"Py[:5]={py_layer0.ravel()[:5].tolist()}"
        )
