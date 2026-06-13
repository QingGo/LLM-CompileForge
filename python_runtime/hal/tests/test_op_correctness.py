"""Pytest entry point for per-operator correctness tests.

Each op in ``OP_TABLE`` is parameterized as a separate test case.
A test passes if the cosine similarity between JIT-compiled output
and PyTorch reference exceeds ``1 - case.rtol``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytest

from tests.op_correctness.registry import OP_TABLE, OpCase
from tests.op_correctness.runner import Runner

_log = logging.getLogger(__name__)


@pytest.mark.parametrize("case", OP_TABLE, ids=lambda c: c.name)
def test_op_correctness(case: object) -> None:
    """Compare JIT output of a single sf op against the PyTorch reference."""
    result = Runner(case).run()
    min_cos = 1.0 - case.rtol
    assert result.cos > min_cos, (
        f"{case.name}: cos={result.cos:.8f} < {min_cos} (rtol={case.rtol}) shape={result.output.shape}"
    )


def test_op_sdpa() -> None:
    """Test sf.scaled_dot_product_attention with a boolean causal mask.

    The C++ lowering expects the 4th input (attn_mask) to be a boolean mask
    where 1.0 = attend and 0.0 = masked — *not* positional index values.
    The MHA model pipeline creates this boolean mask via ``sf.ge`` beforehand.
    We pre-compute a lower-triangular causal mask and use ``is_causal=True``
    for the PyTorch reference.
    """
    import torch.nn.functional as F  # noqa: N812

    case = OpCase(
        "sf.scaled_dot_product_attention",
        lambda q, k, v, m: F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            scale=1.0,
        ),
        [(2, 12, 4, 64), (2, 12, 4, 64), (2, 12, 4, 64), (2, 1, 4, 4)],
        1e-4,
        "scaled_dot_product_attention",
        kwargs={"scale": 1.0},
        output_shapes=[(2, 12, 4, 64)],
    )

    rng = np.random.RandomState(42)
    inputs = [rng.randn(*s).astype(np.float32) for s in case.input_shapes]
    causal_mask = np.tril(np.ones((4, 4), dtype=np.float32))
    inputs[3] = np.broadcast_to(causal_mask, (2, 1, 4, 4)).copy()

    runner = Runner(case, custom_inputs=inputs)
    result = runner.run()
    min_cos = 1.0 - case.rtol
    assert result.cos > min_cos, f"SDPA: cos={result.cos:.6f} < {min_cos} shape={result.output.shape}"


def test_op_sdpa_explicit_mask() -> None:
    """Test sf.scaled_dot_product_attention with is_causal=False + explicit boolean mask.

    Unlike test_op_sdpa which uses is_causal=True and relies on PyTorch's
    internal causal masking, this test passes the boolean mask EXPLICITLY
    to F.scaled_dot_product_attention via attn_mask= parameter.
    This matches how the actual model pipeline uses SDPA (the model does
    NOT set is_causal).
    """
    import torch
    import torch.nn.functional as F  # noqa: N812

    case = OpCase(
        "sf.scaled_dot_product_attention",
        lambda q, k, v, m: F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=torch.where(m > 0.5, 0.0, float("-inf")),
            is_causal=False,
            scale=1.0,
        ),
        [(2, 12, 4, 64), (2, 12, 4, 64), (2, 12, 4, 64), (2, 1, 4, 4)],
        1e-4,
        "scaled_dot_product_attention",
        kwargs={"scale": 1.0},
        output_shapes=[(2, 12, 4, 64)],
    )

    rng = np.random.RandomState(42)
    inputs = [rng.randn(*s).astype(np.float32) for s in case.input_shapes]
    causal_mask = np.tril(np.ones((4, 4), dtype=np.float32))
    inputs[3] = np.broadcast_to(causal_mask, (2, 1, 4, 4)).copy()

    runner = Runner(case, custom_inputs=inputs)
    result = runner.run()
    min_cos = 1.0 - case.rtol
    assert result.cos > min_cos, f"SDPA explicit mask: cos={result.cos:.6f} < {min_cos} shape={result.output.shape}"


def test_op_sdpa_model_mask() -> None:
    """Test sf.scaled_dot_product_attention with the actual model mask from dylib.

    The model's mask uses a compressed format (only diagonal elements stored)
    that requires ``mask >= mask.T`` expansion.  This test checks whether
    the C++ lowering handles this compressed format correctly.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F  # noqa: N812

    from scripts.ctypes_forward import run_ctypes

    model_dir = "outputs/compiled/opt_125m_fresh"
    dylib = run_ctypes(model_dir, dylib_path=f"{model_dir}/libopt_125m.dylib")
    model_mask = dylib._func_outputs[0][13]

    case = OpCase(
        "sf.scaled_dot_product_attention",
        lambda q, k, v, m: F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=torch.where(m > 0.5, 0.0, float("-inf")),
            is_causal=False,
            scale=1.0,
        ),
        [(2, 12, 4, 64), (2, 12, 4, 64), (2, 12, 4, 64), (2, 1, 4, 4)],
        1e-4,
        "scaled_dot_product_attention",
        kwargs={"scale": 1.0},
        output_shapes=[(2, 12, 4, 64)],
    )

    rng = np.random.RandomState(42)
    inputs = [rng.randn(*s).astype(np.float32) for s in case.input_shapes]
    inputs[3] = model_mask.copy()

    runner = Runner(case, custom_inputs=inputs)
    result = runner.run()

    print(f"\nSDPA with model mask: cos={result.cos:.6f}")
    print("  Model mask values (first batch):")
    print(f"  {model_mask[0, 0]}")

    min_cos = 1.0 - case.rtol
    assert result.cos > min_cos, (
        f"SDPA model mask: cos={result.cos:.6f} < {min_cos}\n"
        f"The C++ lowering fails to expand the compressed mask format.\n"
        f"Model mask:\n{model_mask[0, 0]}"
    )


def test_mask_construction() -> None:
    """Test mask construction: arange → unsqueeze → le → expand.

    Isolates the causal mask pipeline from func[0].output[13] to determine
    which op produces wrong output.  PyTorch reference produces a
    lower-triangular causal mask.
    """
    import numpy as np
    import torch

    from python_runtime.hal.pytorch_backend import PyTorchBackend
    from scripts._cos import cosine_similarity
    from tests.op_correctness.runner import invoke_and_extract, lower_and_jit

    # MLIR for: arange(4) -> unsqueeze -> le -> expand
    # simulates func[0]'s mask construction pipeline
    mlir_text = """\
module {
  func.func @main(%arg0: tensor<1xi64>) -> tensor<4x4xf32> {
    %0 = "sf.arange"(%arg0) {device = "cpu", pin_memory = false} : (tensor<1xi64>) -> tensor<4xi64>
    %1 = "sf.unsqueeze"(%0) {dim = 0 : i64} : (tensor<4xi64>) -> tensor<1x4xi64>
    %2 = "sf.unsqueeze"(%0) {dim = 1 : i64} : (tensor<4xi64>) -> tensor<4x1xi64>
    %3 = "sf.le"(%2, %1) : (tensor<4x1xi64>, tensor<1x4xi64>) -> tensor<4x4xf32>
    return %3 : tensor<4x4xf32>
  }
}"""

    seq_len = np.array([4], dtype=np.int64)

    engine, output_shape = lower_and_jit(mlir_text)
    jit_mask = invoke_and_extract(engine, "main", [seq_len], output_shape)

    backend = PyTorchBackend("cpu")
    arange_out = backend.execute("arange", [torch.tensor(seq_len)], device="cpu", pin_memory=False)  # type: ignore[arg-type]
    row = backend.execute("unsqueeze", [arange_out], dim=0)
    col = backend.execute("unsqueeze", [arange_out], dim=1)
    causal = backend.execute("le", [col, row])
    ref_mask = causal.numpy()

    cos = cosine_similarity(jit_mask, ref_mask)
    print("\nmask construction test:")
    print(f"  JIT shape: {list(jit_mask.shape)}")
    print(f"  REF shape: {list(ref_mask.shape)}")
    print(f"  cos = {cos:.6f}")
    print("  JIT mask:")
    print(jit_mask)
    print("  REF mask:")
    print(ref_mask)

    assert cos > 0.99, f"mask construction: cos={cos:.6f} < 0.99\nJIT:\n{jit_mask}\nREF:\n{ref_mask}"


def test_mask_full_chain() -> None:
    """Test extended mask construction: arange → unsqueeze → le → logical_and.

    Extends test_mask_construction with ``sf.logical_and`` chained after
    ``sf.le``.  This tests whether the ops after ``sf.le`` correctly
    propagate the causal mask.  ``sf.logical_and`` with two identical inputs
    is a no-op, so the output should match the causal mask exactly.

    If cos < 0.99, the ``sf.logical_and`` lowering is truncating or
    corrupting the mask values.
    """
    import numpy as np
    import torch

    from python_runtime.hal.pytorch_backend import PyTorchBackend
    from scripts._cos import cosine_similarity
    from tests.op_correctness.runner import invoke_and_extract, lower_and_jit

    # MLIR for: arange(4) -> unsqueeze -> le -> logical_and(le, le)
    # logical_and with identical inputs = identity (no-op).
    mlir_text = """\
module {
  func.func @main(%arg0: tensor<1xi64>) -> tensor<4x4xf32> {
    %0 = "sf.arange"(%arg0) {device = "cpu", pin_memory = false} : (tensor<1xi64>) -> tensor<4xi64>
    %1 = "sf.unsqueeze"(%0) {dim = 0 : i64} : (tensor<4xi64>) -> tensor<1x4xi64>
    %2 = "sf.unsqueeze"(%0) {dim = 1 : i64} : (tensor<4xi64>) -> tensor<4x1xi64>
    %3 = "sf.le"(%2, %1) : (tensor<4x1xi64>, tensor<1x4xi64>) -> tensor<4x4xf32>
    %4 = "sf.logical_and"(%3, %3) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    return %4 : tensor<4x4xf32>
  }
}"""

    seq_len = np.array([4], dtype=np.int64)

    engine, output_shape = lower_and_jit(mlir_text)
    jit_mask = invoke_and_extract(engine, "main", [seq_len], output_shape)

    backend = PyTorchBackend("cpu")
    arange_out = backend.execute("arange", [torch.tensor(seq_len)], device="cpu", pin_memory=False)  # type: ignore[arg-type]
    row = backend.execute("unsqueeze", [arange_out], dim=0)
    col = backend.execute("unsqueeze", [arange_out], dim=1)
    causal = backend.execute("le", [col, row])
    and_out = backend.execute("logical_and", [causal, causal])
    ref_mask = and_out.float().numpy()

    cos = cosine_similarity(jit_mask, ref_mask)
    print("\nmask full chain test:")
    print(f"  JIT shape: {list(jit_mask.shape)}")
    print(f"  REF shape: {list(ref_mask.shape)}")
    print(f"  cos = {cos:.6f}")
    print("  JIT mask:")
    print(jit_mask)
    print("  REF mask:")
    print(ref_mask)

    assert cos > 0.99, f"mask full chain: cos={cos:.6f} < 0.99\nJIT:\n{jit_mask}\nREF:\n{ref_mask}"


def test_ones_like_dtype_chain() -> None:
    """Test dtype chain: ones_like (f32→f32) → cumsum (f32→f32).

    Verifies that sf.ones_like produces f32 output (inherits self dtype)
    and sf.cumsum correctly operates on the f32 result.
    """
    import numpy as np
    import torch

    from scripts._cos import cosine_similarity
    from tests.op_correctness.runner import invoke_and_extract, lower_and_jit

    mlir_text = """\
module {
  func.func @main(%arg0: tensor<4x768xf32>) -> tensor<4x768xf32> {
    %0 = "sf.ones_like"(%arg0) : (tensor<4x768xf32>) -> tensor<4x768xf32>
    %1 = "sf.cumsum"(%0) {dim = 0 : i64} : (tensor<4x768xf32>) -> tensor<4x768xf32>
    return %1 : tensor<4x768xf32>
  }
}"""

    input_data = np.random.randn(4, 768).astype(np.float32)

    engine, output_shape = lower_and_jit(mlir_text)
    output = invoke_and_extract(engine, "main", [input_data], output_shape)

    ref = torch.cumsum(torch.ones_like(torch.from_numpy(input_data)), dim=0).numpy()

    cos = cosine_similarity(output, ref)
    print(f"\nones_like dtype chain: cos = {cos:.6f}")
    print(f"  JIT shape: {list(output.shape)}")
    print(f"  REF shape: {list(ref.shape)}")

    assert cos > 0.99, f"ones_like dtype chain: cos={cos:.6f} < 0.99\nJIT:\n{output}\nREF:\n{ref}"


def test_arange_to_index_chain() -> None:
    """Test dtype chain: arange (i64) → cumsum (i64) → sub (i64) → unsqueeze.

    sf.arange always produces i64. cumsum promotes integer input to i64,
    so the chain stays in integer arithmetic. The reference uses f32 since
    the JIT uses f32 memref output format for comparison.
    """
    import numpy as np
    import torch

    from scripts._cos import cosine_similarity
    from tests.op_correctness.runner import invoke_and_extract, lower_and_jit

    mlir_text = """\
module {
  func.func @main(%arg0: tensor<1xi64>) -> tensor<8xi64> {
    %0 = "sf.arange"(%arg0) {device = "cpu", pin_memory = false} : (tensor<1xi64>) -> tensor<8xi64>
    %1 = "sf.cumsum"(%0) {dim = 0 : i64} : (tensor<8xi64>) -> tensor<8xi64>
    return %1 : tensor<8xi64>
  }
}"""

    seq_len = np.array([8], dtype=np.int64)

    engine, output_shape = lower_and_jit(mlir_text)
    output = invoke_and_extract(engine, "main", [seq_len], output_shape)

    a = torch.arange(8, dtype=torch.int64)
    ref = torch.cumsum(a, dim=0).to(dtype=torch.float32).numpy()

    cos = cosine_similarity(output, ref)
    print(f"\narange to index chain: cos = {cos:.6f}")
    print(f"  JIT shape: {list(output.shape)}")
    print(f"  REF shape: {list(ref.shape)}")

    assert cos > 0.99, (
        f"arange to index chain: cos={cos:.6f} < 0.99\nJIT[:8]: {output.ravel()[:8]}\nREF[:8]: {ref.ravel()[:8]}"
    )


def test_sf_index_float_acceptance(capfd: Any) -> None:
    """Test sf.index with f32 index tensor — verifies auto-conversion and warning.

    The ``sf.index`` lowering auto-converts f32 indices to i64 via FPToUIOp.
    This test bypasses the Python-level type check by writing MLIR directly,
    and verifies:
      1. The output matches PyTorch reference exactly (cos == 1.0).
      2. A WARNING is emitted to stderr about the auto-conversion.
    """
    import numpy as np
    import torch

    from scripts._cos import cosine_similarity
    from tests.op_correctness.runner import invoke_and_extract, lower_and_jit

    # MLIR with f32 index tensor (bypasses Python-level sf.index type check)
    mlir_text = """\
module {
  func.func @main(%data: tensor<4x768xf32>, %index: tensor<2xf32>) -> tensor<2x768xf32> {
    %0 = "sf.index"(%data, %index) : (tensor<4x768xf32>, tensor<2xf32>) -> tensor<2x768xf32>
    return %0 : tensor<2x768xf32>
  }
}"""

    rng = np.random.RandomState(42)
    data = rng.randn(4, 768).astype(np.float32)
    index = np.array([1.0, 3.0], dtype=np.float32)

    # Lower/JIT, capture stderr (llvm::errs) for the WARNING
    engine, output_shape = lower_and_jit(mlir_text)
    _out, err = capfd.readouterr()

    jit_out = invoke_and_extract(engine, "main", [data, index], output_shape)

    ref_out = torch.from_numpy(data)[[1, 3]].numpy()

    cos = cosine_similarity(jit_out, ref_out)

    print("\nsf.index f32 acceptance test:")
    print(f"  JIT shape: {list(jit_out.shape)}")
    print(f"  REF shape: {list(ref_out.shape)}")
    print(f"  cos = {cos:.8f}")

    # Near-exact match (cos may exceed 1.0 by epsilon due to FP rounding)
    atol = 1e-6
    assert abs(cos - 1.0) < atol, (
        f"sf.index f32: cos={cos:.8f} != 1.0\nJIT[:5]: {jit_out.ravel()[:5]}\nREF[:5]: {ref_out.ravel()[:5]}"
    )

    # Verify the WARNING was emitted to stderr
    assert "WARNING: auto-converting f32 index to i64" in err, (
        f"Expected WARNING about f32 auto-conversion in stderr, got:\n{err}"
    )
