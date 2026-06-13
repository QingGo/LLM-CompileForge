# ruff: noqa: E402
"""L1.5 forward correctness tests for the compiler pipeline.

Tests compile models through the pipeline and verify correctness:
  - ``tiny_llama`` (2-layer, hidden_size=16): minimal compilation check
  - ``opt-125m`` (12-layer, hidden_size=768): full forward + cos > 0.99

Known limitations (documented tech debt):
  1. tiny_llama executor fails on ``sf.index`` with SSA-ref shape attrs
     (fx_to_mlir shape inference fallback generates invalid attributes).
  2. tiny_llama passes (canonicalize, CSE, fusion) fail due to unregistered
     ``sf.pow`` and ``sf.add`` type mismatches in C++ dialect lowering.
  These are tracked as pipeline bugs — fix the fx_to_mlir shape inference
  and sf-dialect op registry to resolve.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest
import torch

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

from tests.helpers import cosine_similarity as _cos_sim

# ── Skip conditions ──────────────────────────────────────────


def _tiny_llama_cached() -> bool:
    hub = os.path.expanduser("~/.cache/huggingface/hub/models--hf-internal-testing--tiny-random-LlamaForCausalLM")
    snapshots = os.path.join(hub, "snapshots")
    return os.path.isdir(snapshots) and len(os.listdir(snapshots)) > 0


def _opt125m_cached() -> bool:
    hub = os.path.expanduser("~/.cache/huggingface/hub/models--facebook--opt-125m")
    snapshots = os.path.join(hub, "snapshots")
    return os.path.isdir(snapshots) and len(os.listdir(snapshots)) > 0


requires_tiny = pytest.mark.skipif(
    not _tiny_llama_cached(),
    reason="hf-internal-testing/tiny-random-LlamaForCausalLM not cached",
)
requires_opt = pytest.mark.skipif(
    not _opt125m_cached(),
    reason="facebook/opt-125m not cached",
)


# ── Test: tiny_llama compiles (structural) ───────────────────


@pytest.mark.xfail(
    reason="pre-existing: FX converter type inference doesn't handle i64→f32 in embed_prefix matmul; see test file docstring"
)
@pytest.mark.integration
@pytest.mark.timeout(60)
@requires_tiny
def test_tiny_llama_compiles() -> None:
    """Verify tiny_llama exports and compiles to a valid MlirModule.

    Uses ``apply_fusion=False`` to avoid C++ dialect issues
    (``sf.pow`` unregistered, ``sf.add`` type mismatch).
    """
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    from tests.helpers import patch_transformers_torch

    patch_transformers_torch()

    config = LlamaConfig.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
    config.use_cache = False
    model = LlamaForCausalLM(config)
    model.eval()

    from compiler.pipeline import compile_mlir

    input_ids = torch.randint(0, 32000, (2, 4), dtype=torch.long)
    with tempfile.TemporaryDirectory(prefix="l15_") as tmpdir:
        mlir_mod = compile_mlir(
            model,
            example_args=(input_ids,),
            output_dir=tmpdir,
            apply_fusion=False,
        )
        assert len(mlir_mod.functions) > 0, "MlirModule has no functions"
        total_ops = sum(len(f.ops) for f in mlir_mod.functions)
        print(f"  {len(mlir_mod.functions)} functions, {total_ops} ops")
        assert total_ops > 0, "MlirModule has no ops"


# ── Test: tiny_llama config ─────────────────────────────────


@pytest.mark.unit
@requires_tiny
def test_tiny_llama_config() -> None:
    from transformers.models.llama.configuration_llama import LlamaConfig

    config = LlamaConfig.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
    print(
        f"  hidden_size={config.hidden_size} layers={config.num_hidden_layers} "
        f"heads={config.num_attention_heads} vocab={config.vocab_size}"
    )
    assert config.num_hidden_layers == 2
    assert config.hidden_size == 16


# ── Test: opt-125m forward + cos > 0.99 (full pipeline) ─────


@pytest.mark.integration
@pytest.mark.timeout(120)
@requires_opt
def test_opt125m_compile_and_forward_cosine() -> None:
    """Compile opt-125m, run forward, compare with HF reference.

    Uses ``apply_fusion=False`` to skip broken MLIR passes
    (``sf.ones_like`` with 0 operands, unregistered ops).
    The core pipeline (export -> fx -> MlirModule -> executor)
    is tested and verified.
    """
    from transformers.models.opt.configuration_opt import OPTConfig
    from transformers.models.opt.modeling_opt import OPTForCausalLM

    from tests.helpers import patch_transformers_torch

    patch_transformers_torch()

    hub_dir = os.path.expanduser("~/.cache/huggingface/hub/models--facebook--opt-125m")
    snapshots = os.path.join(hub_dir, "snapshots")
    snap = os.listdir(snapshots)[0]
    config_path = os.path.join(snapshots, snap, "config.json")
    config = OPTConfig.from_pretrained(config_path)
    config.use_cache = False

    model_path = os.path.join(snapshots, snap, "pytorch_model.bin")
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    model = OPTForCausalLM(config)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    input_ids = torch.tensor([[2, 32826, 85, 4129]], dtype=torch.long)

    # HF reference
    with torch.no_grad():
        hf_output = model(input_ids)
    hf_logits = hf_output.logits

    # Compile through full pipeline
    from compiler.pipeline import compile_mlir
    from python_runtime.engine.mlir_executor import MlirExecutor
    from python_runtime.hal.pytorch_backend import PyTorchBackend

    with tempfile.TemporaryDirectory(prefix="l15_opt_") as tmpdir:
        mlir_mod = compile_mlir(
            model,
            example_args=(input_ids,),
            output_dir=tmpdir,
            apply_fusion=False,
        )
        print(f"  {len(mlir_mod.functions)} functions, {sum(len(f.ops) for f in mlir_mod.functions)} ops")

        executor = MlirExecutor(mlir_mod, PyTorchBackend("cpu"))
        py_logits = executor.forward(input_ids)
        py_np = py_logits.detach().numpy()

        # Cosine comparison
        cos = _cos_sim(hf_logits, py_logits)
        print(f"  Cosine(compiled vs HF): {cos:.10f}")

        assert cos > 0.99, f"Cosine similarity {cos:.6f} < 0.99"
        assert py_np.shape == hf_logits.shape, f"Shape mismatch: compiled {py_np.shape} vs HF {hf_logits.shape}"
        assert not np.isnan(py_np).any(), "NaN in compiled output"
        assert not np.isinf(py_np).any(), "Inf in compiled output"
        print(f"  Passed: cos={cos:.6f} shape={py_np.shape}")


# ── Test: opt-125m forward smoke (batch=1, seq=4) ───────────


@pytest.mark.integration
@pytest.mark.timeout(60)
@requires_opt
def test_opt125m_forward_smoke() -> None:
    """Verify opt-125m compiles and forward runs without errors."""
    from transformers.models.opt.configuration_opt import OPTConfig
    from transformers.models.opt.modeling_opt import OPTForCausalLM

    from tests.helpers import patch_transformers_torch

    patch_transformers_torch()

    hub_dir = os.path.expanduser("~/.cache/huggingface/hub/models--facebook--opt-125m")
    snapshots = os.path.join(hub_dir, "snapshots")
    snap = os.listdir(snapshots)[0]
    config = OPTConfig.from_pretrained(os.path.join(snapshots, snap, "config.json"))
    config.use_cache = False
    state_dict = torch.load(
        os.path.join(snapshots, snap, "pytorch_model.bin"),
        map_location="cpu",
        weights_only=False,
    )
    model = OPTForCausalLM(config)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    from compiler.pipeline import compile_mlir
    from python_runtime.engine.mlir_executor import MlirExecutor
    from python_runtime.hal.pytorch_backend import PyTorchBackend

    input_ids = torch.randint(0, 50272, (1, 4), dtype=torch.long)
    with tempfile.TemporaryDirectory(prefix="l15_opt2_") as tmpdir:
        mlir_mod = compile_mlir(
            model,
            example_args=(input_ids,),
            output_dir=tmpdir,
            apply_fusion=False,
        )
        executor = MlirExecutor(mlir_mod, PyTorchBackend("cpu"))
        logits = executor.forward(input_ids)

        assert logits.shape[-1] == 50272, f"Bad vocab dim: {logits.shape[-1]}"
        assert not torch.isnan(logits).any(), "NaN in output"
        assert not torch.isinf(logits).any(), "Inf in output"
        print(f"  Forward OK: shape {list(logits.shape)}")
