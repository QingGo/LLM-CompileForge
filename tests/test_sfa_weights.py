"""TDD tests for compiler/sfa_weights.py — protobuf SfaWeightData serialization."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from gen.proto.python.sfa_abi_pb2 import SfaWeightData

# Unit tests — fast, no external deps
pytestmark = pytest.mark.unit

_DTYPE_MAP: dict[int, torch.dtype] = {
    0: torch.float32,
    1: torch.float16,
    2: torch.bfloat16,
    3: torch.int64,
    4: torch.int32,
    5: torch.int8,
    6: torch.uint8,
}


def _deserialize_weight_data(data: bytes) -> tuple[dict[str, str], dict[str, Any]]:
    """Parse protobuf SfaWeightData binary back into (name_mapping, constants).

    Returns:
        tuple of (name_mapping: {compiled_name: hf_key}, constants: {name: tensor})
    """
    msg = SfaWeightData()
    msg.ParseFromString(data)

    name_mapping: dict[str, str] = {}
    for entry in msg.weight_entries:
        name_mapping[entry.compiled_name] = entry.hf_key

    constants: dict[str, Any] = {}
    for entry in msg.constant_entries:
        shape = tuple(entry.shape)
        dtype = _DTYPE_MAP.get(entry.dtype_code, torch.float32)

        tensor_data = entry.data
        if dtype == torch.float16:
            arr = np.frombuffer(tensor_data, dtype=np.float16).reshape(shape)
        elif dtype == torch.bfloat16:
            element_count = int(np.prod(shape))
            arr = np.zeros(element_count, dtype=np.float32)
            arr_view = arr.view(np.uint16)
            for i in range(len(arr_view) // 2):
                arr_view[i * 2 + 1] = np.frombuffer(tensor_data[i * 2 : i * 2 + 2], dtype=np.uint16)[0]
            arr = arr.reshape(shape).view(torch.bfloat16)
        elif dtype == torch.int64:
            arr = np.frombuffer(tensor_data, dtype=np.int64).reshape(shape)
        elif dtype == torch.int32:
            arr = np.frombuffer(tensor_data, dtype=np.int32).reshape(shape)
        elif dtype == torch.int8:
            arr = np.frombuffer(tensor_data, dtype=np.int8).reshape(shape)
        elif dtype == torch.uint8:
            arr = np.frombuffer(tensor_data, dtype=np.uint8).reshape(shape)
        else:
            arr = np.frombuffer(tensor_data, dtype=np.float32).reshape(shape)

        constants[entry.name] = torch.from_numpy(arr.copy())

    return name_mapping, constants


# ── Test fixtures ─────────────────────────────────────────────────────


def _mock_name_mapping() -> dict[str, str]:
    """3 weight mappings typical for a small model."""
    return {
        "wte_weight": "model.decoder.embed_tokens.weight",
        "layer_0_self_attn_k_proj_weight": "model.decoder.layers.0.self_attn.k_proj.weight",
        "layer_0_self_attn_v_proj_weight": "model.decoder.layers.0.self_attn.v_proj.weight",
    }


def _mock_constants() -> dict[str, torch.Tensor]:
    """2 constant tensors: a small float scalar and a 1D int64 tensor."""
    return {
        "_const_0": torch.tensor(0.125, dtype=torch.float32),
        "_const_causal_mask": torch.tensor([0, -65504, -65504, -65504], dtype=torch.float16),
    }


# ── Tests ─────────────────────────────────────────────────────────────


class TestBuildWeightData:
    """build_weight_data: serialize name_mapping + constants to protobuf SfaWeightData."""

    def test_empty_inputs(self):
        """Empty name_mapping and constants → produces valid protobuf binary."""
        from compiler.sfa_weights import build_weight_data

        data = build_weight_data({}, {})
        msg = SfaWeightData()
        msg.ParseFromString(data)
        assert len(msg.weight_entries) == 0
        assert len(msg.constant_entries) == 0

    def test_weights_only_no_constants(self):
        """3 weight mappings, no constants → serializes all weight entries."""
        from compiler.sfa_weights import build_weight_data

        name_mapping = _mock_name_mapping()
        data = build_weight_data(name_mapping, {})

        parsed_mapping, parsed_constants = _deserialize_weight_data(data)
        assert parsed_mapping == name_mapping
        assert parsed_constants == {}

    def test_constants_only_no_weights(self):
        """2 constant tensors, no weights → serializes all constant entries."""
        from compiler.sfa_weights import build_weight_data

        constants = _mock_constants()
        data = build_weight_data({}, constants)

        parsed_mapping, parsed_constants = _deserialize_weight_data(data)
        assert parsed_mapping == {}
        assert len(parsed_constants) == 2

        # Verify constant values
        for name, tensor in constants.items():
            assert name in parsed_constants
            parsed = parsed_constants[name]
            assert parsed.shape == tensor.shape
            assert parsed.dtype == tensor.dtype
            assert torch.equal(parsed, tensor), f"Constant {name} value mismatch"

    def test_roundtrip_weights_and_constants(self):
        """3 weights + 2 constants → roundtrip preserves all entries and values."""
        from compiler.sfa_weights import build_weight_data

        name_mapping = _mock_name_mapping()
        constants = _mock_constants()
        data = build_weight_data(name_mapping, constants)

        parsed_mapping, parsed_constants = _deserialize_weight_data(data)

        # Verify all weight mappings preserved
        assert len(parsed_mapping) == 3
        assert parsed_mapping == name_mapping

        # Verify all constants preserved
        assert len(parsed_constants) == 2
        for name, tensor in constants.items():
            assert name in parsed_constants
            parsed = parsed_constants[name]
            assert parsed.shape == tensor.shape, f"Shape mismatch for {name}"
            assert parsed.dtype == tensor.dtype, f"Dtype mismatch for {name}"
            assert torch.equal(parsed, tensor), f"Value mismatch for {name}"

    def test_unicode_hf_keys(self):
        """Weight mapping with Unicode paths in HF keys → roundtrip preserves UTF-8."""
        from compiler.sfa_weights import build_weight_data

        name_mapping = {
            "wte_weight": "modèle.décodeur.embed_tokens.poids",
            "ln_weight": "layer.0.layer_norm.weight",
        }
        data = build_weight_data(name_mapping, {})

        parsed_mapping, _ = _deserialize_weight_data(data)
        assert parsed_mapping == name_mapping

    def test_constants_multiple_dtypes(self):
        """Constants with float32, float16, int64, int32 → all roundtrip correctly."""
        from compiler.sfa_weights import build_weight_data

        constants = {
            "_f32": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
            "_f16": torch.tensor([0.5, -1.5], dtype=torch.float16),
            "_i64": torch.tensor([42, 99], dtype=torch.int64),
            "_i32": torch.tensor([7], dtype=torch.int32),
        }
        data = build_weight_data({}, constants)

        _, parsed_constants = _deserialize_weight_data(data)
        assert len(parsed_constants) == 4
        for name, tensor in constants.items():
            assert name in parsed_constants
            assert torch.equal(parsed_constants[name], tensor), f"{name} mismatch"

    def test_build_weight_data_function_exists(self):
        """Smoke test: function is importable and callable with correct signature."""
        from compiler.sfa_weights import build_weight_data

        result = build_weight_data({}, {})
        assert isinstance(result, bytes)
