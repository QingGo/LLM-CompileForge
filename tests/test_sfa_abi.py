"""TDD tests for compiler/sfa_abi.py — SfaAbiHeader generation from LLVM IR."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from gen.proto.python.sfa_abi_pb2 import (
    SfaAbiHeader,
    SfaInputKind,
)

# Unit tests — fast, no external deps
pytestmark = pytest.mark.unit


# ── Constants ────────────────────────────────────────────────────────

SFA_MAGIC = 0x41464253  # "SFBA" in LE bytes


# ── Mock LLVM IR generators ──────────────────────────────────────────


def _mock_llvm_ir_single_func() -> str:
    """Mock LLVM IR with one ciface wrapper (sret + 2 inputs, rank-3 output)."""
    return """
; ModuleID = 'mock'
target triple = "arm64-apple-macosx"

define void @_mlir_ciface_main_0(ptr %0, ptr %1, ptr %2) {
  %4 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %1, align 8
  %5 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %4, 0
  %6 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %2, align 8
  %7 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, 0
  %8 = call { ptr, ptr, i64, [3 x i64], [3 x i64] } @main_0(ptr %5, ptr %7)
  store { ptr, ptr, i64, [3 x i64], [3 x i64] } %8, ptr %0, align 8
  ret void
}
"""


def _mock_llvm_ir_two_funcs() -> str:
    """Mock LLVM IR with two ciface wrappers (different output ranks)."""
    return """
; ModuleID = 'mock'
target triple = "arm64-apple-macosx"

define void @_mlir_ciface_main_0(ptr %0, ptr %1, ptr %2) {
  %4 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %1, align 8
  %5 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %4, 0
  %6 = call { ptr, ptr, i64, [3 x i64], [3 x i64] } @main_0(ptr %5)
  store { ptr, ptr, i64, [3 x i64], [3 x i64] } %6, ptr %0, align 8
  ret void
}

define void @_mlir_ciface_main_1(ptr %0, ptr %1, ptr %2, ptr %3) {
  %5 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %1, align 8
  %6 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %5, 0
  %7 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %2, align 8
  %8 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %7, 0
  %9 = call { ptr, ptr, i64, [1 x i64], [1 x i64] } @main_1(ptr %6, ptr %8)
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %9, ptr %0, align 8
  ret void
}
"""


def _mock_llvm_ir_many_args() -> str:
    """Mock LLVM IR with ciface wrapper that has 7 ptr args (sret + 6 inputs)."""
    return """
; ModuleID = 'mock'

define void @_mlir_ciface_main_0(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6) {
  %8 = call { ptr, ptr, i64, [2 x i64], [2 x i64] } @main_0(ptr null, ptr null, ptr null, ptr null, ptr null, ptr null)
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, ptr %0, align 8
  ret void
}
"""


def _mock_pre_lowering_module():
    """Build a mock pre-lowering module with 2 functions for merge_with_semantics."""
    return {
        "functions": [
            {
                "name": "main_0",
                "inputs": [
                    # (name, type_str)
                    ("input_ids", "tensor<4x64xf32>"),
                    ("position_ids", "tensor<4xf32>"),
                ],
                "weights": {
                    "wte.weight": "tensor<50257x64xf32>",
                },
                "weight_ops": [
                    {"name": "wte.weight", "output_types": ["tensor<50257x64xf32>"]},
                ],
            },
            {
                "name": "main_1",
                "inputs": [
                    ("hidden_states", "tensor<4x64x64xf32>"),
                ],
                "weights": {},
                "weight_ops": [],
            },
        ],
    }


def _write_temp_ll(content: str) -> str:
    """Write LLVM IR content to a temp file, return path."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ll", delete=False) as f:
        f.write(content)
        f.flush()
        return f.name


def _parse_abi(data: bytes) -> SfaAbiHeader:
    """Deserialize protobuf SfaAbiHeader from bytes."""
    header = SfaAbiHeader()
    header.ParseFromString(data)
    return header


# ── Tests: parse_ciface_signatures ───────────────────────────────────


class TestParseCifaceSignatures:
    """parse_ciface_signatures: extract function name → (num_args, output_rank)."""

    def test_single_func(self):
        """One ciface wrapper → correct (num_args=3, output_rank=3)."""
        from compiler.sfa_abi import parse_ciface_signatures

        path = _write_temp_ll(_mock_llvm_ir_single_func())
        try:
            sigs = parse_ciface_signatures(path)
            assert "_mlir_ciface_main_0" in sigs
            num_args, output_rank = sigs["_mlir_ciface_main_0"]
            # 3 ptr params: %0 (sret), %1 (input), %2 (input)
            assert num_args == 3
            assert output_rank == 3
        finally:
            Path(path).unlink()

    def test_two_funcs(self):
        """Two ciface wrappers → both signatures extracted."""
        from compiler.sfa_abi import parse_ciface_signatures

        path = _write_temp_ll(_mock_llvm_ir_two_funcs())
        try:
            sigs = parse_ciface_signatures(path)
            assert len(sigs) == 2
            # main_0: 3 ptr params, rank-3 output
            assert sigs["_mlir_ciface_main_0"] == (3, 3)
            # main_1: 4 ptr params, rank-1 output
            assert sigs["_mlir_ciface_main_1"] == (4, 1)
        finally:
            Path(path).unlink()

    def test_many_args(self):
        """7 ptr args → num_args=7, output_rank=2."""
        from compiler.sfa_abi import parse_ciface_signatures

        path = _write_temp_ll(_mock_llvm_ir_many_args())
        try:
            sigs = parse_ciface_signatures(path)
            assert sigs["_mlir_ciface_main_0"] == (7, 2)
        finally:
            Path(path).unlink()

    def test_empty_file(self):
        """No ciface functions → empty dict."""
        from compiler.sfa_abi import parse_ciface_signatures

        path = _write_temp_ll("; no ciface functions here\n")
        try:
            sigs = parse_ciface_signatures(path)
            assert sigs == {}
        finally:
            Path(path).unlink()


# ── Tests: merge_with_semantics ──────────────────────────────────────


class TestMergeWithSemantics:
    """merge_with_semantics: combine LLVM signatures with pre-lowering module."""

    def test_basic(self):
        """Merge → correct SfaFuncMeta list with input fields."""
        from compiler.sfa_abi import merge_with_semantics

        sigs = {
            "_mlir_ciface_main_0": (3, 3),
            "_mlir_ciface_main_1": (4, 1),
        }
        module = _mock_pre_lowering_module()

        metas = merge_with_semantics(sigs, module)
        assert len(metas) == 2

        # func 0
        assert metas[0]["symbol"] == "_mlir_ciface_main_0"
        assert metas[0]["num_inputs"] == 3
        assert metas[0]["output_rank"] == 3
        assert "input_fields" in metas[0]
        # First two inputs of func 0 are global inputs
        assert metas[0]["input_fields"][0]["kind"] == 0  # GLOBAL
        assert metas[0]["input_fields"][1]["kind"] == 0  # GLOBAL

        # func 1
        assert metas[1]["symbol"] == "_mlir_ciface_main_1"
        assert metas[1]["num_inputs"] == 4
        assert metas[1]["output_rank"] == 1

    def test_ssa_connection(self):
        """When input %name matches a producer output → SSA kind."""
        from compiler.sfa_abi import merge_with_semantics

        sigs = {
            "_mlir_ciface_main_0": (3, 3),
            "_mlir_ciface_main_1": (3, 1),
        }
        module = {
            "functions": [
                {
                    "name": "main_0",
                    "inputs": [("input_ids", "tensor<4x64xf32>")],
                    "weights": {},
                    "weight_ops": [],
                },
                {
                    "name": "main_1",
                    "inputs": [
                        ("hidden_states_0_main_0_0", "tensor<4x64x64xf32>"),
                    ],
                    "weights": {},
                    "weight_ops": [],
                },
            ],
        }

        metas = merge_with_semantics(sigs, module)
        # func 1's input should be SSA from func 0
        assert metas[1]["input_fields"][0]["kind"] == 2  # SFA_INPUT_SSA

    def test_weight_input(self):
        """Weight ops in pre-lowering module → WEIGHT kind."""
        from compiler.sfa_abi import merge_with_semantics

        sigs = {
            "_mlir_ciface_main_0": (3, 3),
        }
        module = {
            "functions": [
                {
                    "name": "main_0",
                    "inputs": [("input_ids", "tensor<4x64xf32>")],
                    "weights": {},
                    "weight_ops": [
                        {"name": "wte.weight", "output_types": ["tensor<50257x64xf32>"]},
                    ],
                },
            ],
        }

        metas = merge_with_semantics(sigs, module)
        fields = metas[0]["input_fields"]
        assert len(fields) == 2  # global input + weight
        assert fields[0]["kind"] == 0  # GLOBAL
        assert fields[1]["kind"] == 1  # WEIGHT
        assert fields[1]["weight_name"] == "wte.weight"


# ── Tests: serialize_abi ─────────────────────────────────────────────


def _default_meta(symbol="_mlir_ciface_main_0", num_inputs=2, output_rank=3):
    return {
        "symbol": symbol,
        "num_inputs": num_inputs,
        "output_rank": output_rank,
        "input_fields": [
            {"kind": SfaInputKind.Value("SFA_INPUT_GLOBAL")},
            {"kind": SfaInputKind.Value("SFA_INPUT_WEIGHT"), "weight_name": "wte.weight"},
        ],
    }


class TestSerializeAbi:
    """serialize_abi: produce protobuf SfaAbiHeader binary."""

    def test_magic(self):
        """Protobuf header has correct magic value."""
        from compiler.sfa_abi import serialize_abi

        metas = [_default_meta()]
        data = serialize_abi(metas)
        header = _parse_abi(data)
        assert header.magic == SFA_MAGIC

    def test_header(self):
        """Header has correct magic, version=1, num_funcs."""
        from compiler.sfa_abi import serialize_abi

        metas = [_default_meta(), _default_meta(symbol="_mlir_ciface_main_1")]
        data = serialize_abi(metas)
        header = _parse_abi(data)

        assert header.magic == SFA_MAGIC
        assert header.version == 1
        assert len(header.funcs) == 2

    def test_func_meta(self):
        """SfaFuncMeta fields are preserved."""
        from compiler.sfa_abi import serialize_abi

        metas = [_default_meta()]
        data = serialize_abi(metas)
        header = _parse_abi(data)

        func = header.funcs[0]
        assert func.num_inputs == 2
        assert func.output_rank == 3
        assert func.symbol == "_mlir_ciface_main_0"

    def test_string_table(self):
        """Function symbols are preserved in protobuf fields."""
        from compiler.sfa_abi import serialize_abi

        metas = [
            _default_meta(symbol="_mlir_ciface_main_0"),
            _default_meta(symbol="_mlir_ciface_main_1"),
        ]
        data = serialize_abi(metas)
        header = _parse_abi(data)

        assert header.funcs[0].symbol == "_mlir_ciface_main_0"
        assert header.funcs[1].symbol == "_mlir_ciface_main_1"

    def test_input_fields(self):
        """SfaInputField array has correct kind values."""
        from compiler.sfa_abi import serialize_abi

        metas = [_default_meta(num_inputs=2)]
        data = serialize_abi(metas)
        header = _parse_abi(data)

        func = header.funcs[0]
        assert len(func.input_fields) == 2
        assert func.input_fields[0].kind == SfaInputKind.SFA_INPUT_GLOBAL
        assert func.input_fields[1].kind == SfaInputKind.SFA_INPUT_WEIGHT
        assert func.input_fields[1].weight_name == "wte.weight"

    def test_ssa_field(self):
        """SSA input field has producer_func and producer_out."""
        from compiler.sfa_abi import serialize_abi

        metas = [
            {
                "symbol": "_mlir_ciface_main_0",
                "num_inputs": 1,
                "output_rank": 3,
                "input_fields": [
                    {
                        "kind": SfaInputKind.Value("SFA_INPUT_SSA"),
                        "producer_func": 0,
                        "producer_out": 0,
                    }
                ],
            }
        ]
        data = serialize_abi(metas)
        header = _parse_abi(data)

        field = header.funcs[0].input_fields[0]
        assert field.kind == SfaInputKind.SFA_INPUT_SSA
        assert field.ssa.producer_func == 0
        assert field.ssa.producer_out == 0

    def test_roundtrip(self):
        """Full round-trip: multi-func metas → serialize → parse all fields."""
        from compiler.sfa_abi import serialize_abi

        metas = [
            {
                "symbol": "_mlir_ciface_main_0",
                "num_inputs": 3,
                "output_rank": 3,
                "input_fields": [
                    {"kind": SfaInputKind.Value("SFA_INPUT_GLOBAL")},
                    {"kind": SfaInputKind.Value("SFA_INPUT_GLOBAL")},
                    {"kind": SfaInputKind.Value("SFA_INPUT_WEIGHT"), "weight_name": "wte.weight"},
                ],
            },
            {
                "symbol": "_mlir_ciface_main_1",
                "num_inputs": 2,
                "output_rank": 1,
                "input_fields": [
                    {
                        "kind": SfaInputKind.Value("SFA_INPUT_SSA"),
                        "producer_func": 0,
                        "producer_out": 0,
                    },
                    {"kind": SfaInputKind.Value("SFA_INPUT_WEIGHT"), "weight_name": "ln.weight"},
                ],
            },
        ]
        data = serialize_abi(metas)
        header = _parse_abi(data)

        # Header
        assert header.magic == SFA_MAGIC
        assert header.version == 1
        assert len(header.funcs) == 2

        # Func 0
        f0 = header.funcs[0]
        assert f0.symbol == "_mlir_ciface_main_0"
        assert f0.num_inputs == 3
        assert f0.output_rank == 3
        assert len(f0.input_fields) == 3
        assert f0.input_fields[0].kind == SfaInputKind.SFA_INPUT_GLOBAL
        assert f0.input_fields[1].kind == SfaInputKind.SFA_INPUT_GLOBAL
        assert f0.input_fields[2].kind == SfaInputKind.SFA_INPUT_WEIGHT

        # Func 1
        f1 = header.funcs[1]
        assert f1.symbol == "_mlir_ciface_main_1"
        assert f1.num_inputs == 2
        assert f1.output_rank == 1
        assert len(f1.input_fields) == 2
        assert f1.input_fields[0].kind == SfaInputKind.SFA_INPUT_SSA
        assert f1.input_fields[1].kind == SfaInputKind.SFA_INPUT_WEIGHT
