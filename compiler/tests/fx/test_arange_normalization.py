"""TDD tests: torch.arange → sf.arange operand normalization.

Contract (see sf-dialect/lib/Sf/SfLowerGenOps.cpp + compiler/tests/test_arange_fix.py):

  ``sf.arange(start, [size])`` produces ``[start, start+1, ..., start+size-1]``.

``converter._collect_arange_args`` must therefore emit the FIRST operand as
the START value (default 0) and the SECOND (dyn_shape[0]) as the SIZE
(= end - start), for both ``torch.arange(end)`` and
``torch.arange(start, end)``.  step != 1 and symbolic starts are rejected.
"""

from __future__ import annotations

import pytest
import torch

from compiler.fx.converter import fx_graph_to_mlir


def _arange_ops(
    model: torch.nn.Module,
    x: torch.Tensor,
    dynamic_shapes: dict | None = None,
) -> list:
    exported = torch.export.export(model, (x,), dynamic_shapes=dynamic_shapes)
    mod = fx_graph_to_mlir(exported)
    ops = []
    for f in mod.functions:
        for op in f.ops:
            if op.op_name == "arange":
                ops.append(op)
    return ops


class _EndOnly(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.arange(5).sum()


class _StartEnd(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.arange(2, 7).sum()


class _FloatStart(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # .to(int64) keeps the add type-consistent (sf.arange emits i64;
        # float-start arange is only exercised for the START operand dtype).
        return x + torch.arange(2.0, 7).to(torch.int64).sum()


class _SymbolicEnd(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.arange(x.shape[1]).sum()


class _StepTwo(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.arange(1, 5, 2).sum()


class TestArangeNormalization:
    def test_end_only_emits_start_zero_and_size(self) -> None:
        """torch.arange(5) → input=0 (start), dyn_shape=[5] (size)."""
        x = torch.randint(0, 10, (2, 3))
        mod = fx_graph_to_mlir(torch.export.export(_EndOnly(), (x,)))
        ops = [o for f in mod.functions for o in f.ops if o.op_name == "arange"]
        assert len(ops) == 1
        op = ops[0]
        assert len(op.operands) == 2, f"expected [start, size], got {op.operands}"

        # Find the const weights' values by scanning module weights
        vals = {}
        for f in mod.functions:
            for wname, tensor in f.weights.items():
                if tensor.numel() == 1:
                    vals[wname] = tensor.item()
        assert vals[op.operands[0]] == 0
        assert vals[op.operands[1]] == 5

    def test_start_end_emits_start_and_size(self) -> None:
        """torch.arange(2, 7) → input=2, dyn_shape=[5]. (Not [2, 7].)"""
        mod = fx_graph_to_mlir(torch.export.export(_StartEnd(), (torch.randint(0, 10, (2, 3)),)))
        vals = {}
        op = None
        for f in mod.functions:
            for wname, tensor in f.weights.items():
                if tensor.numel() == 1:
                    vals[wname] = tensor.item()
            for o in f.ops:
                if o.op_name == "arange":
                    op = o
        assert op is not None and len(op.operands) == 2
        assert vals[op.operands[0]] == 2
        assert vals[op.operands[1]] == 5

    def test_float_start_keeps_f32_start_and_i64_size(self) -> None:
        """torch.arange(2.0, 7) → f32 start const, i64 size const."""
        mod = fx_graph_to_mlir(torch.export.export(_FloatStart(), (torch.randint(0, 10, (2, 3)),)))
        op = None
        vals = {}
        for f in mod.functions:
            for wname, tensor in f.weights.items():
                if tensor.numel() == 1:
                    vals[wname] = (tensor.item(), str(tensor.dtype))
            for o in f.ops:
                if o.op_name == "arange":
                    op = o
        assert op is not None and len(op.operands) == 2
        assert vals[op.operands[0]][0] == 2.0
        assert "float32" in vals[op.operands[0]][1]
        assert vals[op.operands[1]] == (5, "torch.int64")

    def test_symbolic_end_with_zero_start_emits_ssa_size(self) -> None:
        """torch.arange(sym_seq) → input=0 const, dyn_shape=[sym_seq ssa].

        Requires a dynamic seq dim so x.shape[1] stays symbolic in the
        export (a static shape would fold to an int and emit a const).
        """
        from torch.export import Dim

        ops = _arange_ops(
            _SymbolicEnd(),
            torch.randint(0, 10, (2, 3)),
            dynamic_shapes={"x": {1: Dim("seq")}},
        )
        assert len(ops) == 1
        assert len(ops[0].operands) == 2
        assert ops[0].operands[0].startswith("_const_")
        assert ops[0].operands[1].startswith("%")

    def test_step_not_one_rejected(self) -> None:
        """torch.arange with step != 1 must fail loudly, not silently miscompile."""
        with pytest.raises(NotImplementedError):
            _arange_ops(_StepTwo(), torch.randint(0, 10, (2, 3)))
