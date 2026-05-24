"""Pytest entry point for per-operator correctness tests.

Each op in ``OP_TABLE`` is parameterized as a separate test case.
A test passes if the cosine similarity between JIT-compiled output
and PyTorch reference exceeds ``1 - case.rtol``.
"""

from __future__ import annotations

import logging

import pytest

from tests.op_correctness.registry import OP_TABLE
from tests.op_correctness.runner import Runner

_log = logging.getLogger(__name__)


@pytest.mark.parametrize("case", OP_TABLE, ids=lambda c: c.name)
def test_op_correctness(case: object) -> None:
    """Compare JIT output of a single sf op against the PyTorch reference."""
    result = Runner(case).run()
    min_cos = 1.0 - case.rtol
    assert result.cos > min_cos, (
        f"{case.name}: cos={result.cos:.8f} < {min_cos} (rtol={case.rtol}) "
        f"shape={result.output.shape}"
    )
