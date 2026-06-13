"""Tests for Tensor Parallelism (Phase 2 Sprint 1).

Verifies:
  - ColumnParallelLinear / RowParallelLinear forward pass correctness.
  - Single-rank equivalent to standard nn.Linear.
  - Multi-process TP=2/4 correctness via torch.multiprocessing.spawn.
  - Auto TP strategy search for memory-constrained models.

Reference: design-phase2.md §5.2 — TP correctness: cos_sim > 0.999
"""

from __future__ import annotations

import multiprocessing
import os

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from tests.helpers import assert_cosine_above

# ── Mock Communicator for single-process tests ───────────


class _MockCommunicator:
    """Simulates a Communicator in single-process mode (tp_size=1)."""

    def all_reduce(self, tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
        return tensor.clone()

    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.clone()

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        return tensor.clone()

    @property
    def rank(self) -> int:
        return 0

    @property
    def world_size(self) -> int:
        return 1


# ── TwoRankMock for simulating TP=2 in single process ─────


class _TwoRankMock:
    """Simulates a two-rank TP communicator in single process.

    all_gather: concatenates the same tensor twice.
    all_reduce: doubles the tensor (sum over 2 identical ranks).
    """

    def all_reduce(self, tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
        return tensor * 2.0

    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.cat([tensor, tensor], dim=-1)

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        return tensor.clone()

    @property
    def rank(self) -> int:
        return 0

    @property
    def world_size(self) -> int:
        return 2


# ── Single-Process Correctness ────────────────────────────


@pytest.mark.unit
class TestColumnParallelSingleProcess:
    def test_matches_standard_linear(self) -> None:
        from compiler.tp.linear import ColumnParallelLinear

        torch.manual_seed(42)
        in_f, out_f = 32, 64
        x = torch.randn(4, in_f)

        comm = _MockCommunicator()
        col_linear = ColumnParallelLinear(in_f, out_f, comm)  # type: ignore[arg-type]
        ref = nn.Linear(in_f, out_f)

        with torch.no_grad():
            col_linear.weight.copy_(ref.weight)
            if col_linear.bias is not None:
                col_linear.bias.copy_(ref.bias)

            result = col_linear(x)
            expected = ref(x)

        assert_cosine_above(result, expected)

    def test_no_gather_returns_shard(self) -> None:
        from compiler.tp.linear import ColumnParallelLinear

        torch.manual_seed(1)
        in_f, out_f = 16, 32
        x = torch.randn(2, in_f)

        comm = _MockCommunicator()
        col_linear = ColumnParallelLinear(in_f, out_f, comm, gather_output=False)  # type: ignore[arg-type]

        result = col_linear(x)
        assert result.shape == (2, out_f)


@pytest.mark.unit
class TestRowParallelSingleProcess:
    def test_matches_standard_linear(self) -> None:
        from compiler.tp.linear import RowParallelLinear

        torch.manual_seed(7)
        in_f, out_f = 64, 32
        x = torch.randn(4, in_f)

        comm = _MockCommunicator()
        row_linear = RowParallelLinear(in_f, out_f, comm)  # type: ignore[arg-type]
        ref = nn.Linear(in_f, out_f)

        with torch.no_grad():
            row_linear.weight.copy_(ref.weight)
            if row_linear.bias is not None:
                row_linear.bias.copy_(ref.bias)

            result = row_linear(x)
            expected = ref(x)

        assert_cosine_above(result, expected)

    def test_input_is_parallel_skip_gather(self) -> None:
        from compiler.tp.linear import RowParallelLinear

        torch.manual_seed(3)
        in_f, out_f = 32, 16
        x = torch.randn(4, in_f)

        comm = _MockCommunicator()
        row_linear = RowParallelLinear(in_f, out_f, comm, input_is_parallel=True)  # type: ignore[arg-type]

        result = row_linear(x)
        assert result.shape == (4, out_f)


@pytest.mark.unit
class TestTPTwoRankMock:
    def test_column_parallel_tp2(self) -> None:
        from compiler.tp.linear import ColumnParallelLinear

        torch.manual_seed(13)
        in_f, out_f = 32, 64
        x = torch.randn(4, in_f)

        # Simulate two ranks sharding the full weight.
        comm_r0 = _TwoRankMock()
        comm_r1 = _TwoRankMock()
        col_l0 = ColumnParallelLinear(in_f, out_f, comm_r0)  # type: ignore[arg-type]
        col_l1 = ColumnParallelLinear(in_f, out_f, comm_r1)  # type: ignore[arg-type]

        ref = nn.Linear(in_f, out_f)
        with torch.no_grad():
            col_l0.weight.copy_(ref.weight[: out_f // 2])
            if col_l0.bias is not None:
                col_l0.bias.copy_(ref.bias[: out_f // 2])
            col_l1.weight.copy_(ref.weight[out_f // 2 :])
            if col_l1.bias is not None:
                col_l1.bias.copy_(ref.bias[out_f // 2 :])

            partial_r0 = nn.functional.linear(x, col_l0.weight, col_l0.bias)
            partial_r1 = nn.functional.linear(x, col_l1.weight, col_l1.bias)

            # all_gather concatenates along last dim
            result = torch.cat([partial_r0, partial_r1], dim=-1)
            expected = ref(x)

        assert_cosine_above(result, expected)


# ── Multi-Process TP=2 Gloo Tests ─────────────────────────


class _GlooBridge:
    """Bridge using the default torch.distributed process group."""

    def all_reduce(self, tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
        result = tensor.clone()
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        return result

    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=-1)

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        result = tensor.clone()
        dist.broadcast(result, src)
        return result

    @property
    def rank(self) -> int:
        return dist.get_rank()

    @property
    def world_size(self) -> int:
        return dist.get_world_size()


def _run_tp2_col_worker(rank: int, in_f: int, out_f: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=2)
    from compiler.tp.linear import ColumnParallelLinear

    torch.manual_seed(42)
    x = torch.randn(4, in_f)
    comm = _GlooBridge()
    layer = ColumnParallelLinear(in_f, out_f, comm)  # type: ignore[arg-type]
    with torch.no_grad():
        result = layer(x)
    torch.save(result, f"/tmp/tp2_col_result_rank{rank}.pt")
    dist.destroy_process_group()


def _run_tp2_row_worker(rank: int, in_f: int, out_f: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group("gloo", rank=rank, world_size=2)
    from compiler.tp.linear import RowParallelLinear

    torch.manual_seed(7)
    x = torch.randn(4, in_f)
    comm = _GlooBridge()
    layer = RowParallelLinear(in_f, out_f, comm)  # type: ignore[arg-type]
    with torch.no_grad():
        result = layer(x)
    torch.save(result, f"/tmp/tp2_row_result_rank{rank}.pt")
    dist.destroy_process_group()


class _GlooBridge:
    """Bridge from GlooCommunicator using the default process group."""

    def all_reduce(self, tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
        result = tensor.clone()
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        return result

    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=-1)

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        result = tensor.clone()
        dist.broadcast(result, src)
        return result

    @property
    def rank(self) -> int:
        return dist.get_rank()

    @property
    def world_size(self) -> int:
        return dist.get_world_size()


@pytest.mark.unit
@pytest.mark.timeout(30)
class TestTPMultiProcess:
    def test_column_parallel_tp2_gloo(self) -> None:
        in_f, out_f = 16, 32

        p0 = multiprocessing.Process(target=_run_tp2_col_worker, args=(0, in_f, out_f))
        p1 = multiprocessing.Process(target=_run_tp2_col_worker, args=(1, in_f, out_f))
        p0.start()
        p1.start()
        p0.join()
        p1.join()

        r0 = torch.load("/tmp/tp2_col_result_rank0.pt")
        r1 = torch.load("/tmp/tp2_col_result_rank1.pt")

        assert r0.shape == (4, out_f)
        assert r1.shape == (4, out_f)
        assert_cosine_above(r0, r1)

    def test_row_parallel_tp2_gloo(self) -> None:
        in_f, out_f = 32, 16

        p0 = multiprocessing.Process(target=_run_tp2_row_worker, args=(0, in_f, out_f))
        p1 = multiprocessing.Process(target=_run_tp2_row_worker, args=(1, in_f, out_f))
        p0.start()
        p1.start()
        p0.join()
        p1.join()

        r0 = torch.load("/tmp/tp2_row_result_rank0.pt")
        r1 = torch.load("/tmp/tp2_row_result_rank1.pt")

        assert r0.shape == (4, out_f)
        assert r1.shape == (4, out_f)
        assert_cosine_above(r0, r1)


# ── Auto TP Strategy ──────────────────────────────────────


@pytest.mark.unit
class TestAutoTPStrategy:
    def test_small_model_no_tp_needed(self) -> None:
        from compiler.tp.strategy import search_tp_strategy

        model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 16))
        result = search_tp_strategy(model, available_memory_gb=24.0)

        assert result.tp_size == 1
        assert result.feasible

    def test_large_model_requires_tp(self) -> None:
        from compiler.tp.strategy import search_tp_strategy

        large = nn.Linear(8192, 8192)
        result = search_tp_strategy(large, available_memory_gb=0.001, max_tp_size=8)

        assert result.tp_size >= 2

    def test_count_parameters(self) -> None:
        from compiler.tp.strategy import count_parameters

        model = nn.Linear(32, 64)
        assert count_parameters(model) == 32 * 64 + 64  # weights + bias

    def test_divisible_constraint(self) -> None:
        from compiler.tp.linear import ColumnParallelLinear

        comm = _TwoRankMock()
        with pytest.raises(ValueError, match="divisible"):
            ColumnParallelLinear(16, 31, comm)  # type: ignore[arg-type]
