"""Tensor Parallelism — Megatron-LM style column/row-parallel layers.

Provides ColumnParallelLinear and RowParallelLinear that shard model
weights across ranks and use HAL Communicator for collective ops.

Reference: design-phase2.md §2.4.1, §2.4.2
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from compiler.tp.communicator_protocol import TPCommunicator


class ColumnParallelLinear(nn.Module):
    """Linear layer with output dimension sharded across ranks.

    The weight [out_features, in_features] is split by rows:
    each rank holds [out_features // tp_size, in_features].
    Outputs are gathered via all_gather.

    Args:
        in_features: Input feature dimension.
        out_features: Output feature dimension.
        comm: HAL Communicator for the TP group.
        gather_output: If True, all-gather partial outputs (default).
                       If False, return local shard only.
        bias: Whether to include bias (sharded like weight).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        comm: TPCommunicator,
        gather_output: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.comm = comm
        tp_size = comm.world_size
        if out_features % tp_size != 0:
            raise ValueError(f"out_features ({out_features}) must be divisible by tp_size ({tp_size})")
        self.out_per_rank = out_features // tp_size
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output

        self.weight = nn.Parameter(torch.empty(self.out_per_rank, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_per_rank))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = self.weight.size(1)
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        partial_out = F.linear(x, self.weight, self.bias)
        if self.gather_output:
            return self.comm.all_gather(partial_out)
        return partial_out


class RowParallelLinear(nn.Module):
    """Linear layer with input dimension sharded across ranks.

    The weight [out_features, in_features] is split by columns:
    each rank holds [out_features, in_features // tp_size].
    Inputs are gathered via all_gather (if not already sharded),
    and partial outputs are summed via all_reduce.

    Args:
        in_features: Input feature dimension.
        out_features: Output feature dimension.
        comm: HAL Communicator for the TP group.
        input_is_parallel: If True, input is already sharded along last dim.
                           If False, input is gathered first.
        bias: Whether to include bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        comm: TPCommunicator,
        input_is_parallel: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.comm = comm
        tp_size = comm.world_size
        if in_features % tp_size != 0:
            raise ValueError(f"in_features ({in_features}) must be divisible by tp_size ({tp_size})")
        self.in_per_rank = in_features // tp_size
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel

        self.weight = nn.Parameter(torch.empty(out_features, self.in_per_rank))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = self.weight.size(1)
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.input_is_parallel:
            start = self.comm.rank * self.in_per_rank
            end = start + self.in_per_rank
            x = x[..., start:end]
        partial_out = F.linear(x, self.weight, self.bias)
        return self.comm.all_reduce(partial_out)


class VocabParallelEmbedding(nn.Module):
    """Embedding sharded along the vocabulary dimension.

    Each rank holds embeddings for vocab_size // tp_size tokens.

    Args:
        num_embeddings: Total vocabulary size.
        embedding_dim: Embedding dimension.
        comm: HAL Communicator for the TP group.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        comm: TPCommunicator,
    ) -> None:
        super().__init__()
        self.comm = comm
        tp_size = comm.world_size
        if num_embeddings % tp_size != 0:
            raise ValueError(f"num_embeddings ({num_embeddings}) must be divisible by tp_size ({tp_size})")
        self.vocab_start = comm.rank * (num_embeddings // tp_size)
        self.vocab_end = self.vocab_start + num_embeddings // tp_size
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(torch.empty(num_embeddings // tp_size, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = (x >= self.vocab_start) & (x < self.vocab_end)
        x_local = x.clone()
        x_local = x_local - self.vocab_start
        x_local[~mask] = 0
        out = F.embedding(x_local, self.weight)
        out[~mask] = 0.0
        return self.comm.all_reduce(out)
