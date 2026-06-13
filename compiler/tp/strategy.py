"""Auto TP strategy search — heuristic model analysis for tensor parallelism.

Phase 2 MVP provides heuristic-based automatic parallel strategy
selection (§2.4.4):
  1. Scan model structure, identify shardable linear layers + param counts.
  2. Check memory constraint: can each GPU hold the sharded model?
  3. Generate minimum viable TP configuration.
  4. Estimate communication overhead from bandwidth parameters.

Reference: design-phase2.md §2.4.4
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn


@dataclass
class TPConfig:
    """Tensor parallelism configuration for a single layer type.

    Attributes:
        parallel_style: "column" or "row".
        gather_output: For column-parallel, whether to all-gather output.
        input_is_parallel: For row-parallel, whether input is already sharded.
    """

    parallel_style: str
    gather_output: bool = True
    input_is_parallel: bool = False


DEFAULT_LLAMA_TP_CONFIG: dict[str, TPConfig] = {
    "q_proj": TPConfig("column"),
    "k_proj": TPConfig("column"),
    "v_proj": TPConfig("column"),
    "o_proj": TPConfig("row", input_is_parallel=True),
    "gate_proj": TPConfig("column"),
    "up_proj": TPConfig("column"),
    "down_proj": TPConfig("row", input_is_parallel=True),
    "embed_tokens": TPConfig("column"),
    "lm_head": TPConfig("column"),
}


@dataclass
class AutoTPResult:
    """Result of automatic TP strategy search.

    Attributes:
        tp_size: Recommended tensor parallel degree.
        total_params: Total model parameter count.
        per_rank_params: Parameters per rank after sharding.
        memory_bytes_per_rank: Estimated memory per rank (FP16).
        communication_overhead_pct: Estimated all-reduce/all-gather overhead.
        feasible: Whether the model fits in the given memory budget.
    """

    tp_size: int
    total_params: int
    per_rank_params: int
    memory_bytes_per_rank: int
    communication_overhead_pct: float
    feasible: bool


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters())


def search_tp_strategy(
    model: nn.Module,
    available_memory_gb: float = 24.0,
    max_comm_overhead_pct: float = 15.0,
    max_tp_size: int = 8,
    bandwidth_gb_s: float = 600.0,
) -> AutoTPResult:
    """Search for the minimum viable TP configuration.

    Args:
        model: PyTorch model to analyze.
        available_memory_gb: Available GPU HBM per card (GB).
        max_comm_overhead_pct: Maximum acceptable communication overhead (%).
        max_tp_size: Upper bound for TP degree.
        bandwidth_gb_s: Interconnect bandwidth (GB/s), e.g. NVLink = 600.

    Returns:
        AutoTPResult with recommended configuration.
    """
    total_params = count_parameters(model)
    fp16_bytes = total_params * 2  # FP16 = 2 bytes per param

    needed_memory_gb = fp16_bytes / (1024**3)

    if needed_memory_gb <= available_memory_gb:
        return AutoTPResult(
            tp_size=1,
            total_params=total_params,
            per_rank_params=total_params,
            memory_bytes_per_rank=fp16_bytes,
            communication_overhead_pct=0.0,
            feasible=True,
        )

    for tp in range(2, max_tp_size + 1, 2):
        per_rank_params = total_params // tp
        per_rank_mem_gb = (per_rank_params * 2) / (1024**3)

        comm_overhead = _estimate_comm_overhead(tp, total_params, bandwidth_gb_s)

        if per_rank_mem_gb <= available_memory_gb and comm_overhead <= max_comm_overhead_pct:
            return AutoTPResult(
                tp_size=tp,
                total_params=total_params,
                per_rank_params=per_rank_params,
                memory_bytes_per_rank=per_rank_params * 2,
                communication_overhead_pct=comm_overhead,
                feasible=True,
            )

    min_tp = max_tp_size
    return AutoTPResult(
        tp_size=min_tp,
        total_params=total_params,
        per_rank_params=total_params // min_tp,
        memory_bytes_per_rank=(total_params // min_tp) * 2,
        communication_overhead_pct=_estimate_comm_overhead(min_tp, total_params, bandwidth_gb_s),
        feasible=(total_params // min_tp) * 2 / (1024**3) <= available_memory_gb,
    )


def _estimate_comm_overhead(tp_size: int, total_params: int, bandwidth_gb_s: float) -> float:
    """Estimate communication overhead as percentage of total compute.

    Simplified heuristic: communication scales with total_params / tp_size
    (each all-reduce is proportional to shard size), and is inversely
    proportional to available bandwidth.

    For typical models with NVLink (600 GB/s), TP=2 → ~5-8%, TP=4 → ~10-12%.
    """
    params_per_rank = total_params // tp_size
    comm_bytes_per_step = params_per_rank * 2  # FP16: 2 bytes per param, one all-reduce

    # Rough estimate: comm_time / compute_time ratio
    # Compute: ~ total_params * 2 ops (multiply + add) / GPU TFLOPS
    # We approximate with bandwidth-based ratio
    base_overhead = (comm_bytes_per_step / (bandwidth_gb_s * 1e9)) * 1e3
    return min(base_overhead * 100.0, 50.0)
