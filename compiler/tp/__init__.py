"""Tensor Parallelism — Phase 2 Module D.

Megatron-LM style column/row-parallel linear layers with HAL
Communicator integration for hardware-agnostic distributed inference.

Reference: design-phase2.md §2.4
"""

from compiler._lazy_imports import lazy_imports

lazy_imports(
    __name__,
    globals(),
    {
        "ColumnParallelLinear": ("compiler.tp.linear", "ColumnParallelLinear"),
        "RowParallelLinear": ("compiler.tp.linear", "RowParallelLinear"),
        "VocabParallelEmbedding": ("compiler.tp.linear", "VocabParallelEmbedding"),
        "search_tp_strategy": ("compiler.tp.strategy", "search_tp_strategy"),
        "count_parameters": ("compiler.tp.strategy", "count_parameters"),
    },
)
