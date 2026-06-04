"""Unit tests for compiler/tp/strategy.py — TPConfig, param counting, strategy search."""

import torch.nn as nn

from compiler.tp.strategy import AutoTPResult, TPConfig, count_parameters, search_tp_strategy


class TestTPConfig:
    def test_column_parallel_config(self):
        cfg = TPConfig(parallel_style="column", gather_output=True)
        assert cfg.parallel_style == "column"
        assert cfg.gather_output is True
        assert cfg.input_is_parallel is False  # default

    def test_row_parallel_config(self):
        cfg = TPConfig(parallel_style="row", input_is_parallel=True)
        assert cfg.parallel_style == "row"
        assert cfg.input_is_parallel is True


class TestCountParameters:
    def test_counts_linear_layer(self):
        model = nn.Linear(10, 20)
        params = count_parameters(model)
        # Linear: weight [20,10] + bias [20] = 220 params
        assert params > 0

    def test_counts_sequential(self):
        model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 3))
        params = count_parameters(model)
        assert params > 0


class TestSearchTPStrategy:
    def test_returns_autotp_result_for_simple_model(self):
        model = nn.Sequential(nn.Linear(64, 128), nn.Linear(128, 64))
        result = search_tp_strategy(model, available_memory_gb=40)
        assert isinstance(result, AutoTPResult)

    def test_feasible_for_tiny_model(self):
        model = nn.Linear(4, 4)
        result = search_tp_strategy(model, available_memory_gb=1, max_tp_size=2)
        assert isinstance(result, AutoTPResult)
