"""Tests for RWKV-7 model class and weight loading."""

from __future__ import annotations

import os

import pytest
import torch


@pytest.mark.unit
class TestRWKV7Model:
    def test_model_creation(self) -> None:
        from models.RWKV.rwkv_model import RWKV7Config, RWKV7Model

        config = RWKV7Config(vocab_size=1024, hidden_size=64, num_layers=2)
        model = RWKV7Model(config)
        x = torch.randint(0, 1024, (1, 4))
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 4, 1024)

    def test_group_norm(self) -> None:
        from models.RWKV.rwkv_model import RWKV7GroupNorm

        gn = RWKV7GroupNorm(num_groups=4, num_channels=64)
        x = torch.randn(2, 8, 64)
        out = gn(x)
        assert out.shape == (2, 8, 64)

    def test_time_mix_forward(self) -> None:
        from models.RWKV.rwkv_model import RWKV7TimeMix

        tm = RWKV7TimeMix(hidden_size=64, head_size=16)
        x = torch.randn(1, 4, 64)
        v_first = torch.zeros(1, 4, 64)
        out, v = tm(x, v_first)
        assert out.shape == (1, 4, 64)
        assert v.shape == (1, 4, 64)

    def test_channel_mix_forward(self) -> None:
        from models.RWKV.rwkv_model import RWKV7ChannelMix

        cm = RWKV7ChannelMix(hidden_size=64, intermediate_size=256)
        x = torch.randn(1, 4, 64)
        out = cm(x)
        assert out.shape == (1, 4, 64)

    def test_full_layer_forward(self) -> None:
        from models.RWKV.rwkv_model import RWKV7Layer

        layer = RWKV7Layer(layer_idx=0, hidden_size=64, head_size=16, intermediate_size=256)
        x = torch.randn(1, 4, 64)
        v_first = torch.zeros(1, 4, 64)
        out, v = layer(x, v_first)
        assert out.shape == (1, 4, 64)

    @pytest.mark.timeout(120)
    def test_weight_loading(self) -> None:
        from models.RWKV.rwkv_model import RWKV7Config, RWKV7Model

        pth_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "models", "RWKV", "rwkv7-g1",
            "rwkv7-g1d-0.4b-20260210-ctx8192.pth",
        )
        if not os.path.isfile(pth_path):
            pytest.skip("RWKV model weights not available")

        config = RWKV7Config(vocab_size=65536, hidden_size=1024, num_layers=2)
        model = RWKV7Model(config)
        model.load_weights_from_pth(pth_path)
        x = torch.randint(0, 65536, (1, 4))
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 4, 65536)

    def test_convert_param_name(self) -> None:
        from models.RWKV.rwkv_model import _convert_param_name

        assert _convert_param_name("emb.weight") == "emb.weight"
        assert _convert_param_name("head.weight") == "head.weight"
        assert _convert_param_name("blocks.0.ln1.weight") == "blocks.0.ln1.weight"
        assert _convert_param_name("blocks.0.att.w0") == "blocks.0.att.w0"
        assert _convert_param_name("ln0.weight") == "blocks.0.ln0.weight"
