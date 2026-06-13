"""Unit tests for compiler/rwkv/dialect.py — RWKV dialect op definitions."""

from compiler.rwkv.dialect import RwkvChannelMixOp, RwkvStateEvolveOp, RwkvTimeMixOp


class TestRwkvTimeMixOp:
    def test_default_name(self):
        op = RwkvTimeMixOp()
        assert op.name == "rwkv.time_mix"

    def test_custom_name(self):
        op = RwkvTimeMixOp(name="rwkv.custom")
        assert op.name == "rwkv.custom"


class TestRwkvChannelMixOp:
    def test_default_name(self):
        op = RwkvChannelMixOp()
        assert op.name == "rwkv.channel_mix"


class TestRwkvStateEvolveOp:
    def test_default_name(self):
        op = RwkvStateEvolveOp()
        assert op.name == "rwkv.state_evolve"
