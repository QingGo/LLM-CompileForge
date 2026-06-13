"""Tests for VirtualDevice, hardware specs, and Roofline simulator."""

from __future__ import annotations

import pytest
import torch


@pytest.mark.unit
class TestHardwareSpec:
    def test_from_yaml(self) -> None:
        import os

        from python_runtime.hal.hardware_spec import HardwareSpec

        spec_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "hal",
            "hardware_specs",
            "a100.yaml",
        )
        spec = HardwareSpec.from_yaml(spec_path)
        assert spec.name == "NVIDIA A100 80GB"
        assert spec.peak_tflops > 300
        assert spec.bandwidth_gbs > 2000
        assert spec.memory_gb >= 80
        assert len(spec.compute_units) == 2
        assert len(spec.memory_levels) == 4
        assert len(spec.interconnects) == 2

    def test_m2_pro_spec(self) -> None:
        import os

        from python_runtime.hal.hardware_spec import HardwareSpec

        spec_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "hal",
            "hardware_specs",
            "m2_pro.yaml",
        )
        spec = HardwareSpec.from_yaml(spec_path)
        assert spec.peak_tflops == 0.5
        assert spec.bandwidth_gbs == 100.0

    def test_npu_spec(self) -> None:
        import os

        from python_runtime.hal.hardware_spec import HardwareSpec

        spec_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "hal",
            "hardware_specs",
            "hypothetical_npu.yaml",
        )
        spec = HardwareSpec.from_yaml(spec_path)
        assert spec.memory_gb > 0

    def test_backward_compat(self) -> None:
        from python_runtime.hal.hardware_spec import HardwareSpec

        spec = HardwareSpec(name="Old", peak_tflops=10.0, bandwidth_gbs=50.0)
        assert spec.peak_tflops == 10.0
        assert spec.bandwidth_gbs == 50.0
        assert len(spec.compute_units) == 1

    def test_dynamic_latency(self) -> None:
        from python_runtime.hal.hardware_spec import HardwareSpec

        spec = HardwareSpec(name="Test", peak_tflops=1.0, bandwidth_gbs=10.0)
        import torch

        a = torch.randn(1, 1024)
        b = torch.randn(1024, 4096)
        latency = spec.predict_latency("matmul", [a, b])
        assert latency > 0

    def test_peak_tflops_for(self) -> None:
        from python_runtime.hal.hardware_spec import ComputeUnit, HardwareSpec

        cu = ComputeUnit(name="Tensor", fp16_tflops=312.0, applies_to=["matmul"])
        spec = HardwareSpec(name="GPU", compute_units=[cu])
        tf = spec.peak_tflops_for("matmul", "float16")
        assert tf == 312.0
        tf_default = spec.peak_tflops_for("add", "float16")
        assert tf_default == 312.0


@pytest.mark.unit
class TestRooflineSimulator:
    def test_basic_analysis(self) -> None:
        from compiler.tools.verification.roofline_sim import RooflineSimulator

        sim = RooflineSimulator(peak_tflops=312.0, bandwidth_gbs=2039.0, name="A100")
        profiles = {
            "matmul": {"flops": 2.0 * 1024 * 1024 * 1024, "bytes": 3 * 1024 * 1024 * 2},
            "add": {"flops": 1024.0, "bytes": 2 * 1024 * 2},
        }
        report = sim.analyze(profiles)
        assert report.hardware_name == "A100"
        assert len(report.points) == 2
        assert report.compute_bound_count + report.memory_bound_count == 2

    def test_from_yaml(self) -> None:
        import os

        from compiler.tools.verification.roofline_sim import RooflineSimulator

        spec_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "hal",
            "hardware_specs",
            "a100.yaml",
        )
        sim = RooflineSimulator.from_yaml(spec_path)
        assert sim.peak_tflops >= 300
        assert sim.bandwidth_gbs > 1000

    def test_compare(self) -> None:
        from compiler.tools.verification.roofline_sim import RooflineSimulator

        sim_a = RooflineSimulator(peak_tflops=312.0, bandwidth_gbs=2039.0, name="GPU")
        sim_b = RooflineSimulator(peak_tflops=0.5, bandwidth_gbs=100.0, name="CPU")
        profiles = {"matmul": {"flops": 1e12, "bytes": 3 * 1024 * 1024}}
        reports = sim_a.compare(sim_b, op_profiles=profiles)
        assert len(reports) == 2
        assert reports[0].hardware_name == "GPU"
        assert reports[1].hardware_name == "CPU"

    def test_roofline_point(self) -> None:
        from compiler.tools.verification.roofline_sim import RooflinePoint

        pt = RooflinePoint(
            op_name="matmul",
            operational_intensity=512.0,
            attainable_gflops=200000.0,
            peak_gflops=312000.0,
            is_compute_bound=True,
        )
        assert pt.efficiency_pct > 60.0
        assert pt.is_compute_bound

    def test_approximate_op_profile(self) -> None:
        from compiler.tools.verification.roofline_sim import approximate_op_profile

        p = approximate_op_profile("matmul", (1, 1024, 4096), dtype_bytes=2)
        assert p["flops"] > 0
        assert p["bytes"] > 0

    def test_summary(self) -> None:
        from compiler.tools.verification.roofline_sim import RooflineSimulator

        sim = RooflineSimulator(peak_tflops=10.0, bandwidth_gbs=50.0, name="Test")
        profiles = {"add": {"flops": 1024, "bytes": 2048}}
        report = sim.analyze(profiles)
        summary = report.summary()
        assert "Test" in summary
        assert "TFLOPS" in summary


@pytest.mark.unit
class TestVirtualDevice:
    def test_create_and_execute(self) -> None:
        from python_runtime.hal.hardware_spec import HardwareSpec
        from python_runtime.hal.virtual_device import VirtualDevice

        spec = HardwareSpec(
            name="TestDevice",
            peak_tflops=1.0,
            bandwidth_gbs=10.0,
            op_latency_us={"add": 5.0},
        )
        dev = VirtualDevice("test", spec)
        a = torch.randn(1, 16)
        b = torch.randn(1, 16)
        out = dev.execute("add", [a, b])
        assert isinstance(out, torch.Tensor)
        assert dev.total_time_ns > 0

    def test_weight_op(self) -> None:
        from python_runtime.hal.hardware_spec import HardwareSpec
        from python_runtime.hal.virtual_device import VirtualDevice

        spec = HardwareSpec("T", 1.0, 10.0)
        dev = VirtualDevice("test", spec)
        w = torch.randn(64, 64)
        out = dev.execute("weight", [], _weight_name="fc.weight", _weight_tensor=w)
        assert out.shape == (64, 64)

    def test_matmul_shape_inference(self) -> None:
        from python_runtime.hal.hardware_spec import HardwareSpec
        from python_runtime.hal.virtual_device import VirtualDevice

        spec = HardwareSpec("T", 1.0, 10.0)
        dev = VirtualDevice("test", spec)
        a = torch.randn(1, 64)
        b = torch.randn(64, 128)
        out = dev.execute("matmul", [a, b])
        assert out.shape == (1, 128)

    def test_sdpa_shape_inference(self) -> None:
        from python_runtime.hal.hardware_spec import HardwareSpec
        from python_runtime.hal.virtual_device import VirtualDevice

        spec = HardwareSpec("T", 1.0, 10.0)
        dev = VirtualDevice("test", spec)
        q = torch.randn(1, 8, 16, 64)
        k = torch.randn(1, 8, 16, 64)
        v = torch.randn(1, 8, 16, 64)
        out = dev.execute("scaled_dot_product_attention", [q, k, v])
        assert out.shape == (1, 8, 16, 64)

    def test_reset_stats(self) -> None:
        from python_runtime.hal.hardware_spec import HardwareSpec
        from python_runtime.hal.virtual_device import VirtualDevice

        spec = HardwareSpec("T", 1.0, 10.0, op_latency_us={"add": 5.0})
        dev = VirtualDevice("test", spec)
        dev.execute("add", [torch.randn(1, 16), torch.randn(1, 16)])
        assert dev.total_time_ns > 0
        dev.reset_stats()
        assert dev.total_time_ns == 0

    def test_multiple_op_counts(self) -> None:
        from python_runtime.hal.hardware_spec import HardwareSpec
        from python_runtime.hal.virtual_device import VirtualDevice

        spec = HardwareSpec("T", 1.0, 10.0, op_latency_us={"add": 1.0, "mul": 2.0})
        dev = VirtualDevice("test", spec)
        dev.execute("add", [torch.randn(1, 4), torch.randn(1, 4)])
        dev.execute("mul", [torch.randn(1, 4), torch.randn(1, 4)])
        dev.execute("mul", [torch.randn(1, 4), torch.randn(1, 4)])
        assert dev._op_counts.get("add", 0) == 1
        assert dev._op_counts.get("mul", 0) == 2
