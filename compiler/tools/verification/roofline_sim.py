"""Roofline simulator — predicts performance upper bounds for arbitrary hardware.

Given a hardware specification (peak FLOPS, bandwidth) and an operation profile
(FLOP count, bytes accessed), computes the attainable performance ceiling and
identifies whether the operation is compute-bound or memory-bound.

Reference: design-phase3.md §1.5.2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RooflinePoint:
    """A single point on the Roofline chart.

    Attributes:
        op_name: Operation name.
        operational_intensity: FLOPs / bytes accessed.
        attainable_gflops: Performance ceiling for this op on this hardware.
        peak_gflops: Hardware peak compute throughput.
        is_compute_bound: True if the op is compute-bound.
    """

    op_name: str
    operational_intensity: float
    attainable_gflops: float
    peak_gflops: float
    is_compute_bound: bool

    @property
    def ridge_point(self) -> float:
        return self.peak_gflops / max(self.operational_intensity, 1e-9)

    @property
    def efficiency_pct(self) -> float:
        return 100.0 * self.attainable_gflops / max(self.peak_gflops, 1e-9)


@dataclass
class RooflineReport:
    """Complete Roofline analysis for a hardware + model combination.

    Attributes:
        hardware_name: Name of the analyzed hardware.
        peak_tflops: Peak FP32 compute (TFLOPS).
        bandwidth_gbs: Peak memory bandwidth (GB/s).
        points: Per-operation analysis points.
        ridge_gflops_per_byte: Ridge point (compute-bound threshold).
    """

    hardware_name: str
    peak_tflops: float
    bandwidth_gbs: float
    points: list[RooflinePoint] = field(default_factory=list)

    @property
    def peak_gflops(self) -> float:
        return self.peak_tflops * 1000.0

    @property
    def ridge_gflops_per_byte(self) -> float:
        return self.peak_gflops / max(self.bandwidth_gbs, 1e-9)

    @property
    def compute_bound_count(self) -> int:
        return sum(1 for p in self.points if p.is_compute_bound)

    @property
    def memory_bound_count(self) -> int:
        return sum(1 for p in self.points if not p.is_compute_bound)

    @property
    def avg_efficiency_pct(self) -> float:
        if not self.points:
            return 0.0
        return sum(p.efficiency_pct for p in self.points) / len(self.points)

    def summary(self) -> str:
        lines = [
            f"Roofline Report — {self.hardware_name}",
            f"  Peak:  {self.peak_tflops:.1f} TFLOPS, "
            f"{self.bandwidth_gbs:.0f} GB/s",
            f"  Ridge: {self.ridge_gflops_per_byte:.1f} GFLOPs/byte",
            f"  {len(self.points)} ops: "
            f"{self.compute_bound_count} compute-bound, "
            f"{self.memory_bound_count} memory-bound",
            f"  Avg efficiency: {self.avg_efficiency_pct:.1f}%",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hardware_name": self.hardware_name,
            "peak_tflops": self.peak_tflops,
            "bandwidth_gbs": self.bandwidth_gbs,
            "ridge_gflops_per_byte": self.ridge_gflops_per_byte,
            "num_ops": len(self.points),
            "compute_bound": self.compute_bound_count,
            "memory_bound": self.memory_bound_count,
            "avg_efficiency_pct": round(self.avg_efficiency_pct, 1),
            "points": [
                {
                    "op": p.op_name,
                    "oi": round(p.operational_intensity, 2),
                    "attainable_gflops": round(p.attainable_gflops, 1),
                    "bound": "compute" if p.is_compute_bound else "memory",
                    "efficiency_pct": round(p.efficiency_pct, 1),
                }
                for p in self.points
            ],
        }


class RooflineSimulator:
    """Simulates Roofline model for a given hardware specification.

    Args:
        peak_tflops: Peak FP32 compute throughput (TFLOPS).
        bandwidth_gbs: Peak memory bandwidth (GB/s).
        name: Optional name for reporting.
    """

    def __init__(
        self,
        peak_tflops: float,
        bandwidth_gbs: float,
        name: str = "unknown",
    ) -> None:
        self.peak_tflops = peak_tflops
        self.bandwidth_gbs = bandwidth_gbs
        self.name = name

    @classmethod
    def from_spec(cls, spec: Any) -> RooflineSimulator:
        from python_runtime.hal.hardware_spec import HardwareSpec

        if isinstance(spec, HardwareSpec):
            return cls(spec.peak_tflops, spec.bandwidth_gbs, spec.name)
        if hasattr(spec, "peak_tflops"):
            return cls(spec.peak_tflops, spec.bandwidth_gbs, getattr(spec, "name", "unknown"))
        raise TypeError(f"Unsupported spec type: {type(spec)}")

    @classmethod
    def from_yaml(cls, path: str) -> RooflineSimulator:
        from python_runtime.hal.hardware_spec import HardwareSpec

        spec = HardwareSpec.from_yaml(path)
        return cls(spec.peak_tflops, spec.bandwidth_gbs, spec.name)

    def analyze(
        self, op_profiles: dict[str, dict[str, float]]
    ) -> RooflineReport:
        """Analyze a set of operations on this hardware.

        Args:
            op_profiles: dict of op_name → {"flops": total_FLOPs, "bytes": bytes_accessed}
                         Each value represents a single execution of the operation with
                         typical shapes.

        Returns:
            RooflineReport with per-operation analysis.
        """
        peak_gflops = self.peak_tflops * 1000.0
        report = RooflineReport(
            hardware_name=self.name,
            peak_tflops=self.peak_tflops,
            bandwidth_gbs=self.bandwidth_gbs,
        )
        for op_name, profile in op_profiles.items():
            flops = profile.get("flops", 0.0)
            byte_count = profile.get("bytes", 0.0)
            oi = flops / max(byte_count, 1.0)
            attainable = min(peak_gflops, self.bandwidth_gbs * oi)
            report.points.append(
                RooflinePoint(
                    op_name=op_name,
                    operational_intensity=oi,
                    attainable_gflops=attainable,
                    peak_gflops=peak_gflops,
                    is_compute_bound=oi >= (peak_gflops / max(self.bandwidth_gbs, 1e-9)),
                )
            )
        return report

    def compare(
        self, *others: RooflineSimulator, op_profiles: dict[str, dict[str, float]]
    ) -> list[RooflineReport]:
        """Compare multiple hardware configurations on the same operation set."""
        reports = [self.analyze(op_profiles)]
        for other in others:
            reports.append(other.analyze(op_profiles))
        return reports


def approximate_op_profile(
    op_name: str, shape: tuple[int, ...], dtype_bytes: int = 2
) -> dict[str, float]:
    """Estimate FLOPs and bytes for a single operation with given shapes.

    These are approximate formulas for typical tensor dimensions:
    - matmul[M,K] × [K,N] = 2*M*K*N FLOPs, (M*K + K*N + M*N)*dtype_bytes bytes
    - element-wise op with shape S: S FLOPs, 2*S*dtype_bytes bytes

    Args:
        op_name: Operation name (e.g. "matmul", "add").
        shape: Representative output tensor shape.
        dtype_bytes: Bytes per element (2 for FP16, 4 for FP32).

    Returns:
        dict with "flops" and "bytes".
    """
    total_elems = 1
    for d in shape:
        total_elems *= d

    if op_name in ("matmul", "linear"):
        if len(shape) >= 2:
            m, n = shape[-2], shape[-1]
            k = n if len(shape) == 2 else shape[-1]
            return {
                "flops": 2.0 * m * k * n,
                "bytes": (m * k + k * n + m * n) * dtype_bytes,
            }
        return {"flops": float(total_elems), "bytes": 3.0 * total_elems * dtype_bytes}

    if op_name in ("scaled_dot_product_attention",):
        if len(shape) >= 3:
            b, h, s, d = shape[0], shape[1], shape[2], shape[3] if len(shape) >= 4 else shape[2]
            return {
                "flops": 4.0 * b * h * s * s * d,
                "bytes": 4.0 * b * h * s * d * dtype_bytes,
            }
        return {"flops": float(total_elems), "bytes": float(total_elems) * dtype_bytes}

    if op_name in ("layer_norm", "rms_norm"):
        return {"flops": 5.0 * total_elems, "bytes": 3.0 * total_elems * dtype_bytes}

    return {"flops": float(total_elems), "bytes": 2.0 * total_elems * dtype_bytes}
