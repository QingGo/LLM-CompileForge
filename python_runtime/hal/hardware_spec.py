"""Hardware simulation types — compute, memory, interconnect models.

Provides the data structures for realistic hardware performance simulation:
  - Multi-precision compute units (vector, tensor core, scalar)
  - Multi-level memory hierarchy (HBM, SRAM, registers)
  - Multi-device interconnects (NVLink, PCIe, CXL)

Reference: design-phase3.md §1.5 (Hardware Verification Framework)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ComputeUnit:
    """A compute unit on a hardware device (e.g. vector ALU, tensor core).

    Attributes:
        name: Human-readable name ("Vector Unit", "Tensor Core").
        fp32_tflops: Peak FP32 throughput in TFLOPS.
        fp16_tflops: Peak FP16 throughput in TFLOPS.
        bf16_tflops: Peak BF16 throughput in TFLOPS.
        int8_tops: Peak INT8 throughput in TOPS.
        int4_tops: Peak INT4 throughput in TOPS.
        fp8_tflops: Peak FP8 throughput in TFLOPS.
        applies_to: List of op patterns this unit applies to.
            Empty = "all ops", ["matmul", "linear"] = only gemm ops.
    """

    name: str = "ComputeUnit"
    fp32_tflops: float = 10.0
    fp16_tflops: float = 0.0
    bf16_tflops: float = 0.0
    int8_tops: float = 0.0
    int4_tops: float = 0.0
    fp8_tflops: float = 0.0
    applies_to: list[str] = field(default_factory=list)

    def peak_tflops_for(self, op_name: str, dtype_str: str = "float32") -> float:
        """Return peak TFLOPS for the given op and precision."""
        if self.applies_to and op_name not in self.applies_to:
            return 0.0
        lut: dict[str, float] = {
            "float32": self.fp32_tflops,
            "float16": self.fp16_tflops,
            "bfloat16": self.bf16_tflops,
            "int8": self.int8_tops,
            "int4": self.int4_tops,
            "float8": self.fp8_tflops,
        }
        return lut.get(dtype_str, self.fp32_tflops)

    @property
    def max_tflops(self) -> float:
        return max(
            self.fp32_tflops, self.fp16_tflops, self.bf16_tflops, self.int8_tops, self.int4_tops, self.fp8_tflops
        )


@dataclass
class MemoryLevel:
    """A level in the memory hierarchy.

    Attributes:
        name: "HBM", "SRAM", "L2 Cache", "Register File"
        size_gb: Capacity in GB.
        bandwidth_gbs: Peak bandwidth in GB/s.
        latency_ns: Access latency in nanoseconds.
    """

    name: str = "Memory"
    size_gb: float = 16.0
    bandwidth_gbs: float = 100.0
    latency_ns: float = 100.0


@dataclass
class Interconnect:
    """Inter-device communication link.

    Attributes:
        name: "NVLink", "PCIe Gen5", "CXL", "InfiniBand"
        bandwidth_gbs: Bidirectional bandwidth in GB/s.
        latency_us: One-way transfer latency in microseconds.
        topology: Connection topology ("all_to_all", "ring", "tree", "custom").
    """

    name: str = "Interconnect"
    bandwidth_gbs: float = 50.0
    latency_us: float = 5.0
    topology: str = "all_to_all"


@dataclass
class HardwareSpec:
    """Complete hardware specification for latency simulation.

    Attributes:
        name: Human-readable device name.
        compute_units: List of compute units on this device.
        memory_levels: Memory hierarchy from fastest/smallest to slowest/largest.
        interconnects: Inter-device communication links.
        op_latency_us: Per-operation fixed latency overrides (bypasses estimation).
    """

    def __init__(
        self,
        name: str = "Unknown",
        peak_tflops: float | None = None,
        bandwidth_gbs: float | None = None,
        memory_gb: float | None = None,
        op_latency_us: dict[str, float] | None = None,
        compute_units: list[ComputeUnit] | None = None,
        memory_levels: list[MemoryLevel] | None = None,
        interconnects: list[Interconnect] | None = None,
    ) -> None:
        self.name = name
        self.op_latency_us: dict[str, float] = dict(op_latency_us or {})

        # Backward compat: old flat args → new structured format
        if compute_units is not None:
            self.compute_units = compute_units
        else:
            self.compute_units = []
            if peak_tflops is not None:
                self.compute_units.append(
                    ComputeUnit(
                        name="Default",
                        fp32_tflops=peak_tflops,
                    )
                )

        if memory_levels is not None:
            self.memory_levels = memory_levels
        else:
            self.memory_levels = []
            if bandwidth_gbs is not None or memory_gb is not None:
                self.memory_levels.append(
                    MemoryLevel(
                        name="Main Memory",
                        size_gb=memory_gb if memory_gb is not None else 16.0,
                        bandwidth_gbs=bandwidth_gbs if bandwidth_gbs is not None else 50.0,
                    )
                )

        self.interconnects: list[Interconnect] = interconnects or []

    # ── Convenience properties (backward compat) ──────────

    @property
    def peak_tflops(self) -> float:
        """Max FP compute throughput across all compute units."""
        if not self.compute_units:
            return 10.0
        return max(max(cu.fp32_tflops, cu.fp16_tflops, cu.bf16_tflops, cu.fp8_tflops) for cu in self.compute_units)

    @property
    def bandwidth_gbs(self) -> float:
        """Return the primary memory bandwidth (largest level, not fastest)."""
        if not self.memory_levels:
            return 50.0
        return self.primary_memory().bandwidth_gbs

    @property
    def memory_gb(self) -> float:
        if not self.memory_levels:
            return 16.0
        return sum(ml.size_gb for ml in self.memory_levels)

    def primary_memory(self) -> MemoryLevel:
        """Return the largest memory level (HBM/main memory)."""
        if not self.memory_levels:
            return MemoryLevel()
        return max(self.memory_levels, key=lambda m: m.size_gb)

    def fastest_memory(self) -> MemoryLevel:
        """Return the fastest memory level (SRAM/cache)."""
        if not self.memory_levels:
            return MemoryLevel()
        return max(self.memory_levels, key=lambda m: m.bandwidth_gbs)

    def peak_tflops_for(self, op_name: str, dtype_str: str = "float32") -> float:
        """Return the best compute throughput for this op+dtype."""
        best = 0.0
        for cu in self.compute_units:
            t = cu.peak_tflops_for(op_name, dtype_str)
            if t > best:
                best = t
        return best if best > 0 else self.peak_tflops

    # ── Latency estimation ────────────────────────────────

    def predict_latency(self, op_name: str, inputs: list[Any], **kwargs: Any) -> float:
        """Predict execution latency in nanoseconds.

        Uses fixed latency from op_latency_us if available.
        Otherwise estimates via Roofline model: max(compute_time, memory_time).
        """
        us = self.op_latency_us.get(op_name)
        if us is not None:
            return us * 1000.0
        us = self.op_latency_us.get("default")
        if us is not None:
            return us * 1000.0

        import torch

        dtype_str = "float32"
        for v in inputs:
            if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
                if v.dtype == torch.float16:
                    dtype_str = "float16"
                elif v.dtype == torch.bfloat16:
                    dtype_str = "bfloat16"
                break

        flops = _estimate_flops(op_name, inputs, kwargs)
        byte_count = _estimate_bytes(op_name, inputs, kwargs)

        peak = self.peak_tflops_for(op_name, dtype_str) * 1e9
        bw = self.primary_memory().bandwidth_gbs * 1e9  # GB/s → B/s

        compute_ns = (flops / peak) * 1e9 if peak > 0 else 0
        memory_ns = (byte_count / bw) * 1e9 if bw > 0 else 0

        return max(compute_ns, memory_ns, 1.0)

    # ── Serialization ─────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str) -> HardwareSpec:
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HardwareSpec:
        cu_list = []
        for cu_data in data.get("compute_units", []):
            cu_list.append(
                ComputeUnit(
                    name=cu_data.get("name", ""),
                    fp32_tflops=float(cu_data.get("fp32_tflops", 0)),
                    fp16_tflops=float(cu_data.get("fp16_tflops", 0)),
                    bf16_tflops=float(cu_data.get("bf16_tflops", 0)),
                    int8_tops=float(cu_data.get("int8_tops", 0)),
                    int4_tops=float(cu_data.get("int4_tops", 0)),
                    fp8_tflops=float(cu_data.get("fp8_tflops", 0)),
                    applies_to=cu_data.get("applies_to", []),
                )
            )

        mem_list = []
        for m_data in data.get("memory_levels", []):
            mem_list.append(
                MemoryLevel(
                    name=m_data.get("name", ""),
                    size_gb=float(m_data.get("size_gb", 0)),
                    bandwidth_gbs=float(m_data.get("bandwidth_gbs", 0)),
                    latency_ns=float(m_data.get("latency_ns", 0)),
                )
            )

        ic_list = []
        for ic_data in data.get("interconnects", []):
            ic_list.append(
                Interconnect(
                    name=ic_data.get("name", ""),
                    bandwidth_gbs=float(ic_data.get("bandwidth_gbs", 0)),
                    latency_us=float(ic_data.get("latency_us", 0)),
                    topology=ic_data.get("topology", "all_to_all"),
                )
            )

        # Backward compat: old YAML format with peak_tflops/bandwidth_gbs
        if not cu_list and "peak_tflops" in data:
            cu_list.append(
                ComputeUnit(
                    name="Default",
                    fp32_tflops=float(data.get("peak_tflops", 10)),
                    fp16_tflops=float(data.get("peak_tflops", 10)),
                )
            )
        if not mem_list and "bandwidth_gbs" in data:
            mem_list.append(
                MemoryLevel(
                    name="Main Memory",
                    size_gb=float(data.get("memory_gb", 16)),
                    bandwidth_gbs=float(data.get("bandwidth_gbs", 50)),
                )
            )

        return cls(
            name=data.get("name", "Unknown"),
            compute_units=cu_list,
            memory_levels=mem_list,
            interconnects=ic_list,
            op_latency_us=data.get("op_latency_us", {}),
        )


def _estimate_flops(op_name: str, inputs: list[Any], kwargs: dict[str, Any]) -> float:
    """Estimate FLOP count for an operation based on tensor inputs."""
    import torch

    def _total(t: torch.Tensor) -> int:
        return int(t.numel())

    if op_name in ("matmul", "linear"):
        if len(inputs) >= 2 and isinstance(inputs[0], torch.Tensor) and isinstance(inputs[1], torch.Tensor):
            a, b = inputs[0], inputs[1]
            if a.dim() == 2 and b.dim() == 2:
                return 2.0 * a.shape[0] * a.shape[1] * b.shape[1]
            if a.dim() >= 3 and b.dim() >= 2:
                return 2.0 * a.shape[-2] * a.shape[-1] * b.shape[-1]
        return 1e9

    if op_name == "scaled_dot_product_attention":
        if len(inputs) >= 3 and isinstance(inputs[0], torch.Tensor) and inputs[0].dim() >= 3:
            q = inputs[0]
            batch, heads, seq, dim = q.shape[0], q.shape[1], q.shape[2], q.shape[3]
            return 4.0 * batch * heads * seq * seq * dim
        return 1e9

    if op_name in ("layer_norm", "rms_norm"):
        if inputs and isinstance(inputs[0], torch.Tensor):
            return 5.0 * _total(inputs[0])
        return 1000

    if op_name in ("softmax",):
        if inputs and isinstance(inputs[0], torch.Tensor):
            return 5.0 * _total(inputs[0])
        return 1000

    if op_name in (
        "add",
        "sub",
        "mul",
        "div",
        "neg",
        "relu",
        "gelu",
        "silu",
        "sigmoid",
        "softplus",
        "exp",
        "tanh",
        "sqrt",
    ):
        if inputs and isinstance(inputs[0], torch.Tensor):
            return float(_total(inputs[0]))
        return 100

    for v in inputs:
        if isinstance(v, torch.Tensor):
            return float(_total(v))
    return 100


def _estimate_bytes(op_name: str, inputs: list[Any], kwargs: dict[str, Any]) -> float:
    """Estimate bytes accessed (read + write) for an operation."""
    import torch

    total_bytes = 0.0
    element_size = 2.0  # default FP16

    for v in inputs:
        if isinstance(v, torch.Tensor):
            if v.dtype == torch.float32:
                element_size = 4.0
            elif v.dtype == torch.float16:
                element_size = 2.0
            elif v.dtype == torch.bfloat16:
                element_size = 2.0
            total_bytes += float(v.numel()) * element_size

    if total_bytes == 0:
        return 1024.0

    return total_bytes * 1.5 if op_name in ("matmul", "linear", "scaled_dot_product_attention") else total_bytes * 2.0
