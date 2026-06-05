"""Precision contract fixtures — shared test vectors for all subprojects.

Each fixture is a PrecisionContract proto serialized to binary (.pb).
Subprojects import these via their generated proto bindings and verify
their artifacts against the same expected values.

Usage:
    # Generate fixture
    python tests/contract/fixtures/generate.py

    # Load in compiler tests
    from tests.contract.fixtures import load_precision_fixture
    contract = load_precision_fixture("matmul_2x2")

    # Load in Rust tests
    let contract = load_precision_fixture("matmul_2x2");
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gen.proto.python.sfa_precision_pb2 import PrecisionContract

FIXTURES_DIR = Path(__file__).resolve().parent


def make_contract(version: str = "1.0") -> PrecisionContract:
    contract = PrecisionContract(version=version)
    return contract


def add_matmul_2x2(contract: PrecisionContract) -> None:
    """Simple 2x2 matmul: input=[1,2] @ W=[[1,2],[3,4]] = [7,10]"""
    c = contract.cases.add()
    c.name = "matmul_2x2_f32"
    c.input_data.extend([1.0, 2.0])
    c.input_shape.extend([1, 2])
    c.weight_data.extend([1.0, 2.0, 3.0, 4.0])
    c.weight_shape.extend([2, 2])
    c.expected_output.extend([7.0, 10.0])
    c.expected_shape.extend([1, 2])
    c.min_cosine = 0.9999
    c.max_abs_error = 1e-5


def add_rms_norm_2x4(contract: PrecisionContract) -> None:
    """RMSNorm on 2x4 input with weight [1,1,1,1]."""
    x = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
    w = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    # RMS = sqrt(mean(x_i^2)) for each row
    # Row 0: sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
    # Row 1: sqrt((25+36+49+64)/4) = sqrt(43.5) ≈ 6.5955
    rms0 = np.sqrt(np.mean(x[0]**2))
    rms1 = np.sqrt(np.mean(x[1]**2))
    expected = np.array([
        x[0] / rms0 * w,
        x[1] / rms1 * w,
    ], dtype=np.float32)

    c = contract.cases.add()
    c.name = "rms_norm_2x4_f32"
    c.input_data.extend(x.ravel().tolist())
    c.input_shape.extend([2, 4])
    c.weight_data.extend(w.ravel().tolist())
    c.weight_shape.extend([4])
    c.expected_output.extend(expected.ravel().tolist())
    c.expected_shape.extend([2, 4])
    c.min_cosine = 0.9995
    c.max_abs_error = 1e-4


def save_fixture(contract: PrecisionContract, name: str) -> None:
    path = FIXTURES_DIR / f"{name}.pb"
    path.write_bytes(contract.SerializeToString())
    print(f"  Saved {path} ({len(path.read_bytes())} bytes, {len(contract.cases)} cases)")


def load_fixture(name: str) -> PrecisionContract:
    path = FIXTURES_DIR / f"{name}.pb"
    contract = PrecisionContract()
    contract.ParseFromString(path.read_bytes())
    return contract


def load_precision_fixture(name: str) -> PrecisionContract:
    """Load a precision contract fixture by name.

    Used by compiler tests and runtime tests to get shared test vectors.
    """
    return load_fixture(name)


if __name__ == "__main__":
    contract = make_contract("1.0")
    add_matmul_2x2(contract)
    add_rms_norm_2x4(contract)
    save_fixture(contract, "precision_cases")
    print(f"Generated precision fixtures with {len(contract.cases)} cases")
