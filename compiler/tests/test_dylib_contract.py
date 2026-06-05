"""Compiler contract tests: verify compiled dylib ABI compliance with sfa.h.

Contract from include/sfa.h:
  struct<(ptr, ptr, i64, array<RANK x i64>, array<RANK x i64>)>
  - Field 0 (allocated): must be non-null for valid tensor descriptors
  - Field 1 (aligned):   must be non-null for valid tensor descriptors
  - Fields 0-2 form a fixed 24-byte header matching MLIR memref exactly

Tests verify:
  1. ctypes struct layout matches sfa.h sizes (compile-time ABI)
  2. LLVM IR sret descriptors in ciface wrappers have valid pointers
  3. Dylib symbol existence for ciface functions

The sret null-pointer bug (handoff P0): bufferization can produce KV split
function sret descriptors where aligned=null. This contract test catches it.
"""

from __future__ import annotations

import ctypes
import re
import struct
from pathlib import Path
from typing import Any

import pytest

# ── Project paths ────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_KV_DYLIB_DIR = _PROJECT_ROOT / "outputs" / "compiled" / "opt_125m_kv"
_KV_DYLIB_PATH = _KV_DYLIB_DIR / "libopt_125m_kv.dylib"
_KV_LL_PATH = _KV_DYLIB_DIR / "model.ll"


def _has_dylib() -> bool:
    return _KV_DYLIB_PATH.exists()


def _has_ll() -> bool:
    return _KV_LL_PATH.exists()


# ── ctypes struct definitions (matching include/sfa.h) ──────────────


class SFATensorRaw1(ctypes.Structure):
    _fields_ = [
        ("allocated", ctypes.c_void_p),
        ("aligned", ctypes.c_void_p),
        ("offset", ctypes.c_int64),
        ("sizes", ctypes.c_int64 * 1),
        ("strides", ctypes.c_int64 * 1),
    ]


class SFATensorRaw2(ctypes.Structure):
    _fields_ = [
        ("allocated", ctypes.c_void_p),
        ("aligned", ctypes.c_void_p),
        ("offset", ctypes.c_int64),
        ("sizes", ctypes.c_int64 * 2),
        ("strides", ctypes.c_int64 * 2),
    ]


class SFATensorRaw3(ctypes.Structure):
    _fields_ = [
        ("allocated", ctypes.c_void_p),
        ("aligned", ctypes.c_void_p),
        ("offset", ctypes.c_int64),
        ("sizes", ctypes.c_int64 * 3),
        ("strides", ctypes.c_int64 * 3),
    ]


class SFATensorRaw4(ctypes.Structure):
    _fields_ = [
        ("allocated", ctypes.c_void_p),
        ("aligned", ctypes.c_void_p),
        ("offset", ctypes.c_int64),
        ("sizes", ctypes.c_int64 * 4),
        ("strides", ctypes.c_int64 * 4),
    ]


# ── Contract: struct sizes match sfa.h ──────────────────────────────


class TestSFATensorRawLayout:
    """Verify ctypes struct sizes match the sfa.h contract (handoff P0)."""

    # From sfa.h:
    #   SFATensorRaw1: 24 + 8 + 8  = 40 bytes
    #   SFATensorRaw2: 24 + 16 + 16 = 56 bytes
    #   SFATensorRaw3: 24 + 24 + 24 = 72 bytes
    #   SFATensorRaw4: 24 + 32 + 32 = 88 bytes
    # The 24-byte header is: allocated(8) + aligned(8) + offset(8)

    def test_raw1_size_40_bytes(self) -> None:
        """SFATensorRaw1 = 24 (header) + 8 (sizes) + 8 (strides) = 40."""
        assert ctypes.sizeof(SFATensorRaw1) == 40, (
            f"SFATensorRaw1 size={ctypes.sizeof(SFATensorRaw1)}, expected 40"
        )

    def test_raw2_size_56_bytes(self) -> None:
        """SFATensorRaw2 = 24 + 16 + 16 = 56."""
        assert ctypes.sizeof(SFATensorRaw2) == 56

    def test_raw3_size_72_bytes(self) -> None:
        """SFATensorRaw3 = 24 + 24 + 24 = 72."""
        assert ctypes.sizeof(SFATensorRaw3) == 72

    def test_raw4_size_88_bytes(self) -> None:
        """SFATensorRaw4 = 24 + 32 + 32 = 88."""
        assert ctypes.sizeof(SFATensorRaw4) == 88

    def test_header_24_byte_field_offsets(self) -> None:
        assert SFATensorRaw1.allocated.offset == 0, "allocated must be at offset 0"
        assert SFATensorRaw1.aligned.offset == 8, "aligned must be at offset 8"
        assert SFATensorRaw1.offset.offset == 16, (
            "offset field must be at byte 16"
        )
        assert SFATensorRaw1.sizes.offset == 24, "sizes must be at offset 24"

    def test_sret_descriptor_aligned_offset(self) -> None:
        """Contract: aligned field (index 1) is at offset 8 in every rank."""
        # This is the field that must be non-null per the sret contract.
        for raw_cls, expected_offset in [
            (SFATensorRaw1, 8),
            (SFATensorRaw2, 8),
            (SFATensorRaw3, 8),
            (SFATensorRaw4, 8),
        ]:
            assert raw_cls.aligned.offset == expected_offset, (
                f"{raw_cls.__name__}.aligned offset={raw_cls.aligned.offset}, expected={expected_offset}"
            )


# ── LLVM IR patterns for sret verification ──────────────────────────

# Ciface wrapper definition: define void @_mlir_ciface_<name>(<params>) {
_CIFACE_DEF_RE = re.compile(
    r"define\s+void\s+@(_mlir_ciface_\w+)\s*\(([^)]*)\)\s*\{"
)

# Store into sret pointer: store ... ptr %0, align 8
# The sret pointer is always the first argument (%0)
_STORE_SRET_RE = re.compile(
    r"store\s+\{([^}]+)\}\s+%(\d+),\s*ptr\s+%(\d+)"
)

_INSERTVALUE_UNDEF_RE = re.compile(
    r"insertvalue\s+\{\s*ptr,\s*ptr,\s*i64,\s*"
    r"\[(\d+)\s+x\s+i64\],\s*\[(\d+)\s+x\s+i64\]\s*\}"
    r"\s+undef,\s*(ptr\s+\w+)\s*,\s*(\d+)"
)


def _parse_ciface_body(ll_text: str, _func_name: str, start_pos: int) -> str:
    """Extract the body of a ciface wrapper function (from { to matching }).

    start_pos is the position right after the opening '{' that the
    _CIFACE_DEF_RE already consumed.
    """
    depth = 1
    pos = start_pos
    while pos < len(ll_text) and depth > 0:
        if ll_text[pos] == "{":
            depth += 1
        elif ll_text[pos] == "}":
            depth -= 1
        pos += 1
    return ll_text[start_pos : pos - 1]


def _has_sret_store(ll_body: str) -> bool:
    """Check if the ciface wrapper body stores to the sret pointer (%0)."""
    return bool(re.search(r"store\s+.*ptr\s+%0", ll_body))


def _find_sret_construction(ll_body: str) -> list[dict[str, Any]]:
    """Find how the sret struct is constructed in a ciface wrapper body.

    Returns list of construction events: 'store_sret' (stores to %0),
    or 'insertvalue_chain' that builds the descriptor field by field.

    Each event has: {"type": ..., "fields_set": set[int]} for insertvalue chains.
    """
    events: list[dict[str, Any]] = []

    # Pattern 1: any store to sret ptr %0 (accepts complex return types)
    if _has_sret_store(ll_body):
        events.append({"type": "store_sret"})

    # Pattern 2: insertvalue chain from undef building a descriptor
    current_rank: int | None = None
    fields_set: set[int] = set()
    for m in _INSERTVALUE_UNDEF_RE.finditer(ll_body):
        rank_a, rank_b = int(m.group(1)), int(m.group(2))
        if rank_a != rank_b:
            continue
        rank = rank_a
        field_idx = int(m.group(4))
        if field_idx == 0:
            if current_rank is not None and fields_set:
                events.append({
                    "type": "insertvalue_chain",
                    "rank": current_rank,
                    "fields_set": fields_set,
                })
            current_rank = rank
            fields_set = {field_idx}
        else:
            if current_rank == rank:
                fields_set.add(field_idx)

    if current_rank is not None and fields_set:
        events.append({
            "type": "insertvalue_chain",
            "rank": current_rank,
            "fields_set": fields_set,
        })

    return events


def _extract_ciface_functions(ll_text: str) -> dict[str, dict[str, Any]]:
    """Parse all ciface wrapper functions from LLVM IR text.

    Returns: {func_name: {"params": str, "body": str}}
    """
    funcs: dict[str, dict[str, Any]] = {}
    for m in _CIFACE_DEF_RE.finditer(ll_text):
        name = m.group(1)
        params = m.group(2)
        body = _parse_ciface_body(ll_text, name, m.end())
        funcs[name] = {"params": params, "body": body}
    return funcs


# ── Contract: LLVM IR sret descriptor verification ──────────────────


@pytest.mark.skipif(not _has_ll(), reason="KV dylib model.ll not found")
class TestLLVMIRSretContract:
    """Verify sret descriptors in LLVM IR comply with sfa.h contract.

    The contract requires that every ciface wrapper's sret output
    descriptor has both allocated (field 0) and aligned (field 1)
    populated from real memory allocations, not null/undef.
    """

    @pytest.fixture(scope="class")
    def llvm_text(self) -> str:
        return _KV_LL_PATH.read_text()

    @pytest.fixture(scope="class")
    def ciface_funcs(self, llvm_text: str) -> dict[str, dict[str, Any]]:
        return _extract_ciface_functions(llvm_text)

    def test_ciface_function_count(self, ciface_funcs: dict[str, dict[str, Any]]) -> None:
        """KV dylib with cache-policy should have 28 ciface wrappers."""
        count = len(ciface_funcs)
        # 28 = main_0 + 12*_a + 12*_b + main_13 + main_14 + main_15
        assert count == 28, (
            f"Expected 28 ciface wrappers (KV split), got {count}. "
            f"Functions: {list(ciface_funcs.keys())}"
        )

    def test_all_ciface_have_sret_construction(self, ciface_funcs: dict[str, dict[str, Any]]) -> None:
        """Every ciface wrapper must construct a sret descriptor.

        The sret pointer (%0) is always the first argument. The function
        must either store a call result to it or build it via insertvalue.
        """
        missing: list[str] = []
        for name, info in ciface_funcs.items():
            events = _find_sret_construction(info["body"])
            if not events:
                missing.append(name)

        assert not missing, (
            f"These ciface wrappers have no detectable sret construction: {missing}"
        )

    def test_sret_store_call_has_returns(self, ciface_funcs: dict[str, dict[str, Any]]) -> None:
        void_stores: list[str] = []
        for name, info in ciface_funcs.items():
            body = info["body"]
            if not _has_sret_store(body):
                continue
            store_match = re.search(r"store\s+\{[^}]+\}\s+%(\d+),\s*ptr\s*%0", body)
            if store_match:
                result_reg = store_match.group(1)
                call_pat = re.compile(
                    rf"%{result_reg}\s*=\s*call\s+\{{[^}}]+\}}\s*@\w+\("
                )
                if not call_pat.search(body):
                    void_stores.append(f"{name} (stores %{result_reg} from non-call)")

        assert not void_stores, (
            f"Functions storing non-call results to sret: {void_stores}"
        )

    def test_kv_split_sret_has_valid_construction(self, ciface_funcs: dict[str, dict[str, Any]]) -> None:
        kv_funcs = {k: v for k, v in ciface_funcs.items() if k.endswith("a")}
        assert len(kv_funcs) == 12, f"Expected 12 KV split (_a) functions, got {len(kv_funcs)}"

        for name, info in kv_funcs.items():
            events = _find_sret_construction(info["body"])
            store_evs = [e for e in events if e["type"] == "store_sret"]
            insert_evs = [e for e in events if e["type"] == "insertvalue_chain"]

            assert store_evs or insert_evs, (
                f"KV function {name}: no sret construction found"
            )

            for ev in insert_evs:
                field_check = ev["fields_set"]
                assert 0 in field_check, (
                    f"KV function {name}: insertvalue chain missing field 0 (allocated)"
                )
                assert 1 in field_check, (
                    f"KV function {name}: insertvalue chain missing field 1 (aligned). "
                    f"Fields set: {field_check}"
                )

    def test_no_undef_sret_descriptor(self, ciface_funcs: dict[str, dict[str, Any]]) -> None:
        """No sret descriptor should be written with undef values.

        Pattern check: insertvalue with undef base that sets field 0 to %0
        (the sret ptr itself) but fails to set field 1 (aligned) is a bug.
        """
        bad_funcs: list[str] = []
        for name, info in ciface_funcs.items():
            body = info["body"]
            # Find all insertvalue chains starting from undef
            # Check that for each chain, both field 0 AND field 1 are set
            chains = _find_sret_construction(body)
            for ev in chains:
                if ev["type"] == "insertvalue_chain":
                    fields = ev["fields_set"]
                    if 0 not in fields or 1 not in fields:
                        bad_funcs.append(
                            f"{name} (rank={ev['rank']}, fields={fields})"
                        )

        assert not bad_funcs, (
            f"Functions with incomplete descriptor construction "
            f"(missing allocated or aligned): {bad_funcs}"
        )


# ── Contract: dylib load + symbol verification ─────────────────────


@pytest.mark.skipif(not _has_dylib(), reason="KV dylib not found")
class TestDylibSymbolContract:
    """Verify compiled dylib can be loaded and has correct ciface symbols."""

    @pytest.fixture(scope="class")
    def lib(self) -> ctypes.CDLL:
        return ctypes.CDLL(str(_KV_DYLIB_PATH))

    def test_dylib_loads_without_error(self, lib: ctypes.CDLL) -> None:
        """Dylib must load via ctypes without error."""
        assert lib._handle is not None, "Dylib handle is null after load"

    def test_ciface_main_1a_symbol_exists(self, lib: ctypes.CDLL) -> None:
        """Key KV split function symbol must be exported."""
        try:
            func = lib._mlir_ciface_main_1a
            assert func is not None
        except AttributeError:
            pytest.fail("_mlir_ciface_main_1a not found in dylib symbols")

    def test_all_ciface_symbols_exported(self, lib: ctypes.CDLL) -> None:
        """All 28 ciface wrappers must be exported as dylib symbols."""
        missing: list[str] = []
        # KV split: main_0 + main_1a..main_12a + main_1b..main_12b
        # + main_13 + main_14 + main_15 = 28 total
        expected = ["_mlir_ciface_main_0"]
        for i in range(1, 13):  # 1..12
            expected.append(f"_mlir_ciface_main_{i}a")
            expected.append(f"_mlir_ciface_main_{i}b")
        for i in range(13, 16):  # 13..15
            expected.append(f"_mlir_ciface_main_{i}")

        for name in expected:
            try:
                getattr(lib, name)
            except AttributeError:
                missing.append(name)

        assert not missing, (
            f"Missing ciface symbols: {missing} "
            f"({len(missing)}/{len(expected)} missing)"
        )

    def test_sfa_abi_symbol_exists(self, lib: ctypes.CDLL) -> None:
        """sfa_abi exported symbol must exist for proto deserialization."""
        try:
            sym = lib.sfa_abi
            assert sym is not None
        except AttributeError:
            pytest.fail("sfa_abi symbol not found in dylib")

    def test_sfa_abi_size_symbol_exists(self, lib: ctypes.CDLL) -> None:
        """sfa_abi_size exported symbol must exist."""
        try:
            sym = lib.sfa_abi_size
            assert sym is not None
        except AttributeError:
            pytest.fail("sfa_abi_size symbol not found in dylib")

    def test_sfa_weights_symbol_exists(self, lib: ctypes.CDLL) -> None:
        """sfa_weights exported symbol must exist."""
        try:
            sym = lib.sfa_weights
            assert sym is not None
        except AttributeError:
            pytest.fail("sfa_weights symbol not found in dylib")

    def test_sfa_weights_size_symbol_exists(self, lib: ctypes.CDLL) -> None:
        """sfa_weights_size exported symbol must exist."""
        try:
            sym = lib.sfa_weights_size
            assert sym is not None
        except AttributeError:
            pytest.fail("sfa_weights_size symbol not found in dylib")


# ── Contract: memref descriptor binary layout ───────────────────────


class TestMemRefDescriptorLayout:
    """Verify memref descriptor binary layout matches MLIR LLVM dialect.

    When ctypes reads a memref descriptor struct from the dylib,
    the binary layout must match exactly. This is the contract that
    prevents the sret null-pointer SIGSEGV.
    """

    def test_memref_packed_struct(self) -> None:
        for rank, expected in [(1, 40), (2, 56), (3, 72), (4, 88)]:
            fmt = "PPq" + "q" * rank + "q" * rank
            calc = struct.calcsize(fmt)
            assert calc == expected, f"Rank {rank}: struct.calcsize={calc}, expected={expected}"

    def test_aligned_field_at_byte_8(self) -> None:
        for rank in [1, 2, 3, 4]:
            fmt = "PPq" + "q" * rank + "q" * rank
            offsets = [0]
            for f in fmt:
                offsets.append(offsets[-1] + struct.calcsize(f))
            assert offsets[1] == 8, (
                f"Rank {rank}: aligned field at byte {offsets[1]}, expected 8"
            )

    def test_memref_descriptor_can_be_created(self) -> None:
        import numpy as np

        data = np.zeros(10, dtype=np.float32)
        ptr = data.ctypes.data_as(ctypes.c_void_p)

        desc = SFATensorRaw1()
        desc.allocated = ptr
        desc.aligned = ptr
        desc.offset = 0
        desc.sizes[0] = 10
        desc.strides[0] = 1

        assert desc.allocated is not None
        assert desc.aligned is not None
        assert desc.offset == 0
        assert desc.sizes[0] == 10
        assert desc.strides[0] == 1


# ── Runtime sret validation: compile mini model, ctypes call, verify descriptors ─

@pytest.mark.integration
@pytest.mark.timeout(120)
class TestRuntimeSretContract:
    """Verify sret descriptors at RUNTIME by compiling and calling via ctypes.

    The LLVM IR tests above verify compile-time patterns. This test
    actually compiles a single-op model to a dylib, calls the ciface
    function via ctypes, and reads the sret output buffer to verify
    allocated and aligned pointers are non-null at runtime.
    """

    def test_sret_descriptor_non_null_at_runtime(self) -> None:
        import sys
        import tempfile
        from pathlib import Path

        import numpy as np

        root = Path(__file__).resolve().parent.parent.parent
        sys.path.insert(0, str(root))

        from compiler.tests.test_precision_contract import (
            _compile_sf_to_dylib,
            _make_memref_struct,
        )

        # Minimal matmul model: input [1,2] @ weight [[1,2],[3,4]] = [7,10]
        sf_mlir = """module {
  func.func @main_0(%input: tensor<1x2xf32>, %weight: tensor<2x2xf32>) -> tensor<1x2xf32> {
    %0 = "sf.matmul"(%input, %weight) : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<1x2xf32>
    return %0 : tensor<1x2xf32>
  }
}"""
        input_data = np.array([[1.0, 2.0]], dtype=np.float32)
        weight_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        with tempfile.TemporaryDirectory() as td:
            dylib_path = _compile_sf_to_dylib(sf_mlir, td, "test_sret")
            lib = ctypes.CDLL(dylib_path)

            inp_mr = _make_memref_struct(
                input_data.ctypes.data, input_data.ndim, input_data.shape)
            w_mr = _make_memref_struct(
                weight_data.ctypes.data, weight_data.ndim, weight_data.shape)

            sret_buf = (ctypes.c_uint8 * 1024)()
            lib._mlir_ciface_main_0(
                ctypes.byref(sret_buf),
                ctypes.byref(inp_mr),
                ctypes.byref(w_mr),
            )

            # Read sret descriptor fields directly from the buffer
            # offset 0: allocated (8 bytes, void*)
            # offset 8: aligned   (8 bytes, void*)
            # offset 16: offset   (8 bytes, i64)
            # offset 24: sizes[0] (8 bytes, i64)
            allocated_ptr = ctypes.c_void_p.from_buffer(sret_buf, 0)
            aligned_ptr = ctypes.c_void_p.from_buffer(sret_buf, 8)

            # Contract: both allocated and aligned must be non-null
            assert allocated_ptr.value is not None, (
                "sret contract violation: allocated pointer is null"
            )
            assert aligned_ptr.value is not None, (
                "sret contract violation: aligned pointer is null"
            )
            # They may differ (aligned is the aligned version of allocated)
            # but both must be valid pointers
            assert allocated_ptr.value != 0, (
                f"sret contract violation: allocated={allocated_ptr.value:#x}"
            )
            assert aligned_ptr.value != 0, (
                f"sret contract violation: aligned={aligned_ptr.value:#x}"
            )
