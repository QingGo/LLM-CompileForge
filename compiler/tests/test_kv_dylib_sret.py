"""TDD test for Bug 2: KV-split dylib rank-4 sret aligned pointer null.

L2: Validates LLVM IR sret pattern for multi-return rank-4 functions.
"""

import os
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.mark.integration
@pytest.mark.timeout(30)
def test_kv_dylib_llvm_ir_has_malloc_for_rank4_outputs():
    """L2 test: The inner function @main_1a must use malloc for rank-4 outputs."""
    ll_path = os.path.join(ROOT, "outputs", "compiled", "opt_125m_kv", "model.ll")
    if not os.path.exists(ll_path):
        pytest.skip(f"LLVM IR not found. Run: make rebuild-dylib for opt_125m_kv")

    with open(ll_path) as f:
        ll_text = f.read()

    # Find @main_1a function
    lines = ll_text.split("\n")
    inner_start = None
    for i, line in enumerate(lines):
        if "@main_1a(" in line and "define" in line:
            inner_start = i
            break
    assert inner_start is not None, "@main_1a not found in LLVM IR"

    # Find end of @main_1a
    brace_count = 0
    inner_end = None
    for i in range(inner_start, len(lines)):
        brace_count += lines[i].count("{") - lines[i].count("}")
        if brace_count == 0 and i > inner_start:
            inner_end = i
            break

    inner_text = "\n".join(lines[inner_start:inner_end + 1])

    # Assert: at least 3 malloc calls for 3 rank-4 output buffers
    malloc_count = inner_text.count("@malloc")
    assert malloc_count >= 3, (
        f"BUG: @main_1a has only {malloc_count} malloc calls, "
        f"expected >=3 for 3 rank-4 outputs"
    )

    # Assert: return type contains rank-4 memref structs
    ret_line = None
    for line in reversed(lines[inner_start:inner_end + 1]):
        if "ret" in line and not line.strip().startswith(";"):
            ret_line = line.strip()
            break
    assert "[4 x i64], [4 x i64]" in ret_line, (
        f"BUG: @main_1a return type should be rank-4 structs: {ret_line[:120]}"
    )

    # Assert: ciface wrapper stores result to sret ptr %0
    wrapper_start = None
    for i, line in enumerate(lines):
        if "_mlir_ciface_main_1a" in line and "define" in line:
            wrapper_start = i
            break
    wrapper_text = "\n".join(lines[wrapper_start:inner_start])
    assert "store" in wrapper_text, "ciface wrapper must store result to sret"
    assert "ptr %0" in wrapper_text, "ciface wrapper must reference sret ptr %0"

    print(f"PASS: LLVM IR validated for _mlir_ciface_main_1a")
    print(f"  malloc calls: {malloc_count}")
    print(f"  ret type: rank-4 structs confirmed")
    print(f"  ciface wrapper: store to sret ptr %0 confirmed")


@pytest.mark.integration
@pytest.mark.timeout(30)
def test_kv_dylib_all_symbols_loadable():
    """L3 test: All _mlir_ciface_* symbols from ABI proto must exist in dylib."""
    import ctypes
    from gen.proto.python.sfa_abi_pb2 import SfaAbiHeader

    dylib_path = os.path.join(ROOT, "outputs", "compiled", "opt_125m_kv", "libopt_125m.dylib")
    if not os.path.exists(dylib_path):
        pytest.skip("KV dylib not found")

    lib = ctypes.CDLL(dylib_path)
    size_sym = ctypes.c_uint64.in_dll(lib, "sfa_abi_size")
    data_ptr = (ctypes.c_uint8 * size_sym.value).in_dll(lib, "sfa_abi")
    abi = SfaAbiHeader()
    abi.ParseFromString(bytes(data_ptr))

    missing = []
    for f in abi.funcs:
        try:
            getattr(lib, f.symbol)
        except AttributeError:
            missing.append(f.symbol)

    assert not missing, f"BUG: {len(missing)} symbols missing from dylib: {missing[:5]}..."
    print(f"PASS: All {len(abi.funcs)} symbols loadable from dylib")
