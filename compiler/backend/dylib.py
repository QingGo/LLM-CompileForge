"""Dylib compilation and SFA ABI embedding utilities."""

import os
import subprocess
import textwrap
from pathlib import Path

from compiler.backend.compile_utils import (
    _compile_serveforge_free,
    _compile_sfa_blas_bridge,
    _find_cc,
)


def _check_sf_dialect_freshness(compiled_dir: str) -> None:
    """Check model.mlir is not stale vs current sf-dialect.

    Hard error if sf_dialect_hash in metadata differs from HEAD.
    Silent pass if metadata has no hash (old model) or not in git repo.
    """
    import json as _json
    import os as _os
    import subprocess as _subprocess
    import sys as _sys

    meta_path = _os.path.join(compiled_dir, "metadata.json")
    if not _os.path.isfile(meta_path):
        return  # no metadata, skip check

    with open(meta_path) as _f:
        meta = _json.load(_f)

    old_hash = meta.get("sf_dialect_hash")
    if old_hash is None:
        print(
            "WARNING: model.mlir has no sf_dialect_hash (pre-staleness-check era)",
            file=_sys.stderr,
        )
        return

    try:
        result = _subprocess.run(
            ["git", "rev-parse", "HEAD:sf-dialect/"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        print(
            "WARNING: could not verify sf-dialect freshness (not a git repo?)",
            file=_sys.stderr,
        )
        return

    if result.returncode != 0:
        print(
            "WARNING: could not verify sf-dialect freshness (git error)",
            file=_sys.stderr,
        )
        return

    new_hash = result.stdout.strip()
    if old_hash != new_hash:
        print("ERROR: model.mlir is stale!", file=_sys.stderr)
        print(f"  Generated with sf-dialect: {old_hash[:8]}", file=_sys.stderr)
        print(f"  Current sf-dialect HEAD:  {new_hash[:8]}", file=_sys.stderr)
        print(
            f"  Run 'python compiler/compile.py ... --output-dir {compiled_dir}' first.",
            file=_sys.stderr,
        )
        print(
            "  Or use 'make test-dylib-cos' which handles this automatically.",
            file=_sys.stderr,
        )
        _sys.exit(1)


def _compile_blob_to_o(
    data: bytes,
    symbol_name: str,
    size_symbol_name: str,
    work_dir: str,
) -> str:
    """Compile a binary blob to a .o file exporting named symbols.

    Returns the path to the generated .o file.
    """
    hex_lines: list[str] = []
    for i in range(0, len(data), 12):
        chunk = data[i : i + 12]
        hex_lines.append(", ".join(f"0x{b:02X}" for b in chunk))

    c_source = textwrap.dedent(f"""\
    #include <stdint.h>
    const uint8_t {symbol_name}[{len(data)}] = {{
        {",".join(hex_lines)}
    }};
    const uint64_t {size_symbol_name} = {len(data)};
    """)

    c_path = os.path.join(work_dir, f"{symbol_name}.c")
    o_path = os.path.join(work_dir, f"{symbol_name}.o")
    with open(c_path, "w") as f:
        f.write(c_source)

    cc_bin = _find_cc()
    result = subprocess.run(
        [cc_bin, "-c", c_path, "-o", o_path],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to compile {symbol_name} object (exit {result.returncode}):\n{result.stderr[:2000]}"
        )
    return o_path


def _sfa_relink_dylib(
    compiled_path: Path,
    model_name: str,
    sfa_abi_data: bytes,
    sfa_weights_data: bytes,
    sfa_cache_policy_data: bytes | None = None,
    sfa_op_plan_data: bytes | None = None,
) -> None:
    """Re-link the dylib with SFA ABI and weights object files embedded.

    When ``sfa_cache_policy_data`` is provided, the ``sfa_cache_policy`` /
    ``sfa_cache_policy_size`` symbol pair is exported too.  Its presence is
    the runtime's feature-detect signal for the proto cache-policy contract
    (absent on legacy dylibs, which keep the JSON/heuristic fallback).

    When ``sfa_op_plan_data`` is provided, the additive ``sfa_op_plan`` /
    ``sfa_op_plan_size`` pair is exported for the Phase 5 HAL kernel graph.
    """
    from compiler.backend.llvm_backend import (  # type: ignore[attr-defined]
        _compile_embedded_data,
        link_dylib,
        llc_compile,
    )

    work_dir = str(compiled_path)

    sfa_abi_o = _compile_blob_to_o(
        sfa_abi_data,
        "sfa_abi",
        "sfa_abi_size",
        work_dir,
    )
    sfa_weights_o = _compile_blob_to_o(
        sfa_weights_data,
        "sfa_weights",
        "sfa_weights_size",
        work_dir,
    )

    model_ll = os.path.join(work_dir, "model.ll")
    model_o = llc_compile(model_ll, arch="native", opt_level=3)

    const_bin = os.path.join(work_dir, "constants.bin")
    const_o = _compile_embedded_data(const_bin, work_dir)

    free_o = _compile_serveforge_free(work_dir)
    blas_o = _compile_sfa_blas_bridge(work_dir)

    dylib_path = os.path.join(work_dir, f"lib{model_name}.dylib")
    obj_files = [model_o, const_o, free_o, blas_o, sfa_abi_o, sfa_weights_o]
    if sfa_cache_policy_data is not None:
        sfa_cache_policy_o = _compile_blob_to_o(
            sfa_cache_policy_data,
            "sfa_cache_policy",
            "sfa_cache_policy_size",
            work_dir,
        )
        obj_files.append(sfa_cache_policy_o)
    if sfa_op_plan_data is not None:
        sfa_op_plan_o = _compile_blob_to_o(
            sfa_op_plan_data,
            "sfa_op_plan",
            "sfa_op_plan_size",
            work_dir,
        )
        obj_files.append(sfa_op_plan_o)
    link_dylib(obj_files, dylib_path)
