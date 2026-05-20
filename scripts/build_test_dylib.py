"""Build synthetic test dylib + safetensors for weight f16→f32 TDD.

Produces:
  tests/data/libtest_weight.dylib   — single-function dylib with embedded SFCF blob
  tests/data/test_weight.safetensors — F16 weight tensor

Kernel: copies weight element-wise to output (identity).
Used by Rust test test_weight_forward_via_dylib.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _find_cc() -> str:
    for candidate in ["cc", "gcc", "clang"]:
        try:
            result = subprocess.run(
                [candidate, "--version"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return candidate
        except FileNotFoundError:
            continue
    raise RuntimeError("no C compiler found")


def _compile_embedded_data(bin_path: str, work_dir: str) -> str:
    """Create a .o file with embedded binary data (same as compile_utils)."""
    with open(bin_path, "rb") as f:
        data = f.read()
    hex_lines = []
    for i in range(0, len(data), 12):
        chunk = data[i: i + 12]
        hex_lines.append(", ".join(f"0x{b:02X}" for b in chunk))
    c_source = textwrap.dedent(f"""\
    #include <stdint.h>
    const uint8_t serveforge_constants_data[{len(data)}] = {{
        {",".join(hex_lines)}
    }};
    const uint64_t serveforge_constants_size = {len(data)};
    """)
    c_path = os.path.join(work_dir, "serveforge_constants.c")
    o_path = os.path.join(work_dir, "serveforge_constants.o")
    with open(c_path, "w") as f:
        f.write(c_source)
    cc_bin = _find_cc()
    result = subprocess.run(
        [cc_bin, "-c", c_path, "-o", o_path],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to compile embedded constants (exit {result.returncode}):\n"
            f"{result.stderr[:2000]}"
        )
    return o_path


def _link_dylib(obj_files: list[str], output: str) -> str:
    cc_bin = _find_cc()
    cmd = [cc_bin, "-shared", "-o", output] + obj_files
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(
            f"Link failed (exit {result.returncode}):\n{result.stderr[:2000]}"
        )
    return output


def _emit_string(buf: bytearray, s: str) -> None:
    encoded = s.encode("utf-8")
    buf.extend(struct.pack("<I", len(encoded)))
    buf.extend(encoded)


def build_sfcf_blob() -> bytes:
    """Build SFCF binary blob for a single weight-copy function."""
    buf = bytearray()
    buf.extend(b"SFCF")
    buf.extend(struct.pack("<I", 2))  # version

    # Name mapping: 1 entry
    buf.extend(struct.pack("<I", 1))
    _emit_string(buf, "weight.0")
    _emit_string(buf, "test_weight")

    # Constants: 0
    buf.extend(struct.pack("<I", 0))

    # Compute graph: 1 function
    buf.extend(struct.pack("<I", 1))

    # Function 0
    _emit_string(buf, "_mlir_ciface_mul_weight_input")
    buf.extend(struct.pack("<I", 2))  # num_inputs
    buf.extend(struct.pack("<I", 1))  # num_outputs

    # Input 0: weight (binding_type=0)
    buf.extend(struct.pack("<B", 0))
    _emit_string(buf, "weight.0")
    buf.extend(struct.pack("<B", 1))  # rank=1
    buf.extend(struct.pack("<I", 1))  # num_dims
    buf.extend(struct.pack("<Q", 4))  # shape[0]=4

    # Input 1: global_input (binding_type=2)
    buf.extend(struct.pack("<B", 2))
    buf.extend(struct.pack("<B", 2))  # rank=2
    buf.extend(struct.pack("<I", 2))  # num_dims
    buf.extend(struct.pack("<Q", 1))  # shape[0]=1
    buf.extend(struct.pack("<Q", 4))  # shape[1]=4

    # Output 0
    buf.extend(struct.pack("<B", 1))  # rank=1
    buf.extend(struct.pack("<I", 1))  # num_dims
    buf.extend(struct.pack("<Q", 4))  # shape[0]=4

    # Global input: func=0, arg_idx=1
    buf.extend(struct.pack("<I", 0))
    buf.extend(struct.pack("<I", 1))

    # Global output: func=0, output_idx=0
    buf.extend(struct.pack("<I", 0))
    buf.extend(struct.pack("<I", 0))

    return bytes(buf)


def build_safetensors(output_dir: Path) -> Path:
    """Create a safetensors file with known F16 values for 'test_weight'."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # F16 values: [1.0, 2.0, 3.0, 4.0]
    f16_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    raw_bytes = f16_data.tobytes()

    header = {
        "test_weight": {
            "dtype": "F16",
            "shape": [4],
            "data_offsets": [0, len(raw_bytes)],
        }
    }
    header_json = json.dumps(header, separators=(",", ":"))
    header_bytes = header_json.encode("utf-8")

    # Pad header to 8-byte alignment (safetensors spec)
    padded_len = ((8 + len(header_bytes) + 7) // 8) * 8
    padding = b" " * (padded_len - 8 - len(header_bytes))

    st_path = output_dir / "test_weight.safetensors"
    with open(st_path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes) + len(padding)))
        f.write(header_bytes)
        f.write(padding)
        f.write(raw_bytes)

    return st_path


def build_test_dylib(output_dir: Path) -> Path:
    """Build the full test dylib with embedded SFCF blob."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── C kernel source ──
    c_source = textwrap.dedent("""\
    #include <stdint.h>
    #include <string.h>

    typedef struct {
        float* allocated;
        float* aligned;
        int64_t offset;
        int64_t sizes[1];
        int64_t strides[1];
    } memref_1d_f32_t;

    typedef struct {
        int64_t* allocated;
        int64_t* aligned;
        int64_t offset;
        int64_t sizes[2];
        int64_t strides[2];
    } memref_2d_i64_t;

    void _mlir_ciface_mul_weight_input(
        void* sret,
        memref_1d_f32_t* weight,
        memref_2d_i64_t* input
    ) {
        (void)input;
        int64_t n = weight->sizes[0];

        float* out_data = (float*)((char*)sret + 1024);
        for (int64_t i = 0; i < n; i++) {
            out_data[i] = weight->aligned[weight->strides[0] * i];
        }

        memref_1d_f32_t* out_desc = (memref_1d_f32_t*)sret;
        out_desc->allocated = out_data;
        out_desc->aligned = out_data;
        out_desc->offset = 0;
        out_desc->sizes[0] = n;
        out_desc->strides[0] = 1;
    }
    """)
    c_path = output_dir / "test_kernel.c"
    c_path.write_text(c_source)

    # Compile to .o
    o_path = output_dir / "test_kernel.o"
    cc_bin = _find_cc()
    subprocess.run(
        [cc_bin, "-c", str(c_path), "-o", str(o_path), "-O0"],
        capture_output=True, text=True, check=True, timeout=30,
    )

    # ── SFCF blob → embedded .o ──
    sfcf_bytes = build_sfcf_blob()
    sfcf_path = output_dir / "constants.bin"
    sfcf_path.write_bytes(sfcf_bytes)
    constants_o = _compile_embedded_data(str(sfcf_path), str(output_dir))

    # ── Link → .dylib ──
    dylib_path = output_dir / "libtest_weight.dylib"
    _link_dylib([str(o_path), constants_o], str(dylib_path))

    return dylib_path


if __name__ == "__main__":
    data_dir = ROOT / "tests" / "data"

    dylib_path = build_test_dylib(data_dir)
    print(f"dylib: {dylib_path} ({dylib_path.stat().st_size} bytes)")

    st_path = build_safetensors(data_dir)
    print(f"safetensors: {st_path} ({st_path.stat().st_size} bytes)")

    result = subprocess.run(
        ["nm", "-gU", str(dylib_path)], capture_output=True, text=True
    )
    print("\nSymbols:")
    print(result.stdout)
