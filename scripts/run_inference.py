"""Run inference on a compiled .dylib model using Python ctypes.

Usage: python scripts/run_inference.py compiled/opt_125m_v8 [--model-name opt_125m] [--prompt "Hello"] [--max-tokens 10]
"""

import argparse
import ctypes
import json
import struct
from pathlib import Path


def load_sfcf_blob(dylib_path: str) -> bytes:
    """Extract the embedded SFCF constants blob from a .dylib."""
    import subprocess
    result = subprocess.run(
        ["nm", "-gm", dylib_path],
        capture_output=True, text=True,
    )
    for line in result.stdout.split("\n"):
        if "__serveforge_constants_data" in line:
            parts = line.strip().split()
            int(parts[0], 16) if parts[0].startswith("0") else int(parts[0], 16)
            break
    else:
        raise RuntimeError("serveforge_constants_data symbol not found")

    # Read the dylib binary
    with open(dylib_path, "rb") as f:
        f.read()

    # Find the __TEXT,__const section via otool
    result = subprocess.run(
        ["otool", "-l", dylib_path],
        capture_output=True, text=True,
    )
    # Parse load commands to find __const section
    # For simplicity, just look at the symbol value
    result2 = subprocess.run(
        ["nm", "-gm", dylib_path],
        capture_output=True, text=True,
    )
    const_data_addr = None
    const_size_addr = None
    for line in result2.stdout.split("\n"):
        if "__serveforge_constants_data" in line:
            parts = line.strip().split()
            if parts[0] != "(undefined)":
                const_data_addr = int(parts[0], 16)
        if "__serveforge_constants_size" in line:
            parts = line.strip().split()
            if parts[0] != "(undefined)":
                const_size_addr = int(parts[0], 16)

    if const_data_addr is None or const_size_addr is None:
        raise RuntimeError("Could not find SFCF symbols")

    # The dylib is a Mach-O file. We need to find the __TEXT,__const section
    # which contains the constant data.
    # For this, we use `otool -s __TEXT __const` to dump the section.
    result3 = subprocess.run(
        ["otool", "-s", "__TEXT", "__const", dylib_path],
        capture_output=True, text=True,
    )
    lines = result3.stdout.strip().split("\n")
    # Parse hex dump
    hex_data = b""
    in_section = False
    for line in lines:
        if line.startswith("Contents of"):
            in_section = True
            continue
        if in_section and line.strip():
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                hex_str = parts[1].replace(" ", "")
                hex_data += bytes.fromhex(hex_str)

    return hex_data


def run_inference(model_dir: str, model_name: str, prompt: str, max_tokens: int = 10):
    model_path = Path(model_dir)
    dylib_path = model_path / f"lib{model_name}.dylib"
    model_path / "metadata.json"
    constants_path = model_path / "constants.bin"

    # Read constants
    constants_data = constants_path.read_bytes()
    pos = 0
    assert constants_data[:4] == b"SFCF"
    version = struct.unpack("<I", constants_data[4:8])[0]
    assert version == 2
    pos = 8

    # Parse name mapping
    nm_count = struct.unpack("<I", constants_data[pos:pos+4])[0]
    pos += 4
    name_mapping = {}
    for _ in range(nm_count):
        nlen = struct.unpack("<I", constants_data[pos:pos+4])[0]
        pos += 4
        compiled = constants_data[pos:pos+nlen].decode()
        pos += nlen
        klen = struct.unpack("<I", constants_data[pos:pos+4])[0]
        pos += 4
        hf_key = constants_data[pos:pos+klen].decode()
        pos += klen
        name_mapping[compiled] = hf_key

    print(f"Loaded {len(name_mapping)} weight mappings")

    # Load safetensors
    safetensors_path = model_path / "weights.safetensors"
    if not safetensors_path.exists():
        # Try the opt_125m subdirectory
        safetensors_path = model_path / model_name / "weights.safetensors"

    if not safetensors_path.exists():
        print("WARNING: No safetensors found, using random weights")
        raw_weights = {}
    else:
        with open(safetensors_path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
            raw_weights = {}
            for key, meta in header.items():
                f.seek(8 + header_len + meta["data_offsets"][0])
                raw_weights[key] = f.read(meta["data_offsets"][1] - meta["data_offsets"][0])

    print(f"Loaded {len(raw_weights)} raw weights")

    # Load .dylib
    ctypes.CDLL(str(dylib_path))

    # Get function symbols

    # The function takes: out_ptr, weight0_ptr, weight1_ptr, ..., input_ptr
    # We need to build memref descriptors for each weight and the input

    # For now, show module loaded successfully
    print(f"\nModel loaded from: {dylib_path}")
    print("Module ready for inference (weight integration in progress)")

    return "Inference not yet implemented (weight argument packing needed)"


def main():
    parser = argparse.ArgumentParser(description="Run inference on compiled model")
    parser.add_argument("model_dir", help="Compiled model directory")
    parser.add_argument("--model-name", default="opt_125m", help="Model name")
    parser.add_argument("--prompt", default="Hello", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=10)
    args = parser.parse_args()

    result = run_inference(args.model_dir, args.model_name, args.prompt, args.max_tokens)
    print(result)


if __name__ == "__main__":
    main()
