#!/usr/bin/env python3
"""Inspect a compiled .dylib: list functions, weights, compute graph.

Usage:
    python scripts/inspect_dylib.py compiled/opt_125m_fresh/libopt_125m.dylib
    python scripts/inspect_dylib.py compiled/opt_125m_fresh --summary
    python scripts/inspect_dylib.py compiled/opt_125m_fresh --functions
    python scripts/inspect_dylib.py compiled/opt_125m_fresh --compute-graph
"""

import argparse
import os
import sys
from typing import Any


def find_sfcf_blob(dylib_path: str) -> bytes | None:
    """Extract the SFCF binary blob from a .dylib.

    Reads the dylib as binary, searches for 'SFCF' magic bytes.
    """
    with open(dylib_path, "rb") as f:
        data = f.read()

    sfcf_magic = b"SFCF"
    idx = data.find(sfcf_magic)
    if idx == -1:
        return None

    # Read version (2 bytes after magic)
    int.from_bytes(data[idx+4:idx+6], "little")

    # The blob extends to the end of the embedded data section
    # For now, return from magic to end of file
    return data[idx:]


def list_functions(dylib_path: str) -> list[dict[str, Any]]:
    import subprocess

    result = subprocess.run(
        ["nm", "-g", dylib_path],
        capture_output=True, text=True
    )

    functions = []
    for line in result.stdout.splitlines():
        if "_mlir_ciface_" in line:
            parts = line.split()
            if len(parts) >= 3:
                addr = parts[0]
                kind = parts[1]
                name = parts[2].replace("_mlir_ciface_", "")
                functions.append({
                    "address": addr,
                    "type": kind,
                    "name": name,
                })

    return functions


def list_sections(dylib_path: str) -> list[dict[str, Any]]:
    import subprocess

    result = subprocess.run(
        ["nm", "-gm", dylib_path],
        capture_output=True, text=True
    )

    sections = []
    for line in result.stdout.splitlines():
        if "SECTION" in line or "symbol" in line.lower():
            sections.append(line.strip())

    return sections


def inspect_summary(dylib_path: str) -> dict[str, Any]:

    # File size
    size = os.path.getsize(dylib_path)

    # Function count
    funcs = list_functions(dylib_path)

    # Has SFCF blob
    has_sfcf = find_sfcf_blob(dylib_path) is not None

    return {
        "path": dylib_path,
        "size_bytes": size,
        "size_mb": size / (1024 * 1024),
        "num_functions": len(funcs),
        "has_sfcf_blob": has_sfcf,
        "function_names": [f["name"] for f in funcs],
    }


def print_compute_graph(dylib_path: str, compiled_dir: str = "") -> None:
    """Try to parse and print the compute graph from the SFCF blob."""
    import struct

    blob = find_sfcf_blob(dylib_path)
    if blob is None:
        print("No SFCF blob found (trying compiled dir constants.bin)")
        # Try constants.bin
        if compiled_dir:
            bin_path = os.path.join(compiled_dir, "constants.bin")
            if os.path.exists(bin_path):
                with open(bin_path, "rb") as f:
                    blob = f.read()

    if blob is None:
        print("No SFCF blob found.")
        return

    print("\n=== SFCF Blob ===")
    print(f"Size: {len(blob)} bytes")
    print(f"Magic: {blob[:4]}")

    if len(blob) >= 6:
        version = struct.unpack("<H", blob[4:6])[0]
        print(f"Version: {version}")

    # Try to parse name_mapping entries
    pos = 6  # After SFCF + version
    try:
        num_mappings = struct.unpack("<I", blob[pos:pos+4])[0]
        pos += 4
        print(f"Name mappings: {num_mappings}")

        for _i in range(min(num_mappings, 5)):  # Show first 5
            # Read compiled_name length + string
            name_len = struct.unpack("<I", blob[pos:pos+4])[0]
            pos += 4
            compiled_name = blob[pos:pos+name_len].decode("utf-8", errors="replace")
            pos += name_len

            # Read hf_key length + string
            key_len = struct.unpack("<I", blob[pos:pos+4])[0]
            pos += 4
            hf_key = blob[pos:pos+key_len].decode("utf-8", errors="replace")
            pos += key_len

            print(f"  {compiled_name} \u2192 {hf_key}")

        if num_mappings > 5:
            print(f"  ... and {num_mappings - 5} more")

        # Show constant count
        num_constants = struct.unpack("<I", blob[pos:pos+4])[0]
        pos += 4
        print(f"Constants: {num_constants}")

        for _i in range(min(num_constants, 3)):
            name_len = struct.unpack("<I", blob[pos:pos+4])[0]
            pos += 4
            const_name = blob[pos:pos+name_len].decode("utf-8", errors="replace")
            pos += name_len
            dtype = struct.unpack("<I", blob[pos:pos+4])[0]
            pos += 4
            ndim = struct.unpack("<I", blob[pos:pos+4])[0]
            pos += 4
            shape = struct.unpack(f"<{ndim}I", blob[pos:pos+4*ndim])
            pos += 4*ndim
            print(f"  {const_name}: dtype={dtype}, shape={shape}")

        print(f"\n  Total blob parsed: {pos}/{len(blob)} bytes ({100*pos//len(blob)}%)")
    except (struct.error, IndexError) as e:
        print(f"  Parse error at position {pos}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect a compiled .dylib"
    )
    parser.add_argument("path", help="Path to .dylib or compiled directory")
    parser.add_argument("--summary", action="store_true", help="Show summary only")
    parser.add_argument("--functions", action="store_true", help="List functions")
    parser.add_argument("--compute-graph", action="store_true", help="Parse compute graph")
    args = parser.parse_args()

    path = args.path
    compiled_dir = ""

    # Check if path is a directory or dylib
    if os.path.isdir(path):
        compiled_dir = path
        for f in os.listdir(path):
            if f.endswith(".dylib"):
                path = os.path.join(path, f)
                break

    if not os.path.exists(path):
        print(f"File not found: {path}")
        return 1

    print(f"=== .dylib Inspection: {path} ===")
    print(f"Size: {os.path.getsize(path) / 1024:.1f} KB")

    if args.summary or not (args.functions or args.compute_graph):
        summary = inspect_summary(path)
        print("\nSummary:")
        print(f"  Functions: {summary['num_functions']}")
        print(f"  Has SFCF blob: {summary['has_sfcf_blob']}")
        if summary['function_names']:
            print("  Function names:")
            for name in summary['function_names']:
                print(f"    {name}")

    if args.functions:
        funcs = list_functions(path)
        print(f"\nFunctions ({len(funcs)}):")
        for f in funcs:
            print(f"  {f['name']}")

    if args.compute_graph:
        print_compute_graph(path, compiled_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
