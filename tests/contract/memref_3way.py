"""
memref_3way.py — 3-way MemRef layout contract test (Python side)

Verifies that the per-rank SFATensorRaw struct sizes and field offsets
match the C ABI from include/sfa.h using struct.Struct.

Output format (one value per line, same keys as C and Rust):
  Python:40
  Python:56
  Python:72
  Python:88
  Python:allocated=0
  Python:aligned=8
  Python:offset=16
"""

import struct
import sys


def main() -> int:
    ptr_size = struct.Struct("P").size  # void* — 8 bytes on 64-bit
    i64_size = struct.Struct("q").size  # int64_t — 8 bytes
    header = ptr_size + ptr_size + i64_size  # allocated + aligned + offset

    # Compute sizes: header + rank * 8 (sizes array) + rank * 8 (strides array)
    sizes = {}
    for rank in range(1, 5):
        total = header + 2 * rank * i64_size
        sizes[rank] = total
        print(f"Python:{total}")

    # Field offsets for the 24-byte memref header (SFATensorRaw2).
    # The layout is struct<(void*, void*, int64_t, int64_t[2], int64_t[2])>
    assert struct.calcsize("PPq2q2q") == 56, f"SFATensorRaw2 expected 56, got {struct.calcsize('PPq2q2q')}"
    print("Python:allocated=0")
    print(f"Python:aligned={ptr_size}")
    print(f"Python:offset={ptr_size + ptr_size}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
