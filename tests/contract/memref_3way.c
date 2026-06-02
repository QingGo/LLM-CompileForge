/**
 * memref_3way.c — 3-way MemRef layout contract test (C side)
 *
 * Verifies that sizeof(SFATensorRaw1-4) and field offsets in
 * include/sfa.h match the expected MLIR memref descriptor layout:
 *   struct<(ptr, ptr, i64, array<RANK x i64>, array<RANK x i64>)>
 *
 * Output format (one value per line):
 *   C:40
 *   C:56
 *   C:72
 *   C:88
 *   C:allocated=0
 *   C:aligned=8
 *   C:offset=16
 *
 * Build: cc -I../../include -o memref_3way_c memref_3way.c
 */

#include <stddef.h>
#include <stdio.h>
#include "sfa.h"

int main(void) {
    /* sizeof assertions (must match the header comment + Rust + Python) */
    printf("C:%zu\n", sizeof(SFATensorRaw1));
    printf("C:%zu\n", sizeof(SFATensorRaw2));
    printf("C:%zu\n", sizeof(SFATensorRaw3));
    printf("C:%zu\n", sizeof(SFATensorRaw4));

    /* Field offset assertions for the 24-byte memref header.
     * SFATensorRaw2 is used as the canonical rank since the header
     * layout is identical across all ranks. */
    printf("C:allocated=%zu\n", offsetof(SFATensorRaw2, allocated));
    printf("C:aligned=%zu\n",   offsetof(SFATensorRaw2, aligned));
    printf("C:offset=%zu\n",    offsetof(SFATensorRaw2, offset));

    return 0;
}
