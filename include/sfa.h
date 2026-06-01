/**
 * sfa.h — Stable Function ABI header for LLM-CompileForge
 *
 * Phase 0: C-level struct definitions for tensor descriptors.
 * Binary layout matches MLIR LLVM dialect memref descriptor:
 *
 *   struct<(ptr, ptr, i64, array<RANK x i64>, array<RANK x i64>)>
 *
 * Per-rank structs (SFATensorRaw1..4) guarantee the correct ABI
 * for every rank.  The first three fields (allocated, aligned, offset)
 * are a fixed 24-byte header that mirrors the MLIR memref exactly.
 *
 * Rank-parameterized sizes:
 *   SFATensorRaw1: 24 + 8 + 8  = 40 bytes
 *   SFATensorRaw2: 24 + 16 + 16 = 56 bytes
 *   SFATensorRaw3: 24 + 24 + 24 = 72 bytes
 *   SFATensorRaw4: 24 + 32 + 32 = 88 bytes
 */

#ifndef SFA_H
#define SFA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Device type ─────────────────────────────────────────────────── */

typedef enum {
    SFA_CPU  = 1,
    SFA_CUDA = 2,
} SFADevice;

/* ── Per-rank raw tensor descriptors ────────────────────────────────
 *
 * Each struct is binary-compatible with the MLIR memref descriptor
 * for that rank.  Do NOT merge into a single struct with fixed-size
 * arrays — the ABI requires sizes/strides to match the function
 * signature's declared rank exactly.
 */

typedef struct {
    void*    allocated;
    void*    aligned;
    int64_t  offset;
    int64_t  sizes[1];
    int64_t  strides[1];
} SFATensorRaw1;

typedef struct {
    void*    allocated;
    void*    aligned;
    int64_t  offset;
    int64_t  sizes[2];
    int64_t  strides[2];
} SFATensorRaw2;

typedef struct {
    void*    allocated;
    void*    aligned;
    int64_t  offset;
    int64_t  sizes[3];
    int64_t  strides[3];
} SFATensorRaw3;

typedef struct {
    void*    allocated;
    void*    aligned;
    int64_t  offset;
    int64_t  sizes[4];
    int64_t  strides[4];
} SFATensorRaw4;

/* ── Unified tensor wrapper ──────────────────────────────────────── */

typedef struct {
    SFADevice device;
    int32_t   rank;
    int32_t   elem_size;   /* sz of one element in bytes (e.g. 4 for f32) */

    union {
        SFATensorRaw1 r1;
        SFATensorRaw2 r2;
        SFATensorRaw3 r3;
        SFATensorRaw4 r4;
    };
} SFATensor;

#ifdef __cplusplus
}
#endif

#endif /* SFA_H */
