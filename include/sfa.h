/**
 * sfa.h — Stable Function ABI header for LLM-CompileForge
 *
 * SINGLE SOURCE OF TRUTH for tensor descriptor cross-component interfaces:
 *   compiler → dylib → runtime (Python, C, Rust)
 *
 * Sections:
 *   1. Tensor descriptors       — SFATensorRaw1-4, SFATensor, SFADevice
 *   2. SFA ABI metadata         — protobuf schema (see include/sfa_abi.proto)
 *
 * Binary layout of Section 1 matches MLIR LLVM dialect memref descriptor:
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
 *
 * Function metadata and weight data (Sections 2-4) are now defined
 * as protobuf in include/sfa_abi.proto and serialized via protoc.
 * No hand-rolled C structs — the proto schema is the sole source of truth.
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

/* ── Section 2: SFA ABI metadata ─────────────────────────────────────
 *
 * Function metadata, input bindings, and weight data are encoded as
 * protobuf (see include/sfa_abi.proto) and embedded in the compiled
 * dylib as exported symbols:
 *
 *   sfa_abi      — SfaAbiHeader protobuf binary
 *   sfa_abi_size  — u64 byte length
 *   sfa_weights  — SfaWeightData protobuf binary
 *   sfa_weights_size — u64 byte length
 *
 * The runtime decodes these with prost::Message::decode().
 * No hand-rolled C structs — the proto schema is the sole source of truth.
 */

#ifdef __cplusplus
}
#endif

#endif /* SFA_H */
