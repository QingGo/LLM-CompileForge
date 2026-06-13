/* test_matmul: C = A @ B  (f32, row-major)
 *
 * A: m×k, B: k×n, C: m×n
 * Uses naive triple-loop matmul — no BLAS dependency.
 *
 * Exports: _mlir_ciface_matmul_f32
 *   arg0: sret  → output descriptor (memref<2xf32> → rank-2, m×n)
 *   arg1: a     → input A descriptor (memref<2xf32>)
 *   arg2: b     → input B descriptor (memref<2xf32>)
 *
 * Descriptor layout (matches MLIR memref struct):
 *   offset  0: allocated ptr (8 bytes)
 *   offset  8: aligned  ptr (8 bytes)
 *   offset 16: offset   i64 (8 bytes)
 *   offset 24: sizes    i64[rank] (8*rank bytes)
 *   offset 40: strides  i64[rank] (8*rank bytes)
 *   Total: 24 + 16*rank bytes
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    float* allocated;
    float* aligned;
    int64_t offset;
    int64_t sizes[2];
    int64_t strides[2];
} memref_2d_f32_t;

static void* sret_alloc(void* sret, int64_t n_bytes) {
    /* Place output data immediately after the 56-byte descriptor. */
    (void)n_bytes;
    return (char*)sret + 56;
}

void _mlir_ciface_matmul_f32(
    void* sret,
    memref_2d_f32_t* a,
    memref_2d_f32_t* b
) {
    int64_t m = a->sizes[0];
    int64_t k = a->sizes[1];
    int64_t n = b->sizes[1];

    float* out_data = (float*)sret_alloc(sret, m * n);
    float* a_data = a->aligned;
    float* b_data = b->aligned;
    int64_t a_stride0 = a->strides[0];
    int64_t a_stride1 = a->strides[1];
    int64_t b_stride0 = b->strides[0];
    int64_t b_stride1 = b->strides[1];

    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int64_t p = 0; p < k; p++) {
                sum += a_data[i * a_stride0 + p * a_stride1]
                     * b_data[p * b_stride0 + j * b_stride1];
            }
            out_data[i * n + j] = sum;
        }
    }

    memref_2d_f32_t* out_desc = (memref_2d_f32_t*)sret;
    out_desc->allocated = out_data;
    out_desc->aligned = out_data;
    out_desc->offset = 0;
    out_desc->sizes[0] = m;
    out_desc->sizes[1] = n;
    out_desc->strides[0] = n;
    out_desc->strides[1] = 1;
}
