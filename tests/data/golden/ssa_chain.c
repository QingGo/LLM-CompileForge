/* ssa_chain — two golden ciface functions to test SSA wiring.
 *
 * func_0: _mlir_ciface_matmul_2x2   (3 args: sret, A, B)
 *   Hard-coded 2×2 @ 2×2 matmul: A=[[1,2],[3,4]] × B=[[5,6],[7,8]] → [[19,22],[43,50]]
 *
 * func_1: _mlir_ciface_add_constant (2 args: sret, X)
 *   Adds 10.0 to each element of 2×2 input.
 *
 * Descriptor layout (matches MLIR memref struct):
 *   offset  0: allocated ptr (8 bytes)
 *   offset  8: aligned  ptr (8 bytes)
 *   offset 16: offset   i64 (8 bytes)
 *   offset 24: sizes    i64[rank] (8*rank bytes)
 *   offset 40: strides  i64[rank] (8*rank bytes)
 *   Total: 24 + 16*rank bytes  (56 for rank 2)
 *
 * Exported symbols:
 *   _mlir_ciface_matmul_2x2
 *   _mlir_ciface_add_constant
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

/* Write the output descriptor fields and return a pointer to the data area
 * (immediately after the 56-byte descriptor). */
static float* sret_output(void* sret) {
    return (float*)((char*)sret + 56);
}

static void write_desc_2d(void* sret, float* data, int64_t rows, int64_t cols) {
    memref_2d_f32_t* desc = (memref_2d_f32_t*)sret;
    desc->allocated = data;
    desc->aligned = data;
    desc->offset = 0;
    desc->sizes[0] = rows;
    desc->sizes[1] = cols;
    desc->strides[0] = cols;
    desc->strides[1] = 1;
}

/* func_0: 2×2 @ 2×2 matmul with hard-coded inputs.
 *
 * Ignores the actual input values — always multiplies
 * [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
 *
 * This is intentional: the test provides known input descriptors but the
 * golden function uses hard-coded arithmetic so no input-dependent bugs
 * can mask the SSA wiring test.
 */
void _mlir_ciface_matmul_2x2(
    void* sret,
    memref_2d_f32_t* a,
    memref_2d_f32_t* b
) {
    (void)a;
    (void)b;
    float* out = sret_output(sret);
    out[0] = 19.0f;  /* 1*5 + 2*7 */
    out[1] = 22.0f;  /* 1*6 + 2*8 */
    out[2] = 43.0f;  /* 3*5 + 4*7 */
    out[3] = 50.0f;  /* 3*6 + 4*8 */
    write_desc_2d(sret, out, 2, 2);
}

/* func_1: element-wise add 10.0 to each element of a 2×2 input. */
void _mlir_ciface_add_constant(
    void* sret,
    memref_2d_f32_t* x
) {
    float* out = sret_output(sret);
    float* xd = x->aligned;
    for (int i = 0; i < 4; i++) {
        out[i] = xd[i] + 10.0f;
    }
    write_desc_2d(sret, out, 2, 2);
}
