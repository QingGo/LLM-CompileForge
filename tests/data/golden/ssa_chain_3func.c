/* ssa_chain_3func — three-function SSA chain: matmul → add → relu
 *
 * func_0: _mlir_ciface_chain3_matmul (3 args: sret, A, B)
 *   Hard-coded [2,3] @ [3,2] matmul:
 *     A=[[1,0,0],[0,1,0]] × B=[[1,2],[3,4],[5,6]] → [[1,2],[3,4]]
 *
 * func_1: _mlir_ciface_chain3_add (3 args: sret, X, bias)
 *   Element-wise adds bias to func_0 output.
 *   bias=[[10,-10],[10,-10]]
 *   → [[11,-8],[13,-6]]
 *
 * func_2: _mlir_ciface_chain3_relu (2 args: sret, X)
 *   Element-wise max(0, x) on func_1 output.
 *   → [[11,0],[13,0]]
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
 *   _mlir_ciface_chain3_matmul
 *   _mlir_ciface_chain3_add
 *   _mlir_ciface_chain3_relu
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

/* func_0: [2,3] @ [3,2] matmul with hard-coded inputs.
 *
 * Ignores the actual input values — always multiplies
 * [[1,0,0],[0,1,0]] @ [[1,2],[3,4],[5,6]] = [[1,2],[3,4]]
 *
 * This is intentional: the test provides known input descriptors but the
 * golden function uses hard-coded arithmetic so no input-dependent bugs
 * can mask the SSA wiring test.
 */
void _mlir_ciface_chain3_matmul(
    void* sret,
    memref_2d_f32_t* a,
    memref_2d_f32_t* b
) {
    (void)a;
    (void)b;
    float* out = sret_output(sret);
    /* row 0: 1*1+0*3+0*5=1, 1*2+0*4+0*6=2 */
    out[0] = 1.0f;
    out[1] = 2.0f;
    /* row 1: 0*1+1*3+0*5=3, 0*2+1*4+0*6=4 */
    out[2] = 3.0f;
    out[3] = 4.0f;
    write_desc_2d(sret, out, 2, 2);
}

/* func_1: element-wise add bias=[[10,-10],[10,-10]] to 2×2 input. */
void _mlir_ciface_chain3_add(
    void* sret,
    memref_2d_f32_t* x,
    memref_2d_f32_t* bias
) {
    (void)bias;
    float* out = sret_output(sret);
    float* xd = x->aligned;
    /* add: [1, 2, 3, 4] + [10, -10, 10, -10] = [11, -8, 13, -6] */
    float bias_vals[4] = {10.0f, -10.0f, 10.0f, -10.0f};
    for (int i = 0; i < 4; i++) {
        out[i] = xd[i] + bias_vals[i];
    }
    write_desc_2d(sret, out, 2, 2);
}

/* func_2: element-wise ReLU max(0, x) on 2×2 input. */
void _mlir_ciface_chain3_relu(
    void* sret,
    memref_2d_f32_t* x
) {
    float* out = sret_output(sret);
    float* xd = x->aligned;
    for (int i = 0; i < 4; i++) {
        out[i] = xd[i] > 0.0f ? xd[i] : 0.0f;
    }
    write_desc_2d(sret, out, 2, 2);
}
