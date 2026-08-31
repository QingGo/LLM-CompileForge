/* ssa_chain_fork — branch chain with multi-consumer SSA
 *
 * func_0: _mlir_ciface_fork_matmul (3 args: sret, A, B)
 *   Hard-coded 2×2 @ 2×2 matmul:
 *     A=[[1,2],[3,4]] × B=[[0.1,0.2],[0.3,0.4]]
 *     → [[0.7, 1.0], [1.5, 2.2]]
 *
 * func_1: _mlir_ciface_fork_relu (2 args: sret, X)
 *   Element-wise max(0, x) on func_0 output.
 *   → [[0.7, 1.0], [1.5, 2.2]]  (all positive)
 *
 * func_2: _mlir_ciface_fork_sigmoid (2 args: sret, X)
 *   Element-wise 1/(1+exp(-x)) on func_0 output.
 *   sigmoid(0.7)≈0.668188, sigmoid(1.0)≈0.731059,
 *   sigmoid(1.5)≈0.817574, sigmoid(2.2)≈0.900250
 *
 * Key property: func_0's sret is consumed by BOTH func_1 AND func_2
 * (fork/join SSA pattern). The Rust test passes the same descriptor
 * to both downstream functions.
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
 *   _mlir_ciface_fork_matmul
 *   _mlir_ciface_fork_relu
 *   _mlir_ciface_fork_sigmoid
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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
 * Ignores actual input values — always multiplies
 * [[1,2],[3,4]] @ [[0.1,0.2],[0.3,0.4]] = [[0.7,1.0],[1.5,2.2]]
 */
void _mlir_ciface_fork_matmul(
    void* sret,
    memref_2d_f32_t* a,
    memref_2d_f32_t* b
) {
    (void)a;
    (void)b;
    float* out = sret_output(sret);
    /* [0,0]: 1*0.1 + 2*0.3 = 0.1 + 0.6 = 0.7 */
    out[0] = 0.7f;
    /* [0,1]: 1*0.2 + 2*0.4 = 0.2 + 0.8 = 1.0 */
    out[1] = 1.0f;
    /* [1,0]: 3*0.1 + 4*0.3 = 0.3 + 1.2 = 1.5 */
    out[2] = 1.5f;
    /* [1,1]: 3*0.2 + 4*0.4 = 0.6 + 1.6 = 2.2 */
    out[3] = 2.2f;
    write_desc_2d(sret, out, 2, 2);
}

/* func_1: element-wise ReLU max(0, x) on 2×2 input. */
void _mlir_ciface_fork_relu(
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

/* func_2: element-wise sigmoid 1/(1+exp(-x)) on 2×2 input. */
void _mlir_ciface_fork_sigmoid(
    void* sret,
    memref_2d_f32_t* x
) {
    float* out = sret_output(sret);
    float* xd = x->aligned;
    for (int i = 0; i < 4; i++) {
        out[i] = 1.0f / (1.0f + expf(-xd[i]));
    }
    write_desc_2d(sret, out, 2, 2);
}
