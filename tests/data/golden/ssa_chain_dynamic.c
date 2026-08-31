/* ssa_chain_dynamic — dynamic dimension SSA chain
 *
 * func_0: _mlir_ciface_dynamic_matmul (3 args: sret, A, B)
 *   Reads input dimensions from A and B descriptors and produces
 *   a [2,2] output (matmul result). The golden function ignores
 *   actual input values and produces hard-coded output — BUT it
 *   reads the sizes from the input descriptors to verify they
 *   were passed correctly.
 *
 *   This tests that dynamic dimension descriptors (sizes=0
 *   initially, resolved to actual dims) work in the SSA chain.
 *
 *   Hard-coded output: [[4,4]]?  No — let's use identity-like:
 *   Always produces [[10, 20], [30, 40]] for any [2,2] input.
 *   Verifies the shapes from both inputs are [2,2].
 *
 * func_1: _mlir_ciface_dynamic_scale (2 args: sret, X)
 *   Multiplies each element by 2.5f (scalar operation).
 *   → [[25, 50], [75, 100]]
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
 *   _mlir_ciface_dynamic_matmul
 *   _mlir_ciface_dynamic_scale
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

/* func_0: matmul that validates input dimensions and produces
 * hard-coded output. The function checks that both A and B have
 * shape [2,2] (rows=2, cols=2). This is the key difference from
 * other golden functions: it reads descriptor sizes to verify
 * dynamic dims were resolved by the caller.
 *
 * Ignores actual input data values.
 */
void _mlir_ciface_dynamic_matmul(
    void* sret,
    memref_2d_f32_t* a,
    memref_2d_f32_t* b
) {
    /* Validate input dimensions were passed correctly.
     * For dynamic dims (0 in the descriptor), proper callers
     * should have resolved them to actual values before calling.
     * We verify the resolved dims are [2,2] for both inputs. */
    if (a->sizes[0] != 2 || a->sizes[1] != 2) {
        /* Invalid shape — write zeroed output */
        float* out = sret_output(sret);
        for (int i = 0; i < 4; i++) out[i] = -1.0f;
        write_desc_2d(sret, out, 2, 2);
        return;
    }
    if (b->sizes[0] != 2 || b->sizes[1] != 2) {
        float* out = sret_output(sret);
        for (int i = 0; i < 4; i++) out[i] = -1.0f;
        write_desc_2d(sret, out, 2, 2);
        return;
    }

    (void)a;
    (void)b;
    float* out = sret_output(sret);
    out[0] = 10.0f;
    out[1] = 20.0f;
    out[2] = 30.0f;
    out[3] = 40.0f;
    write_desc_2d(sret, out, 2, 2);
}

/* func_1: element-wise multiply by 2.5f on 2×2 input.
 *
 * Reads actual input data from func_0's output descriptor.
 */
void _mlir_ciface_dynamic_scale(
    void* sret,
    memref_2d_f32_t* x
) {
    float* out = sret_output(sret);
    float* xd = x->aligned;
    for (int i = 0; i < 4; i++) {
        out[i] = xd[i] * 2.5f;
    }
    write_desc_2d(sret, out, 2, 2);
}
