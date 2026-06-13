/* sret_write — writes known MemRef descriptor + data to sret.
 *
 * Contract test fixture: C side writes pre-determined descriptor fields
 * and output data.  Rust side reads back and verifies every byte matches.
 *
 * Exports: _mlir_ciface_sret_write_f32  (1 arg: sret only)
 */

#include <stdint.h>
#include <stdlib.h>

typedef struct {
    float* allocated;
    float* aligned;
    int64_t offset;
    int64_t sizes[2];
    int64_t strides[2];
} memref_2d_f32_t;

void _mlir_ciface_sret_write_f32(void* sret) {
    float* data = (float*)malloc(12 * sizeof(float));
    for (int i = 0; i < 12; i++) data[i] = (float)(i + 1);

    memref_2d_f32_t* desc = (memref_2d_f32_t*)sret;
    desc->allocated = data;
    desc->aligned   = data;
    desc->offset    = 0;
    desc->sizes[0]  = 3;
    desc->sizes[1]  = 4;
    desc->strides[0] = 4;
    desc->strides[1] = 1;
}
