#include <stdint.h>
#include <string.h>

typedef struct {
    float* allocated;
    float* aligned;
    int64_t offset;
    int64_t sizes[1];
    int64_t strides[1];
} memref_1d_f32_t;

typedef struct {
    int64_t* allocated;
    int64_t* aligned;
    int64_t offset;
    int64_t sizes[2];
    int64_t strides[2];
} memref_2d_i64_t;

void _mlir_ciface_mul_weight_input(
    void* sret,
    memref_1d_f32_t* weight,
    memref_2d_i64_t* input
) {
    (void)input;
    int64_t n = weight->sizes[0];

    float* out_data = (float*)((char*)sret + 1024);
    for (int64_t i = 0; i < n; i++) {
        out_data[i] = weight->aligned[weight->strides[0] * i];
    }

    memref_1d_f32_t* out_desc = (memref_1d_f32_t*)sret;
    out_desc->allocated = out_data;
    out_desc->aligned = out_data;
    out_desc->offset = 0;
    out_desc->sizes[0] = n;
    out_desc->strides[0] = 1;
}
