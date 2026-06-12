/* test_rms_norm: y = x / sqrt(mean(x^2) + eps) * w
 *
 * x: shape [batch, hidden] — rmsnorm over last dim (axis=1)
 * w: shape [hidden]
 *
 * Exports: _mlir_ciface_rms_norm_f32
 *   arg0: sret  → output descriptor (rank 2)
 *   arg1: x     → input descriptor (rank 2)
 *   arg2: w     → weight descriptor (rank 1)
 *
 * eps = 1e-5
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

typedef struct {
    float* allocated;
    float* aligned;
    int64_t offset;
    int64_t sizes[1];
    int64_t strides[1];
} memref_1d_f32_t;

void _mlir_ciface_rms_norm_f32(
    void* sret,
    memref_2d_f32_t* x,
    memref_1d_f32_t* w
) {
    int64_t batch  = x->sizes[0];
    int64_t hidden = x->sizes[1];
    float eps = 1e-5f;

    float* out_data = (float*)((char*)sret + 56); /* descriptor for rank-2 = 56 bytes */
    float* x_data = x->aligned;
    float* w_data = w->aligned;

    int64_t x_s0 = x->strides[0];
    int64_t x_s1 = x->strides[1];
    int64_t w_s0 = w->strides[0];

    for (int64_t b = 0; b < batch; b++) {
        /* Compute mean of squares over hidden dim */
        double sum_sq = 0.0;
        for (int64_t h = 0; h < hidden; h++) {
            float val = x_data[b * x_s0 + h * x_s1];
            sum_sq += (double)(val * val);
        }
        float rms = sqrtf((float)(sum_sq / hidden) + eps);

        for (int64_t h = 0; h < hidden; h++) {
            float val = x_data[b * x_s0 + h * x_s1];
            out_data[b * hidden + h] = (val / rms) * w_data[h * w_s0];
        }
    }

    memref_2d_f32_t* out_desc = (memref_2d_f32_t*)sret;
    out_desc->allocated = out_data;
    out_desc->aligned = out_data;
    out_desc->offset = 0;
    out_desc->sizes[0] = batch;
    out_desc->sizes[1] = hidden;
    out_desc->strides[0] = hidden;
    out_desc->strides[1] = 1;
}
