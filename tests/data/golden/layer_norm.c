/* test_layer_norm: y = (x - mean) / sqrt(var + eps) * w + b
 *
 * x: shape [batch, seq, hidden] — layernorm over last dim (axis=2)
 * w: shape [hidden]
 * b: shape [hidden]
 *
 * Exports: _mlir_ciface_layer_norm_f32
 *   arg0: sret  → output descriptor (rank 3)
 *   arg1: x     → input descriptor (rank 3)
 *   arg2: w     → weight descriptor (rank 1)
 *   arg3: b     → bias descriptor (rank 1)
 *
 * eps is hardcoded to 1e-5 (matching the sf.layer_norm default).
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    float* allocated;
    float* aligned;
    int64_t offset;
    int64_t sizes[3];
    int64_t strides[3];
} memref_3d_f32_t;

typedef struct {
    float* allocated;
    float* aligned;
    int64_t offset;
    int64_t sizes[1];
    int64_t strides[1];
} memref_1d_f32_t;

static void* sret_alloc(void* sret, int64_t n_bytes) {
    /* Place output data immediately after the 72-byte rank-3 descriptor. */
    (void)n_bytes;
    return (char*)sret + 72;
}

void _mlir_ciface_layer_norm_f32(
    void* sret,
    memref_3d_f32_t* x,
    memref_1d_f32_t* w,
    memref_1d_f32_t* b
) {
    int64_t batch  = x->sizes[0];
    int64_t seq    = x->sizes[1];
    int64_t hidden = x->sizes[2];
    float eps = 1e-5f;

    float* out_data = (float*)sret_alloc(sret, batch * seq * hidden);
    float* x_data = x->aligned;
    float* w_data = w->aligned;
    float* b_data = b->aligned;

    int64_t x_s0 = x->strides[0];
    int64_t x_s1 = x->strides[1];
    int64_t x_s2 = x->strides[2];

    for (int64_t bs = 0; bs < batch; bs++) {
        for (int64_t s = 0; s < seq; s++) {
            /* Compute mean over hidden dim */
            double sum = 0.0;
            for (int64_t h = 0; h < hidden; h++) {
                sum += (double)x_data[bs * x_s0 + s * x_s1 + h * x_s2];
            }
            float mean = (float)(sum / hidden);

            /* Compute variance */
            double var_sum = 0.0;
            for (int64_t h = 0; h < hidden; h++) {
                float diff = x_data[bs * x_s0 + s * x_s1 + h * x_s2] - mean;
                var_sum += (double)(diff * diff);
            }
            float var = (float)(var_sum / hidden);

            /* Normalize + scale + shift */
            float inv_std = 1.0f / sqrtf(var + eps);
            for (int64_t h = 0; h < hidden; h++) {
                float val = x_data[bs * x_s0 + s * x_s1 + h * x_s2];
                float norm = (val - mean) * inv_std;
                int64_t out_idx = (bs * seq + s) * hidden + h;
                out_data[out_idx] = norm * w_data[h * w->strides[0]] + b_data[h * b->strides[0]];
            }
        }
    }

    memref_3d_f32_t* out_desc = (memref_3d_f32_t*)sret;
    out_desc->allocated = out_data;
    out_desc->aligned = out_data;
    out_desc->offset = 0;
    out_desc->sizes[0] = batch;
    out_desc->sizes[1] = seq;
    out_desc->sizes[2] = hidden;
    out_desc->strides[0] = seq * hidden;
    out_desc->strides[1] = hidden;
    out_desc->strides[2] = 1;
}
