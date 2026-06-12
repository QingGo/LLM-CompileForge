/* test_layer_norm_multi: multi-output layer_norm.
 *
 * Produces 2 outputs from a single ciface call:
 *   out1 = layer_norm(x, w, b)
 *   out2 = layer_norm(x + 1.0, w, b)
 *
 * x: shape [batch, seq, hidden] — layernorm over last dim (axis=2)
 * w: shape [hidden]  (gamma)
 * b: shape [hidden]  (beta)
 *
 * Exports: _mlir_ciface_layer_norm_f32
 *   arg0: sret  → packed output descriptors (rank 3 × 2)
 *   arg1: x     → input descriptor (rank 3)
 *   arg2: w     → weight descriptor (rank 1)
 *   arg3: b     → bias descriptor (rank 1)
 *
 * sret layout:
 *   [0..72):    output 1 descriptor (rank-3: 24 + 48 = 72 bytes)
 *   [72..144):  output 2 descriptor
 *   [144..):    output 1 data (batch*seq*hidden*4 bytes)
 *   ...         output 2 data
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

static void layer_norm_kernel(
    float* out,
    float* x_data,
    int64_t x_s0, int64_t x_s1, int64_t x_s2,
    int64_t batch, int64_t seq_len, int64_t hidden,
    float* w_data, int64_t w_s0,
    float* b_data, int64_t b_s0,
    float eps,
    int add_one
) {
    for (int64_t bs = 0; bs < batch; bs++) {
        for (int64_t s = 0; s < seq_len; s++) {
            /* Compute mean over hidden dim */
            double sum = 0.0;
            for (int64_t h = 0; h < hidden; h++) {
                float val = x_data[bs * x_s0 + s * x_s1 + h * x_s2];
                if (add_one) val += 1.0f;
                sum += (double)val;
            }
            float mean = (float)(sum / hidden);

            /* Compute variance */
            double var_sum = 0.0;
            for (int64_t h = 0; h < hidden; h++) {
                float val = x_data[bs * x_s0 + s * x_s1 + h * x_s2];
                if (add_one) val += 1.0f;
                float diff = val - mean;
                var_sum += (double)(diff * diff);
            }
            float var = (float)(var_sum / hidden);

            /* Normalize + scale + shift */
            float inv_std = 1.0f / sqrtf(var + eps);
            for (int64_t h = 0; h < hidden; h++) {
                float val = x_data[bs * x_s0 + s * x_s1 + h * x_s2];
                if (add_one) val += 1.0f;
                float norm = (val - mean) * inv_std;
                int64_t out_idx = (bs * seq_len + s) * hidden + h;
                out[out_idx] = norm * w_data[h * w_s0] + b_data[h * b_s0];
            }
        }
    }
}

static void write_desc(void* desc_ptr, float* data, int64_t d0, int64_t d1, int64_t d2) {
    memref_3d_f32_t* desc = (memref_3d_f32_t*)desc_ptr;
    desc->allocated = data;
    desc->aligned = data;
    desc->offset = 0;
    desc->sizes[0] = d0;
    desc->sizes[1] = d1;
    desc->sizes[2] = d2;
    desc->strides[0] = d1 * d2;
    desc->strides[1] = d2;
    desc->strides[2] = 1;
}

void _mlir_ciface_layer_norm_f32(
    void* sret,
    memref_3d_f32_t* x,
    memref_1d_f32_t* w,
    memref_1d_f32_t* b
) {
    int64_t batch   = x->sizes[0];
    int64_t seq_len = x->sizes[1];
    int64_t hidden  = x->sizes[2];
    float eps = 1e-5f;

    float* x_data = x->aligned;
    float* w_data = w->aligned;
    float* b_data = b->aligned;
    int64_t x_s0 = x->strides[0];
    int64_t x_s1 = x->strides[1];
    int64_t x_s2 = x->strides[2];
    int64_t w_s0 = w->strides[0];
    int64_t b_s0 = b->strides[0];

    int64_t data_bytes = batch * seq_len * hidden * (int64_t)sizeof(float);

    /* sret layout: desc1(72B) + desc2(72B) + data1 + data2 */
    float* out1 = (float*)((char*)sret + 144);
    float* out2 = (float*)((char*)sret + 144 + data_bytes);

    layer_norm_kernel(out1, x_data, x_s0, x_s1, x_s2, batch, seq_len, hidden,
                      w_data, w_s0, b_data, b_s0, eps, 0);
    layer_norm_kernel(out2, x_data, x_s0, x_s1, x_s2, batch, seq_len, hidden,
                      w_data, w_s0, b_data, b_s0, eps, 1);

    write_desc(sret, out1, batch, seq_len, hidden);
    write_desc((char*)sret + 72, out2, batch, seq_len, hidden);
}
