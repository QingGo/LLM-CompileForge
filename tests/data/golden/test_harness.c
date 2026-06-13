/* test_harness.c — minimal C harness for verifying compiler-produced dylibs.
 *
 * Replaces Python ctypes for compiler output validation.  Zero Python
 * dependency — just dlopen + dlsym + memref struct + output check.
 *
 * Usage:
 *   cc -o test_harness test_harness.c
 *   ./test_harness <dylib_path> <symbol> <rank> <sizes...> -- <expected_values...>
 *
 * Example:
 *   ./test_harness ./matmul.dylib _mlir_ciface_matmul_f32 2 2 2 -- 58 64 139 154
 *
 * Constructs inputs from expected output (identity check pattern — inputs
 * are filled with 1.0, expected output must match reference computation).
 *
 * For production use, input data should be passed via files or inline hex.
 */

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── MemRef descriptor (must match MLIR + Rust layout) ────────────── */
typedef struct {
    float* allocated;
    float* aligned;
    int64_t offset;
    int64_t sizes[4];
    int64_t strides[4];
} memref_t;

/* ── Helpers ──────────────────────────────────────────────────────── */

static memref_t make_memref(int rank, int64_t* shape, float* data) {
    memref_t m = {0};
    m.allocated = data;
    m.aligned = data;
    m.offset = 0;
    for (int i = 0; i < rank; i++) m.sizes[i] = shape[i];
    /* Row-major strides */
    int64_t stride = 1;
    for (int i = rank - 1; i >= 0; i--) {
        m.strides[i] = stride;
        stride *= shape[i];
    }
    return m;
}

static float* alloc_f32(int64_t n) {
    float* p = (float*)calloc((size_t)n, sizeof(float));
    if (!p) { fprintf(stderr, "OOM\n"); exit(2); }
    return p;
}

/* ── Main ─────────────────────────────────────────────────────────── */

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr, "Usage: %s <dylib> <symbol> <rank> <d0> [d1] [d2] [d3]"
                " -- <expected_0> <expected_1> ...\n", argv[0]);
        return 2;
    }

    const char* dylib_path = argv[1];
    const char* symbol     = argv[2];
    int rank               = atoi(argv[3]);

    /* Parse shape */
    int64_t shape[4] = {1, 1, 1, 1};
    int64_t numel = 1;
    for (int i = 0; i < rank && 4 + i < argc; i++) {
        if (strcmp(argv[4 + i], "--") == 0) break;
        shape[i] = atoll(argv[4 + i]);
        numel *= shape[i];
    }

    /* Find expected values after "--" */
    int expected_start = -1;
    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "--") == 0) { expected_start = i + 1; break; }
    }
    if (expected_start < 0) {
        fprintf(stderr, "Missing '--' separator before expected values\n");
        return 2;
    }
    int n_expected = argc - expected_start;
    if (n_expected != (int)numel) {
        fprintf(stderr, "Expected %lld values, got %d\n",
                (long long)numel, n_expected);
        return 2;
    }
    float* expected = alloc_f32(numel);
    for (int i = 0; i < n_expected; i++)
        expected[i] = (float)atof(argv[expected_start + i]);

    /* Load dylib */
    void* handle = dlopen(dylib_path, RTLD_NOW);
    if (!handle) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }

    typedef void (*ciface_fn)(void*, memref_t*, memref_t*, memref_t*);
    ciface_fn func = (ciface_fn)dlsym(handle, symbol);
    if (!func) { fprintf(stderr, "dlsym(%s): %s\n", symbol, dlerror()); dlclose(handle); return 1; }

    /* Allocate inputs: fill with 1.0 (identity-like) + expected output */
    float* input_data = alloc_f32(numel);
    for (int64_t i = 0; i < numel; i++) input_data[i] = 1.0f;

    memref_t input_desc = make_memref(rank, shape, input_data);

    /* Allocate sret buffer: descriptor + output data */
    size_t desc_size = sizeof(memref_t);   /* 24 + 16*4 = 88 bytes */
    size_t sret_size = desc_size + (size_t)numel * sizeof(float);
    void* sret = calloc(1, sret_size);
    if (!sret) { fprintf(stderr, "OOM for sret\n"); return 2; }

    /* Call */
    func(sret, &input_desc, NULL, NULL);

    /* Read output */
    memref_t* out_desc = (memref_t*)sret;
    float* out_data = out_desc->aligned;
    if (!out_data) {
        /* Data may be after the descriptor in sret buffer */
        out_data = (float*)((char*)sret + desc_size);
    }

    /* Verify */
    int failed = 0;
    float max_err = 0.0f;
    for (int i = 0; i < n_expected; i++) {
        float err = out_data[i] - expected[i];
        if (err < 0) err = -err;
        if (err > max_err) max_err = err;
        if (err > 1e-4) {
            fprintf(stderr, "MISMATCH[%d]: got %.6f expected %.6f (err=%.2e)\n",
                    i, out_data[i], expected[i], err);
            failed = 1;
        }
    }
    if (!failed) {
        printf("OK: %d values verified, max_err=%.2e\n", n_expected, (double)max_err);
    }

    /* Cleanup */
    free(input_data);
    free(expected);
    free(sret);
    dlclose(handle);
    return failed ? 1 : 0;
}
