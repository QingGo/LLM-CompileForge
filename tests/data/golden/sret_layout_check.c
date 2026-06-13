/* sret_layout_check.c — compile-time assert for MemRef descriptor layout.
 *
 * Verifies that the C struct layout matches what the Rust runtime expects.
 * Compile with: cc -c sret_layout_check.c -o /dev/null
 * If it compiles (no static_assert failures), the layout is correct.
 */

#include <stdint.h>
#include <stddef.h>

/* Must match MemRefDesc<RANK> in runtime/src/hal/cpu/memref.rs */
typedef struct {
    float* allocated;
    float* aligned;
    int64_t offset;
    int64_t sizes[2];
    int64_t strides[2];
} memref_2d_f32_t;

/* Size check: descriptor = 24 + 16*rank bytes */
_Static_assert(sizeof(memref_2d_f32_t) == 56, "MemRefDesc2 must be 56 bytes");

/* Field offset checks: match MLIR memref struct layout */
_Static_assert(offsetof(memref_2d_f32_t, allocated) == 0,  "allocated at offset 0");
_Static_assert(offsetof(memref_2d_f32_t, aligned)   == 8,  "aligned at offset 8");
_Static_assert(offsetof(memref_2d_f32_t, offset)    == 16, "offset at offset 16");
_Static_assert(offsetof(memref_2d_f32_t, sizes)     == 24, "sizes at offset 24");
_Static_assert(offsetof(memref_2d_f32_t, strides)   == 40, "strides at offset 40");

/* Alignment: all fields are 8-byte aligned on 64-bit */
_Static_assert(_Alignof(memref_2d_f32_t) == 8, "MemRefDesc2 must be 8-byte aligned");

int main(void) { return 0; }
