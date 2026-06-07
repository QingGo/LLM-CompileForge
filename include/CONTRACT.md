# CONTRACT.md — SFA Interface Contract

<!--
  Auto-generated from code analysis of:
    include/sfa.h
    include/sfa_abi.proto
    compiler/sfa_abi.py         (Writer side — proto serialization)
    runtime/src/abi.rs          (Reader side — proto parsing → ComputeGraph)
    runtime/src/compute_graph_runner.rs  (Reader side — actual SfaMemRef usage)
    runtime/src/hal/sfa.rs      (SfaMemRef definition)
    runtime/src/compute_graph.rs (FuncDef, InputBinding, IOTensorDef)
    sf-dialect/lib/Sf/SfaContractPass.cpp  (C++ post-lowering verification)
-->

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Verified working: Writer populates, Reader honors |
| ⚠️ | Populated but Reader partially ignores or uses only in limited contexts |
| ❌ | Contract violation: Writer populates field not honored at all, or mismatch |

---

## sfa.h — Hot-path memory layout

Binary-compatible structs matching MLIR LLVM dialect memref descriptors.
Layout: `struct<(ptr, ptr, i64, array<RANK x i64>, array<RANK x i64>)>`.

### SFATensorRaw1 (40 bytes)

| Field | Writer | Reader | Status |
|-------|--------|--------|--------|
| `allocated` | sf-dialect (C++ dylib — inner function memref allocation) / runtime (Rust — `SfaMemRef` constructors, buffer allocation) | dylib (ciface wrappers read via ptr argument); runtime (`SfaMemRefRaw::data_ptr()`) | ✅ |
| `aligned` | Same as `allocated` (always set equal) | dylib (may use for aligned loads); runtime (`to_memref_desc_any()`) | ✅ |
| `offset` | Always 0 | dylib (memref base offset); runtime (copied in `to_memref_desc_any()`) | ✅ |
| `sizes[1]` | sf-dialect (from MLIR memref type) / runtime (from buffer shape) | dylib (reads dim sizes from descriptor); runtime (`sizes_i64()`) | ✅ |
| `strides[1]` | sf-dialect (row-major strides) / runtime (`from_shape()` computes strides) | dylib (reads for indexing); runtime (`strides_i64()`) | ✅ |

### SFATensorRaw2 (56 bytes)

Identical structure to R1, with `sizes[2]` and `strides[2]`.

| Field | Writer | Reader | Status |
|-------|--------|--------|--------|
| `allocated` | sf-dialect / runtime | dylib / runtime | ✅ |
| `aligned` | sf-dialect / runtime | dylib / runtime | ✅ |
| `offset` | Always 0 | dylib / runtime | ✅ |
| `sizes[2]` | sf-dialect / runtime | dylib / runtime | ✅ |
| `strides[2]` | sf-dialect / runtime | dylib / runtime | ✅ |

### SFATensorRaw3 (72 bytes)

Identical structure to R1, with `sizes[3]` and `strides[3]`.

| Field | Writer | Reader | Status |
|-------|--------|--------|--------|
| `allocated` | sf-dialect / runtime | dylib / runtime | ✅ |
| `aligned` | sf-dialect / runtime | dylib / runtime | ✅ |
| `offset` | Always 0 | dylib / runtime | ✅ |
| `sizes[3]` | sf-dialect / runtime | dylib / runtime | ✅ |
| `strides[3]` | sf-dialect / runtime | dylib / runtime | ✅ |

### SFATensorRaw4 (88 bytes)

Identical structure to R1, with `sizes[4]` and `strides[4]`.

| Field | Writer | Reader | Status |
|-------|--------|--------|--------|
| `allocated` | sf-dialect / runtime | dylib / runtime | ✅ |
| `aligned` | sf-dialect / runtime | dylib / runtime | ✅ |
| `offset` | Always 0 | dylib / runtime | ✅ |
| `sizes[4]` | sf-dialect / runtime | dylib / runtime | ✅ |
| `strides[4]` | sf-dialect / runtime | dylib / runtime | ✅ |

### SFATensor (unified wrapper)

| Field | Writer | Reader | Status |
|-------|--------|--------|--------|
| `device` | Runtime (`SFATensor::from_vec_f32()` — Rust only) | Runtime internal only (not passed across binary boundary) | ⚠️ Defined in sfa.h but never validated by any cross-component reader; runtime always assumes CPU |
| `rank` | Runtime (derived from shape) | Runtime internal (`SFATensor` accessors) | ✅ Internal use only |
| `elem_size` | Runtime (derived from Dtype) | Runtime internal (`as_buffer_ref()`) | ✅ Internal use only |
| `union { r1..r4 }` | Runtime (`SFATensor::from_vec_f32()`) | Runtime (`as_buffer_ref()`, `SFATensorRawAny` match) | ✅ Internal use only |

### SFADevice enum

| Value | Writer | Reader | Status |
|-------|--------|--------|--------|
| `SFA_CPU = 1` | Implicit (runtime assumes) | No cross-component reader validates | ⚠️ Defined but not enforced; runtime operates as CPU-only |
| `SFA_CUDA = 2` | Never populated | Never read | ⚠️ Reserved for future use; no implementation |

### Dylib Exported Symbols (convention, not struct fields)

| Symbol | Writer | Reader | Status |
|--------|--------|--------|--------|
| `sfa_abi` | compiler (`serialize_abi()` → embedded as `const uint8_t[]`) | runtime (`load_sfa_abi()` via `read_byte_slice_from_symbol()`) | ✅ |
| `sfa_abi_size` | compiler (computed from proto byte length) | runtime (`read_u64_symbol()`) | ✅ |
| `sfa_weights` | compiler (`sfa_weights.py` → embedded as `const uint8_t[]`) | runtime (`load_sfa_weights()` via `read_byte_slice_from_symbol()`) | ✅ |
| `sfa_weights_size` | compiler | runtime | ✅ |
| `_mlir_ciface_*` | sf-dialect (C++: MLIR bufferization + `llvm.emit_c_interface` → dylib export) | runtime (`executable.execute(symbol, ...)`) | ✅ |

---

## sfa_abi.proto — Cold-path metadata

### SfaAbiHeader

Top-level protobuf message embedded as `sfa_abi` symbol.

| Field | Writer | Reader | Status |
|-------|--------|--------|--------|
| `magic` | compiler (`sfa_abi.py:serialize_abi()` → `0x41464253`) | runtime (`abi.rs:load_sfa_abi()` → validates against `SFA_MAGIC`) | ✅ |
| `version` | compiler (`sfa_abi.py:SFA_VERSION = 1`) | runtime (`abi.rs` — validates against `SFA_VERSION`) | ✅ |
| `funcs` | compiler (`serialize_abi()` — one `SfaFuncMeta` per ciface function) | runtime (`build_compute_graph()` — iterates to build `Vec<FuncDef>`) | ✅ |

### SfaFuncMeta

Metadata for a single `_mlir_ciface_*` wrapper function.

| Field | Writer | Reader | Status |
|-------|--------|--------|--------|
| `symbol` | compiler (`merge_with_semantics()` — e.g. `"_mlir_ciface_main_0"`) | runtime (`abi.rs:build_compute_graph()` → `FuncDef.symbol`; used as op_name for `executable.execute()`) | ✅ |
| `num_inputs` | compiler (`parse_ciface_signatures()` — count of `ptr` args in LLVM IR, excluding sret) | runtime (`abi.rs:build_compute_graph()` → `FuncDef.num_inputs`; used as `Vec::with_capacity()` hint, then validated against `inputs.len()` via `anyhow::ensure!`) | ✅ Validated: mismatch triggers error at graph build time |
| `output_rank` | compiler (`parse_ciface_signatures()` — rank of sret return struct from LLVM IR) | runtime (`abi.rs:build_compute_graph()` → used **only as fallback** when `outputs` list is empty; ignored when `OutputDescriptor` entries are present) | ⚠️ Only used as fallback; when outputs list populated, output_rank is dead data |
| `input_fields` | compiler (`merge_with_semantics()` — one per function argument) | runtime (`build_compute_graph()` — iterates to build `(InputBinding, IOTensorDef)` pairs) | ✅ |
| `outputs` | compiler (`merge_with_semantics()` — from lowered MLIR output types via `parse_lowered_output_types()`) | runtime (`build_compute_graph()` — preferred source for output IOTensorDef; falls back to output_rank when empty) | ✅ |

### SfaInputField

Describes kind and binding of each function input argument.

| Field | Writer | Reader | Status |
|-------|--------|--------|--------|
| `kind` | compiler (`merge_with_semantics()` — `SFA_INPUT_GLOBAL` for first-function first-inputs; `SFA_INPUT_WEIGHT` for weight ops; `SFA_INPUT_SSA` for producer references) | runtime (`build_compute_graph()` — maps to `InputBinding` enum) | ✅ |
| `binding.weight_name` | compiler (`merge_with_semantics()` — from `weight_ops[].name`, filtered to exclude `_const_*` scalars) | runtime (`build_compute_graph()` — reads as `InputBinding::Weight(name)`; used by `load_weight_tensor()` to look up in WeightProvider) | ✅ |
| `binding.ssa.producer_func` | compiler (`merge_with_semantics()` — index into `SfaAbiHeader.funcs` based on producer_map or fallback heuristics) | runtime (`build_compute_graph()` — reads as `InputBinding::Ssa { producer_func, output_idx }`; used by `run_function_graph()` to look up `func_outputs[producer_func][output_idx]`) | ✅ |
| `binding.ssa.producer_out` | compiler (`merge_with_semantics()` — output index from producer_map or type-based matching) | runtime (same as producer_func above) | ✅ |
| `rank` | compiler (`merge_with_semantics()` — from `parse_lowered_argument_types()` analysis of lowered MLIR tensor types) | runtime (`build_compute_graph()` — for `Weight`/`SSA`/`GlobalInput` bindings: sets `IOTensorDef.rank` from proto, with fallback to 2 if proto rank is 0. In `run_function_graph()`, io_def.rank is **not** used for SfaMemRef construction — buffer's native rank is used instead (see G2).) | ⚠️ Issue: io_def.rank not used for SfaMemRef descriptor construction (buffer rank used instead per explicit design choice — see G2) |
| `dims` | compiler (`merge_with_semantics()` — from `parse_lowered_argument_types()`, dynamic dims as 0) | runtime (`build_compute_graph()` — for `Weight`/`SSA`/`GlobalInput`: sets `IOTensorDef.shape` from proto, with fallback to `[0,0]` if proto dims empty. In `run_function_graph()`, GlobalInput shape resolved independently via `fill_global_input()`. SSA/Weight dims used for io_def.shape but not for SfaMemRef construction.) | ⚠️ SSA/Weight dims used for io_def.shape but not for SfaMemRef construction |

### OutputDescriptor

Per-output tensor descriptor.

| Field | Writer | Reader | Status |
|-------|--------|--------|--------|
| `rank` | compiler (`parse_lowered_output_types()` — from lowered MLIR return tensor rank; fallback to `output_rank` from LLVM IR sret) | runtime (`build_compute_graph()` → `IOTensorDef.rank`; used by `allocate_output_buffers()` for buffer sizing and `extract_output_tensor()` for shape resolution) | ✅ |
| `dims` | compiler (`parse_lowered_output_types()` — from lowered MLIR return tensor dims; 0 = dynamic; fallback to zeros) | runtime (`build_compute_graph()` → `IOTensorDef.shape`; used by `allocate_output_buffers()` with dynamic→concrete fallback and `extract_output_tensor()` for sret shape parsing) | ✅ |

### SfaSsaRef

SSA dataflow edge: reference to a producer function's output.

| Field | Writer | Reader | Status |
|-------|--------|--------|--------|
| `producer_func` | compiler — index into `SfaAbiHeader.funcs` | runtime — used as array index into `func_outputs[]` | ✅ |
| `producer_out` | compiler — output index of producer function | runtime — used as array index into `func_outputs[producer_func][]` | ✅ |

### SfaWeightEntry

Maps compiled weight name → HuggingFace safetensors key.

| Field | Writer | Reader | Status |
|-------|--------|--------|--------|
| `compiled_name` | compiler (from `fx_to_mlir.py` weight ops, e.g. `"wte_weight"`) | runtime (`load_sfa_weights()` → `SfaWeightProvider.name_mapping` key; used by `get_weight_memref()`) | ✅ |
| `hf_key` | compiler (from `fx_to_mlir.py` HF key map, e.g. `"model.decoder.embed_tokens.weight"`) | runtime (`load_sfa_weights()` → `SfaWeightProvider.name_mapping` value; used to look up safetensors mmap offset) | ✅ |

### SfaConstant

Embedded constant tensor (compiler-synthesized, not from model weights).

| Field | Writer | Reader | Status |
|-------|--------|--------|--------|
| `name` | compiler (e.g. `"_const_causal_mask"`) | runtime (`load_sfa_weights()` → `ConstantTensor` key in `constants` map) | ✅ |
| `dtype_code` | compiler (`sfa_weights.py` — 0=f32, 1=f16, 2=bf16, 3=i64, 4=i32, 5=i8, 6=u8) | runtime (`Dtype::from_code(dtype_code)` in `load_sfa_weights()`) | ✅ |
| `shape` | compiler (tensor dimensions) | runtime (`ConstantTensor.shape` — used for memref descriptor construction) | ✅ |
| `data` | compiler (raw element bytes) | runtime (`ConstantTensor.data` — used as byte source for `constant_as_memref()`) | ✅ |

### SfaWeightData

Top-level weight data message.

| Field | Writer | Reader | Status |
|-------|--------|--------|--------|
| `weight_entries` | compiler (`sfa_weights.py` — name mapping entries) | runtime (`load_sfa_weights()` — builds `HashMap<String, String>` for name_mapping) | ✅ |
| `constant_entries` | compiler (`sfa_weights.py` — embedded constants) | runtime (`load_sfa_weights()` — builds `HashMap<String, ConstantTensor>` for constants) | ✅ |

---

## Known Contract Gaps & Violations

### G1: SfaInputField.rank ignored for GlobalInput ✅ RESOLVED

**Location**: `runtime/src/abi.rs:build_compute_graph()` lines 214-224
```rust
SfaInputKind::SfaInputGlobal => IOTensorDef {
    rank: if field.rank > 0 {
        field.rank as u8
    } else {
        log::warn!("GlobalInput rank=0 in proto, defaulting to 2");
        2
    },
    shape: if !field.dims.is_empty() {
        field.dims.clone()
    } else {
        log::warn!("GlobalInput dims empty in proto, defaulting to [0,0]");
        vec![0, 0]
    },
    consumed_internally: false,
},
```
**Resolution**: Runtime now reads `field.rank` and `field.dims` from proto. Falls back to hardcoded defaults (2, [0,0]) only when proto values are zero/empty, logging a warning.
**Severity**: Resolved — no longer a latent bug.

### G2: SfaInputField.rank not used for SfaMemRef ABI construction

**Location**: `runtime/src/compute_graph_runner.rs` lines 330-365
```rust
// IMPORTANT: use the buffer's native rank, NOT io_def.rank.
// The dylib's LLVM IR reads a fixed-size struct per argument —
// providing a smaller struct ... causes out-of-bounds reads → SIGSEGV.
// A larger struct ... works because the dylib only reads the first
// expected-size bytes.  The per-input rank/dims in io_def are
// informational metadata for the compiler and for future per-input
// buffer allocation.
```
**Impact**: The proto's `SfaInputField.rank` records what the compiler *expects* the dylib to read, but the runtime intentionally constructs a potentially different-sized SfaMemRef. If the buffer's native rank is *smaller* than what the dylib expects, it causes SIGSEGV. The runtime works around this by promoting rank-1 to rank-2.
**Severity**: Medium — the runtime's workaround is fragile; a future dylib expecting rank-4 with a runtime providing rank-2 buffer would crash.
**Status**: ⚠️ Partially mitigated — `debug_assert!` + `log::warn` added to detect/prevent rank mismatches at runtime; full fix requires dylib-level ABI changes.

### G3: ~~SfaFuncMeta.num_inputs not validated against input_fields.len()~~ ✅ RESOLVED

**Location**: `runtime/src/abi.rs:build_compute_graph()` — validated after input_fields iteration
```rust
anyhow::ensure!(
    inputs.len() == num_inputs,
    "num_inputs mismatch: proto field says {}, but found {} input_fields",
    num_inputs,
    inputs.len()
);
```
**Resolution**: Runtime now validates `inputs.len() == num_inputs` immediately after processing input_fields. Mismatch causes an error return instead of silent corruption.
**Severity**: Resolved — guardrail in place.

### G4: SfaAbiHeader.version not validated ✅ RESOLVED

**Location**: `runtime/src/abi.rs:load_sfa_abi()` — version field now validated against `SFA_VERSION`.
**Resolution**: `load_sfa_abi()` checks `abi.version == SFA_VERSION` (1) after magic validation. Rejects mismatched versions with a clear error message.
**Severity**: Low — current version is always 1; no version 2 exists.

### G5: SFADevice enum not enforced

**Location**: `include/sfa.h` defines `SFA_CPU=1`, `SFA_CUDA=2`. `runtime/src/hal/sfa.rs:SfaMemRef` has no device field. Runtime assumes CPU.
**Impact**: CUDA enum defined but no reader honors it; cross-device dispatch would require contract extension.
**Severity**: Low — CPU-only MVP; reserved for future.

---

## Subproject Contract Obligations

### sf-dialect (C++)

1. **MUST** produce dylib ciface wrappers via `llvm.emit_c_interface` where each input argument is a pointer to a binary-compatible memref descriptor matching `SFATensorRaw{N}` layout (`{ptr, ptr, i64, [N]i64, [N]i64}`).
2. **MUST** ensure all function arguments are lowered to `memref<...>` types with rank 1-4 before bufferization (verified by `SfaContractVerifyPass`).
3. **MUST** promote all `sf.weight` and `sf.constant` ops to function arguments before bufferization (verified by `SfaContractVerifyPass` — residual ops are errors).
4. **MUST NOT** emit memref descriptors with rank outside [1, 4] — `SFATensorRaw` supports R1-R4 only.
5. **MUST** produce sret output descriptors (via `_mlir_ciface_*` wrappers) where the sret struct's rank matches the post-bufferization packed output memref.
6. **MUST** record input semantics (`sf.func_metadata` module attribute) with `input_kinds` distinguishing "global" vs "weight" args (used by compiler for proto generation).

### compiler (Python)

1. **MUST** faithfully record per-input rank/dims from lowered MLIR into `SfaInputField.rank` and `SfaInputField.dims` (via `parse_lowered_argument_types()`).
2. **MUST** faithfully record per-output rank/dims from lowered MLIR into `OutputDescriptor.rank` and `OutputDescriptor.dims` (via `parse_lowered_output_types()`).
3. **MUST** compute correct SSA wiring: `SfaSsaRef.producer_func` (index into `SfaAbiHeader.funcs`) and `SfaSsaRef.producer_out` (output index) from the dataflow graph (via `merge_with_semantics()` producer_map and fallback heuristics).
4. **MUST** correctly classify each input binding: `SFA_INPUT_GLOBAL` for model inputs (first function, first inputs), `SFA_INPUT_WEIGHT` for weight ops, `SFA_INPUT_SSA` for intermediate dataflow edges.
5. **MUST** embed `SfaAbiHeader` as protobuf binary in the dylib via `sfa_abi` / `sfa_abi_size` symbols.
6. **MUST** embed `SfaWeightData` as protobuf binary in the dylib via `sfa_weights` / `sfa_weights_size` symbols.
7. **MUST** set `SfaAbiHeader.magic = 0x41464253` and `SfaAbiHeader.version = 1`.

### runtime (Rust)

1. **MUST** construct `SfaMemRef` descriptors that are binary-compatible with `SFATensorRaw{N}` layout for all ciface calls.
2. **MUST** allocate one output buffer per `OutputDescriptor` (via `allocate_output_buffers()`).
3. **MUST** parse exactly `OutputDescriptor.count` descriptors from the sret output.
4. **MUST** use SSA binding (`SfaSsaRef.producer_func`, `SfaSsaRef.producer_out`) to wire data between functions (via `func_outputs[][]` lookup in `run_function_graph()`).
5. **MUST** validate `SfaAbiHeader.magic` before processing (currently implemented).
6. **MUST** validate `SfaAbiHeader.version` against expected version (currently implemented).
7. **MUST** validate that `SfaFuncMeta.num_inputs == input_fields.len()` (implemented — see Gap G3 ✅).
8. **SHOULD** use `SfaInputField.rank` when constructing `SfaMemRef` descriptors (currently ignored; runtime uses buffer's native rank — see Gap G2).
9. **MUST** honor `SfaInputField.rank` and `SfaInputField.dims` for `GlobalInput` bindings instead of hardcoding rank=2, shape=[0,0] (see Gap G1 — ✅ RESOLVED).

---

## Dataflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ COMPILER (Python)                                               │
│                                                                 │
│  lowered MLIR ──► parse_lowered_argument_types()                │
│  lowered MLIR ──► parse_lowered_output_types()                  │
│  LLVM IR     ──► parse_ciface_signatures()                      │
│  pre-lowering ──► merge_with_semantics()                        │
│       │                                                         │
│       ▼                                                         │
│  SfaFuncMeta[] ──► serialize_abi() ──► sfa_abi (protobuf)      │
│                                                                 │
│  weight ops   ──► sfa_weights.py ──► sfa_weights (protobuf)    │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼  (embedded in .dylib as symbols)
┌─────────────────────────────────────────────────────────────────┐
│ RUNTIME (Rust)                                                  │
│                                                                 │
│  sfa_abi ──► load_sfa_abi() ──► SfaAbiHeader (proto decode)    │
│  sfa_weights ──► load_sfa_weights() ──► SfaWeightProvider       │
│       │                                                         │
│       ▼                                                         │
│  build_compute_graph() ──► ComputeGraph { FuncDef[] }           │
│       │                                                         │
│       ▼                                                         │
│  run_function_graph()                                           │
│    ├── GlobalInput: fill_global_input() → SFATensor → SfaMemRef │
│    ├── Weight:      load_weight_tensor() → wrap → SfaMemRef     │
│    ├── SSA:         func_outputs[][] → wrap → SfaMemRef         │
│    └── dispatch:    executable.execute(symbol, &inputs, &outputs)│
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ DYLIB (C++ via sf-dialect → MLIR → LLVM → .dylib)               │
│                                                                 │
│  _mlir_ciface_<name>(ptr %sret, ptr %arg0, ptr %arg1, ...)     │
│    └── reads SFATensorRaw{N} structs from arg pointers          │
│    └── writes results to sret memref                            │
└─────────────────────────────────────────────────────────────────┘
```
