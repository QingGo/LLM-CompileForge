#ifndef HAL_OPS
#define HAL_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// HAL Dialect
//===----------------------------------------------------------------------===//

def HAL_Dialect : Dialect {
  let name = "hal";
  let cppNamespace = "::mlir::hal";
  let summary = "Hardware Abstraction Layer ops — lowering target for sf dialect";
  let description = [{
    The `hal` dialect defines the primitive operations for the CPU inference
    runtime. Ops operate on MemRef types (SfaMemRef) and map 1:1 to HAL kernel
    implementations. Lowering from `sf` to `hal` decomposes high-level tensor
    ops into explicit memory-buffer operations.

    Each hal op is the unit of dispatch in the HAL executor.
  }];
}

class HAL_Op<string mnemonic, list<Trait> traits = []> :
    Op<HAL_Dialect, mnemonic, traits>;

// SFA MemRef type constraints — float/int memref for HAL compute ops
// SfaMemRef mirrors the SFATensorRaw1-4 ABI layout from include/sfa.h
def SfaMemRef : MemRefOf<[AnyFloat]>;
def SfaIntMemRef : MemRefOf<[AnyInteger]>;
def SfaAnyMemRef : AnyTypeOf<[SfaMemRef, SfaIntMemRef]>;

//===----------------------------------------------------------------------===//
// Compute-Intensive Ops
//===----------------------------------------------------------------------===//

def HAL_MatmulOp : HAL_Op<"matmul", [Pure]> {
  let summary = "matrix multiplication: C = A × B";
  let description = [{
    Dense matrix multiplication. CPU path dispatches to BLAS FFI
    (Accelerate/OpenBLAS). GPU path uses precompiled CUTLASS cubin.
  }];
  let arguments = (ins SfaMemRef:$lhs, SfaMemRef:$rhs);
  let results = (outs SfaMemRef:$result);
}

def HAL_BatchMatmulOp : HAL_Op<"batch_matmul", [Pure]> {
  let summary = "batched matrix multiplication over batch dimension";
  let description = [{
    Batched GEMM. CPU path dispatches to BLAS FFI with batch loop.
    GPU path uses CUTLASS grouped GEMM cubin.
  }];
  let arguments = (ins SfaMemRef:$lhs, SfaMemRef:$rhs);
  let results = (outs SfaMemRef:$result);
}

//===----------------------------------------------------------------------===//
// Normalization Ops
//===----------------------------------------------------------------------===//

def HAL_RmsNormOp : HAL_Op<"rms_norm", [Pure]> {
  let summary = "RMS normalization";
  let description = [{
    Root Mean Square normalization. Memory-bandwidth bound — always
    a candidate for fusion with the downstream matmul or cache_write.
  }];
  let arguments = (ins SfaMemRef:$input, SfaMemRef:$weight);
  let results = (outs SfaMemRef:$result);
}

def HAL_LayerNormOp : HAL_Op<"layer_norm", [Pure]> {
  let summary = "layer normalization";
  let description = [{
    Layer normalization with learnable weight and bias. Memory-bandwidth
    bound — candidate for fusion with downstream ops.
  }];
  let arguments = (ins SfaMemRef:$input, SfaMemRef:$weight,
                       SfaMemRef:$bias);
  let results = (outs SfaMemRef:$result);
}

//===----------------------------------------------------------------------===//
// Softmax
//===----------------------------------------------------------------------===//

def HAL_SoftmaxOp : HAL_Op<"softmax", [Pure]> {
  let summary = "softmax (stable, online)";
  let description = [{
    Numerically stable softmax. In attention contexts this is always
    fused into the attention op; standalone softmax is used for
    non-attention classification heads.
  }];
  let arguments = (ins SfaMemRef:$input);
  let results = (outs SfaMemRef:$result);
}

//===----------------------------------------------------------------------===//
// Element-Wise Ops
//===----------------------------------------------------------------------===//

def HAL_ElementWiseOp : HAL_Op<"element_wise", [Pure]> {
  let summary = "generic element-wise operation";
  let description = [{
    Covers all element-wise ops: add, mul, sub, div, pow, max, relu,
    gelu, silu, sigmoid, tanh, exp, neg, softplus, sqrt, rsqrt, cos,
    sin, clamp_min, le, gt, lt, eq, ne, logical_and.
    The specific op is encoded as an `op_type` string attribute.
    Always fused — never executed standalone in the final schedule.
  }];
  let arguments = (ins SfaMemRef:$lhs, SfaMemRef:$rhs,
                       StrAttr:$op_type);
  let results = (outs SfaMemRef:$result);
}

//===----------------------------------------------------------------------===//
// Shape / Data Movement Ops (zero-copy)
//===----------------------------------------------------------------------===//

def HAL_ReshapeOp : HAL_Op<"reshape", [Pure]> {
  let summary = "reshape tensor (zero-copy, metadata only)";
  let description = [{
    Change tensor shape without moving data. May be eliminated by
    fusion passes when the producer/consumer can operate on the
    original shape directly.
  }];
  let arguments = (ins SfaAnyMemRef:$input, I64ArrayAttr:$shape);
  let results = (outs SfaAnyMemRef:$result);
}

def HAL_TransposeOp : HAL_Op<"transpose", [Pure]> {
  let summary = "transpose two dimensions of a tensor";
  let description = [{
    Permute two dimensions. Zero-copy when strides allow; otherwise
    materialized by the runtime.
  }];
  let arguments = (ins SfaMemRef:$input, I64Attr:$dim0, I64Attr:$dim1);
  let results = (outs SfaMemRef:$result);
}

//===----------------------------------------------------------------------===//
// Gather / Reduce Ops
//===----------------------------------------------------------------------===//

def HAL_GatherOp : HAL_Op<"gather", [Pure]> {
  let summary = "gather elements along a dimension by index";
  let description = [{
    Index-based gather along the specified dimension.
  }];
  let arguments = (ins SfaMemRef:$input, SfaIntMemRef:$indices,
                       I64Attr:$dim);
  let results = (outs SfaMemRef:$result);
}

def HAL_ReduceOp : HAL_Op<"reduce", [Pure]> {
  let summary = "reduce operation (sum, mean, var, linalg_norm)";
  let description = [{
    Generic reduction. The `reduce_type` attribute selects the
    specific reduction: "sum", "mean", "var", or "linalg_norm".
  }];
  let arguments = (ins SfaMemRef:$input, StrAttr:$reduce_type);
  let results = (outs SfaMemRef:$result);
}

//===----------------------------------------------------------------------===//
// RoPE (Rotary Position Embedding)
//===----------------------------------------------------------------------===//

def HAL_RopeOp : HAL_Op<"rope", [Pure]> {
  let summary = "rotary position embedding";
  let description = [{
    Apply rotary position encoding. Candidate for fusion into the
    QKV projection (hal.fused_qkv_rope).
  }];
  let arguments = (ins SfaMemRef:$input, SfaIntMemRef:$position_ids);
  let results = (outs SfaMemRef:$result);
}

//===----------------------------------------------------------------------===//
// FUTURE Ops (commented-out placeholders — Phase 1+)
//===----------------------------------------------------------------------===//

// FUTURE: Attention + KV Cache
// FUTURE: def HAL_PagedAttentionOp : HAL_Op<"paged_attention"> {
// FUTURE:   let summary = "paged attention with block table";
// FUTURE:   let arguments = (ins SfaMemRef:$query, SfaMemRef:$key_cache,
// FUTURE:                        SfaMemRef:$value_cache, SfaIntMemRef:$block_table);
// FUTURE:   let results = (outs SfaMemRef:$result);
// FUTURE: }
// FUTURE: def HAL_CacheReadOp : HAL_Op<"cache_read"> {
// FUTURE:   let summary = "read full sequence from KV cache";
// FUTURE:   let arguments = (ins SfaIntMemRef:$block_table);
// FUTURE:   let results = (outs SfaMemRef:$key, SfaMemRef:$value);
// FUTURE: }
// FUTURE: def HAL_CacheWriteOp : HAL_Op<"cache_write"> {
// FUTURE:   let summary = "write to KV cache";
// FUTURE:   let arguments = (ins SfaMemRef:$key, SfaMemRef:$value,
// FUTURE:                        SfaIntMemRef:$positions);
// FUTURE: }

// FUTURE: Quantization
// FUTURE: def HAL_QuantizeOp : HAL_Op<"quantize"> {
// FUTURE:   let summary = "dynamic quantization (fp8/int8)";
// FUTURE:   let arguments = (ins SfaMemRef:$input, StrAttr:$dtype);
// FUTURE:   let results = (outs SfaAnyMemRef:$result);
// FUTURE: }
// FUTURE: def HAL_DequantizeOp : HAL_Op<"dequantize"> {
// FUTURE:   let summary = "dequantize";
// FUTURE:   let arguments = (ins SfaAnyMemRef:$input);
// FUTURE:   let results = (outs SfaMemRef:$result);
// FUTURE: }

// FUTURE: Fused Ops (compiler fusion products)
// FUTURE: def HAL_FusedSiluMulOp : HAL_Op<"fused_silu_mul"> {
// FUTURE:   let summary = "fused silu(gate) * up";
// FUTURE:   let arguments = (ins SfaMemRef:$gate, SfaMemRef:$up);
// FUTURE:   let results = (outs SfaMemRef:$output);
// FUTURE: }
// FUTURE: def HAL_FusedRmsNormMatmulOp : HAL_Op<"fused_rms_norm_matmul"> {
// FUTURE:   let summary = "fused rms_norm(x) * weight followed by matmul";
// FUTURE:   let arguments = (ins SfaMemRef:$input, SfaMemRef:$rms_weight,
// FUTURE:                        SfaMemRef:$matmul_weight);
// FUTURE:   let results = (outs SfaMemRef:$output);
// FUTURE: }
// FUTURE: def HAL_FusedSwiGLUFfnOp : HAL_Op<"fused_swiglu_ffn"> {
// FUTURE:   let summary = "fused FFN block: gate→silu→mul(up)→down";
// FUTURE:   let arguments = (ins SfaMemRef:$input, SfaMemRef:$gate_weight,
// FUTURE:                        SfaMemRef:$up_weight, SfaMemRef:$down_weight);
// FUTURE:   let results = (outs SfaMemRef:$output);
// FUTURE: }
// FUTURE: def HAL_FusedQKVRopeOp : HAL_Op<"fused_qkv_rope"> {
// FUTURE:   let summary = "fused Q/K/V projection + RoPE";
// FUTURE:   let arguments = (ins SfaMemRef:$input, SfaMemRef:$q_weight,
// FUTURE:                        SfaMemRef:$k_weight, SfaMemRef:$v_weight,
// FUTURE:                        SfaIntMemRef:$position_ids);
// FUTURE:   let results = (outs SfaMemRef:$q, SfaMemRef:$k, SfaMemRef:$v);
// FUTURE: }
// FUTURE: def HAL_FusedAttentionOutputOp : HAL_Op<"fused_attention_output"> {
// FUTURE:   let summary = "fused attention + output projection";
// FUTURE:   let arguments = (ins SfaMemRef:$attn_output, SfaMemRef:$o_weight);
// FUTURE:   let results = (outs SfaMemRef:$output);
// FUTURE: }

// FUTURE: Data Movement (zero-copy, may be eliminated by fusion)
// FUTURE: def HAL_BroadcastOp : HAL_Op<"broadcast"> { ... }
// FUTURE: def HAL_ConcatOp : HAL_Op<"concat"> { ... }
// FUTURE: def HAL_SliceOp : HAL_Op<"slice"> { ... }
// FUTURE: def HAL_ScatterOp : HAL_Op<"scatter"> { ... }
// FUTURE: def HAL_CopyOp : HAL_Op<"copy"> { ... }
// FUTURE: def HAL_TransferOp : HAL_Op<"transfer"> { ... }

// FUTURE: Reduction / Creation / Sampling
// FUTURE: def HAL_ScanOp : HAL_Op<"scan"> { ... }
// FUTURE: def HAL_FillOp : HAL_Op<"fill"> { ... }
// FUTURE: def HAL_TopkOp : HAL_Op<"topk"> { ... }
// FUTURE: def HAL_ArgmaxOp : HAL_Op<"argmax"> { ... }

// FUTURE: Communication (distributed, future Phase)
// FUTURE: def HAL_AllReduceOp : HAL_Op<"all_reduce"> { ... }
// FUTURE: def HAL_AllGatherOp : HAL_Op<"all_gather"> { ... }
// FUTURE: def HAL_ReduceScatterOp : HAL_Op<"reduce_scatter"> { ... }
// FUTURE: def HAL_SendOp : HAL_Op<"send"> { ... }
// FUTURE: def HAL_RecvOp : HAL_Op<"recv"> { ... }

#endif // HAL_OPS
