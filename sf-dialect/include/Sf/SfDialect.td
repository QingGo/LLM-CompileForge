#ifndef SF_DIALECT
#define SF_DIALECT

include "mlir/IR/OpBase.td"

def Sf_Dialect : Dialect {
  let name = "sf";
  let cppNamespace = "::mlir::sf";
  let summary = "ServeForge AOT compilation dialect for LLM model ops";
  let description = [{
    The `sf` dialect represents neural network operations extracted from
    PyTorch FX graphs. It serves as the high-level IR for the ServeForge
    AOT compilation pipeline. Ops are lowered to `linalg`/`arith`/`math`
    before LLVM code generation.

    Each op carries traits (`Pure`, `Commutative`, `SameOperandsAndResultType`,
    etc.) that enable DCE, CSE, constant folding, and commutative ordering
    by the MLIR canonicalizer.
  }];
  let useDefaultTypePrinterParser = 0;
  let hasConstantMaterializer = 0;
}

class Sf_Op<string mnemonic, list<Trait> traits = []> :
    Op<Sf_Dialect, mnemonic, traits>;

// Floating-point tensor constraint — used by matmul, linear, normalization, attention ops
def Sf_FloatTensor : TensorOf<[AnyFloat]>;
// Scalar float or float tensor — used for type_as/conversion ops
def Sf_FloatOrFloatTensor : AnyTypeOf<[AnyFloat, Sf_FloatTensor]>;

// i32 tensor — used for shape/index ops
def Sf_Int32Tensor : TensorOf<[I32]>;
// i64 tensor — used for arange output, position indices
def Sf_Int64Tensor : TensorOf<[I64]>;
// Any integer tensor — used for bitwise/logical ops
def Sf_IntegerTensor : TensorOf<[AnyInteger]>;

// f16/f32/f64 scalar — used for type_as, constant scalar values
def Sf_FloatScalar : AnyTypeOf<[F16, F32, F64]>;
// i32/i64 scalar — used for index scalar values
def Sf_IntegerScalar : AnyTypeOf<[I32, I64]>;

// Float or integer tensor — used by binary arithmetic ops (add/mul/sub/div)
def Sf_FloatOrIntegerTensor : TensorOf<[AnyFloat, AnyInteger]>;

#endif // SF_DIALECT
