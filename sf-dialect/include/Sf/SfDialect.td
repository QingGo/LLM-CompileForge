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

def Sf_FloatTensor : TensorOf<[AnyFloat]>;
def Sf_FloatOrFloatTensor : AnyTypeOf<[AnyFloat, Sf_FloatTensor]>;

// Integer tensor type constraint — for arange output, index operands, position indices
def Sf_Int32Tensor : TensorOf<[I32]>;
def Sf_Int64Tensor : TensorOf<[I64]>;
def Sf_IntegerTensor : TensorOf<[AnyInteger]>;

// Scalar types (for type_as, constants)
def Sf_FloatScalar : AnyTypeOf<[F16, F32, F64]>;
def Sf_IntegerScalar : AnyTypeOf<[I32, I64]>;

// Numeric tensors — float or integer, used for arithmetic ops (add, mul, sub, div)
def Sf_FloatOrIntegerTensor : TensorOf<[AnyFloat, AnyInteger]>;

#endif // SF_DIALECT
