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

#endif // SF_DIALECT
