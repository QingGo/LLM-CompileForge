//===- SfLoweringPatterns.td - DRR canonicalization/folding patterns --------===//

#ifndef SF_LOWERING_PATTERNS
#define SF_LOWERING_PATTERNS

include "mlir/IR/PatternBase.td"
include "Sf/SfOps.td"

def ConstantZeroF : ConstantAttr<F32Attr, "0.0">;
def ConstantOneF : ConstantAttr<F32Attr, "1.0">;

def FoldAddZero : Pat<(Sf_AddOp $x, (Sf_ConstantOp ConstantZeroF)), (replaceWithValue $x)>;
def FoldMulOne : Pat<(Sf_MulOp $x, (Sf_ConstantOp ConstantOneF)), (replaceWithValue $x)>;

// Named linalg lowering (sf.exp → linalg.exp etc.) via DRR is blocked by
// linalg's AttrSizedOperandSegments + Variadic operands which DRR can't represent.

#endif
