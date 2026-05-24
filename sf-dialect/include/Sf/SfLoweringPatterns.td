//===- SfLoweringPatterns.td - DRR canonicalization/folding patterns --------===//

#ifndef SF_LOWERING_PATTERNS
#define SF_LOWERING_PATTERNS

include "mlir/IR/PatternBase.td"
include "Sf/SfOps.td"

def ConstantZeroF : ConstantAttr<F32Attr, "0.0">;
def ConstantOneF : ConstantAttr<F32Attr, "1.0">;

// ── Arithmetic folding ──────────────────────────────────────────────

def FoldAddZero : Pat<(Sf_AddOp $x, (Sf_ConstantOp ConstantZeroF)), (replaceWithValue $x)>;
def FoldMulOne : Pat<(Sf_MulOp $x, (Sf_ConstantOp ConstantOneF)), (replaceWithValue $x)>;
def FoldSubZero : Pat<(Sf_SubOp $x, (Sf_ConstantOp ConstantZeroF)), (replaceWithValue $x)>;
def FoldDivOne : Pat<(Sf_DivOp $x, (Sf_ConstantOp ConstantOneF)), (replaceWithValue $x)>;

// ── Canonicalization ────────────────────────────────────────────────

def FoldNegNeg : Pat<(Sf_NegOp (Sf_NegOp:$inner $x)), (replaceWithValue $x)>;

// ── Fusion (within existing sf ops) ─────────────────────────────────

def FoldReluRelu : Pat<(Sf_ReluOp (Sf_ReluOp:$inner $x)), (replaceWithValue $inner)>;
def FuseSilu : Pat<(Sf_MulOp (Sf_SigmoidOp:$sig $x), $x), (Sf_SiluOp $x)>;

#endif
