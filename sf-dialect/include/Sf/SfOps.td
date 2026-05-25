#ifndef SF_OPS
#define SF_OPS

include "Sf/SfDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// Base classes
//===----------------------------------------------------------------------===//

class Sf_BinaryOp<string mnemonic, list<Trait> traits = []> :
    Sf_Op<mnemonic, !listconcat(traits, [SameOperandsAndResultElementType, Pure,
      DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>])> {
  let arguments = (ins Sf_FloatOrIntegerTensor:$lhs, Sf_FloatOrIntegerTensor:$rhs);
  let results = (outs Sf_FloatOrIntegerTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasVerifier = 1;
}

class Sf_ActivationOp<string mnemonic, list<Trait> traits = []> :
    Sf_Op<mnemonic, !listconcat([Pure, SameOperandsAndResultType, Elementwise,
      DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>], traits)> {
  let arguments = (ins Sf_FloatTensor:$input);
  let results = (outs Sf_FloatTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($result)";
}

class Sf_ShapeOp<string mnemonic, list<Trait> traits = []> :
    Sf_Op<mnemonic, !listconcat([Pure], traits)> {
}

class Sf_CompareOp<string mnemonic, list<Trait> traits = []> :
    Sf_Op<mnemonic, !listconcat(traits, [SameOperandsElementType, Pure])> {
  let arguments = (ins Sf_FloatOrIntegerTensor:$lhs, Sf_FloatOrIntegerTensor:$rhs);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Binary arithmetic ops
//===----------------------------------------------------------------------===//

def Sf_AddOp : Sf_BinaryOp<"add", [
    Commutative,
    DeclareOpInterfaceMethods<InferTypeOpInterface, ["isCompatibleReturnTypes"]>,
]> {
  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(::mlir::TypeRange inferred,
                                        ::mlir::TypeRange actual);
  }];
  let extraClassDefinition = [{
    bool AddOp::isCompatibleReturnTypes(::mlir::TypeRange inferred,
                                        ::mlir::TypeRange actual) {
      if (inferred.size() != actual.size() || inferred.size() != 1)
        return false;
      auto inf = ::llvm::dyn_cast<::mlir::RankedTensorType>(inferred[0]);
      auto act = ::llvm::dyn_cast<::mlir::RankedTensorType>(actual[0]);
      if (!inf || !act) return false;
      return inf.getElementType() == act.getElementType() &&
             ::mlir::succeeded(::mlir::verifyCompatibleShape(
                 inf.getShape(), act.getShape()));
    }
  }];
}
def Sf_MulOp : Sf_BinaryOp<"mul", [Commutative]>;
def Sf_SubOp : Sf_BinaryOp<"sub">;
def Sf_PowOp : Sf_BinaryOp<"pow">;
def Sf_DivOp : Sf_BinaryOp<"div">;
def Sf_MaxOp : Sf_BinaryOp<"max", [Commutative]>;

//===----------------------------------------------------------------------===//
// Activation ops
//===----------------------------------------------------------------------===//

def Sf_ReluOp : Sf_ActivationOp<"relu">;
def Sf_GeluOp : Sf_ActivationOp<"gelu">;
def Sf_SiluOp : Sf_ActivationOp<"silu">;
def Sf_SigmoidOp : Sf_ActivationOp<"sigmoid">;
def Sf_TanhOp : Sf_ActivationOp<"tanh">;
def Sf_ExpOp : Sf_ActivationOp<"exp">;
def Sf_NegOp : Sf_ActivationOp<"neg">;

//===----------------------------------------------------------------------===//
// Matmul / Linear
//===----------------------------------------------------------------------===//

def Sf_MatmulOp : Sf_Op<"matmul", [Pure]> {
  let arguments = (ins Sf_FloatTensor:$lhs, Sf_FloatTensor:$rhs);
  let results = (outs Sf_FloatTensor:$result);
}

def Sf_LinearOp : Sf_Op<"linear", [Pure]> {
  let arguments = (ins Sf_FloatTensor:$input, Sf_FloatTensor:$weight,
                       Optional<Sf_FloatTensor>:$bias);
  let results = (outs Sf_FloatTensor:$result);
}

//===----------------------------------------------------------------------===//
// Shape / transform ops
//===----------------------------------------------------------------------===//

def Sf_IdentityOp : Sf_ShapeOp<"identity"> {
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
}

def Sf_ViewOp : Sf_ShapeOp<"view"> {
  let arguments = (ins AnyTensor:$input, Variadic<AnyTensor>:$dyn_shape);
  let results = (outs AnyTensor:$result);
}

def Sf_ExpandOp : Sf_ShapeOp<"expand"> {
  let arguments = (ins AnyTensor:$input, Variadic<AnyTensor>:$dyn_shape);
  let results = (outs AnyTensor:$result);
}

def Sf_UnsqueezeOp : Sf_ShapeOp<"unsqueeze"> {
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
}

//===----------------------------------------------------------------------===//
// Reduction ops
//===----------------------------------------------------------------------===//

def Sf_SumOp : Sf_Op<"sum", [Pure]> {
  let arguments = (ins Sf_FloatTensor:$input);
  let results = (outs Sf_FloatTensor:$result);
}

def Sf_MeanOp : Sf_Op<"mean", [Pure]> {
  let arguments = (ins Sf_FloatTensor:$input);
  let results = (outs Sf_FloatTensor:$result);
}

def Sf_SoftmaxOp : Sf_Op<"softmax", [Pure]> {
  let arguments = (ins Sf_FloatTensor:$input);
  let results = (outs Sf_FloatTensor:$result);
}

//===----------------------------------------------------------------------===//
// Special ops (no Pure trait — survived by lowering)
//===----------------------------------------------------------------------===//

def Sf_ConstantOp : Sf_Op<"constant"> {
  let arguments = (ins ElementsAttr:$value);
  let results = (outs AnyTensor:$result);
  let hasFolder = 1;
  let extraClassDefinition = [{
    ::mlir::OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
      return getValue();
    }
  }];
}

def Sf_WeightOp : Sf_Op<"weight"> {
  let arguments = (ins StrAttr:$name);
  let results = (outs AnyTensor:$result);
}

//===----------------------------------------------------------------------===//
// Attention / Norm / Fused ops (passthrough in lowering)
//===----------------------------------------------------------------------===//

def Sf_ScaledDotProductAttentionOp : Sf_Op<"scaled_dot_product_attention"> {
  let arguments = (ins Sf_FloatTensor:$query, Sf_FloatTensor:$key,
                       Sf_FloatTensor:$value, Optional<Sf_FloatTensor>:$attn_mask);
  let results = (outs Sf_FloatTensor:$result);
}

def Sf_LayerNormOp : Sf_Op<"layer_norm"> {
  let arguments = (ins Sf_FloatTensor:$input, Sf_FloatTensor:$weight,
                       Sf_FloatTensor:$bias);
  let results = (outs Sf_FloatTensor:$result);
}

def Sf_RmsNormOp : Sf_Op<"rms_norm"> {
  let arguments = (ins Sf_FloatTensor:$input, Sf_FloatTensor:$weight);
  let results = (outs Sf_FloatTensor:$result);
}

//===----------------------------------------------------------------------===//
// Shape/Comparison/Misc ops (missing from first pass, needed for model coverage)
//===----------------------------------------------------------------------===//

def Sf_TransposeOp : Sf_ShapeOp<"transpose"> {
  let arguments = (ins Sf_FloatTensor:$input, I64Attr:$dim0, I64Attr:$dim1);
  let results = (outs Sf_FloatTensor:$result);
}

def Sf_SliceOp : Sf_ShapeOp<"slice"> {
  let arguments = (ins AnyTensor:$input, I64Attr:$dim,
                       I64Attr:$start, I64Attr:$end);
  let results = (outs AnyTensor:$result);
}

def Sf_LogicalAndOp : Sf_Op<"logical_and", [
    Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>
]> {
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
  let hasVerifier = 1;
}

def Sf_LeOp : Sf_CompareOp<"le">;

def Sf_OnesLikeOp : Sf_ShapeOp<"ones_like"> {
  let arguments = (ins AnyTensor:$input, Variadic<AnyTensor>:$dyn_shape);
  let results = (outs AnyTensor:$result);
}

// Ops that survive lowering (marked legal in pass)
def Sf_ArangeOp : Sf_ShapeOp<"arange"> {
  let arguments = (ins AnyTensor:$input, Variadic<AnyTensor>:$dyn_shape);
  let results = (outs Sf_Int64Tensor:$result);
}

def Sf_SymSizeOp : Sf_ShapeOp<"sym_size"> {
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
}

def Sf_EmbeddingOp : Sf_ShapeOp<"embedding"> {
  let arguments = (ins Sf_FloatTensor:$weight, AnyTensor:$indices);
  let results = (outs Sf_FloatTensor:$result);
}

def Sf_IndexOp : Sf_ShapeOp<"index"> {
  let arguments = (ins Sf_FloatOrIntegerTensor:$input, Variadic<AnyTensor>:$dyn_index);
  let results = (outs Sf_FloatOrIntegerTensor:$result);
  let hasVerifier = 1;
}

def Sf_CumsumOp : Sf_ShapeOp<"cumsum"> {
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
}

def Sf_NewOnesOp : Sf_ShapeOp<"new_ones"> {
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
}

//===----------------------------------------------------------------------===//
// Activation ops missing from first pass
//===----------------------------------------------------------------------===//

def Sf_SoftplusOp : Sf_ActivationOp<"softplus">;
def Sf_SqrtOp : Sf_ActivationOp<"sqrt">;
def Sf_CosOp : Sf_ActivationOp<"cos">;
def Sf_SinOp : Sf_ActivationOp<"sin">;
def Sf_RsqrtOp : Sf_ActivationOp<"rsqrt">;

//===----------------------------------------------------------------------===//
// Comparison ops (missing from first pass, output is bool/integer tensor)
//===----------------------------------------------------------------------===//

def Sf_GtOp : Sf_CompareOp<"gt">;
def Sf_LtOp : Sf_CompareOp<"lt">;
def Sf_EqOp : Sf_CompareOp<"eq">;
def Sf_NeOp : Sf_CompareOp<"ne">;

//===----------------------------------------------------------------------===//
// Shape / transform ops missing from first pass
//===----------------------------------------------------------------------===//

def Sf_PermuteOp : Sf_ShapeOp<"permute"> {
  let arguments = (ins AnyTensor:$input, I64ArrayAttr:$dims);
  let results = (outs AnyTensor:$result);
}

def Sf_SelectOp : Sf_ShapeOp<"select"> {
  let arguments = (ins AnyTensor:$input, I64Attr:$dim, I64Attr:$index);
  let results = (outs AnyTensor:$result);
}

def Sf_CatOp : Sf_ShapeOp<"cat"> {
  let arguments = (ins Variadic<AnyTensor>:$inputs, I64Attr:$dim);
  let results = (outs AnyTensor:$result);
}

#endif // SF_OPS
