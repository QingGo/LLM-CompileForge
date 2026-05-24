//===- SfShapeInference.cpp ----------------------------------------------===//
//
// This file implements shape verification for sf dialect binary ops and
// a reusable NumPy-style broadcast shape computation utility for use by
// lowering passes (e.g., SfLowerCompare.cpp).
//
//===----------------------------------------------------------------------===//

#include "Sf/SfOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/ArrayRef.h"

using namespace mlir;
using namespace mlir::sf;

namespace {

// Maximum supported rank for binary broadcast ops.
constexpr int64_t kMaxBinaryRank = 4;

} // namespace

//===----------------------------------------------------------------------===//
// Public utility: computeBroadcastShape
//===----------------------------------------------------------------------===//

LogicalResult mlir::sf::computeBroadcastShape(RankedTensorType lhsType,
                                               RankedTensorType rhsType,
                                               SmallVectorImpl<int64_t> &outShape) {
  int64_t lhsRank = lhsType.getRank();
  int64_t rhsRank = rhsType.getRank();
  int64_t outRank = std::max(lhsRank, rhsRank);

  outShape.resize(outRank);

  for (int64_t i = 1; i <= outRank; ++i) {
    bool lhsHas = (i <= lhsRank);
    bool rhsHas = (i <= rhsRank);
    int64_t lhsDim = lhsHas ? lhsType.getDimSize(lhsRank - i)
                            : ShapedType::kDynamic;
    int64_t rhsDim = rhsHas ? rhsType.getDimSize(rhsRank - i)
                            : ShapedType::kDynamic;

    int64_t &resultDim = outShape[outRank - i];

    // Missing dim: like size-1 broadcast, output = the only present dim.
    if (!lhsHas) {
      resultDim = rhsDim;
    } else if (!rhsHas) {
      resultDim = lhsDim;
    }
    // Both static
    else if (!ShapedType::isDynamic(lhsDim) && !ShapedType::isDynamic(rhsDim)) {
      if (lhsDim == rhsDim || lhsDim == 1) {
        resultDim = rhsDim;
      } else if (rhsDim == 1) {
        resultDim = lhsDim;
      } else {
        return failure();
      }
    }
    // Both dynamic
    else if (ShapedType::isDynamic(lhsDim) && ShapedType::isDynamic(rhsDim)) {
      resultDim = ShapedType::kDynamic;
    }
    // lhs dynamic, rhs static — rhs=1 broadcasts, no concrete info
    else if (ShapedType::isDynamic(lhsDim)) {
      resultDim = (rhsDim == 1) ? ShapedType::kDynamic : rhsDim;
    }
    // rhs dynamic, lhs static — lhs=1 broadcasts, no concrete info
    else {
      resultDim = (lhsDim == 1) ? ShapedType::kDynamic : lhsDim;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Verifiers for binary float ops (add, mul, sub, div, pow, max)
//===----------------------------------------------------------------------===//

static LogicalResult verifyBinaryOpShapes(Operation *op) {
  auto lhsType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto rhsType = dyn_cast<RankedTensorType>(op->getOperand(1).getType());

  if (!lhsType || !rhsType)
    return op->emitOpError("operands must be ranked tensors");

  if (lhsType.getRank() > kMaxBinaryRank)
    return op->emitOpError("lhs rank ")
           << lhsType.getRank() << " exceeds max supported "
           << kMaxBinaryRank;
  if (rhsType.getRank() > kMaxBinaryRank)
    return op->emitOpError("rhs rank ")
           << rhsType.getRank() << " exceeds max supported "
           << kMaxBinaryRank;

  SmallVector<int64_t> shape;
  if (failed(mlir::sf::computeBroadcastShape(lhsType, rhsType, shape)))
    return op->emitOpError("incompatible shapes for broadcasting: ")
           << lhsType << " and " << rhsType;

  return success();
}

#define IMPL_BINARY_FLOAT_VERIFY(OpName)                                \
  LogicalResult OpName::verify() {                                      \
    return verifyBinaryOpShapes(getOperation());                        \
  }

IMPL_BINARY_FLOAT_VERIFY(AddOp)
IMPL_BINARY_FLOAT_VERIFY(MulOp)
IMPL_BINARY_FLOAT_VERIFY(SubOp)
IMPL_BINARY_FLOAT_VERIFY(DivOp)
IMPL_BINARY_FLOAT_VERIFY(PowOp)
IMPL_BINARY_FLOAT_VERIFY(MaxOp)

// Real inferReturnTypes for ALL binary/comparison ops.
// LLVM 22.1 instantiates InferTypeOpInterface trait models for ALL registered
// ops (via RegisteredOperationName::Model<T>), so every op needs an impl.
// Binary ops compute broadcast shape; comparison ops do the same (both
// operands are Sf_FloatTensor).  Element type: f32 for binary/comparison,
// AnyTensor element type for logical_and (use lhs element type).
#define SF_BINARY_INFER(OpName)                                               \
  LogicalResult OpName::inferReturnTypes(                                     \
      ::mlir::MLIRContext *context, ::std::optional<::mlir::Location>,        \
      ::mlir::ValueRange operands, ::mlir::DictionaryAttr,                    \
      ::mlir::OpaqueProperties, ::mlir::RegionRange,                          \
      ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {           \
    auto lhs = dyn_cast<RankedTensorType>(operands[0].getType());             \
    auto rhs = dyn_cast<RankedTensorType>(operands[1].getType());             \
    if (!lhs || !rhs) return failure();                                       \
    SmallVector<int64_t> shape;                                               \
    if (failed(computeBroadcastShape(lhs, rhs, shape))) return failure();     \
    inferredReturnTypes.push_back(                                            \
        RankedTensorType::get(shape, Builder(context).getF32Type()));         \
    return success();                                                         \
  }

SF_BINARY_INFER(AddOp)
SF_BINARY_INFER(MulOp)
SF_BINARY_INFER(SubOp)
SF_BINARY_INFER(DivOp)
SF_BINARY_INFER(PowOp)
SF_BINARY_INFER(MaxOp)

SF_BINARY_INFER(LeOp)
SF_BINARY_INFER(GtOp)
SF_BINARY_INFER(LtOp)
SF_BINARY_INFER(EqOp)
SF_BINARY_INFER(NeOp)

// logical_and: uses element type from lhs (AnyTensor operands)
LogicalResult LogicalAndOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location>,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr,
    ::mlir::OpaqueProperties, ::mlir::RegionRange,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto lhs = dyn_cast<RankedTensorType>(operands[0].getType());
  auto rhs = dyn_cast<RankedTensorType>(operands[1].getType());
  if (!lhs || !rhs) return failure();
  SmallVector<int64_t> shape;
  if (failed(computeBroadcastShape(lhs, rhs, shape))) return failure();
  inferredReturnTypes.push_back(
      RankedTensorType::get(shape, lhs.getElementType()));
  return success();
}
#undef SF_BINARY_INFER

#define IMPL_COMPARE_VERIFY(OpName)                                     \
  LogicalResult OpName::verify() {                                      \
    return verifyBinaryOpShapes(getOperation());                        \
  }

IMPL_COMPARE_VERIFY(LeOp)
IMPL_COMPARE_VERIFY(GtOp)
IMPL_COMPARE_VERIFY(LtOp)
IMPL_COMPARE_VERIFY(EqOp)
IMPL_COMPARE_VERIFY(NeOp)

LogicalResult LogicalAndOp::verify() {
  return verifyBinaryOpShapes(getOperation());
}

//===----------------------------------------------------------------------===//
// Activation op shape inference — SameOperandsAndResultType
//===----------------------------------------------------------------------===//

#define SF_ACTIVATION_INFER(OpName)                                            \
  ::mlir::LogicalResult OpName::inferReturnTypes(                              \
      ::mlir::MLIRContext *, ::std::optional<::mlir::Location>,                \
      ::mlir::ValueRange operands, ::mlir::DictionaryAttr,                     \
      ::mlir::OpaqueProperties, ::mlir::RegionRange,                           \
      ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {            \
    inferredReturnTypes.push_back(operands[0].getType());                      \
    return ::mlir::success();                                                  \
  }

SF_ACTIVATION_INFER(ReluOp)
SF_ACTIVATION_INFER(GeluOp)
SF_ACTIVATION_INFER(SiluOp)
SF_ACTIVATION_INFER(SigmoidOp)
SF_ACTIVATION_INFER(TanhOp)
SF_ACTIVATION_INFER(ExpOp)
SF_ACTIVATION_INFER(NegOp)
SF_ACTIVATION_INFER(SoftplusOp)
SF_ACTIVATION_INFER(SqrtOp)
SF_ACTIVATION_INFER(CosOp)
SF_ACTIVATION_INFER(SinOp)
SF_ACTIVATION_INFER(RsqrtOp)

#undef SF_ACTIVATION_INFER
