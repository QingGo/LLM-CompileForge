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

  // Check element type compatibility
  if (lhsType.getElementType() != rhsType.getElementType()) {
    return op->emitOpError("lhs element type (")
           << lhsType.getElementType() << ") does not match rhs ("
           << rhsType.getElementType() << ")";
  }

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

// Infer return types for binary arithmetic ops (SameOperandsAndResultElementType).
// Compute broadcast shape and use lhs element type.
#define SF_BINARY_ARITH_INFER(OpName)                                         \
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
        RankedTensorType::get(shape, lhs.getElementType()));                  \
    return success();                                                         \
  }

SF_BINARY_ARITH_INFER(AddOp)
SF_BINARY_ARITH_INFER(MulOp)
SF_BINARY_ARITH_INFER(SubOp)
SF_BINARY_ARITH_INFER(DivOp)
SF_BINARY_ARITH_INFER(PowOp)
SF_BINARY_ARITH_INFER(MaxOp)

// logical_and: output is always f32 to match the lowering which
// converts both operands to bool, andi's them, then uitofp → f32.
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
  auto f32 = Float32Type::get(context);
  inferredReturnTypes.push_back(RankedTensorType::get(shape, f32));
  return success();
}

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
  // Relaxed verifier: logical_and accepts mixed numeric types
  // (integer and float) since the lowering handles promotion.
  auto lhsType = dyn_cast<RankedTensorType>(getOperation()->getOperand(0).getType());
  auto rhsType = dyn_cast<RankedTensorType>(getOperation()->getOperand(1).getType());

  if (!lhsType || !rhsType)
    return emitOpError("operands must be ranked tensors");

  // Both must be numeric (integer or float), but don't require exact match
  auto lhsElt = lhsType.getElementType();
  auto rhsElt = rhsType.getElementType();
  if (!lhsElt.isIntOrFloat() || !rhsElt.isIntOrFloat())
    return emitOpError("operands must have numeric element types, got ")
           << lhsElt << " and " << rhsElt;

  SmallVector<int64_t> shape;
  if (failed(mlir::sf::computeBroadcastShape(lhsType, rhsType, shape)))
    return emitOpError("incompatible shapes for broadcasting: ")
           << lhsType << " and " << rhsType;

  return success();
}

//===----------------------------------------------------------------------===//
// IndexOp verifier: check index operands have integer or float element type
// (float is converted via FPToUIOp in the lowering with a WARNING)
//===----------------------------------------------------------------------===//

LogicalResult IndexOp::verify() {
  for (auto idx : getDynIndex()) {
    auto idxType = dyn_cast<RankedTensorType>(idx.getType());
    if (!idxType ||
        !(isa<IntegerType>(idxType.getElementType()) ||
          isa<FloatType>(idxType.getElementType()))) {
      return emitOpError("index tensor must have integer or float element type, got ")
             << idx.getType();
    }
  }
  return success();
}
