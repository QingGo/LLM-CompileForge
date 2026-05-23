//===- SfShapeInference.cpp ----------------------------------------------===//
//
// This file implements shape verification for sf dialect binary ops and
// a reusable NumPy-style broadcast shape computation utility for use by
// lowering passes (e.g., SfLowerCompare.cpp).
//
//===----------------------------------------------------------------------===//

#include "Sf/SfOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
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
    int64_t lhsDim =
        (i <= lhsRank) ? lhsType.getDimSize(lhsRank - i) : 1;
    int64_t rhsDim =
        (i <= rhsRank) ? rhsType.getDimSize(rhsRank - i) : 1;

    int64_t &resultDim = outShape[outRank - i];

    if (ShapedType::isDynamic(lhsDim) && ShapedType::isDynamic(rhsDim)) {
      resultDim = ShapedType::kDynamic;
    } else if (ShapedType::isDynamic(lhsDim)) {
      resultDim = rhsDim;
    } else if (ShapedType::isDynamic(rhsDim)) {
      resultDim = lhsDim;
    } else if (lhsDim == rhsDim || lhsDim == 1) {
      resultDim = rhsDim;
    } else if (rhsDim == 1) {
      resultDim = lhsDim;
    } else {
      return failure();
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

// Stub: tablegen may still reference InferTypeOpInterface for these ops
// from cached .inc files.  Return success() without modifying the result
// type — the ops already have explicit types from the Python frontend.
#define STUB_INFER_RETURN_TYPES(OpName)                                   \
  LogicalResult OpName::inferReturnTypes(                                 \
      ::mlir::MLIRContext *, ::std::optional<::mlir::Location>,           \
      ::mlir::ValueRange, ::mlir::DictionaryAttr,                         \
      ::mlir::OpaqueProperties, ::mlir::RegionRange,                      \
      ::llvm::SmallVectorImpl<::mlir::Type> &) { return success(); }

STUB_INFER_RETURN_TYPES(AddOp)
STUB_INFER_RETURN_TYPES(MulOp)
STUB_INFER_RETURN_TYPES(SubOp)
STUB_INFER_RETURN_TYPES(DivOp)
STUB_INFER_RETURN_TYPES(PowOp)
STUB_INFER_RETURN_TYPES(MaxOp)

#define IMPL_COMPARE_VERIFY(OpName)                                     \
  LogicalResult OpName::verify() {                                      \
    return verifyBinaryOpShapes(getOperation());                        \
  }

IMPL_COMPARE_VERIFY(LeOp)
IMPL_COMPARE_VERIFY(GtOp)
IMPL_COMPARE_VERIFY(LtOp)
IMPL_COMPARE_VERIFY(EqOp)
IMPL_COMPARE_VERIFY(NeOp)

STUB_INFER_RETURN_TYPES(LeOp)
STUB_INFER_RETURN_TYPES(GtOp)
STUB_INFER_RETURN_TYPES(LtOp)
STUB_INFER_RETURN_TYPES(EqOp)
STUB_INFER_RETURN_TYPES(NeOp)

LogicalResult LogicalAndOp::verify() {
  return verifyBinaryOpShapes(getOperation());
}

STUB_INFER_RETURN_TYPES(LogicalAndOp)
