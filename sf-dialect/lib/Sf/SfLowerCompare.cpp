#define NDEBUG
#include "SfLoweringHelpers.h"
#include "Sf/SfDialect.h"
#include "Sf/SfOps.h"

#define DEBUG_TYPE "sf-lower-to-linalg"

using namespace mlir;

namespace {

// Le comparison -> arith.cmpf in generic with explicit linalg.broadcast
// Computes lhs <= rhs element-wise. Output is f32 (0.0/1.0) to avoid
// i1->f32 unrealized_conversion_cast downstream.
// Uses explicit linalg.broadcast + identity-map linalg.generic instead of
// broadcast affine maps to avoid kDynamic leaks from InferStaticShapeOfOperands.
struct SfLeOpLowering : public OpRewritePattern<sf::LeOp> {
  using OpRewritePattern<sf::LeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sf::LeOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type rt = op.getResult().getType();
    if (!isa<ShapedType>(rt)) return failure();

    auto lhsType = cast<RankedTensorType>(op.getLhs().getType());
    auto rhsType = cast<RankedTensorType>(op.getRhs().getType());
    int64_t lhsRank = lhsType.getRank();
    int64_t rhsRank = rhsType.getRank();
    int64_t outRank = std::max(lhsRank, rhsRank);
    MLIRContext *ctx = rewriter.getContext();

    // 1. Compute broadcast output shape (numpy rules)
    SmallVector<int64_t> outShape(outRank, ShapedType::kDynamic);
    SmallVector<Value> dynSizes;
    for (int64_t i = 0; i < outRank; ++i) {
      int64_t lhsDim = (i >= outRank - lhsRank)
          ? lhsType.getDimSize(i - (outRank - lhsRank)) : 1;
      int64_t rhsDim = (i >= outRank - rhsRank)
          ? rhsType.getDimSize(i - (outRank - rhsRank)) : 1;

      bool lhsDynamic = ShapedType::isDynamic(lhsDim);
      bool rhsDynamic = ShapedType::isDynamic(rhsDim);

      if (lhsDynamic || rhsDynamic) {
        outShape[i] = ShapedType::kDynamic;
        if (i >= outRank - lhsRank) {
          int64_t li = i - (outRank - lhsRank);
          if (lhsType.isDynamicDim(li)) {
            dynSizes.push_back(tensor::DimOp::create(rewriter, loc, op.getLhs(), li));
            continue;
          }
        }
        if (i >= outRank - rhsRank) {
          int64_t ri = i - (outRank - rhsRank);
          if (rhsType.isDynamicDim(ri)) {
            dynSizes.push_back(tensor::DimOp::create(rewriter, loc, op.getRhs(), ri));
          }
        }
      } else {
        outShape[i] = std::max(lhsDim, rhsDim);
      }
    }

    auto f32Type = rewriter.getF32Type();
    auto outTensorType = RankedTensorType::get(outShape, f32Type);
    Value genericInit = tensor::EmptyOp::create(rewriter, loc, outTensorType, dynSizes);

    // 2. Helper: broadcast an operand to the output shape using
    //    linalg.broadcast (never broadcast affine maps).
    auto broadcastOperand = [&](Value operand, RankedTensorType operandType,
                                 int64_t operandRank) -> Value {
      if (operandRank < outRank) {
        // Lower-rank operand: use linalg.broadcast with leading added dims.
        SmallVector<int64_t> addedDims;
        for (int64_t i = 0; i < outRank - operandRank; ++i)
          addedDims.push_back(i);
        Value bcInit = tensor::EmptyOp::create(rewriter, loc, outTensorType, dynSizes);
        return linalg::BroadcastOp::create(rewriter, loc, operand, bcInit, addedDims)->getResult(0);
      }

      if (operandRank == outRank) {
        // Same-rank: check if any dim needs broadcasting (size-1 -> larger).
        SmallVector<int64_t> broadcastDims;
        for (int64_t i = 0; i < outRank; ++i) {
          int64_t opSize = operandType.getDimSize(i);
          if (opSize == 1 && !ShapedType::isDynamic(outShape[i]) && outShape[i] > 1)
            broadcastDims.push_back(i);
        }

        if (broadcastDims.empty())
          return operand;

        // For each broadcast dim: collapse it with a neighbor using
        // tensor.collapse_shape, then linalg.broadcast to add it back.
        Value current = operand;
        int64_t curRank = outRank;

        // Process right-to-left to keep indices stable.
        for (auto k : llvm::reverse(broadcastDims)) {
          // Merge dims (k-1, k) or (0, 1) if k == 0.
          int64_t mergeStart = (k == 0) ? 0 : k - 1;
          int64_t mergeEnd = (k == 0) ? 1 : k;

          // Build reassociation: merge mergeStart..mergeEnd into one.
          SmallVector<ReassociationIndices> reassociation;
          for (int64_t d = 0; d < curRank; ++d) {
            if (d == mergeStart) {
              ReassociationIndices group;
              for (int64_t j = mergeStart; j <= mergeEnd; ++j)
                group.push_back(j);
              reassociation.push_back(group);
              d = mergeEnd;
            } else {
              reassociation.push_back({d});
            }
          }

          // Collapse.
          current = tensor::CollapseShapeOp::create(rewriter, loc, current, reassociation);

          // Compute intermediate init shape:
          // - All dims >= k: output size (current broadcast dim + already processed)
          // - Other broadcast dims < k: operand size (still unprocessed, keep 1)
          // - Non-broadcast dims: output size
          SmallVector<int64_t> interShape(outRank);
          for (int64_t d = 0; d < outRank; ++d) {
            if (llvm::is_contained(broadcastDims, d) && d < k) {
              interShape[d] = operandType.getDimSize(d);
            } else {
              interShape[d] = outShape[d];
            }
          }
          auto interType = RankedTensorType::get(interShape, f32Type);
          SmallVector<int64_t> bcDims = {k};
          Value bcInit = tensor::EmptyOp::create(rewriter, loc, interType, /*dynSizes=*/{});
          current = linalg::BroadcastOp::create(rewriter, loc, current, bcInit, bcDims)->getResult(0);
          curRank = cast<RankedTensorType>(current.getType()).getRank();
        }
        return current;
      }

      llvm_unreachable("operand rank > output rank in sf.le");
      return operand;
    };

    Value broadcastLhs = broadcastOperand(op.getLhs(), lhsType, lhsRank);
    Value broadcastRhs = broadcastOperand(op.getRhs(), rhsType, rhsRank);

    // 3. Create linalg.generic with identity affine maps only.
    auto identityMap = AffineMap::getMultiDimIdentityMap(outRank, ctx);
    SmallVector<AffineMap> genericMaps = {identityMap, identityMap, identityMap};
    SmallVector<utils::IteratorType> iterTypes(outRank, utils::IteratorType::parallel);

    auto g = linalg::GenericOp::create(rewriter, loc, outTensorType,
        ValueRange{broadcastLhs, broadcastRhs}, genericInit,
        genericMaps, iterTypes,
        [&](OpBuilder &b, Location bodyLoc, ValueRange args) {
      Value cmp;
      if (isa<IntegerType>(args[0].getType()) || isa<IntegerType>(args[1].getType())) {
        // Integer comparison: use CmpIOp
        auto lhsInt = arith::IndexCastOp::create(b, loc, b.getIndexType(), args[0]);
        auto rhsInt = arith::IndexCastOp::create(b, loc, b.getIndexType(), args[1]);
        cmp = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sle, lhsInt, rhsInt);
      } else {
        // Float comparison
        cmp = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OLE, args[0], args[1]);
      }
      Value result = arith::UIToFPOp::create(b, loc, rewriter.getF32Type(), cmp);
      linalg::YieldOp::create(b, loc, result);
    });

    rewriter.replaceOp(op, g.getResult(0));
    return success();
  }
};

// LogicalAnd -> linalg.generic with explicit linalg.broadcast
//   bool_a = cmp UGT(a, 0.0), bool_b = cmp UGT(b, 0.0)
//   and = andi(bool_a, bool_b)
//   result = uitofp(and) -> f32
// Uses explicit linalg.broadcast + identity-map linalg.generic to avoid
// kDynamic leaks from InferStaticShapeOfOperands.
struct SfLogicalAndOpLowering : public OpRewritePattern<sf::LogicalAndOp> {
  using OpRewritePattern<sf::LogicalAndOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sf::LogicalAndOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type rt = op.getResult().getType();
    if (!isa<ShapedType>(rt)) return failure();

    auto lhsType = cast<RankedTensorType>(op.getLhs().getType());
    auto rhsType = cast<RankedTensorType>(op.getRhs().getType());
    int64_t lhsRank = lhsType.getRank();
    int64_t rhsRank = rhsType.getRank();
    int64_t outRank = std::max(lhsRank, rhsRank);
    MLIRContext *ctx = rewriter.getContext();

    // 1. Compute broadcast output shape (numpy rules)
    SmallVector<int64_t> outShape(outRank, ShapedType::kDynamic);
    SmallVector<Value> dynSizes;
    for (int64_t i = 0; i < outRank; ++i) {
      int64_t lhsDim = (i >= outRank - lhsRank)
          ? lhsType.getDimSize(i - (outRank - lhsRank)) : 1;
      int64_t rhsDim = (i >= outRank - rhsRank)
          ? rhsType.getDimSize(i - (outRank - rhsRank)) : 1;

      bool lhsDynamic = ShapedType::isDynamic(lhsDim);
      bool rhsDynamic = ShapedType::isDynamic(rhsDim);

      if (lhsDynamic || rhsDynamic) {
        outShape[i] = ShapedType::kDynamic;
        if (i >= outRank - lhsRank) {
          int64_t li = i - (outRank - lhsRank);
          if (lhsType.isDynamicDim(li)) {
            dynSizes.push_back(tensor::DimOp::create(rewriter, loc, op.getLhs(), li));
            continue;
          }
        }
        if (i >= outRank - rhsRank) {
          int64_t ri = i - (outRank - rhsRank);
          if (rhsType.isDynamicDim(ri)) {
            dynSizes.push_back(tensor::DimOp::create(rewriter, loc, op.getRhs(), ri));
          }
        }
      } else {
        outShape[i] = std::max(lhsDim, rhsDim);
      }
    }

    auto f32Type = rewriter.getF32Type();
    auto outTensorType = RankedTensorType::get(outShape, f32Type);
    Value genericInit = tensor::EmptyOp::create(rewriter, loc, outTensorType, dynSizes);

    // 2. Broadcast helper (same pattern as SfLeOpLowering)
    auto broadcastOperand = [&](Value operand, RankedTensorType operandType,
                                 int64_t operandRank) -> Value {
      if (operandRank < outRank) {
        SmallVector<int64_t> addedDims;
        for (int64_t i = 0; i < outRank - operandRank; ++i)
          addedDims.push_back(i);
        Value bcInit = tensor::EmptyOp::create(rewriter, loc, outTensorType, dynSizes);
        return linalg::BroadcastOp::create(rewriter, loc, operand, bcInit, addedDims)->getResult(0);
      }

      if (operandRank == outRank) {
        SmallVector<int64_t> broadcastDims;
        for (int64_t i = 0; i < outRank; ++i) {
          int64_t opSize = operandType.getDimSize(i);
          if (opSize == 1 && !ShapedType::isDynamic(outShape[i]) && outShape[i] > 1)
            broadcastDims.push_back(i);
        }

        if (broadcastDims.empty())
          return operand;

        Value current = operand;
        int64_t curRank = outRank;

        for (auto k : llvm::reverse(broadcastDims)) {
          int64_t mergeStart = (k == 0) ? 0 : k - 1;
          int64_t mergeEnd = (k == 0) ? 1 : k;

          SmallVector<ReassociationIndices> reassociation;
          for (int64_t d = 0; d < curRank; ++d) {
            if (d == mergeStart) {
              ReassociationIndices group;
              for (int64_t j = mergeStart; j <= mergeEnd; ++j)
                group.push_back(j);
              reassociation.push_back(group);
              d = mergeEnd;
            } else {
              reassociation.push_back({d});
            }
          }

          current = tensor::CollapseShapeOp::create(rewriter, loc, current, reassociation);

          // Compute intermediate init shape:
          // - All dims >= k: output size (current broadcast dim + already processed)
          // - Other broadcast dims < k: operand size (still unprocessed, keep 1)
          // - Non-broadcast dims: output size
          SmallVector<int64_t> interShape(outRank);
          for (int64_t d = 0; d < outRank; ++d) {
            if (llvm::is_contained(broadcastDims, d) && d < k) {
              interShape[d] = operandType.getDimSize(d);
            } else {
              interShape[d] = outShape[d];
            }
          }
          auto interType = RankedTensorType::get(interShape, f32Type);
          SmallVector<int64_t> bcDims = {k};
          Value bcInit = tensor::EmptyOp::create(rewriter, loc, interType, /*dynSizes=*/{});
          current = linalg::BroadcastOp::create(rewriter, loc, current, bcInit, bcDims)->getResult(0);
          curRank = cast<RankedTensorType>(current.getType()).getRank();
        }
        return current;
      }

      llvm_unreachable("operand rank > output rank in sf.logical_and");
      return operand;
    };

    Value broadcastLhs = broadcastOperand(op.getLhs(), lhsType, lhsRank);
    Value broadcastRhs = broadcastOperand(op.getRhs(), rhsType, rhsRank);

    // 3. Identity-map linalg.generic
    auto identityMap = AffineMap::getMultiDimIdentityMap(outRank, ctx);
    SmallVector<AffineMap> genericMaps = {identityMap, identityMap, identityMap};
    SmallVector<utils::IteratorType> iterTypes(outRank, utils::IteratorType::parallel);

    auto g = linalg::GenericOp::create(rewriter, loc, outTensorType,
        ValueRange{broadcastLhs, broadcastRhs}, genericInit,
        genericMaps, iterTypes,
        [&](OpBuilder &b, Location bodyLoc, ValueRange args) {
      // Both args are 0.0 (false) or 1.0 (true). Multiply gives AND.
      Value result = arith::MulFOp::create(b, bodyLoc, args[0], args[1]);
      linalg::YieldOp::create(b, bodyLoc, result);
    });

    rewriter.replaceOp(op, g.getResult(0));
    return success();
  }
};

} // namespace

namespace mlir::sf {
void registerComparePatterns(RewritePatternSet &patterns) {
  patterns.add<SfLeOpLowering, SfLogicalAndOpLowering>(patterns.getContext());
}
} // namespace mlir::sf
