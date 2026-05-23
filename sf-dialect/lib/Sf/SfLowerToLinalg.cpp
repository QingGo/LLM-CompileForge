#define NDEBUG
#include "SfLoweringHelpers.h"
#include "SfLoweringPatterns.h"
#include "Sf/SfDialect.h"
#include "Sf/SfOps.h"
#include "Sf/SfPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "sf-lower-to-linalg"

using namespace mlir;
using namespace mlir::sf;

namespace {
#include "Sf/SfLoweringPatterns.inc"
} // namespace

//===----------------------------------------------------------------------===//
// SfLowerToLinalgPass — lower all remaining sf ops via conversion
//===----------------------------------------------------------------------===//
// Assumes sf-promote-weights has already run. All weight/constant ops are
// already converted to func arguments at this point.
//===----------------------------------------------------------------------===//

namespace {
struct SfLowerToLinalgPass
    : public PassWrapper<SfLowerToLinalgPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SfLowerToLinalgPass)

  StringRef getArgument() const final { return "sf-lower-to-linalg"; }
  StringRef getDescription() const final {
    return "Lower remaining sf dialect ops to linalg/arith/math/tensor/scf";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, arith::ArithDialect,
                    math::MathDialect, tensor::TensorDialect,
                    func::FuncDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    llvm::errs() << "  [sf-lower-to-linalg] starting\n";

    // Quick check: no weight/constant should remain
    bool hasWeight = false;
    getOperation()->walk([&](Operation *op) {
      if (isa<sf::WeightOp>(op) || isa<sf::ConstantOp>(op)) {
        op->emitError("weight/constant found — sf-promote-weights must run first");
        hasWeight = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (hasWeight) {
      llvm::errs() << "  [sf-lower-to-linalg] ERROR: weights remain\n";
      signalPassFailure();
      return;
    }

    // Count sf ops before conversion
    int64_t sfCount = 0;
    getOperation()->walk([&](Operation *op) {
      if (op->getDialect() &&
          isa<sf::SfDialect>(op->getDialect()))
        ++sfCount;
    });
    llvm::errs() << "  [sf-lower-to-linalg] found " << sfCount
                 << " sf ops to lower\n";
    if (sfCount == 0) {
      llvm::errs() << "  [sf-lower-to-linalg] nothing to do\n";
      return;
    }

    // Collect functions that contain sf ops
    SmallVector<func::FuncOp> targetFuncs;
    getOperation()->walk([&](func::FuncOp funcOp) {
      bool hasSf = false;
      funcOp.walk([&](Operation *op) {
        if (op->getDialect() && isa<sf::SfDialect>(op->getDialect())) {
          hasSf = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (hasSf)
        targetFuncs.push_back(funcOp);
    });

    // Lower per-function using greedy pattern rewriter (avoids
    // dialect-conversion framework worklist divergence at scale).
    for (auto func : targetFuncs) {
      RewritePatternSet patterns(&getContext());
      registerActivationPatterns(patterns);
      registerMatmulPatterns(patterns);
      registerShapePatterns(patterns);
      registerAttentionPatterns(patterns);
      registerNormalizationPatterns(patterns);
      registerGenOpsPatterns(patterns);
      registerSeqOpsPatterns(patterns);
      registerComparePatterns(patterns);
      registerReducePatterns(patterns);
      populateWithGenerated(patterns);

      int64_t sfBefore = 0;
      func.walk([&](Operation *op) {
        if (op->getDialect() && isa<sf::SfDialect>(op->getDialect())) ++sfBefore;
      });
      int64_t bodyOps = 0;
      func.walk([&](Operation *) { ++bodyOps; });
      llvm::errs() << "  [sf-lower-to-linalg] lowering func '" << func.getName()
                   << "' (" << bodyOps << " body ops, " << sfBefore << " sf ops)\n";
      fprintf(stderr, "  [VERIFY] pre-walk about to start\n");
      // Pre-walk: fix result types for ops whose dyn_shape operands imply a
      // higher rank than the declared result type (e.g. sf.ones_like with
      // 2 dyn_shape operands but tensor<f32> result). Without this fix, the
      // greedy rewriter's LIFO processing would lower type-changing ops after
      // their users, creating type mismatches in the lowered IR.
      llvm::errs() << "  [sf-lower-to-linalg] pre-walk type fixing...\n";
      func.walk([&](Operation *op) {
        if (auto onesLike = dyn_cast<sf::OnesLikeOp>(op)) {
          // Total dimensions = input (scalar tensor containing dim 0's size)
          // + dyn_shape operands (additional dim sizes).
          size_t numDims = 1 + onesLike.getDynShape().size();
          auto resultTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
          if (resultTy && resultTy.getRank() < (int64_t)numDims) {
            auto newTy = RankedTensorType::get(
                SmallVector<int64_t>(numDims, ShapedType::kDynamic),
                resultTy.getElementType());
            op->getResult(0).setType(newTy);
            llvm::errs() << "  [sf-lower-to-linalg] fix ones_like type: "
                         << resultTy << " -> " << newTy << "\n";
          }
        }
        if (auto arangeOp = dyn_cast<sf::ArangeOp>(op)) {
          size_t numDims = 1 + arangeOp.getDynShape().size();
          auto resultTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
          if (resultTy && resultTy.getRank() < (int64_t)numDims) {
            auto newTy = RankedTensorType::get(
                SmallVector<int64_t>(numDims, ShapedType::kDynamic),
                resultTy.getElementType());
            op->getResult(0).setType(newTy);
            llvm::errs() << "  [sf-lower-to-linalg] fix arange type: "
                         << resultTy << " -> " << newTy << "\n";
          }
        }
      });
      LogicalResult result = applyPatternsGreedily(func, std::move(patterns));
      if (failed(result)) {
        llvm::errs() << "  [sf-lower-to-linalg] greedy rewriter did not converge for '"
                     << func.getName() << "'\n";
        signalPassFailure();
      }
      llvm::errs() << "  [sf-lower-to-linalg] after lowering func '" << func.getName() << "'\n";
    }

    // Post-conversion check: report remaining sf ops with their names
    int64_t remaining = 0;
    getOperation()->walk([&](Operation *op) {
      if (op->getDialect() && isa<sf::SfDialect>(op->getDialect())) {
        if (remaining == 0)
          llvm::errs() << "  [sf-lower-to-linalg] remaining sf ops:\n";
        llvm::errs() << "    " << op->getName().getStringRef() << "\n";
        ++remaining;
      }
    });
    if (remaining > 0) {
      llvm::errs() << "  [sf-lower-to-linalg] " << remaining
                   << " sf ops remain unconverted\n";
      signalPassFailure();
    } else {
      llvm::errs() << "  [sf-lower-to-linalg] all sf ops converted\n";
    }
    llvm::errs() << "  [sf-lower-to-linalg] done\n";
  }
};
} // namespace

std::unique_ptr<Pass> mlir::sf::createSfLowerToLinalg() {
  return std::make_unique<SfLowerToLinalgPass>();
}
