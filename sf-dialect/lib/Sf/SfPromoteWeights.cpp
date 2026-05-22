#define NDEBUG
#include "SfLoweringHelpers.h"
#include "Sf/SfDialect.h"
#include "Sf/SfOps.h"
#include "Sf/SfPasses.h"

#define DEBUG_TYPE "sf-lower-to-linalg"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Pass 1: SfPromoteWeightsPass — weight/constant → func arguments
//===----------------------------------------------------------------------===//
// Must run before sf-lower-to-linalg. Uses collect-then-modify pattern
// (not walk-and-erase) to avoid potential iterator invalidation from
// modifying parent FuncOp types during a walk of its child ops.
//===----------------------------------------------------------------------===//

namespace {
struct SfPromoteWeightsPass
    : public PassWrapper<SfPromoteWeightsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SfPromoteWeightsPass)

  StringRef getArgument() const final { return "sf-promote-weights"; }
  StringRef getDescription() const final {
    return "Promote sf.weight and sf.constant ops to function arguments";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    llvm::errs() << "  [sf-promote-weights] collecting weight ops\n";

    // Phase 1a: collect all weight ops first
    SmallVector<sf::WeightOp> weightOps;
    getOperation()->walk([&](sf::WeightOp op) {
      if (op->template getParentOfType<func::FuncOp>())
        weightOps.push_back(op);
    });

    llvm::errs() << "  [sf-promote-weights] found " << weightOps.size()
                 << " weight ops\n";

    // Phase 1b: promote each weight op (outside the walk)
    unsigned promoted = 0;
    for (auto op : weightOps) {
      auto parentFunc = op->template getParentOfType<func::FuncOp>();
      if (!parentFunc) continue;

      auto resultType = op.getResult().getType();
      if (!isa<RankedTensorType>(resultType)) continue;

      auto loc = op.getLoc();
      Block &entry = parentFunc.getBody().front();
      Value newArg = entry.insertArgument(
          entry.getNumArguments(), resultType, loc);

      // Collect weight name for downstream verification
      if (auto nameAttr = op->getAttrOfType<StringAttr>("name")) {
        auto existing =
            parentFunc->getAttrOfType<ArrayAttr>("sf.weight_names");
        SmallVector<Attribute> names;
        if (existing)
          names.assign(existing.begin(), existing.end());
        names.push_back(nameAttr);
        parentFunc->setAttr("sf.weight_names",
                            ArrayAttr::get(&getContext(), names));
      }

      auto origType = parentFunc.getFunctionType();
      SmallVector<Type> newInputs(origType.getInputs());
      newInputs.push_back(resultType);
      parentFunc.setType(
          FunctionType::get(&getContext(), newInputs, origType.getResults()));

      op.replaceAllUsesWith(newArg);
      op.erase();
      ++promoted;
    }

    LLVM_DEBUG(llvm::dbgs() << "[sf-promote-weights] Promoted "
                            << promoted << " weights\n");

    // Phase 2a: collect all constant ops
    SmallVector<sf::ConstantOp> constOps;
    getOperation()->walk([&](sf::ConstantOp op) {
      if (op->template getParentOfType<func::FuncOp>())
        constOps.push_back(op);
    });

    LLVM_DEBUG(llvm::dbgs() << "[sf-promote-weights] Found "
                            << constOps.size() << " constant ops\n");

    // Phase 2b: promote each constant op
    for (auto op : constOps) {
      auto parentFunc = op->template getParentOfType<func::FuncOp>();
      if (!parentFunc) continue;

      auto resultType = op.getResult().getType();
      if (!isa<RankedTensorType>(resultType)) continue;

      auto loc = op.getLoc();
      Block &entry = parentFunc.getBody().front();
      Value newArg = entry.insertArgument(
          entry.getNumArguments(), resultType, loc);

      auto origType = parentFunc.getFunctionType();
      SmallVector<Type> newInputs(origType.getInputs());
      newInputs.push_back(resultType);
      parentFunc.setType(
          FunctionType::get(&getContext(), newInputs, origType.getResults()));

      op.replaceAllUsesWith(newArg);
      op.erase();
    }

    LLVM_DEBUG(llvm::dbgs() << "[sf-promote-weights] Promoted "
                            << promoted << " weight + "
                            << constOps.size() << " constant ops\n");

    llvm::errs() << "  [sf-promote-weights] done, promoted " << promoted << " + "
                 << constOps.size() << " constants\n";

    // Phase 3: verify no remaining weight/constant ops
    bool hasRemaining = false;
    getOperation()->walk([&](Operation *op) {
      if (isa<sf::WeightOp>(op) || isa<sf::ConstantOp>(op)) {
        op->emitError("weight/constant not promoted");
        hasRemaining = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (hasRemaining) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::sf::createSfPromoteWeights() {
  return std::make_unique<SfPromoteWeightsPass>();
}
