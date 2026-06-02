#include "Sf/SfPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#define GEN_PASS_DEF_SFACONTRACTVERIFY
#include "Sf/SfPasses.h.inc"

#include "Sf/SfDialect.h"
#include "Sf/SfOps.h"

#define DEBUG_TYPE "sfa-contract-verify"

using namespace mlir;

//===----------------------------------------------------------------------===//
// SfaContractVerifyPass — post-lowering verification for SFA ABI compliance
//===----------------------------------------------------------------------===//
// Runs AFTER sf-lower-to-linalg, before bufferization. Verifies that the
// lowered IR satisfies SFA ABI constraints (SfaMemRef rank 1-4, no residual
// sf.weight ops, emit_c_interface present). Records per-function input
// semantics as module-level attributes for downstream ABI generation.
//
// Verification policy:
//   - sf.weight/sf.constant residual → ERROR + signalPassFailure
//   - Rank out of [1,4]               → WARNING (SfaMemRef supports R1-R4)
//   - Missing emit_c_interface        → WARNING (expected after bufferization)
//===----------------------------------------------------------------------===//

namespace {
struct SfaContractVerifyPass
    : public ::impl::SfaContractVerifyBase<SfaContractVerifyPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SfaContractVerifyPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    bool hasErrors = false;

    SmallVector<Attribute> funcMetaAttrs;

    module.walk([&](func::FuncOp funcOp) {
      auto funcName = funcOp.getSymName();
      llvm::errs() << "  [sfa-contract-verify] checking " << funcName << "\n";

      auto funcType = funcOp.getFunctionType();
      unsigned numInputs = funcType.getInputs().size();

      // ---- 1a: Verify input types (SfaMemRef rank 1-4) ----

      for (unsigned i = 0; i < numInputs; ++i) {
        Type type = funcType.getInput(i);
        auto memrefType = dyn_cast<MemRefType>(type);
        if (memrefType) {
          int64_t rank = memrefType.getRank();
          if (rank < 1 || rank > 4) {
            funcOp.emitWarning()
                << "input #" << i << " has rank " << rank
                << " (SfaMemRef supports rank 1-4 only)";
          }
          continue;
        }
        auto tensorType = dyn_cast<RankedTensorType>(type);
        if (tensorType) {
          int64_t rank = tensorType.getRank();
          if (rank < 1 || rank > 4) {
            funcOp.emitWarning()
                << "input #" << i << " has rank " << rank
                << " (SfaMemRef supports rank 1-4 only)";
          }
          continue;
        }
        // UnrankedTensorType or other non-MemRef type
        funcOp.emitWarning()
            << "input #" << i << " is not MemRefType (pre-bufferization)";
      }

      // ---- 1b: Verify output types ----

      for (unsigned i = 0; i < funcType.getNumResults(); ++i) {
        Type type = funcType.getResult(i);
        auto memrefType = dyn_cast<MemRefType>(type);
        if (memrefType) {
          int64_t rank = memrefType.getRank();
          if (rank < 1 || rank > 4) {
            funcOp.emitWarning()
                << "output #" << i << " has rank " << rank
                << " (SfaMemRef supports rank 1-4 only)";
          }
          continue;
        }
        auto tensorType = dyn_cast<RankedTensorType>(type);
        if (tensorType) {
          int64_t rank = tensorType.getRank();
          if (rank < 1 || rank > 4) {
            funcOp.emitWarning()
                << "output #" << i << " has rank " << rank
                << " (SfaMemRef supports rank 1-4 only)";
          }
          continue;
        }
      }

      // ---- 2: Scan for residual sf.weight / sf.constant ops ----

      funcOp.walk([&](Operation *op) {
        if (isa<sf::WeightOp>(op)) {
          op->emitError("unpromoted sf.weight found in function body");
          hasErrors = true;
          return WalkResult::interrupt();
        }
        if (isa<sf::ConstantOp>(op)) {
          op->emitError("unpromoted sf.constant found in function body");
          hasErrors = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });

      // ---- 3: Check llvm.emit_c_interface attribute ----

      if (!funcOp->hasAttr("llvm.emit_c_interface")) {
        funcOp.emitWarning()
            << "missing llvm.emit_c_interface attribute"
            << " (expected after bufferization; required for dylib export)";
      }

      // ---- 4: Record input semantics as module-level metadata ----

      // Classify inputs: weight args are appended after all original args
      // by SfPromoteWeights. sf.weight_names records the weight names.
      auto weightNamesAttr =
          funcOp->getAttrOfType<ArrayAttr>("sf.weight_names");
      unsigned numWeightArgs = 0;
      if (weightNamesAttr) {
        numWeightArgs = weightNamesAttr.size();
      }

      // Build input_kinds: "global" for original inputs, "weight" for
      // appended weight arguments. In a future phase, SSA edges will be
      // classified by cross-function analysis.
      SmallVector<Attribute> inputKinds;
      unsigned weightStartIdx =
          (numInputs >= numWeightArgs) ? (numInputs - numWeightArgs) : 0;

      for (unsigned i = 0; i < numInputs; ++i) {
        if (i >= weightStartIdx && numWeightArgs > 0) {
          inputKinds.push_back(StringAttr::get(ctx, "weight"));
        } else {
          inputKinds.push_back(StringAttr::get(ctx, "global"));
        }
      }

      // Build per-function metadata dict
      SmallVector<NamedAttribute> funcMeta;
      funcMeta.push_back(
          NamedAttribute(StringAttr::get(ctx, "symbol"),
                         StringAttr::get(ctx, funcName)));
      funcMeta.push_back(
          NamedAttribute(StringAttr::get(ctx, "num_inputs"),
                         IntegerAttr::get(IntegerType::get(ctx, 32),
                                          numInputs)));
      funcMeta.push_back(
          NamedAttribute(StringAttr::get(ctx, "num_outputs"),
                         IntegerAttr::get(IntegerType::get(ctx, 32),
                                          funcType.getNumResults())));
      funcMeta.push_back(
          NamedAttribute(StringAttr::get(ctx, "input_kinds"),
                         ArrayAttr::get(ctx, inputKinds)));

      // Preserve weight names for ABI generation
      if (weightNamesAttr) {
        funcMeta.push_back(
            NamedAttribute(StringAttr::get(ctx, "weight_names"),
                           weightNamesAttr));
      }

      funcMetaAttrs.push_back(DictionaryAttr::get(ctx, funcMeta));
    });

    // ---- Store metadata as module-level attribute ----

    if (!funcMetaAttrs.empty()) {
      module->setAttr("sfa.func_metadata",
                      ArrayAttr::get(ctx, funcMetaAttrs));
    }

    llvm::errs() << "  [sfa-contract-verify] checked " << funcMetaAttrs.size()
                 << " functions\n";

    if (hasErrors) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::sf::createSfaContractVerify() {
  return std::make_unique<SfaContractVerifyPass>();
}
