#ifndef SF_PASSES_H
#define SF_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
namespace sf {

std::unique_ptr<mlir::Pass> createSfPromoteWeights();
std::unique_ptr<mlir::Pass> createSfLowerToLinalg();
std::unique_ptr<mlir::Pass> createSfStripGEPNuw();
std::unique_ptr<mlir::Pass> createSfFuseSiluPass();
std::unique_ptr<mlir::Pass> createSfFuseRmsNormPass();
std::unique_ptr<mlir::Pass> createSfFuseQKVPass();
std::unique_ptr<mlir::Pass> createSfFuseAttentionPass();

} // namespace sf
} // namespace mlir

#endif // SF_PASSES_H
