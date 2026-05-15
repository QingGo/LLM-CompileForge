#ifndef SF_PASSES_H
#define SF_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
namespace sf {

std::unique_ptr<mlir::Pass> createSfPromoteWeights();
std::unique_ptr<mlir::Pass> createSfLowerToLinalg();

} // namespace sf
} // namespace mlir

#endif // SF_PASSES_H
