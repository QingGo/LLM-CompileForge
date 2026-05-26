//===- DialectPlugin.cpp - mlir-opt dialect plugin entry point ------------===//
//
// Provides mlirGetDialectPluginInfo() so mlir-opt --load-dialect-plugin
// can load the sf dialect without Python.
//
//===----------------------------------------------------------------------===//

#include "Sf-c/Dialects.h"
#include "Sf/SfDialect.h"
#include "Sf/SfPasses.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/Compiler.h"

extern "C" ::mlir::DialectPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
mlirGetDialectPluginInfo() {
  return {
    MLIR_PLUGIN_API_VERSION,
    "SfDialect",
    "0.1",
    [](::mlir::DialectRegistry *registry) {
      registry->insert<::mlir::sf::SfDialect>();
      ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
        return ::mlir::sf::createSfPromoteWeights();
      });
      ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
        return ::mlir::sf::createSfLowerToLinalg();
      });
      ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
        return ::mlir::sf::createSfStripGEPNuw();
      });
    },
  };
}
