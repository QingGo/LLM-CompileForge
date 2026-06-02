#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "Sf/SfDialect.h"
#include "Sf/SfPasses.h"

using namespace mlir;
using namespace llvm;

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "Sf/SfPasses.h.inc"

int main(int argc, char **argv) {
  registerSfPasses();

  DialectRegistry registry;
  registry.insert<sf::SfDialect>();
  registry.insert<func::FuncDialect, arith::ArithDialect,
                  math::MathDialect, tensor::TensorDialect,
                  linalg::LinalgDialect, scf::SCFDialect>();

  return asMainReturnCode(
      MlirOptMain(argc, argv, "sf-opt - MLIR optimizer for sf-dialect\n",
                   registry));
}
