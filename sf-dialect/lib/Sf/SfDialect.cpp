#include "Sf/SfDialect.h"
#include "Sf/SfOps.h"

using namespace mlir;
using namespace mlir::sf;

#include "Sf/SfOpsDialect.cpp.inc"

void SfDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Sf/SfOps.cpp.inc"
      >();
}
