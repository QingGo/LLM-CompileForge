set(LLVM_TARGET_DEFINITIONS SfOps.td)
mlir_tablegen(SfOps.h.inc -gen-op-decls)
mlir_tablegen(SfOps.cpp.inc -gen-op-defs)
add_mlir_dialect(SfOps sf)
add_public_tablegen_target(SfOpsIncGen)

set(LLVM_TARGET_DEFINITIONS SfPasses.td)
mlir_tablegen(SfPasses.h.inc -gen-pass-decls --name Sf)
add_public_tablegen_target(SfPassesIncGen)
