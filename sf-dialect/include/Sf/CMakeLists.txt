set(LLVM_TARGET_DEFINITIONS SfOps.td)
mlir_tablegen(SfOps.h.inc -gen-op-decls)
mlir_tablegen(SfOps.cpp.inc -gen-op-defs)
add_mlir_dialect(SfOps sf)
add_public_tablegen_target(SfOpsIncGen)

set(LLVM_TARGET_DEFINITIONS SfLoweringPatterns.td)
mlir_tablegen(SfLoweringPatterns.inc -gen-rewriters)
add_public_tablegen_target(SfLoweringPatternsIncGen)

set(LLVM_TARGET_DEFINITIONS SfPasses.td)
mlir_tablegen(SfPasses.h.inc -gen-pass-decls --name Sf)
add_public_tablegen_target(SfPassesIncGen)

# Convenience target: regenerate all TableGen .inc files without full build
add_custom_target(tablegen DEPENDS SfOpsIncGen SfPassesIncGen SfLoweringPatternsIncGen)
