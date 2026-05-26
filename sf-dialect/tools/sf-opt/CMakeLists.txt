# sf-opt: standalone mlir-opt-like tool for sf-dialect lowering patterns.
#
# NOTE: This tool requires MLIROptLib (from the llvm-project mlir-opt build).
# If the upstream LLVM build doesn't include mlir-opt, this target is skipped.
# To enable: build mlir-opt from llvm-project first, then reconfigure.
#
#   cd llvm-project/build && ninja mlir-opt
#   cd sf-dialect && cmake . ... && ninja sf-opt

# Check if we can locate the MLIROptLib library.
if(NOT TARGET MLIROptLib)
  message(STATUS "sf-opt: MLIROptLib not found — skipping. "
                 "Run 'make build-mlir-opt' first.")
  return()
endif()

set(LIBS
  MLIROptLib
  MLIRIR
  MLIRSupport
  SfDialect
)

add_mlir_tool(sf-opt sf-opt.cpp)
llvm_update_compile_flags(sf-opt)
target_link_libraries(sf-opt PRIVATE ${LIBS})
mlir_check_all_link_libraries(sf-opt)
