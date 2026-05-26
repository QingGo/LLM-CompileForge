# sf-opt: standalone mlir-opt-like tool for sf-dialect lowering patterns.
#
# NOTE: Requires MLIROptLib (from the upstream mlir-opt build).
# Build with: cd llvm-project/build && ninja mlir-opt first.

set(LIBS
  MLIROptLib
  MLIRIR
  MLIRSupport
  SfDialect
)

if(TARGET MLIROptLib)
  add_mlir_tool(sf-opt sf-opt.cpp)
  llvm_update_compile_flags(sf-opt)
  target_link_libraries(sf-opt PRIVATE ${LIBS})
  mlir_check_all_link_libraries(sf-opt)
else()
  message(WARNING "sf-opt: MLIROptLib not found — run 'make build-mlir-opt' first.")
  add_custom_target(sf-opt
    COMMAND ${CMAKE_COMMAND} -E echo "sf-opt: MLIROptLib not found. Run 'make build-mlir-opt' first."
  )
endif()
