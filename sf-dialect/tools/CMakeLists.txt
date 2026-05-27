add_subdirectory(sf-opt)

# Dialect plugin for mlir-opt: loads sf dialect without Python.
# Usage: mlir-opt --load-dialect-plugin=${CMAKE_BINARY_DIR}/tools/libSfDialectPlugin.dylib
# The plugin exports mlirGetDialectPluginInfo() which mlir-opt's plugin loader
# finds via dlopen. MLIR symbols are resolved at runtime from the mlir-opt
# host process (no explicit link to MLIR needed).
if(TARGET SfDialect)
  add_library(SfDialectPlugin MODULE DialectPlugin.cpp)
  target_include_directories(SfDialectPlugin PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include
  )
  target_link_libraries(SfDialectPlugin PRIVATE SfDialect SfCAPI)
  if(APPLE)
    target_link_options(SfDialectPlugin PRIVATE "LINKER:-undefined,dynamic_lookup")
  endif()
endif()
