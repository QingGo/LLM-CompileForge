include(AddMLIRPython)

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=${MLIR_PYTHON_PACKAGE_PREFIX}.")

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(SfPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT SfPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_sf"
  TD_FILE dialects/SfOps.td
  SOURCES
    dialects/sf.py
    _mlir_libs/__init__.py
    _mlir_libs/_sfDialectsNanobind/py.typed
  DIALECT_NAME sf)

################################################################################
# Instantiation of all Python modules
################################################################################

################################################################################
# Native C++ extension (nanobind)
################################################################################

# Add Sf dialect sources directly to the extension to avoid static linking
# of SfCAPI (which propagates MLIR core libraries transitively, causing
# duplicate MLIR symbol segfaults). MLIR symbols are resolved at runtime
# from the Python process via -undefined dynamic_lookup.
set(SF_EXTRA_SOURCES
  ../lib/CAPI/Dialects.cpp
  ../lib/Sf/SfDialect.cpp
  ../lib/Sf/SfOps.cpp
  ../lib/Sf/SfLowerToLinalg.cpp
)
declare_mlir_python_extension(SfPythonExtensions.nanobind
  MODULE_NAME _sfDialectsNanobind
  ADD_TO_PARENT SfPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES
    SfExtensionNanobind.cpp
    ${SF_EXTRA_SOURCES}
)

# Ensure tablegen-generated headers are built before extension sources
add_dependencies(SfPythonExtensions.nanobind SfOpsIncGen SfPassesIncGen)

# Sf dialect source files must be compiled with -fno-rtti to match MLIR's
# ABI (MLIR is built with RTTI disabled). The extension main source
# (SfExtensionNanobind.cpp) still needs -frtti for nanobind, but the
# Sf dialect code follows MLIR's convention.
# Remove -frtti and -fexceptions for Sf dialect sources by setting
# per-source compile flags.
set_source_files_properties(
  ${SF_EXTRA_SOURCES}
  PROPERTIES
    COMPILE_FLAGS "-fno-rtti -fno-exceptions"
)

set(SfPythonModules_ROOT_PREFIX "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}")

add_mlir_python_modules(SfPythonModules
  ROOT_PREFIX "${SfPythonModules_ROOT_PREFIX}"
  INSTALL_PREFIX "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
  DECLARED_SOURCES
    SfPythonSources
)

# NOTE: AddMLIRPython.cmake sets -twolevel_namespace (line 933), but
# the global HandleLLVMOptions.cmake sets -flat_namespace -undefined
# dynamic_lookup for MODULE libraries. Our extension needs flat namespace
# to resolve MLIR symbols from the Python process at runtime.
if(APPLE)
  target_link_options(SfPythonModules.extension._sfDialectsNanobind.dso
    PRIVATE "LINKER:-flat_namespace"
  )
endif()
