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

set(SfPythonModules_ROOT_PREFIX "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}")

add_mlir_python_modules(SfPythonModules
  ROOT_PREFIX "${SfPythonModules_ROOT_PREFIX}"
  INSTALL_PREFIX "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
  DECLARED_SOURCES
    SfPythonSources
)
