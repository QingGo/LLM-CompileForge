# Install script for directory: /Users/zeng/code/LLM-CompileForge/sf-dialect/python

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "mlir-python-sources" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/src/python/SfPythonSources.sf/dialects" TYPE FILE FILES "/Users/zeng/code/LLM-CompileForge/sf-dialect/python/mlir_sf/dialects/sf.py")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "mlir-python-sources" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/src/python/SfPythonSources.sf/_mlir_libs" TYPE FILE FILES "/Users/zeng/code/LLM-CompileForge/sf-dialect/python/mlir_sf/_mlir_libs/__init__.py")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "mlir-python-sources" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/src/python/SfPythonSources.sf/_mlir_libs/_sfDialectsNanobind" TYPE FILE FILES "/Users/zeng/code/LLM-CompileForge/sf-dialect/python/mlir_sf/_mlir_libs/_sfDialectsNanobind/py.typed")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "mlir-python-sources" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/src/python/SfPythonSources.sf.ops_gen/dialects" TYPE FILE FILES "/Users/zeng/code/LLM-CompileForge/sf-dialect/python/dialects/_sf_ops_gen.py")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "SfPythonModules" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python_packages/sf/mlir_sf/dialects" TYPE FILE FILES "/Users/zeng/code/LLM-CompileForge/sf-dialect/python/mlir_sf/dialects/sf.py")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "SfPythonModules" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python_packages/sf/mlir_sf/_mlir_libs" TYPE FILE FILES "/Users/zeng/code/LLM-CompileForge/sf-dialect/python/mlir_sf/_mlir_libs/__init__.py")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "SfPythonModules" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python_packages/sf/mlir_sf/_mlir_libs/_sfDialectsNanobind" TYPE FILE FILES "/Users/zeng/code/LLM-CompileForge/sf-dialect/python/mlir_sf/_mlir_libs/_sfDialectsNanobind/py.typed")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "SfPythonModules" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python_packages/sf/mlir_sf/dialects" TYPE FILE FILES "/Users/zeng/code/LLM-CompileForge/sf-dialect/python/dialects/_sf_ops_gen.py")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/Users/zeng/code/LLM-CompileForge/sf-dialect/python/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
