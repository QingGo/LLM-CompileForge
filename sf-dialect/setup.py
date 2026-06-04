import sysconfig

from nanobind import get_include as nb_get_include
from setuptools import Extension, setup

sf_include = "include"
sf_capi_include = "."
mlir_include = "../llvm-project/mlir/include"
mlir_build_include = "../llvm-project/build/include"
mlir_tools_include = "../llvm-project/build/tools/mlir/include"
llvm_include = "../llvm-project/llvm/include"
llvm_build_include = "../llvm-project/build/include"
python_include = sysconfig.get_paths()["include"]

sf_extension = Extension(
    "mlir_sf._mlir_libs._sfDialectsNanobind",
    sources=["python/SfExtensionNanobind.cpp"],
    include_dirs=[
        sf_include,
        sf_capi_include,
        nb_get_include(),
        python_include,
        mlir_include,
        mlir_build_include,
        mlir_tools_include,
        llvm_include,
        llvm_build_include,
        "build/include",
    ],
    libraries=["SfCAPI", "SfDialect", "MLIRCAPIIR", "MLIRSupport",
               "MLIRIR", "MLIRCAPIRegistration"],
    library_dirs=["build/lib/Sf", "build/lib/CAPI",
                  "../llvm-project/build/lib"],
    extra_compile_args=[
        "-std=c++17",
        "-fno-rtti",
        "-fno-exceptions",
        "-DMLIR_PYTHON_PACKAGE_PREFIX=mlir_sf.",
        "-DNDEBUG",
        "-O3",
    ],
    extra_link_args=[
        "-Wl,-rpath,@loader_path/../../../../llvm-project/build/lib",
    ],
)

setup(
    name="mlir_sf",
    version="0.1.0",
    ext_modules=[sf_extension],
)
