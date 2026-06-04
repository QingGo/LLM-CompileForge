# ruff: noqa: E501 — long lines in MLIR transform script f-strings

"""External compiler tool orchestration utilities.

Manages subprocess calls to mlir-translate, llc, cc, and C compiler
for LLVM compilation and linking.  All subprocess.run() calls include
explicit timeout to prevent hung processes.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from compiler.backend.fixups import (
    _fixup_unrealized_casts_pass,
)
from compiler.exceptions import (
    LinkError,
    LLCError,
    MissingBindingsError,
    MLIRTranslateError,
    ToolNotFoundError,
)

_log = logging.getLogger(__name__)


def _setup_mlir_path() -> None:
    _mlir_pkg = Path(__file__).resolve().parent.parent.parent / "mlir_binding" / "mlir_package"
    if _mlir_pkg.is_dir() and str(_mlir_pkg) not in sys.path:
        sys.path.insert(0, str(_mlir_pkg))

    # Add sf-dialect Python bindings for C++ passes
    # Check both in-source build and build/ subdirectory build
    _sf_base = Path(__file__).resolve().parent.parent.parent / "sf-dialect"
    for _sf_candidate in [_sf_base / "python_packages" / "sf", _sf_base / "build" / "python_packages" / "sf"]:
        if _sf_candidate.is_dir() and str(_sf_candidate) not in sys.path:
            sys.path.insert(0, str(_sf_candidate))
            break


def _has_bindings() -> bool:
    _setup_mlir_path()
    try:
        import mlir.ir  # noqa: F401
        return True
    except ImportError:
        return False


def _find_mlir_tool(name: str) -> str:
    """Locate an LLVM/MLIR binary (llc, mlir-translate, etc.).

    Checks (in order):
      1. Our build in llvm-project/build/bin/ (preferred — compiled from source)
      2. SERVE_FORGE_LLVM_BIN environment variable
      3. {name} on PATH (may pick up Homebrew version — logs warning)
      4. Common Homebrew paths (fallback)
    """
    # 1. Our build (preferred) — compiled from source with all translation interfaces
    our_build = (
        Path(__file__).resolve().parent.parent.parent
        / "llvm-project" / "build" / "bin" / name
    )
    if our_build.is_file() and os.access(str(our_build), os.X_OK):
        tool_path = str(our_build)
        _log.info("Using %s: %s (LLVM build dir)", name, tool_path)
        return tool_path

    llvm_bin_dir = os.environ.get("SERVE_FORGE_LLVM_BIN", "")
    if llvm_bin_dir:
        candidate = os.path.join(llvm_bin_dir, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            _log.info("Using %s: %s (SERVE_FORGE_LLVM_BIN)", name, candidate)
            return candidate

    path = shutil.which(name)
    if path:
        _log.warning(
            "Using %s from PATH (%s). This may pick up Homebrew version instead "
            "of compiled LLVM build. Prefer: %s",
            name, path, our_build,
        )
        return path

    for prefix in [
        "/usr/local/opt/llvm/bin",
        "/usr/local/opt/llvm@19/bin",
        "/usr/local/opt/llvm@18/bin",
        "/usr/local/opt/llvm/bin",
        "/opt/homebrew/opt/llvm/bin",
    ]:
        candidate = os.path.join(prefix, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            _log.info("Using %s: %s (Homebrew fallback)", name, candidate)
            return candidate

    raise ToolNotFoundError(
        name,
        "Install LLVM (brew install llvm), "
        "or set SERVE_FORGE_LLVM_BIN to the llvm/bin directory.",
    )


def _find_llc() -> str:
    """Locate the llc binary."""
    env_llc = os.environ.get("SERVE_FORGE_LLC")
    if env_llc and shutil.which(env_llc):
        return env_llc
    return _find_mlir_tool("llc")


def _find_mlir_translate() -> str:
    """Locate the mlir-translate binary."""
    return _find_mlir_tool("mlir-translate")


def _find_cc() -> str:
    """Locate the system C compiler for linking object files.

    Checks (in order):
      1. SERVE_FORGE_CC environment variable
      2. 'cc' on PATH
    """
    env_cc = os.environ.get("SERVE_FORGE_CC")
    if env_cc and shutil.which(env_cc):
        return env_cc

    path_cc = shutil.which("cc")
    if path_cc:
        return path_cc

    raise ToolNotFoundError(
        "cc",
        "Set SERVE_FORGE_CC environment variable.",
    )


def mlir_module_to_llvm_ir(ir_module: Any) -> str:
    """Translate an MLIR module (LLVM dialect) to LLVM IR text.

    Uses ``mlir-translate --mlir-to-llvmir`` under the hood, applying
    version-compatibility fixups when the host llvm-tools are older than
    the MLIR bindings (e.g. LLVM 22 bindings + LLVM 20 tools).
    """

    mlir_translate = _find_mlir_translate()
    mlir_text = str(ir_module)

    # Fix up unrealized_conversion_cast ops that cannot be reconciled by
    # MLIR passes alone.  These casts are structurally irreconcilable
    # (5 bare ptr/size values → struct) and remain after finalize-memref-to-llvm
    # + convert-func-to-llvm even with multiple reconcile-unrealized-casts passes.
    # The MLIR-bindings-based pass replaces each with llvm.mlir.undef +
    # llvm.insertvalue chain directly on the ir.Module before translation.
    _fixup_unrealized_casts_pass(ir_module)
    mlir_text = str(ir_module)

    with tempfile.TemporaryDirectory() as td:
        mlir_path = os.path.join(td, "module.mlir")
        with open(mlir_path, "w") as f:
            f.write(mlir_text)

        result = subprocess.run(
            [mlir_translate, "--allow-unregistered-dialect", "--mlir-to-llvmir", mlir_path],
            capture_output=True, text=True,
            timeout=90,
        )

        if result.returncode != 0:
            raise MLIRTranslateError(result.returncode, result.stderr)

        return result.stdout


def emit_llvm_ir_to_file(ir_module: Any, path: str) -> None:
    """Translate an MLIR module to LLVM IR and write it to a .ll file.

    Args:
        ir_module: ir.Module with LLVM dialect (after lowering pipeline).
        path: Output file path (e.g. '/tmp/model.ll').
    """
    ll_text = mlir_module_to_llvm_ir(ir_module)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(ll_text)


def llc_compile(
    ll_file: str,
    arch: str = "",
    output: str | None = None,
    opt_level: int = 3,
) -> str:
    """Invoke llc to compile a .ll file to a .o object file.

    Args:
        ll_file: Path to the .ll input file.
        arch: Target architecture. Empty string means use llc default.
        output: Output .o path. Defaults to ll_file with .o extension.
        opt_level: LLVM optimization level (0-3).

    Returns:
        Path to the generated .o file.

    Raises:
        RuntimeError: If llc is not found or compilation fails.
    """
    llc_bin = _find_llc()

    if output is None:
        output = str(Path(ll_file).with_suffix(".o"))

    cmd = [
        llc_bin,
        "-filetype=obj",
        f"-O{opt_level}",
        ll_file,
        "-o",
        output,
    ]
    if arch and arch != "native":
        cmd.insert(-2, f"-march={arch}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        raise LLCError(result.returncode, result.stderr)

    if not os.path.isfile(output) or os.path.getsize(output) == 0:
        raise RuntimeError(f"llc produced empty or missing .o file: {output}")

    return output


def jit_compile_and_run(ir_module: Any, func_name: str = "main") -> Any:
    """JIT-compile an LLVM IR module and return a callable wrapper.

    Args:
        ir_module: ir.Module with LLVM dialect.
        func_name: Name of the function to look up.

    Returns:
        An ExecutionEngine that can invoke the function.
    """
    if not _has_bindings():
        raise MissingBindingsError()

    from mlir.execution_engine import ExecutionEngine


    engine = ExecutionEngine(ir_module, opt_level=2)
    return engine


def link_dylib(
    obj_files: list[str],
    output: str,
) -> str:
    """Link one or more .o object files into a .dylib shared library.

    Uses the system C compiler with ``-shared`` to produce a dynamically
    loadable library that the Rust runtime can open via ``dlopen`` /
    ``libloading``.  Unresolved symbols (e.g. ``_malloc``) are left to be
    resolved by the host process at load time.

    Args:
        obj_files: Paths to .o files.
        output: Output .dylib path.

    Returns:
        Path to the generated .dylib file.

    Raises:
        RuntimeError: If the linker fails.
    """
    cc_bin = _find_cc()

    # Locate MLIR runner_utils library for memrefCopy and other runtime helpers
    # needed by bufferized dynamic-shaped memref ops.
    build_lib = (
        Path(__file__).resolve().parent.parent.parent.parent
        / "llvm-project" / "build" / "lib"
    )
    runner_lib = build_lib / "libmlir_c_runner_utils.dylib"

    cmd = [
        cc_bin,
        "-shared",
        "-o", output,
        *obj_files,
    ]
    if runner_lib.is_file():
        cmd.extend(["-L", str(build_lib), "-lmlir_c_runner_utils"])
        cmd.append(f"-Wl,-rpath,{build_lib}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise LinkError(result.returncode, result.stderr)

    if not os.path.isfile(output) or os.path.getsize(output) == 0:
        raise RuntimeError(f"link_dylib produced empty or missing file: {output}")

    return output


def compile_mlir_to_dylib(
    ir_module: Any,
    output: str,
    arch: str = "native",
    opt_level: int = 0,
) -> str:
    """Full AOT compilation: MLIR (LLVM dialect) → .ll → .o → .dylib.

    This produces a shared library that the Rust runtime can load to
    execute per-function compute kernels.

    Args:
        ir_module: ir.Module with LLVM dialect (after lowering pipeline).
        output: Output .dylib path.
        arch: Target architecture.
        opt_level: LLVM optimization level (0-3).

    Returns:
        Path to the generated .dylib file.
    """
    with tempfile.TemporaryDirectory() as td:
        ll_path = os.path.join(td, "module.ll")
        emit_llvm_ir_to_file(ir_module, ll_path)
        obj_path = llc_compile(ll_path, arch=arch, opt_level=opt_level)
        return link_dylib([obj_path], output)


def compile_module_to_dylib(
    ir_module: Any,
    output_dir: str,
    model_name: str = "model",
    arch: str = "native",
    opt_level: int = 0,
) -> str:
    """Compile a lowered MLIR module into the outputs/compiled/ artifacts directory.

    Produces:
    * ``<output_dir>/<model_name>.ll``   — LLVM IR text
    * ``<output_dir>/<model_name>.o``    — object file
    * ``<output_dir>/lib<model_name>.dylib`` — shared library

    If ``<output_dir>/constants.bin`` exists, its contents are embedded
    in the dylib as ``serveforge_constants_data`` / ``serveforge_constants_size``
    symbols, accessible by the Rust runtime via ``libloading``.

    Returns the .dylib path.
    """
    os.makedirs(output_dir, exist_ok=True)
    dylib_path = os.path.join(output_dir, f"lib{model_name}.dylib")
    _compile_mlir_to_dylib_with_constants(ir_module, output_dir, dylib_path, arch, opt_level)
    return dylib_path


def _compile_mlir_to_dylib_with_constants(
    ir_module: Any,
    work_dir: str,
    dylib_path: str,
    arch: str,
    opt_level: int,
) -> None:
    """Internal: compile MLIR → .ll → .o → .dylib, embedding constants.bin."""
    const_bin_path = os.path.join(work_dir, "constants.bin")

    obj_files: list[str] = []

    with tempfile.TemporaryDirectory() as td:
        ll_path = os.path.join(td, "module.ll")
        emit_llvm_ir_to_file(ir_module, ll_path)
        # Save a copy of LLVM IR in output dir for debugging
        import shutil
        debug_ll = os.path.join(work_dir, "model.ll")
        shutil.copy2(ll_path, debug_ll)
        obj_path = llc_compile(ll_path, arch=arch, opt_level=opt_level)
        obj_files.append(obj_path)

        if os.path.isfile(const_bin_path):
            obj_files.append(
                _compile_embedded_data(const_bin_path, td)
            )

        obj_files.append(_compile_serveforge_free(td))

        link_dylib(obj_files, dylib_path)


def _compile_embedded_data(bin_path: str, work_dir: str) -> str:
    """Create a .o file with embedded binary data from a file.

    Generates a C source with the binary as a ``const uint8_t`` array,
    compiles it to ``.o``, and returns the path.
    """
    import textwrap

    with open(bin_path, "rb") as f:
        data = f.read()

    hex_lines = []
    for i in range(0, len(data), 12):
        chunk = data[i : i + 12]
        hex_lines.append(", ".join(f"0x{b:02X}" for b in chunk))

    c_source = textwrap.dedent(f"""\
    #include <stdint.h>
    const uint8_t serveforge_constants_data[{len(data)}] = {{
        {",".join(hex_lines)}
    }};
    const uint64_t serveforge_constants_size = {len(data)};
    """)

    c_path = os.path.join(work_dir, "serveforge_constants.c")
    o_path = os.path.join(work_dir, "serveforge_constants.o")
    with open(c_path, "w") as f:
        f.write(c_source)

    cc_bin = _find_cc()
    result = subprocess.run(
        [cc_bin, "-c", c_path, "-o", o_path],
        capture_output=True, text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to compile embedded constants (exit {result.returncode}):\n"
            f"{result.stderr[:2000]}"
        )

    return o_path


def _compile_serveforge_free(work_dir: str) -> str:
    """Create a .o file exporting ``serveforge_free`` for the dylib.

    The dylib allocates output buffers via ``malloc``.  This stub
    exports the matching ``free`` so the Rust runtime can release
    dylib-allocated memory via the dylib's own allocator.
    """
    c_path = os.path.join(work_dir, "serveforge_free.c")
    o_path = os.path.join(work_dir, "serveforge_free.o")
    with open(c_path, "w") as f:
        f.write("""\
#include <stdlib.h>
void serveforge_free(void* ptr) { free(ptr); }
""")
    cc_bin = _find_cc()
    result = subprocess.run(
        [cc_bin, "-c", c_path, "-o", o_path],
        capture_output=True, text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to compile serveforge_free stub (exit {result.returncode}):\n"
            f"{result.stderr[:2000]}"
        )
    return o_path


def _patch_transformers_torch() -> None:
    import torch
    import transformers.utils.generic as _generic
    import transformers.utils.import_utils as _iu
    _iu._torch_available = True  # type: ignore[attr-defined]
    _iu._torch_version = torch.__version__  # type: ignore[attr-defined]
    _generic._torch_pytree = torch.utils._pytree  # type: ignore[attr-defined]
    def _flatten(output: Any) -> tuple[list[Any], list[Any]]:
        return list(output.values()), list(output.keys())
    def _unflatten(values: list[Any], context: Any, output_type: Any = None) -> Any:
        return (output_type or type(context[0]))(**dict(zip(context, values, strict=False)))
    _generic._model_output_flatten = _flatten
    _generic._model_output_unflatten = _unflatten  # type: ignore[assignment]


def _short_shape(shape: Any) -> str:
    return "[" + ", ".join(str(s) for s in shape) + "]"
