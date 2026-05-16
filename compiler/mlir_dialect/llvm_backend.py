"""linalg → LLVM IR lowering pipeline.

Uses standard MLIR passes: one-shot-bufferize → convert-linalg-to-loops
→ lower-affine → convert-scf-to-cf → finalize-memref-to-llvm
→ convert-arith/math/cf/func-to-llvm → reconcile-unrealized-casts.

The pipeline requires that ALL sf dialect ops have been eliminated
before bufferization (sf→linalg lowering must be complete).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _setup_mlir_path() -> None:
    _mlir_pkg = Path(__file__).resolve().parent.parent.parent / "mlir_binding" / "mlir_package"
    if _mlir_pkg.is_dir() and str(_mlir_pkg) not in sys.path:
        sys.path.insert(0, str(_mlir_pkg))


def _has_bindings() -> bool:
    _setup_mlir_path()
    try:
        import mlir.ir  # noqa: F401
        return True
    except ImportError:
        return False


def lower_linalg_to_llvm_ir(ir_module: Any) -> str:
    """Run full linalg→LLVM lowering pipeline on an ir.Module.

    All ops must already be lowered to linalg/arith/math/tensor dialect.
    Any remaining sf.* or other unregistered dialect ops will cause
    bufferization failures.

    Returns LLVM IR text.
    """
    if not _has_bindings():
        raise RuntimeError("MLIR Python bindings not available")

    import mlir.ir as ir
    import mlir.passmanager as pm

    ctx = ir_module.operation.context
    ctx.allow_unregistered_dialects = True

    with ir.Location.unknown(ctx):
        # sf.weight/sf.constant are already promoted to func.func tensor arguments
        # by the C++ sf-lower-to-linalg pass. No Python-level promotion needed.

        # Run bufferization and linalg/math lowering first
        pipeline_pre = (
            "builtin.module("
            "canonicalize,"
            "cse,"
            "one-shot-bufferize{allow-unknown-ops bufferize-function-boundaries},"
            "canonicalize,"
            "cse,"
            "convert-bufferization-to-memref,"
            "convert-linalg-to-loops,"
            "lower-affine,"
            "convert-scf-to-cf,"
            "expand-strided-metadata,"
            "lower-affine,"
            "finalize-memref-to-llvm,"
            "convert-cf-to-llvm,"
            "convert-math-to-llvm,"
            "convert-arith-to-llvm"
            ")"
        )
        pman = pm.PassManager.parse(pipeline_pre, ctx)
        pman.run(ir_module.operation)

        # Add emit_c_interface to all func.func ops AFTER bufferization
        # (trap #31: must be on func.func, not llvm.func)
        for region in ir_module.operation.regions:
            for block in region.blocks:
                for op in block:
                    if str(op.operation.name) == 'func.func':
                        op.operation.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get(ctx)

        # Run func-to-llvm lowering
        pipeline_llvm = (
            "builtin.module("
            "convert-func-to-llvm,"
            "reconcile-unrealized-casts"
            ")"
        )
        pman2 = pm.PassManager.parse(pipeline_llvm, ctx)
        pman2.run(ir_module.operation)

    return str(ir_module)


def lower_linalg_to_llvm_ir_text(mlir_text: str) -> str:
    """Parse MLIR text and run linalg→LLVM lowering."""
    if not _has_bindings():
        raise RuntimeError("MLIR Python bindings not available")

    import mlir.ir as ir

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True

    with ir.Location.unknown(ctx):
        module = ir.Module.parse(mlir_text, ctx)
        return lower_linalg_to_llvm_ir(module)


def _find_mlir_tool(name: str) -> str:
    """Locate an LLVM/MLIR binary (llc, mlir-translate, etc.).

    Checks (in order):
      1. {name} on PATH
      2. Common Homebrew paths
      3. SERVE_FORGE_LLVM_BIN environment variable as directory prefix
    """
    path = shutil.which(name)
    if path:
        return path

    llvm_bin_dir = os.environ.get("SERVE_FORGE_LLVM_BIN", "")
    if llvm_bin_dir:
        candidate = os.path.join(llvm_bin_dir, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    for prefix in [
        "/usr/local/opt/llvm@20/bin",
        "/usr/local/opt/llvm@19/bin",
        "/usr/local/opt/llvm@18/bin",
        "/usr/local/opt/llvm/bin",
        "/opt/homebrew/opt/llvm/bin",
    ]:
        candidate = os.path.join(prefix, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    raise RuntimeError(
        f"{name} not found. Install LLVM (brew install llvm), "
        "or set SERVE_FORGE_LLVM_BIN to the llvm/bin directory."
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


def _fixup_mlir_for_translate(mlir_text: str) -> str:
    """Apply backward-compatibility fixups for LLVM 22 → LLVM 20 translation.

    LLVM 22's MLIR LLVM dialect adds flags (e.g. ``nuw`` on getelementptr)
    that LLVM 20's mlir-translate cannot parse.  These are semantically
    optional / already implicit, so stripping them is safe.
    """
    import re

    # llvm.getelementptr inbounds|nuw → llvm.getelementptr inbounds
    mlir_text = re.sub(
        r"inbounds\|nuw\b", "inbounds", mlir_text
    )

    return mlir_text


def mlir_module_to_llvm_ir(ir_module: Any) -> str:
    """Translate an MLIR module (LLVM dialect) to LLVM IR text.

    Uses ``mlir-translate --mlir-to-llvmir`` under the hood, applying
    version-compatibility fixups when the host llvm-tools are older than
    the MLIR bindings (e.g. LLVM 22 bindings + LLVM 20 tools).
    """
    import tempfile

    mlir_translate = _find_mlir_translate()
    mlir_text = str(ir_module)

    with tempfile.TemporaryDirectory() as td:
        mlir_path = os.path.join(td, "module.mlir")
        with open(mlir_path, "w") as f:
            f.write(mlir_text)

        result = subprocess.run(
            [mlir_translate, "--mlir-to-llvmir", mlir_path],
            capture_output=True, text=True,
        )

        if result.returncode != 0:
            fixed = _fixup_mlir_for_translate(mlir_text)
            if fixed != mlir_text:
                with open(mlir_path, "w") as f:
                    f.write(fixed)
                result = subprocess.run(
                    [mlir_translate, "--mlir-to-llvmir", mlir_path],
                    capture_output=True, text=True,
                )

        if result.returncode != 0:
            raise RuntimeError(
                f"mlir-translate failed (exit {result.returncode}):\n{result.stderr[:1000]}"
            )

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
        "-ffast-math",
        ll_file,
        "-o",
        output,
    ]
    if arch and arch != "native":
        cmd.insert(-2, f"-march={arch}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"llc compilation failed (exit {result.returncode}):\n{result.stderr}"
        )

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
        raise RuntimeError("MLIR Python bindings not available")

    from mlir.execution_engine import ExecutionEngine

    engine = ExecutionEngine(ir_module, opt_level=2)
    return engine


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

    raise RuntimeError(
        "C compiler not found. Set SERVE_FORGE_CC environment variable."
    )


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
    cmd = [
        cc_bin,
        "-shared",
        "-o", output,
        *obj_files,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"link_dylib failed (exit {result.returncode}):\n{result.stderr[:1000]}"
        )

    if not os.path.isfile(output) or os.path.getsize(output) == 0:
        raise RuntimeError(f"link_dylib produced empty or missing file: {output}")

    return output


def compile_mlir_to_dylib(
    ir_module: Any,
    output: str,
    arch: str = "native",
    opt_level: int = 3,
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
    import tempfile

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
    opt_level: int = 3,
) -> str:
    """Compile a lowered MLIR module into the compiled/ artifacts directory.

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
    import tempfile

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
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to compile embedded constants (exit {result.returncode}):\n"
            f"{result.stderr[:500]}"
        )

    return o_path
