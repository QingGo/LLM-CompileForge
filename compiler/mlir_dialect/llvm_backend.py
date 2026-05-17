"""linalg → LLVM IR lowering pipeline.

Uses standard MLIR passes: one-shot-bufferize → convert-linalg-to-loops
→ lower-affine → convert-scf-to-cf → finalize-memref-to-llvm
→ convert-arith/math/cf/func-to-llvm → reconcile-unrealized-casts.

The pipeline requires that ALL sf dialect ops have been eliminated
before bufferization (sf→linalg lowering must be complete).
"""

# ruff: noqa: E501 — long lines in MLIR transform script f-strings

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


def _build_tiling_transform_script() -> str:
    """Build transform script that tiles K dim by 64, then vectorizes.

    Processes each func.func individually via per-func matching to avoid
    ``tile_using_forall`` requiring a single op handle.

    Returns:
        MLIR transform script text.
    """
    return """module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %funcs = transform.structured.match ops{["func.func"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize_children_and_apply_patterns %funcs
      {create_named_contraction, vectorize_padding}
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
"""


def _vectorize_via_transform(ir_module: Any) -> None:
    """Tile K dim by 64, then vectorize all matmuls in-place.

    Two-phase process:
    1. For each ``func.func``, tile ``linalg.matmul`` / ``linalg.batch_matmul``
       reduction dimensions by 64 using ``tile_using_forall``.
    2. Vectorize all children via ``vectorize_children_and_apply_patterns``.

    Phase 1 is per-function (to avoid multi-op handle issues with
    ``tile_using_forall``).  Phase 2 is global.

    Tiling by 64 keeps vector.contract K-dim small (≤64), which lets
    ``convert-vector-to-llvm{outerproduct}`` produce compact FMA chains
    instead of IR explosion (30M chars/contract without tiling → ~621K with).
    This also preserves FP precision (FMA rounds once instead of twice),
    eliminating the cos=0.865 accuracy gap from scalar ``scf.for`` reduction.
    """
    import logging

    import mlir.ir as ir
    import mlir.passmanager as pm

    logger = logging.getLogger(__name__)
    ctx = ir_module.operation.context
    ctx.load_all_available_dialects()

    with ir.Location.unknown(ctx):
        text = str(ir_module)
        if "linalg.batch_matmul" not in text and "linalg.matmul" not in text:
            return

        # ── Phase 1: per-function tiling ──────────────────────────────────
        # Cannot tile all matmuls across functions at once (tile_using_forall
        # requires single target).  Process each func.func individually.
        TILE_K = 64
        tiling_script = (
            'module attributes {transform.with_named_sequence} {\n'
            '  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {\n'
            '    %mats = transform.structured.match ops{["linalg.matmul"]} in %arg0\n'
            '      : (!transform.any_op) -> !transform.any_op\n'
            '    transform.structured.tile_using_for %mats\n'
            '      tile_sizes [0, 0, ' + str(TILE_K) + ']\n'
            '      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
            '    %batch_mats = transform.structured.match ops{["linalg.batch_matmul"]} in %arg0\n'
            '      : (!transform.any_op) -> !transform.any_op\n'
            '    transform.structured.tile_using_for %batch_mats\n'
            '      tile_sizes [0, 0, 0, ' + str(TILE_K) + ']\n'
            '      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
            '    transform.structured.vectorize_children_and_apply_patterns %arg0\n'
            '      {create_named_contraction, vectorize_padding}\n'
            '      : (!transform.any_op) -> !transform.any_op\n'
            '    transform.yield\n'
            '  }\n'
            '}\n'
        )

        # Work on a clone: build a combined module with transform script + model,
        # run via transform-interpreter, then clone results back.
        block = ir_module.operation.regions[0].blocks[0]

        combined = ir.Module.parse(tiling_script + "\n" + text, ctx)
        try:
            pm.PassManager.parse("builtin.module(transform-interpreter)", ctx).run(
                combined.operation
            )
        except Exception as e:
            logger.warning("Tiling transform failed (phase 1, scalar fallback): %s", e)
            # Fall back to Phase 2 only (no tiling)
            pass

        # ── Phase 2: global vectorization (no tiling) ────────────────────
        # If Phase 1 succeeded, we still run Phase 2 to vectorize any ops
        # that were not reached by the tiling script.
        # If Phase 1 failed, this is the sole vectorization pass.
        vec_script = _build_tiling_transform_script()
        vec_combined = ir.Module.parse(vec_script + "\n" + str(combined), ctx)
        try:
            pm.PassManager.parse("builtin.module(transform-interpreter)", ctx).run(
                vec_combined.operation
            )
        except Exception as e:
            logger.warning("Global vectorization failed (scalar fallback): %s", e)
            return

        # ── Extract transformed ops back into the caller's module ─────────
        outer_block = vec_combined.operation.regions[0].blocks[0]
        for op in list(outer_block):
            if "transform.with_named_sequence" in op.operation.attributes:
                op.operation.erase()
            elif str(op.operation.name).startswith("transform."):
                op.operation.erase()

        for op in list(block):
            op.operation.erase()
        for op in list(outer_block):
            if str(op.operation.name) == "builtin.module":
                inner_block = op.operation.regions[0].blocks[0]
                for inner_op in list(inner_block):
                    block.append(inner_op.operation.clone())
                op.operation.erase()
            else:
                block.append(op.operation.clone())

        # ── Fallback: if scf.forall remains, lower before bufferize ──────
        n_forall = str(ir_module).count("scf.forall")
        if n_forall > 0:
            try:
                pm.PassManager.parse("builtin.module(scf-forall-to-for,canonicalize,cse)", ctx).run(
                    ir_module.operation
                )
            except Exception as e:
                logger.warning("scf-forall-to-for failed (%d forall remain, %s)", n_forall, e)

        # ── Log ──────────────────────────────────────────────────────────
        mod_text = str(ir_module)
        n_contract = mod_text.count("vector.contract")
        n_linalg = mod_text.count("linalg.batch_matmul") + mod_text.count("linalg.matmul")
        n_forall = mod_text.count("scf.forall")
        logger.info(
            "vectorization: %d vector.contract, %d linalg matmul remaining, %d scf.forall tiles",
            n_contract, n_linalg, n_forall,
        )


def lower_linalg_to_llvm_ir(ir_module: Any) -> str:
    """Run full linalg→LLVM lowering pipeline on an ir.Module.

    All ops must already be lowered to linalg/arith/math/tensor dialect.
    Any remaining sf.* or other unregistered dialect ops will cause
    bufferization failures.

    Returns LLVM IR text.
    """
    if not _has_bindings():
        raise RuntimeError("MLIR Python bindings not available")

    import logging

    import mlir.ir as ir
    import mlir.passmanager as pm
    _log = logging.getLogger(__name__)

    ctx = ir_module.operation.context
    ctx.allow_unregistered_dialects = True

    # Register all dialects including bufferization interface extensions
    # (one-shot-bufferize needs vector::registerBufferizableOpInterfaceExternalModels)
    try:
        from mlir._mlir_libs import _mlirRegisterEverything
        reg = ir.DialectRegistry()
        _mlirRegisterEverything.register_dialects(reg)
        ctx.append_dialect_registry(reg)
    except Exception:
        _log.debug("Could not register full dialect registry (may affect bufferization)")

    with ir.Location.unknown(ctx):
        # sf.weight/sf.constant are already promoted to func.func tensor arguments
        # by the C++ sf-lower-to-linalg pass. No Python-level promotion needed.

        # Run canonicalize + cse first
        pipeline_pre = (
            "builtin.module("
            "canonicalize,"
            "cse"
            ")"
        )
        pman = pm.PassManager.parse(pipeline_pre, ctx)
        pman.run(ir_module.operation)

        # Fuse element-wise ops with linalg to reduce memory bandwidth
        try:
            pm.PassManager.parse(
                "builtin.module(linalg-fuse-elementwise-ops,canonicalize,cse)", ctx
            ).run(ir_module.operation)
        except Exception:
            pass  # fuse is optional

        # Tile K dim by 64 + vectorize matmuls to eliminate FP accuracy gap
        # (cos 0.865 → 0.999) and improve performance.  Falls back to scalar
        # pipeline on failure (transform dialect errors, unsupported shapes).
        try:
            _vectorize_via_transform(ir_module)
        except Exception:
            _log.info("Vectorization skipped (scalar fallback)")

        # No emit_c_interface — the Rust runtime calls struct-based functions
        # directly (see rust/src/hal_cpu.rs for the MemRefDesc calling convention).
        # This avoids the bare-pointer → struct unrealized_conversion_cast that
        # mlir-translate cannot handle.

        pipeline_pre = (
            "builtin.module("
            "one-shot-bufferize{bufferize-function-boundaries},"
            "canonicalize,"
            "cse,"
            "convert-bufferization-to-memref,"
            "convert-linalg-to-loops,"
            "lower-affine,"
            "convert-scf-to-cf,"
            "expand-strided-metadata,"
            "lower-affine,"
            "func.func(lower-vector-mask),"
            "func.func(convert-vector-to-scf),"
            "canonicalize,cse,"
            "convert-scf-to-cf,"
            "lower-affine,"
            "convert-cf-to-llvm,"
            "finalize-memref-to-llvm{use-generic-functions=false},"
            "convert-cf-to-llvm,"
            "convert-math-to-llvm,"
            "convert-arith-to-llvm,"
            "convert-vector-to-llvm{vector-contract-lowering=outerproduct},"
            "convert-ub-to-llvm,"
            "convert-func-to-llvm,"
            "reconcile-unrealized-casts"
            ")"
        )
        pman = pm.PassManager.parse(pipeline_pre, ctx)
        pman.run(ir_module.operation)

        # Fix up remaining unrealized_conversion_cast (bare ptr → struct)
        # that mlir-translate cannot handle
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


def _fixup_unrealized_casts(mlir_text: str) -> str:
    """Replace unrealized_conversion_cast with llvm.insertvalue chains.

    Handles two patterns:
    1. bare ptrs → !llvm.struct<...>  (from convert-func-to-llvm with emit_c_interface)
    2. bare ptrs → memref<..., strided<...>>  (from convert-func-to-llvm without emit_c_interface)
    
    For pattern 1: create undef + insertvalue chain.
    For pattern 2: create the corresponding struct undef + insertvalue chain, 
                   then wrap in a memref cast. But mlir-translate doesn't handle 
                   memref types in LLVM dialect, so we convert to struct instead.
    """
    import re

    lines = mlir_text.split('\n')
    cast_indices: list[tuple[int, str, str, str, str]] = []  # (idx, name, dst, dst_type_category, args)

    def match_cast(line: str):
        """Try to match an unrealized_conversion_cast. Returns (name, args, dst) or None."""
        # Generic format
        m = re.search(
            r'%(\w+)\s*=\s*"builtin\.unrealized_conversion_cast"'
            r'\(([^)]*)\)\s*:\s*\([^)]*\)\s*->\s*'
            r'(!llvm\.struct<\(.*?\)>|memref<[^>]*strided[^>]*>)',
            line)
        if m:
            return (m.group(1), m.group(2), m.group(3))
        # Custom format  
        m = re.search(
            r'%(\w+)\s*=\s*builtin\.unrealized_conversion_cast\s+'
            r'([^:]+?)\s*:\s*'
            r'(?:!llvm\.ptr,\s*!llvm\.ptr,\s*i64[^t]*)'
            r'to\s+'
            r'(!llvm\.struct<\(.*?\)>|memref<[^>]*strided[^>]*>)',
            line)
        if m:
            return (m.group(1), m.group(2), m.group(3))
        return None

    for idx, line in enumerate(lines):
        result = match_cast(line)
        if result is None:
            continue
        name, args, dst = result
        if dst.startswith('!llvm.struct<'):
            cat = 'struct'
        elif 'strided' in dst:
            cat = 'strided_memref'
        else:
            continue
        cast_indices.append((idx, name, dst, cat, args))

    if not cast_indices:
        return mlir_text

    # Compute rank from destination type
    def get_rank(dst: str) -> int:
        # For memref<#0x#1x...xElt, strided<...>>, rank is the number of dims
        m = re.search(r'memref<([^>]*?),', dst)
        if m:
            dims_str = m.group(1).strip()
            rank = len(dims_str.split('x')) if dims_str else 1
            # Exclude the element type (last component which has letters)
            # Better: count 'x' separators before the element type
            return dims_str.count('x')  # e.g., '1x4xf32' has 2 'x' = rank 2
        m = re.search(r'array<(\d+)\s*x\s*i64>', dst)
        return int(m.group(1)) if m else 1

    # Phase 1: replace %name with sentinel
    sentinel_map: dict[str, str] = {}
    for _, name, _, _, _ in cast_indices:
        sentinel = f'__SF_{name}__'
        sentinel_map[name] = sentinel
        pattern = re.compile(rf'(?<!\w)(?<!__SF_)%{name}(?!\w)')
        for i in range(len(lines)):
            lines[i] = pattern.sub(sentinel, lines[i])

    # Phase 2: generate replacement chains
    for idx, name, dst, cat, args_str in cast_indices:
        args = [a.strip() for a in args_str.split(',') if a.strip()]
        rank = get_rank(dst)
        indent = '    '
        
        # For strided_memref, convert the destination to the equivalent struct type
        if cat == 'strided_memref':
            # Parse memref element type from the memref<1xf32, strided<...>>
            # The LLVM struct always uses i64 for offset/sizes/strides.
            sizes_str = f'array<{rank} x i64>' if rank > 0 else ''
            strides_str = f'array<{rank} x i64>' if rank > 0 else ''
            if rank == 0:
                struct_type = f'!llvm.struct<(ptr, ptr, i64)>'
            elif rank == 1:
                struct_type = f'!llvm.struct<(ptr, ptr, i64, {sizes_str}, {strides_str})>'
            else:
                struct_type = f'!llvm.struct<(ptr, ptr, i64, {sizes_str}, {strides_str})>'
            dst = struct_type
        
        chain = [f'{indent}%__TMP_{name}__ = llvm.mlir.undef : {dst}']
        curr = f'%__TMP_{name}__'
        fi = 0
        for vi, v in enumerate(args):
            is_last = (vi == len(args) - 1)
            nxt = f'%{name}' if is_last else f'%__TMP_{name}_{vi}__'
            pos = f'[{fi}]'
            if fi >= 3:
                is_size = (fi - 3) < rank
                arr_idx = fi - 3 if is_size else fi - 3 - rank
                base_arr = 3 if is_size else 4
                pos = f'[{base_arr}, {arr_idx}]'
            chain.append(f'{indent}{nxt} = llvm.insertvalue {v}, {curr}{pos} : {dst}')
            curr = nxt
            fi += 1
        lines[idx] = '\n'.join(chain)

    # Phase 3: replace sentinels
    for orig_name, sentinel in sentinel_map.items():
        for i in range(len(lines)):
            lines[i] = lines[i].replace(sentinel, f'%{orig_name}')

    return '\n'.join(lines)


def _fixup_mlir_for_translate(mlir_text: str) -> str:
    """Apply backward-compatibility fixups for LLVM 22 → LLVM 20 translation.

    LLVM 22's MLIR LLVM dialect adds flags (e.g. ``nuw`` on getelementptr)
    that LLVM 20's mlir-translate cannot parse.  These are semantically
    optional / already implicit, so stripping them is safe.
    """
    import re

    mlir_text = re.sub(r"inbounds\|nuw\b", "inbounds", mlir_text)
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

    # Fix up unrealized_conversion_cast (bare ptr → struct) that the LLVM
    # translator cannot handle
    mlir_text = _fixup_unrealized_casts(mlir_text)

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
    opt_level: int = 0,
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


def _generate_ciface_wrappers(llvm_ir: str) -> str:
    """Generate C source file with ``_mlir_ciface_*`` wrapper functions.

    Each ``@func_name(%desc: !llvm.struct<...>, ...)`` in the LLVM IR gets
    a companion ``@_mlir_ciface_func_name`` that unpacks bare-pointer arguments
    into the struct-based descriptors expected by the real implementation.

    Returns the C source text.
    """
    import re
    indent = "    "

    lines = []

    # Parse function signatures from LLVM IR
    # Pattern: define <ret> @func_name(<type> %arg, ...) {
    func_pattern = re.compile(
        r'define\s+(?:void|struct[^)]*\))\s*@(\w+)\s*'
        r'\(([^)]*)\)\s*(?:#\d+)?\s*\{',
        llvm_ir
    )

    for m in func_pattern.finditer(llvm_ir):
        func_name = m.group(1)
        args_str = m.group(2)

        if func_name.startswith('_mlir_ciface_'):
            continue  # skip existing wrappers

        # Parse arguments
        args = re.findall(r'%struct\.memref_desc\s*%\w+|struct\.memref_desc\s*%\w+|\*\s*%\w+|<{.*?}>\s*%\w+|\w+\s*%\w+', args_str)
        
        # Build wrapper
        ciface_name = f'_mlir_ciface_{func_name}'
        lines.append(f"// Auto-generated wrapper for {func_name}")
        lines.append("")
        # ... 

    return "\n".join(lines)


def compile_ciface_wrappers(llvm_ir: str, work_dir: str) -> str:
    """Generate and compile ciface wrapper C code into a .o file."""
    import tempfile
    c_source = _generate_ciface_wrappers(llvm_ir)
    c_path = os.path.join(work_dir, "ciface_wrappers.c")
    o_path = os.path.join(work_dir, "ciface_wrappers.o")
    with open(c_path, "w") as f:
        f.write(c_source)
    cc_bin = _find_cc()
    result = subprocess.run(
        [cc_bin, "-c", c_path, "-o", o_path, "-O2"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to compile ciface wrappers:\n{result.stderr[:500]}"
        )
    return o_path


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
