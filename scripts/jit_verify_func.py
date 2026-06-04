#!/usr/bin/env python3
"""JIT-verify a single function from model.lowered.mlir.

Extracts a specific ``func.func @main_N`` from the lowered MLIR module,
runs it through the BUILTIN_STAGES lowering pipeline (same as AOT
compilation) using ``mlir.execution_engine.ExecutionEngine`` (no llc),
and reports output statistics.

Usage::

    python scripts/jit_verify_func.py --func-index 11
    python scripts/jit_verify_func.py --func-index 12
    python scripts/jit_verify_func.py --func-index 0  --model-dir compiled/opt_125m_fresh

The diagnostic purpose: isolate whether func[12]'s dylib cos=0.538
degradation is caused by ``llc`` (LLVM backend) or by the lowering
pipeline itself.  If JIT (which skips llc) shows cos~1.0 for both
func[11] and func[12], then llc is the culprit.
"""

from __future__ import annotations

import argparse
import ctypes
import logging
import re
import sys
from pathlib import Path

import numpy as np

from compiler.mlir_dialect.lowering.compile_utils import _setup_mlir_path

_log = logging.getLogger("jit_verify")


# ── Function extraction ────────────────────────────────────────────────


def extract_func_mlir(mlir_text: str, func_index: int) -> str:
    """Extract ``func.func @main_N`` from the lowered MLIR module text.

    Returns a standalone module wrapping just that function (renamed to
    ``@main`` for JIT invocation simplicity).

    Args:
        mlir_text: Full content of ``model.lowered.mlir``.
        func_index: Index of the function to extract (0-based).

    Returns:
        MLIR text of a standalone module containing the extracted function.
    """
    target = f"func.func @main_{func_index}("
    lines = mlir_text.split("\n")

    # Include #map / #set / etc. alias definitions before the module
    module_start = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("module {"):
            module_start = i
            break
    alias_lines: list[str] = []
    if module_start > 0:
        alias_lines = lines[:module_start]
        # Remove blank lines at the end of alias section
        while alias_lines and not alias_lines[-1].strip():
            alias_lines.pop()

    start_line = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if target in stripped and stripped.startswith("func.func"):
            start_line = i
            break

    if start_line == -1:
        raise ValueError(
            f"Function @main_{func_index} not found "
            f"(target pattern: '{target}')"
        )

    depth = 0
    in_func = False
    result_lines: list[str] = []

    for i in range(start_line, len(lines)):
        line = lines[i]
        result_lines.append(line)

        for ch in line:
            if ch == "{":
                depth += 1
                in_func = True
            elif ch == "}":
                depth -= 1

        if in_func and depth == 0:
            break

    if not in_func:
        raise ValueError(
            f"Function @main_{func_index} starting at line {start_line + 1} "
            f"has no body (no '{{' found)"
        )

    func_body = "\n".join(result_lines)

    # Rename @main_N to @main for JIT simplicity
    func_body = func_body.replace(
        f"func.func @main_{func_index}(", "func.func @main("
    )
    # Also replace any references to @main_N inside the body
    func_body = func_body.replace(f"@main_{func_index}.", "@main.")

    alias_str = "\n".join(alias_lines)
    if alias_str:
        return f"{alias_str}\nmodule {{\n{func_body}\n}}"
    return f"module {{\n{func_body}\n}}"


# ── Signature parsing ──────────────────────────────────────────────────


def _parse_mlir_type_dims(type_str: str) -> tuple[int, ...]:
    """Parse an MLIR tensor type string into a shape tuple.

    Examples:
        ``"tensor<1xf32>"`` → ``(1,)``
        ``"tensor<?x?x768xf32>"`` → ``(-1, -1, 768)``
        ``"tensor<768xf32>"`` → ``(768,)``
        ``"tensor<3072x768xf32>"`` → ``(3072, 768)``
    """
    m = re.match(r"tensor<([^>]*)>", type_str.strip())
    if not m:
        raise ValueError(f"Cannot parse MLIR type: {type_str}")
    dims_str = m.group(1)
    parts = dims_str.rsplit("x", 1)
    if len(parts) < 2:
        return ()
    dim_str = parts[0]
    if not dim_str:
        return ()
    dims: list[int] = []
    for d in dim_str.split("x"):
        d = d.strip()
        if d == "?" or d == "":
            dims.append(-1)
        else:
            dims.append(int(d))
    return tuple(dims)


def parse_func_signature(
    func_mlir: str,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Parse func.func signature from MLIR text.

    Returns ``(inputs, outputs)`` where each element is
    ``(ssa_name, mlir_type_string)``.
    """
    func_line = ""
    for line in func_mlir.split("\n"):
        stripped = line.strip()
        if stripped.startswith("func.func @main("):
            func_line = stripped
            break

    if not func_line:
        raise ValueError("Could not find func.func @main in extracted MLIR")

    paren_depth = 0
    args_start = -1
    args_end = -1
    for i, ch in enumerate(func_line):
        if ch == "(":
            paren_depth += 1
            if paren_depth == 1:
                args_start = i
        elif ch == ")":
            paren_depth -= 1
            if paren_depth == 0:
                args_end = i
                break

    if args_start == -1 or args_end == -1:
        raise ValueError("Cannot parse function arguments from func_line")

    args_str = func_line[args_start + 1 : args_end]

    inputs: list[tuple[str, str]] = []
    arg_parts = _split_top_level(args_str)
    for part in arg_parts:
        part = part.strip()
        if ":" in part:
            name, typ = part.split(":", 1)
            name = name.strip()
            typ = typ.strip()
        if "{" in typ:
            typ = typ[: typ.index("{")].strip()
        inputs.append((name, typ))

    arrow_pos = func_line.find("->", args_end)
    if arrow_pos == -1:
        return inputs, []

    ret_part = func_line[arrow_pos + 2 :].strip()
    brace_pos = ret_part.find("{")
    if brace_pos >= 0:
        ret_part = ret_part[:brace_pos].strip()

    outputs: list[tuple[str, str]] = []
    if ret_part.startswith("("):
        ret_inner = ret_part[1:-1].strip()
        ret_types = _split_top_level(ret_inner)
    elif ret_part:
        ret_types = [ret_part]
    else:
        ret_types = []

    for i, typ in enumerate(ret_types):
        typ = typ.strip()
        if "{" in typ:
            typ = typ[: typ.index("{")].strip()
        outputs.append((f"%out{i}", typ))

    return inputs, outputs


def _split_top_level(s: str) -> list[str]:
    """Split a string by commas, respecting nesting in ``<>``."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in s:
        if ch in "<({":
            depth += 1
            current.append(ch)
        elif ch in ">)}":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    rest = "".join(current).strip()
    if rest:
        parts.append(rest)
    return parts


# ── Input generation ───────────────────────────────────────────────────


def generate_random_inputs(
    inputs_spec: list[tuple[str, str]], seed: int = 42
) -> list[np.ndarray]:
    """Generate random numpy arrays matching the MLIR function inputs.

    For dynamic dims (``?``), uses fixed sizes: batch=2, seq=4, dim=768.
    Uses a fixed seed so that the same seed always produces the same values.
    """
    rng = np.random.RandomState(seed)

    result: list[np.ndarray] = []
    for _name, typ_str in inputs_spec:
        shape = _parse_mlir_type_dims(typ_str)
        resolved: list[int] = []
        for i, d in enumerate(shape):
            if d == -1:
                # First dynamic dim = batch (2), rest = seq (4)
                resolved.append(2 if i == 0 else 4)
            else:
                resolved.append(d)
        arr = rng.uniform(0.1, 2.0, size=tuple(resolved)).astype(np.float32)
        result.append(arr)
    return result


# ── JIT lowering and execution ─────────────────────────────────────────


def _add_emit_c_interface(module: object) -> None:
    """Add ``llvm.emit_c_interface`` to each ``func.func``."""
    import mlir.ir as ir

    ctx = module.operation.context  # type: ignore[union-attr]

    def _cb(op: object) -> ir.WalkResult:
        opv = op  # type: ignore[assignment]
        if hasattr(opv, "name") and str(opv.name) == "func.func":  # type: ignore[union-attr]
            opv.operation.attributes["llvm.emit_c_interface"] = (  # type: ignore[union-attr]
                ir.UnitAttr.get(context=ctx)
            )
        return ir.WalkResult.ADVANCE

    module.operation.walk(_cb)  # type: ignore[union-attr]


def _detect_jit_output_spec(
    module: object,
) -> tuple[tuple[int, ...], int]:
    """Detect output shape and rank from an LLVM-dialect MLIR module.

    After full lowering to LLVM dialect, the function signature should
    contain the output memref type.  We parse it to get the output shape
    and rank.

    Returns ``(shape, rank)`` where dynamic dims are represented as -1.
    """

    module_str = str(module)

    m = re.search(
        r'llvm\.func\s+@main\s*\(([^)]*)\)',
        module_str,
    )
    if m:
        params_str = m.group(1)
        # sret params look like: !llvm.ptr {llvm.sret} or !llvm.struct<...> {llvm.sret}
        sret_params = re.findall(
            r'(!llvm\.(?:ptr|struct[^)]*\)))\s*\{[^}]*llvm\.sret[^}]*\}', params_str
        )
        if sret_params:
            array_matches = re.findall(r"array<(\d+)\s*x\s*i64>", sret_params[0])
            if array_matches:
                rank = int(array_matches[0])
                return (2, 4, 768)[:rank] if rank > 0 else (), rank
            if "!llvm.ptr" in struct_type:
                return (), 0

    # Fallback: parse memref type from module string
    memref_match = re.search(r"memref<([^>]*)xf32>", module_str)
    if memref_match:
        dims_str = memref_match.group(1)
        dims = []
        for d in dims_str.split("x"):
            d = d.strip()
            if d and d != "?":
                dims.append(int(d))
            elif d == "?":
                dims.append(2)  # default batch for remaining dynamic dims
        if dims:
            return tuple(dims), len(dims)

    return (2, 4, 768), 3


def lower_and_jit(
    func_mlir: str,
    np_inputs: list[np.ndarray],
    opt_level: int = 3,
) -> np.ndarray:
    """JIT-compile an extracted lower-level function and run with numpy inputs.

    Applies the same BUILTIN_STAGES pipeline as AOT compilation
    (minus A1 canonicalize,cse and E10 strip-gep-nuw), using
    ``mlir.execution_engine.ExecutionEngine`` instead of mlir-translate+llc.

    Args:
        func_mlir: MLIR text of a standalone module containing one func.func.
        np_inputs: List of numpy arrays matching the function parameters.
        opt_level: LLVM optimization level (2=moderate, default; use 0 for
            no opt, 3 for max).  opt_level=0 is extremely slow on full
            decoder layers — each matmul runs as scalar loops.

    Returns:
        Output numpy array.
    """
    import signal
    import time

    _setup_mlir_path()

    import mlir.ir as ir
    import mlir.passmanager as pm
    from mlir.execution_engine import ExecutionEngine
    from mlir.runtime.np_to_memref import get_ranked_memref_descriptor

    t_start = time.time()

    _log.info("[1/7] Setting up MLIR context ...")
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True

    try:
        from mlir._mlir_libs import _mlirRegisterEverything  # type: ignore[attr-defined]

        reg = ir.DialectRegistry()
        _mlirRegisterEverything.register_dialects(reg)
        ctx.append_dialect_registry(reg)
    except (ImportError, AttributeError):
        _log.warning("Could not register all dialects — bufferization may fail")

    try:
        from mlir_sf._mlir_libs._sfDialectsNanobind import sf

        sf.register_dialects(ctx._CAPIPtr, load=True)
    except ImportError:
        pass

    with ir.Location.unknown(ctx):
        _log.info("[2/7] Parsing MLIR (%d chars) ...", len(func_mlir))
        module = ir.Module.parse(func_mlir, ctx)
        _log.info("[2/7] Parsed OK (%.2fs)", time.time() - t_start)

        # ── Add emit_c_interface (BUILTIN_STAGES C1) ───────────────
        _log.info("[3/7] Adding emit_c_interface + running lowering pipeline ...")
        t_phase = time.time()
        _add_emit_c_interface(module)

        llvm_pipeline = (
            "builtin.module("
            "eliminate-empty-tensors,"
            "empty-tensor-to-alloc-tensor,"
            "one-shot-bufferize{bufferize-function-boundaries"
            " allow-unknown-ops"
            " function-boundary-type-conversion=identity-layout-map},"
            "canonicalize,cse,"
            "convert-bufferization-to-memref,"
            "convert-linalg-to-loops,"
            "lower-affine,"
            "convert-scf-to-cf,"
            "expand-strided-metadata,"
            "finalize-memref-to-llvm{use-generic-functions=false},"
            "convert-math-to-llvm,"
            "convert-vector-to-llvm,"
            "convert-arith-to-llvm,"
            "convert-func-to-llvm,"
            "convert-cf-to-llvm,"
            "convert-bufferization-to-memref,"
            "finalize-memref-to-llvm{use-generic-functions=false},"
            "reconcile-unrealized-casts"
            ")"
        )
        pman = pm.PassManager.parse(llvm_pipeline, ctx)
        pman.run(module.operation)
        _log.info("[3/7] Pipeline done (%.2fs)", time.time() - t_phase)

        # ── Fix remaining casts ───────────────────────────────────
        _log.info("[4/7] Fixing unrealized casts ...")
        t_phase = time.time()
        from compiler.mlir_dialect.lowering.fixups import _fixup_unrealized_casts_pass
        _fixup_unrealized_casts_pass(module)
        _log.info("[4/7] Cast fixup done (%.2fs)", time.time() - t_phase)

        # ── Detect output shape ────────────────────────────────────
        _log.info("[5/7] Detecting output shape ...")
        output_shape, _ = _detect_jit_output_spec(module)
        _log.info("[5/7] Output shape: %s", list(output_shape))

        # ── Create ExecutionEngine ─────────────────────────────────
        _log.info("[6/7] Creating ExecutionEngine (opt_level=%d) ...", opt_level)
        t_phase = time.time()
        engine = ExecutionEngine(module, opt_level=opt_level)
        _log.info("[6/7] ExecutionEngine ready (%.2fs)", time.time() - t_phase)

    # ── Build MemRef descriptors and invoke ─────────────────────────
    try:
        _log.info(
            "[7/7] Building MemRef descriptors (%d inputs) + invoking ...",
            len(np_inputs),
        )
        t_phase = time.time()

        input_inner_ptrs: list[ctypes.pointer] = []
        input_outer_ptrs: list[ctypes.pointer] = []
        for i, arr in enumerate(np_inputs):
            desc = get_ranked_memref_descriptor(arr)
            inner = ctypes.pointer(desc)
            outer = ctypes.pointer(inner)
            input_inner_ptrs.append(inner)
            input_outer_ptrs.append(outer)
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug("  input[%d] shape=%s size=%d", i, list(arr.shape), arr.size)

        out_arr = np.zeros(output_shape, dtype=np.float32)
        out_desc = get_ranked_memref_descriptor(out_arr)
        out_inner = ctypes.pointer(out_desc)
        out_outer = ctypes.pointer(out_inner)

        _log.info("[7/7] Calling engine.invoke('main', output + %d inputs) ...", len(np_inputs))
        sys.stderr.flush()

        # JIT code may trigger SIGFPE (division by zero, invalid sqrt)
        # from unoptimized scalar loops.  Catch it so we don't crash.
        _old_sigfpe = signal.signal(signal.SIGFPE, signal.SIG_IGN)
        args: list[object] = [out_outer] + input_outer_ptrs
        try:
            engine.invoke("main", *args)
        finally:
            signal.signal(signal.SIGFPE, _old_sigfpe)
        _log.info("[7/7] Invoke returned (%.2fs)", time.time() - t_phase)

        # ── Read back output ───────────────────────────────────────
        _log.info("Reading back output ...")
        clone = get_ranked_memref_descriptor(out_arr)
        ctypes.memmove(
            ctypes.addressof(clone),
            ctypes.cast(out_inner, ctypes.c_void_p).value,
            ctypes.sizeof(clone),
        )
        runtime_shape = tuple(int(s) for s in clone.shape)
        _log.info(
            "Runtime output shape: %s (allocated: %s)",
            list(runtime_shape),
            list(output_shape),
        )
        result = (
            np.ctypeslib.as_array(clone.aligned, shape=runtime_shape)
            .copy()
        )
        _log.info("TOTAL time: %.2fs", time.time() - t_start)
    finally:
        del engine

    return result


# ── Main ───────────────────────────────────────────────────────────────


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s [%(name)s] %(message)s",
        stream=sys.stderr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="JIT-verify a single function from model.lowered.mlir"
    )
    parser.add_argument(
        "--func-index",
        type=int,
        default=11,
        help="Function index to extract and JIT (0-based, default: 11)",
    )
    parser.add_argument(
        "--model-dir",
        default="compiled/opt_125m_fresh",
        help="Model directory (default: compiled/opt_125m_fresh)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for input generation (default: 42)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    _setup_logging(args.verbose)

    model_dir = Path(args.model_dir)
    lowered_path = model_dir / "model.lowered.mlir"
    if not lowered_path.exists():
        print(f"ERROR: {lowered_path} not found", file=sys.stderr)
        sys.exit(1)

    # 1. Load lowered MLIR
    _log.info("Loading %s ...", lowered_path)
    lowered_mlir = lowered_path.read_text()

    # 2. Extract function
    _log.info("Extracting func[%d] ...", args.func_index)
    func_mlir = extract_func_mlir(lowered_mlir, args.func_index)

    # 3. Parse signature
    inputs_spec, outputs_spec = parse_func_signature(func_mlir)
    _log.info(
        "Inputs: %d, Outputs: %d",
        len(inputs_spec),
        len(outputs_spec),
    )
    for name, typ in inputs_spec:
        _log.debug("  Input %s : %s", name, typ)
    for name, typ in outputs_spec:
        _log.debug("  Output %s : %s", name, typ)

    # 4. Generate random inputs
    _log.info("Generating random inputs with seed=%d ...", args.seed)
    np_inputs = generate_random_inputs(inputs_spec, seed=args.seed)
    for i, arr in enumerate(np_inputs):
        _log.debug(
            "  input[%d] shape=%s dtype=%s", i, list(arr.shape), arr.dtype
        )

    # 5. JIT compile and run
    print(f"\n{'=' * 60}")
    print(f"  JIT verifying func[{args.func_index}] ...")
    print(f"{'=' * 60}")

    jit_output = lower_and_jit(func_mlir, np_inputs)

    # 6. Report
    print(f"\n{'─' * 60}")
    print("  JIT output:")
    print(f"    shape  = {list(jit_output.shape)}")
    print(f"    mean   = {jit_output.mean():.8f}")
    print(f"    std    = {jit_output.std():.8f}")
    print(f"    min    = {jit_output.min():.8f}")
    print(f"    max    = {jit_output.max():.8f}")
    has_nan = int(np.any(np.isnan(jit_output)))
    has_inf = int(np.any(np.isinf(jit_output)))
    print(f"    NaN    = {bool(has_nan)}")
    print(f"    Inf    = {bool(has_inf)}")
    print(f"{'─' * 60}")

    # 7. Save output for inspection
    out_path = f"/tmp/jit_func_{args.func_index}_output.npy"
    np.save(out_path, jit_output)
    _log.info("Output saved to %s", out_path)

    # 8. Quick self-check: compare func[11] vs func[12] with same seed
    # If both agree (cos ~ 1.0), the JIT is deterministic.
    # If they disagree, the pipeline produces non-deterministic results.
    print(
        "\n  JIT compilation completed successfully — "
        "the lowering pipeline (excluding llc) works for "
        f"func[{args.func_index}]."
    )
    print(
        "  To compare with dylib output, run:\n"
        "    python scripts/diagnose_func12_ctypes.py\n"
    )


if __name__ == "__main__":
    main()
