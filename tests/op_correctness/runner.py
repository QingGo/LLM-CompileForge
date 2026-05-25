"""MLIR lowering + JIT execution + correctness comparison for sf ops.

Core flow for each test case:
  1. Generate MLIR text wrapping the sf op in a ``func.func``.
  2. Lower sf→linalg via C++ passes (sf-promote-weights, sf-lower-to-linalg).
  3. Lower linalg→LLVM dialect via builtin pipeline stages.
  4. JIT-compile via ``mlir.execution_engine.ExecutionEngine``.
  5. Build ctypes MemRef descriptors, invoke the JIT-compiled function.
  6. Extract output as ``numpy.ndarray``, compare with PyTorch reference
     using cosine similarity.
"""

from __future__ import annotations

import ctypes
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from compiler.mlir_dialect.compile_utils import _setup_mlir_path
from scripts._cos import cosine_similarity
from tests.op_correctness.registry import OpCase

_log = logging.getLogger(__name__)


# ── MemRef descriptor helpers (for ExecutionEngine ctypes invocation) ──


def _make_memref_arg(arr: np.ndarray) -> tuple[Any, Any]:
    """Build a double-pointer memref argument for ExecutionEngine.invoke().

    Returns ``(inner_ptr, outer_ptr)``:
        inner_ptr — ctypes pointer to the MemRef descriptor (kept alive).
        outer_ptr — ctypes pointer to inner_ptr (what invoke() receives).
    """
    from mlir.runtime.np_to_memref import get_ranked_memref_descriptor

    desc = get_ranked_memref_descriptor(arr)
    inner = ctypes.pointer(desc)
    outer = ctypes.pointer(inner)
    return inner, outer


def _read_memref_output(inner_ptr: Any, arr_dummy: np.ndarray) -> np.ndarray:
    """Read output from a ctypes MemRef descriptor written by JIT."""
    from mlir.runtime.np_to_memref import get_ranked_memref_descriptor

    clone = get_ranked_memref_descriptor(arr_dummy)
    ctypes.memmove(
        ctypes.addressof(clone),
        ctypes.cast(inner_ptr, ctypes.c_void_p).value,
        ctypes.sizeof(clone),
    )
    if arr_dummy.ndim == 0:
        # Scalar (rank-0) memref: aligned pointer holds the scalar value directly
        return np.array(clone.aligned[0], dtype=np.float32).reshape(())
    return np.ctypeslib.as_array(clone.aligned, shape=tuple(clone.shape)).copy()


# ── MLIR text generation ─────────────────────────────────────────────


def _shape_to_mlir_type(shape: tuple[int, ...]) -> str:
    if not shape:
        return "tensor<f32>"
    dims = "x".join(str(d) for d in shape)
    return f"tensor<{dims}xf32>"


def generate_mlir(case: OpCase) -> str:
    input_types = [_shape_to_mlir_type(s) for s in case.input_shapes]
    output_types = [_shape_to_mlir_type(s) for s in (case.output_shapes or [case.input_shapes[0]])]

    inputs_str = ", ".join(f"%arg{i}: {t}" for i, t in enumerate(input_types))
    input_vals = ", ".join(f"%arg{i}" for i in range(len(input_types)))
    input_type_str = ", ".join(input_types)
    output_type_str = output_types[0]

    attrs_str = ""
    if case.kwargs:
        attr_parts: list[str] = []
        for k, v in case.kwargs.items():
            if isinstance(v, bool):
                attr_parts.append(f'{k} = {"true" if v else "false"}')
            elif isinstance(v, int):
                attr_parts.append(f"{k} = {v} : i64")
            elif isinstance(v, float):
                attr_parts.append(f"{k} = {v} : f64")
            elif isinstance(v, (list, tuple)):
                elts = ", ".join(str(x) for x in v)
                attr_parts.append(f"{k} = [{elts}]")
        if attr_parts:
            attrs_str = "{" + ", ".join(attr_parts) + "} "

    module = (
        f"module {{\n"
        f"  func.func @main({inputs_str}) -> {output_type_str} {{\n"
        f"    %0 = \"{case.sf_op_name}\"({input_vals}) {attrs_str}: "
        f"({input_type_str}) -> {output_type_str}\n"
        f"    return %0 : {output_type_str}\n"
        f"  }}\n"
        f"}}"
    )
    _log.debug("Generated MLIR for %s:\n%s", case.name, module)
    return module


# ── Lowering pipeline ────────────────────────────────────────────────


def _register_sf_dialect(ctx: Any) -> None:
    try:
        from mlir_sf._mlir_libs._sfDialectsNanobind import sf

        sf.register_dialects(ctx._CAPIPtr, load=True)
    except ImportError:
        pass


def _add_emit_c_interface(module: Any, ctx: Any) -> None:
    import mlir.ir as ir

    def _cb(op: Any) -> Any:
        if hasattr(op, "name") and op.name == "func.func":
            op.operation.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get(context=ctx)
        return ir.WalkResult.ADVANCE

    module.operation.walk(_cb)


def lower_and_jit(mlir_text: str) -> tuple[Any, tuple[int, ...]]:
    """Lower MLIR text (sf dialect) through to LLVM dialect and JIT-compile.

    Steps:
      1. Parse MLIR text with sf dialect registered.
      2. Run ``sf-promote-weights``, ``canonicalize``, ``cse``, ``sf-lower-to-linalg``.
      3. Add ``llvm.emit_c_interface`` to all functions.
      4. Run the standard linalg→LLVM lowering pipeline (bufferize + LLVM conv).
      5. Apply ``_fixup_unrealized_casts_pass`` to resolve any remaining casts.
      6. Create an ``mlir.execution_engine.ExecutionEngine``.

    Returns:
        A tuple ``(engine, output_shape)`` where ``engine`` is the
        ``ExecutionEngine`` and ``output_shape`` is the output tensor shape.
    """
    _setup_mlir_path()
    import mlir.ir as ir
    import mlir.passmanager as pm

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    _register_sf_dialect(ctx)

    try:
        from mlir._mlir_libs import _mlirRegisterEverything

        reg = ir.DialectRegistry()
        _mlirRegisterEverything.register_dialects(reg)
        ctx.append_dialect_registry(reg)
    except (ImportError, AttributeError) as e:
        _log.warning("Could not register all dialects: %s", e)

    with ir.Location.unknown(ctx):
        # Step 1: Parse
        module = ir.Module.parse(mlir_text, ctx)

        # Step 2: sf → linalg lowering
        sf_pipeline = (
            "builtin.module("
            "sf-promote-weights,"
            "canonicalize,"
            "cse,"
            "sf-lower-to-linalg"
            ")"
        )
        pman = pm.PassManager.parse(sf_pipeline, ctx)
        pman.enable_verifier(True)
        pman.run(module.operation)
        _log.debug("After sf-lower-to-linalg:\n%s", str(module))

        # Step 3: Add emit_c_interface
        _add_emit_c_interface(module, ctx)

        # Step 4: linalg → LLVM lowering
        # finalize-memref-to-llvm runs twice: first to decompose function
        # signatures, then again after convert-func-to-llvm to handle any
        # remaining memref types (matching BUILTIN_STAGES ordering).
        llvm_pipeline = (
            "builtin.module("
            "canonicalize,"
            "cse,"
            "one-shot-bufferize{bufferize-function-boundaries},"
            "convert-bufferization-to-memref,"
            "convert-linalg-to-loops,"
            "lower-affine,"
            "convert-scf-to-cf,"
            "expand-strided-metadata,"
            "finalize-memref-to-llvm{use-generic-functions=false},"
            "convert-cf-to-llvm,"
            "convert-math-to-llvm,"
            "convert-vector-to-llvm,"
            "convert-arith-to-llvm,"
            "convert-func-to-llvm,"
            "convert-bufferization-to-memref,"
            "finalize-memref-to-llvm{use-generic-functions=false},"
            "reconcile-unrealized-casts"
            ")"
        )
        pman2 = pm.PassManager.parse(llvm_pipeline, ctx)
        pman2.run(module.operation)
        _log.debug("After LLVM lowering:\n%s", str(module))

        output_shape = _detect_output_shape(module)

        # Step 5: Fix remaining unrealized_conversion_cast ops
        from compiler.mlir_dialect.fixups import _fixup_unrealized_casts_pass

        _fixup_unrealized_casts_pass(module)

        # Step 6: JIT compile
        try:
            from pathlib import Path

            from mlir.execution_engine import ExecutionEngine

            _runner_lib = (
                Path(__file__).resolve().parent.parent.parent
                / "llvm-project" / "build" / "lib" / "libmlir_c_runner_utils.dylib"
            )
            shared_libs = [str(_runner_lib)] if _runner_lib.exists() else []
            engine = ExecutionEngine(module, opt_level=0, shared_libs=shared_libs)
            return engine, output_shape
        except Exception as e:
            raise RuntimeError(
                f"ExecutionEngine creation failed: {e}\n"
                f"Module after lowering:\n{str(module)}"
            ) from e


def _detect_output_shape(module: Any) -> tuple[int, ...]:
    import mlir.ir as ir

    output_shape: tuple[int, ...] | None = None

    def _find_func(op: Any) -> Any:
        nonlocal output_shape
        if str(op.operation.name) == "func.func":
            ft = str(op.operation.attributes.get("function_type", ""))
            import re

            # Find the LAST tensor/memref type (the return type in function signature)
            matches = list(re.finditer(r"(tensor|memref)<([\dx]*)xf32>", ft))
            if matches:
                m = matches[-1]
                dims = m.group(2)
                output_shape = tuple(int(d) if d.isdigit() else 0 for d in dims.split("x") if d)
                if not output_shape:
                    output_shape = ()
        return ir.WalkResult.ADVANCE

    module.operation.walk(_find_func)
    if output_shape is not None:
        return output_shape

    return (4, 768)


# ── Invocation and extraction ─────────────────────────────────────────


def invoke_and_extract(
    engine: Any,
    func_name: str,
    input_arrays: list[np.ndarray],
    output_shape: tuple[int, ...],
) -> np.ndarray:
    """Invoke a JIT-compiled function with given numpy inputs and extract output.

    Builds MemRef descriptors for all input tensors, allocates an output
    MemRef, calls ``engine.invoke()``, and reads back the output.
    """
    input_inner_ptrs = []
    input_outer_ptrs = []
    for arr in input_arrays:
        inner, outer = _make_memref_arg(arr)
        input_inner_ptrs.append(inner)
        input_outer_ptrs.append(outer)

    out_arr = np.zeros(output_shape, dtype=np.float32)
    out_inner, out_outer = _make_memref_arg(out_arr)

    args: list[Any] = [out_outer] + input_outer_ptrs
    engine.invoke(func_name, *args)

    return _read_memref_output(out_inner, out_arr)


# ── Result type ───────────────────────────────────────────────────────


@dataclass
class RunResult:
    """Outcome of a single op correctness test."""

    cos: float
    output: np.ndarray
    reference: np.ndarray


# ── Runner ────────────────────────────────────────────────────────────


class Runner:
    """Orchestrates the full op correctness test pipeline.

    Generates the same random inputs for both JIT and PyTorch reference,
    then compares outputs using cosine similarity.
    """

    def __init__(self, case: OpCase, custom_inputs: list[np.ndarray] | None = None) -> None:
        self.case = case
        self.custom_inputs = custom_inputs

    def run(self) -> RunResult:
        """Execute the test pipeline and return the comparison result."""
        case = self.case

        # Step 1: Generate MLIR text
        mlir_text = generate_mlir(case)

        # Step 2-6: Lower and JIT compile
        engine, _ = lower_and_jit(mlir_text)

        # Use known output shape from test case (detection from LLVM IR is fragile)
        output_shape = (case.output_shapes or [case.input_shapes[0]])[0]

        # Step 7: Generate shared input data (same for JIT and torch)
        if self.custom_inputs is not None:
            input_arrays = self.custom_inputs
        else:
            rng = np.random.RandomState(42)
            input_arrays = [rng.randn(*shape).astype(np.float32) for shape in case.input_shapes]
            kws = case.kwargs or {}
            if kws.get("positive_inputs"):
                input_arrays = [np.abs(arr) + 0.1 for arr in input_arrays]

        # Step 8: Invoke JIT and extract output
        try:
            output = invoke_and_extract(engine, "main", input_arrays, output_shape)
        finally:
            del engine

        # Step 9: Compute PyTorch reference using the same input data
        torch_inputs = [torch.from_numpy(arr) for arr in input_arrays]
        reference_torch = case.torch_fn(*torch_inputs)
        reference_np = reference_torch.to(dtype=torch.float32).numpy()

        # Handle shape differences
        if output.shape != reference_np.shape:
            _log.warning(
                "Shape mismatch for %s: JIT=%s ref=%s",
                case.name, output.shape, reference_np.shape,
            )
            if reference_np.ndim == 0:
                reference_np = reference_np.reshape(output.shape)

        cos = cosine_similarity(output, reference_np)
        _log.info(
            "%s: cos=%.8f (rtol=%s) shape=%s",
            case.name, cos, case.rtol, output.shape,
        )

        return RunResult(cos=cos, output=output, reference=reference_np)
