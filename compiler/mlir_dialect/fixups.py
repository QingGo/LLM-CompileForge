# ruff: noqa: E501 — long lines in MLIR fixup f-strings

"""Fixups for MLIR LLVM dialect IR — text-level and MLIR-bindings-based.

These functions perform transformations on LLVM dialect IR text,
replacing unrealized_conversion_cast chains with equivalent LLVM ops,
stripping version-incompatible flags, fixing scalar→tensor arith.constant
type mismatches, and lowering vector-typed arith ops that the pass
pipeline cannot handle.

Functions that require MLIR Python bindings fall back gracefully when
bindings are not available, preserving the string→string contract.
"""

from __future__ import annotations

import logging
import re
import typing as _typing

if _typing.TYPE_CHECKING:
    import mlir.ir as ir


def _strided_to_struct(memref_type: str) -> str:
    """Build the equivalent ``!llvm.struct<...>`` for a memref type."""
    m = re.match(r'memref<([^>]*?)(?:,|>)', memref_type)
    if not m:
        return '!llvm.struct<(ptr, ptr, i64)>'
    dims = m.group(1)
    rank = dims.count('x')
    sizes = f'array<{rank} x i64>' if rank else ''
    strides = f'array<{rank} x i64>' if rank else ''
    return (f'!llvm.struct<(ptr, ptr, i64, {sizes}, {strides})>'
            if rank else '!llvm.struct<(ptr, ptr, i64)>')


def _fixup_unrealized_casts(mlir_text: str) -> str:
    """Replace ``unrealized_conversion_cast`` with equivalent LLVM dialect ops.

    After ``finalize-memref-to-llvm`` + ``convert-func-to-llvm``, two kinds of
    ``unrealized_conversion_cast`` remain (no actual memref ops exist):

    1. **Entry cast** — bare ptrs → strided memref (143 in OPT-125M)::

         %v = builtin.unrealized_conversion_cast %a0, %a1, %a2, ...
           : !llvm.ptr, !llvm.ptr, i64, i64, ...
           to memref<DIMS, strided<...>>

       → Replace with ``undef`` + ``llvm.insertvalue`` chain that builds
         the equivalent ``!llvm.struct<(ptr, ptr, i64, ...)>``.
       %v is now an LLVM struct instead of a strided memref.

    2. **Exit cast** — strided memref → ``!llvm.struct<...>`` (143 pairs)::

         %r = builtin.unrealized_conversion_cast %v
           : memref<..., strided<...>> to !llvm.struct<...>

       After fixing (1), %v IS already ``!llvm.struct<...>``.
       This cast becomes a no-op (struct → struct).  Replace all uses
       of %r with %v and remove the cast line.

    3. **Direct struct cast** — bare ptrs → ``!llvm.struct<...>`` (45)::

         %r = builtin.unrealized_conversion_cast %a0, %a1, ...
           : !llvm.ptr, !llvm.ptr, i64, ... to !llvm.struct<...>

       → ``undef`` + ``llvm.insertvalue`` chain (unchanged).
    """
    _log = logging.getLogger("compiler.compile_utils")

    lines = mlir_text.split('\n')
    indent = '    '

    # ── Pre-compiled regexes ───────────────────────────────────────

    _re_entry = re.compile(
        r'%(\w+)\s*=\s*builtin\.unrealized_conversion_cast\s+'
        r'((?:%\w+(?:\s*,\s*)?)+?)\s*:\s*'
        r'!llvm\.ptr,\s*!llvm\.ptr,\s*i64'
        r'.*?\s*to\s*'
        r'(memref<[^>]*?strided[^>]*>[^>]*>|memref<[^>]*>)'
    )
    _re_exit = re.compile(
        r'%(\w+)\s*=\s*builtin\.unrealized_conversion_cast\s+%(\w+)\s*:\s*'
        r'memref<[^>]*?strided[^>]*>[^>]*>\s*to\s*'
        r'(!llvm\.struct<\([^)]*\)>)'
    )
    _re_exit_identity = re.compile(
        r'%(\w+)\s*=\s*builtin\.unrealized_conversion_cast\s+%(\w+)\s*:\s*'
        r'memref<[^>]*>\s*to\s*'
        r'(!llvm\.struct<\([^)]*\)>)'
    )
    _re_direct = re.compile(
        r'%(\w+)\s*=\s*builtin\.unrealized_conversion_cast\s+'
        r'(%[\w,\s%]+?)\s*:\s*'
        r'(!llvm\.ptr,\s*!llvm\.ptr,\s*i64(?:,\s*(?:!llvm\.ptr|i64))*)\s*to\s*'
        r'(!llvm\.struct<\([^)]*\)>)'
    )

    # ── Phase 1: classify all casts ─────────────────────────────────

    # List of (line_idx, name, [arg_names], memref_type) for entry casts.
    # NOTE: SSA names (%0, %1, ...) are reused across functions. Use a list
    # instead of a dict keyed by name to avoid overwrites between functions.
    entry_casts: list[tuple[int, str, list[str], str]] = []
    # List of (line_idx, dst_name, src_name, struct_type) — exit struct→struct
    exit_casts: list[tuple[int, str, str, str]] = []
    # List of (line_idx, name, dst_struct, args_str) — direct struct
    direct_casts: list[tuple[int, str, str, str]] = []

    for idx, line in enumerate(lines):
        s = line.strip()
        if 'unrealized_conversion_cast' not in s:
            continue

        m = _re_entry.match(s)
        if m:
            name = m.group(1)
            args = [a.strip() for a in m.group(2).split(',') if a.strip()]
            entry_casts.append((idx, name, args, m.group(3)))
            continue

        m = _re_exit.match(s)
        if m:
            exit_casts.append((idx, m.group(1), m.group(2), m.group(3)))
            continue

        m = _re_exit_identity.match(s)
        if m:
            exit_casts.append((idx, m.group(1), m.group(2), m.group(3)))
            continue

        m = _re_direct.match(s)
        if m:
            name = m.group(1)
            args = m.group(2)
            direct_casts.append((idx, name, m.group(4), args))
            continue

    if not entry_casts and not exit_casts and not direct_casts:
        return mlir_text

    # ── Phase 2: build struct type from memref type ─────────────────
    # (uses module-level _strided_to_struct)

    # ── Phase 3: handle exit casts (struct → struct no-op) ─────────
    # For each exit cast %r = cast %v : memref<strided> -> !llvm.struct:
    #   %v comes from an entry cast → after Phase 4 it IS an !llvm.struct.
    #   Instead of renaming (which breaks across-function SSA uniqueness),
    #   build an independent undef+insertvalue chain from %v's entry args.
    #
    #   Key insight: SSA names (%0, %1, ...) are reused across functions.
    #   A global rename would corrupt other functions.  We keep %r's name
    #   and build the struct independently using the entry cast's args.

    for exit_idx, dst_name, src_name, struct_type in exit_casts:
        # Find the entry cast that produced src_name.
        # SSA names are reused across functions — scan REVERSE and use
        # the nearest PRECEDING entry cast (same function).
        entry_args = None
        entry_mtype = None
        for e_idx, e_name, e_args, e_mtype in reversed(entry_casts):
            if e_name == src_name and e_idx < exit_idx:
                entry_args = e_args
                entry_mtype = e_mtype
                break

        if entry_args is None:
            continue

        struct_type = _strided_to_struct(entry_mtype)
        rank = entry_mtype.count('x')
        chain = [f'{indent}%__TMP_{dst_name}__ = llvm.mlir.undef : {struct_type}']
        curr = f'%__TMP_{dst_name}__'
        fi = 0
        for vi, v in enumerate(entry_args):
            is_last = (vi == len(entry_args) - 1)
            nxt = f'%{dst_name}' if is_last else f'%__TMP_{dst_name}_{vi}__'
            if fi < 3:
                pos = f'[{fi}]'
            else:
                arr_idx = fi - 3
                if arr_idx < rank:
                    pos = f'[3, {arr_idx}]'
                else:
                    pos = f'[4, {arr_idx - rank}]'
            chain.append(f'{indent}{nxt} = llvm.insertvalue {v}, {curr}{pos} : {struct_type}')
            curr = nxt
            fi += 1
        lines[exit_idx] = '\n'.join(chain)

    # ── Phase 4: replace entry casts with undef + insertvalue ──────
    for idx, name, args, mtype in entry_casts:
        struct_type = _strided_to_struct(mtype)
        rank = mtype.count('x')
        chain = [f'{indent}%__TMP_{name}__ = llvm.mlir.undef : {struct_type}']
        curr = f'%__TMP_{name}__'
        fi = 0
        for vi, v in enumerate(args):
            is_last = (vi == len(args) - 1)
            nxt = f'%{name}' if is_last else f'%__TMP_{name}_{vi}__'
            if fi < 3:
                pos = f'[{fi}]'
            else:
                arr_idx = fi - 3
                if arr_idx < rank:
                    pos = f'[3, {arr_idx}]'
                else:
                    pos = f'[4, {arr_idx - rank}]'
            chain.append(f'{indent}{nxt} = llvm.insertvalue {v}, {curr}{pos} : {struct_type}')
            curr = nxt
            fi += 1
        lines[idx] = '\n'.join(chain)

    # ── Phase 5: replace direct struct casts ────────────────────────
    for idx, name, dst, args_str in direct_casts:
        args = [a.strip() for a in args_str.split(',') if a.strip()]
        rank = 0
        m = re.search(r'array<(\d+)\s*x\s*i64>', dst)
        if m:
            rank = int(m.group(1))
        chain = [f'{indent}%__TMP_{name}__ = llvm.mlir.undef : {dst}']
        curr = f'%__TMP_{name}__'
        fi = 0
        for vi, v in enumerate(args):
            is_last = (vi == len(args) - 1)
            nxt = f'%{name}' if is_last else f'%__TMP_{name}_{vi}__'
            if fi < 3:
                pos = f'[{fi}]'
            else:
                arr_idx = fi - 3
                if arr_idx < rank:
                    pos = f'[3, {arr_idx}]'
                else:
                    pos = f'[4, {arr_idx - rank}]'
            chain.append(f'{indent}{nxt} = llvm.insertvalue {v}, {curr}{pos} : {dst}')
            curr = nxt
            fi += 1
        lines[idx] = '\n'.join(chain)

    changes = len(entry_casts) + len(direct_casts)
    for exit_idx, _dst_name, src_name, _struct_type in exit_casts:
        for e_idx, e_name, _e_args, _e_mtype in entry_casts:
            if e_name == src_name and e_idx < exit_idx:
                changes += 1
                break

    _log.warning("Removed %d unrealized_conversion_cast ops from LLVM IR", changes)
    return '\n'.join(lines)


def _fixup_mlir_for_translate(mlir_text: str) -> str:
    """Apply backward-compatibility fixups for LLVM 22 → LLVM 20 translation.

    The C++ ``sf-strip-gep-nuw`` pass handles ``nuw`` stripping when the
    sf-dialect bindings are loaded.  This text-level regex fallback exists
    for environments where the C++ pass is not registered.
    """
    original = mlir_text
    mlir_text = re.sub(r"inbounds\|nuw\b", "inbounds", mlir_text)
    if mlir_text != original:
        _log = logging.getLogger("compiler.compile_utils")
        _log.warning("Applied mlir-translate compatibility fixups (text fallback)")
    return mlir_text


def _fixup_arith_constant_scalar_tensor(mlir_text: str) -> str:
    """Fix arith.constant ops with scalar value + tensor result type.

    Root cause: Upstream MLIR passes (canonicalize, one-shot-bufferize) can
    produce ``arith.constant`` ops where the value attribute is a scalar
    (e.g. ``FloatAttr``) but the result type is a tensor.  Example in generic
    op form::

        %r = "arith.constant"() <{value = 1.25 : f32}> : () -> tensor<1xf32>

    MLIR's ``AllTypesMatch<["value", "result"]>`` trait normally prevents this,
    but in generic op form the verifier allows it during creation, and some
    C++ patterns bypass the builder altogether.

    The correct form uses a ``DenseElementsAttr`` (splat) when the result
    type is a tensor::

        %r = "arith.constant"() <{value = dense<1.25> : tensor<1xf32>}> : () -> tensor<1xf32>

    This fixup works in two phases:

    1. **Regex pre-processing**: An improved regex handles all static tensor
       shapes (not just ``tensor<1xT>``), wrapping scalar values in ``dense<>``.
    2. **MLIR bindings verification**: If available, parses the fixed IR, walks
       all ops (including nested regions), and fixes any remaining mismatches
       using the MLIR Python API.  Falls back gracefully if bindings are absent.

    Returns:
        Fixed MLIR text (or original text if no fixes needed / parsing fails).
    """
    fixed = _fixup_arith_tensor_constants_regex(mlir_text)
    return _fixup_arith_tensor_constants_mlir(fixed)


def _fixup_arith_tensor_constants_regex(mlir_text: str) -> str:
    """Phase 1: Regex-based fixup for scalar-valued arith.constant → dense.

    Matches generic-op-form arith.constant ops where the value is a scalar
    FloatAttr/IntegerAttr but the result type is a static-shaped tensor.
    Handles any static tensor rank (not just ``tensor<1xT>``).
    """
    _pattern = re.compile(
        r'(\s*%\w+\s*=\s*"arith\.constant"\s*\(\)\s*<{value\s*=\s*)'     # prefix
        r'(-?(?:inf|nan|[0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?))'      # scalar value
        r'(\s*:\s*(f32|f64|f16|bf16|i1|i8|i16|i32|i64)\s*}>\s*:\s*\(\)\s*->\s*)'   # type annotation
        r'(tensor<([^>]+)>)(\s*)',                                         # result tensor type
    )

    _lines = mlir_text.split('\n')
    _fix_count = 0
    for _i, _line in enumerate(_lines):
        _m = _pattern.search(_line)
        if not _m:
            continue

        _val = _m.group(2)
        _val_type = _m.group(4)  # e.g. 'f32'
        _tensor_full = _m.group(5)  # e.g. 'tensor<1xf32>'
        _tensor_inner = _m.group(6)  # e.g. '1xf32' or '1x2xf32'

        # Extract element type from the tensor (last component after 'x')
        _parts = _tensor_inner.split('x')
        _tensor_elt_type = _parts[-1]

        # Verify element types match
        if _val_type != _tensor_elt_type:
            continue

        # Skip dynamic shapes (tensor<?xf32> cannot have a dense splat)
        if any('?' in p for p in _parts[:-1]):
            continue

        _lines[_i] = (
            f'{_m.group(1)}dense<{_val}> : {_tensor_full}}}> : () -> {_tensor_full}{_m.group(7)}'
        )
        _fix_count += 1

    if _fix_count:
        _log = logging.getLogger("compiler.fixups")
        _log.warning("Fixed %d arith.constant ops with scalar→tensor type mismatch (regex)", _fix_count)
        return '\n'.join(_lines)
    return mlir_text


def _fixup_arith_tensor_constants_mlir(mlir_text: str) -> str:
    """Phase 2: MLIR-bindings-based verification and fix of arith.constant ops.

    Parses the (already regex-fixed) IR, walks all ops in all regions, and
    converts any remaining ``arith.constant`` ops with scalar value + tensor
    result type to use ``DenseElementsAttr``.  Falls back to the input text
    if MLIR bindings are unavailable or parsing fails.
    """
    try:
        import mlir.ir as ir
    except ImportError:
        return mlir_text

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        try:
            module = ir.Module.parse(mlir_text, ctx)
        except Exception:
            # Even after regex fix, the IR might be invalid for other reasons
            return mlir_text

        _fix_count = _walk_and_fix_tensor_constants(module)

        if _fix_count:
            _log = logging.getLogger("compiler.fixups")
            _log.warning(
                "Fixed %d arith.constant ops with scalar→tensor type mismatch (MLIR)", _fix_count
            )

        # Always re-serialize via str(module) to ensure clean custom-format output.
        # This catches cases where the regex produced valid-but-ugly generic format.
        return str(module)


def _walk_and_fix_tensor_constants(module: ir.Module) -> int:
    """Walk all regions/blocks in ``module`` and fix scalar→tensor arith.constant ops.

    Uses the MLIR Python API (``ir.Operation.walk``) to visit every operation
    and convert ``FloatAttr``/``IntegerAttr`` values on ``arith.constant`` ops
    with tensor result types to ``DenseElementsAttr`` (splat).

    Returns the number of ops fixed.
    """
    import mlir.ir as ir  # noqa: F811 — late import inside function

    _fix_count = 0
    _candidates: list[ir.Operation] = []

    # Phase 2a: Collect candidates (avoid modifying IR while walking).
    # The walk callback must return WalkResult.ADVANCE to continue.
    _walk_result = ir.WalkResult

    def _collector(op: ir.Operation) -> ir.WalkResult:
        if str(op.operation.name) != "arith.constant":
            return _walk_result.ADVANCE

        _result_type = op.operation.result.type
        if not isinstance(_result_type, ir.ShapedType):
            return _walk_result.ADVANCE  # scalar result type — valid as-is

        _value_attr = op.operation.attributes.get("value")
        if _value_attr is None:
            return _walk_result.ADVANCE

        if isinstance(_value_attr, ir.DenseElementsAttr):
            return _walk_result.ADVANCE  # already correct

        if isinstance(_value_attr, (ir.FloatAttr, ir.IntegerAttr)):
            _candidates.append(op.operation)

        return _walk_result.ADVANCE

    module.operation.walk(_collector)

    # Phase 2b: Fix each candidate
    for _op in _candidates:
        _result_type = _op.result.type
        _value_attr = _op.attributes["value"]

        if isinstance(_value_attr, ir.DenseElementsAttr):
            continue  # already correct

        if not isinstance(_value_attr, (ir.FloatAttr, ir.IntegerAttr)):
            continue  # unsupported scalar attribute type

        # Build splat dense attribute
        try:
            _new_attr = ir.DenseElementsAttr.get_splat(_result_type, _value_attr)
            _op.attributes["value"] = _new_attr
            _fix_count += 1
        except Exception:
            _log = logging.getLogger("compiler.fixups")
            _log.debug("Failed to fix arith.constant at %s", _op.location, exc_info=True)

    return _fix_count

