# ruff: noqa: E501 — long lines in MLIR fixup f-strings

"""Pure-text-manipulation fixups for MLIR LLVM dialect IR.

These functions perform regex-based text transformations on LLVM dialect
IR text, replacing unrealized_conversion_cast chains with equivalent LLVM
ops, stripping version-incompatible flags, and lowering vector-typed arith
ops that the pass pipeline cannot handle.

All functions operate on strings (not MLIR Python bindings objects) and
can be called safely without an MLIR Context.
"""

from __future__ import annotations

import re


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


def _replace_dense(m: re.Match) -> str:
    """Replace ``arith.constant dense<...> : vector<...>`` with scalar+broadcast."""
    name = m.group(1)
    vec_type = m.group(2)
    return (
        f"%{name}__scalar = arith.constant 0.000000e+00 : f32\n"
        f"  %{name} = vector.broadcast %{name}__scalar : f32 to {vec_type}"
    )


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

    return '\n'.join(lines)


def _fixup_mlir_for_translate(mlir_text: str) -> str:
    """Apply backward-compatibility fixups for LLVM 22 → LLVM 20 translation.

    LLVM 22's MLIR LLVM dialect adds flags (e.g. ``nuw`` on getelementptr)
    that LLVM 20's mlir-translate cannot parse.  These are semantically
    optional / already implicit, so stripping them is safe.
    """
    mlir_text = re.sub(r"inbounds\|nuw\b", "inbounds", mlir_text)
    return mlir_text


def _fixup_vector_arith_constant(mlir_text: str) -> str:
    """Replace ``arith.constant dense<...> : vector<...>`` with
    ``arith.constant scalar + vector.broadcast`` pattern, and convert
    vector-typed ``arith.{mul,add,sub}f`` to ``llvm.{fmul,fadd,fsub}``.

    The in-tree vectorizer (``create_named_contraction``) creates
    ``arith.constant dense<0> : vector<NxMxf32>`` and ``arith.mulf/addf
    : vector<NxMxf32>`` ops that ``convert-arith-to-llvm`` cannot handle
    (it only handles scalar types).  This fixup runs **inside** the
    pipeline, so ``convert-vector-to-llvm`` can lower the
    ``vector.broadcast`` ops produced here.

    Replacements:
    - ``arith.constant dense<0> : vector<A,B,...>``
      → ``arith.constant 0.0 : f32`` + ``vector.broadcast``
    - ``arith.mulf %a, %b : vector<N...>`` → ``llvm.fmul %a, %b``
    - ``arith.addf %a, %b : vector<N...>`` → ``llvm.fadd %a, %b``
    - ``arith.subf %a, %b : vector<N...>`` → ``llvm.fsub %a, %b``
    """
    mlir_text = re.sub(
        r'%(\w+)\s*=\s*arith\.constant\s+dense<[^>]+>\s*:\s*(vector<\d+(?:x\d+)*xf32>)',
        _replace_dense,
        mlir_text,
    )

    # arith.mulf / addf / subf followed by vector type → llvm dialect version
    for arith_op, llvm_op in [
        ("mulf", "fmul"),
        ("addf", "fadd"),
        ("subf", "fsub"),
    ]:
        mlir_text = re.sub(
            rf'arith\.{arith_op}(\s+(?:%\w+\s*,?\s*)+:\s*)vector<',
            rf'llvm.{llvm_op}\1vector<',
            mlir_text,
        )

    return mlir_text
