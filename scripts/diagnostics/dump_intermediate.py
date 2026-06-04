#!/usr/bin/env python3
"""Extract and dump a single SSA value from lowered MLIR using the MLIR API.

Usage:
    python scripts/dump_intermediate.py --list <mlir_file> [function]
    python scripts/dump_intermediate.py <mlir_file> <function> <ssa_name>
"""
import ctypes
import os
import re
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    if sys.argv[1] == "--list":
        _list(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
        return
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    result = _dump(sys.argv[1], sys.argv[2], sys.argv[3])
    if result is not None:
        print(f"Shape: {result.shape}\nValues: {result}")
    else:
        print("ERROR: failed")


# ═══════════════════════════════════════════════════════════════════
#  --list
# ═══════════════════════════════════════════════════════════════════

def _list(mlir_path: str, func_filter: str | None = None) -> None:
    import mlir.ir as ir
    ctx = ir.Context(); ctx.allow_unregistered_dialects = True
    with ctx, open(mlir_path) as f:
        mod = ir.Module.parse(f.read(), ctx)

    for op in mod.body.operations:
        if "sym_name" not in op.attributes: continue
        name = str(op.attributes["sym_name"]).strip('"')
        if func_filter and name != func_filter: continue
        print(f"\n=== {name} ===")
        for blk in op.body.blocks:
            for sub in blk.operations:
                if sub.results:
                    print(f"  {sub.results[0].get_name()}: {sub.name}")


# ═══════════════════════════════════════════════════════════════════
#  Extract + dump
# ═══════════════════════════════════════════════════════════════════

def _dump(mlir_path: str, func_name: str, ssa_name: str) -> np.ndarray | None:
    import mlir.ir as ir

    ssa = ssa_name if ssa_name.startswith("%") else f"%{ssa_name}"
    ctx = ir.Context(); ctx.allow_unregistered_dialects = True
    with ctx, open(mlir_path) as f:
        mod = ir.Module.parse(f.read(), ctx)

    # Find target function
    func_op = None
    for op in mod.body.operations:
        if "sym_name" in op.attributes and str(op.attributes["sym_name"]).strip('"') == func_name:
            func_op = op
            break
    if func_op is None:
        print(f"Function {func_name} not found")
        return None

    # Collect all ops in the function body
    body_ops = list(func_op.body.blocks[0].operations)

    # Find the target value and its index
    target_idx = None
    target_type = None
    for i, op in enumerate(body_ops):
        for r in op.results:
            if r.get_name() == ssa:  # get_name() returns '%name'
                target_idx = i
                target_type = r.type
                break
        if target_idx is not None:
            break

    if target_idx is None:
        print(f"SSA {ssa_name} not found")
        return None

    print(f"Target: {ssa_name} : {target_type}  (op #{target_idx} of {len(body_ops)})")

    # Build new module with ops up to target, returning the target value
    out_mod = ir.Module.create(loc=ir.Location.unknown(ctx))
    with ir.InsertionPoint(out_mod.body), ir.Location.unknown(ctx):
        ftype = ir.FunctionType.get([], [target_type])
        fop = ir.Operation.create(
            "func.func", regions=1, attributes={
                "function_type": ir.TypeAttr.get(ftype),
                "sym_name": ir.StringAttr.get("main"),
            },
        )
        new_blk = fop.regions[0].blocks.append()
        with ir.InsertionPoint(new_blk):
            val_map = {}
            for i in range(target_idx + 1):
                op = body_ops[i]
                operands = []
                for v in op.operands:
                    operands.append(val_map.get(v.get_name(), v))
                new_op = ir.Operation.create(
                    op.name,
                    results=[r.type for r in op.results],
                    operands=operands,
                    regions=len(op.regions),
                )
                for old_r, new_r in zip(op.results, new_op.results):
                    val_map[old_r.get_name()] = new_r
                # Copy regions (scf.for etc)
                for ri, region in enumerate(op.regions):
                    _copy_region(region, new_op.regions[ri], val_map)

            # Build return op using the cloned target value
            result_val = val_map[ssa]  # ssa is the raw name like '%20' (which get_name() also returns)
            ir.Operation.create("func.return", operands=[result_val])

    return _jit_run(out_mod)


def _copy_region(src, dst, val_map):
    """Shallow region copy for scf.for etc."""
    import mlir.ir as ir
    loc = ir.Location.unknown()
    for bi, old_block in enumerate(src.blocks):
        new_block = dst.blocks.append()
        for arg in old_block.arguments:
            new_arg = new_block.add_argument(arg.type, loc)
            val_map[arg.get_name()] = new_arg
        with ir.InsertionPoint(new_block):
            for op in old_block.operations:
                operands = [val_map.get(v.get_name(), v) for v in op.operands]
                new_op = ir.Operation.create(
                    op.name,
                    results=[r.type for r in op.results],
                    operands=operands,
                    regions=len(op.regions),
                )
                for old_r, new_r in zip(op.results, new_op.results):
                    val_map[old_r.get_name()] = new_r
                for ri, region in enumerate(op.regions):
                    _copy_region(region, new_op.regions[ri], val_map)


# ═══════════════════════════════════════════════════════════════════
#  JIT compile + run
# ═══════════════════════════════════════════════════════════════════

def _jit_run(mod) -> np.ndarray | None:
    from compiler.mlir_dialect.lowering.compile_utils import _find_cc
    from compiler.mlir_dialect.lowering.llvm_backend import lower_linalg_to_llvm_ir

    try:
        lower_linalg_to_llvm_ir(mod)
    except Exception as e:
        print(f"Lowering failed: {e}")
        return None

    ll_text = str(mod)

    with tempfile.TemporaryDirectory() as tmp:
        ll = Path(tmp) / "dump.ll"
        obj = Path(tmp) / "dump.o"
        dylib = Path(tmp) / "dump.dylib"
        ll.write_text(ll_text)
        cc = _find_cc()

        if os.system(f"{cc} -c -O2 {ll} -o {obj} 2>/dev/null") != 0:
            print("Compile failed")
            return None

        src = Path(__file__).resolve()
        lib_dir = src.parent.parent / "llvm-project" / "build" / "lib"
        if os.system(f"{cc} -shared {obj} -o {dylib} -L {lib_dir} -lmlir_c_runner_utils -Wl,-rpath,{lib_dir}") != 0:
            print("Link failed")
            return None

        lib = ctypes.CDLL(str(dylib))
        sret = (ctypes.c_uint8 * 131072)()
        lib._mlir_ciface_main(ctypes.byref(sret))

        return _parse_sret(sret, mod)


def _parse_sret(sret: bytes, mod) -> np.ndarray | None:
    import mlir.ir as ir

    # Get return type from module
    for op in mod.body.operations:
        if "sym_name" in op.attributes:
            ftype = ir.FunctionType(ir.TypeAttr(op.attributes["function_type"]).value)
            type_str = str(ftype.results[0])
            break
    else:
        return None

    m = re.match(r"tensor<([^>]+)>", type_str)
    if not m: return None

    parts = m.group(1).split("x")
    elt = parts[-1]; rank = len(parts)

    aligned = struct.unpack_from("<Q", sret, 8)[0]
    if aligned == 0: return None

    sizes = []
    for i in range(rank):
        sz = struct.unpack_from("<q", sret, 24 + 8 * i)[0]
        sizes.append(max(sz, 1))

    n = 1
    for s in sizes: n *= s

    ct_map = {"f32": ctypes.c_float, "f64": ctypes.c_double, "i64": ctypes.c_int64}
    np_map = {"f32": np.float32, "f64": np.float64, "i64": np.int64}
    ct = ct_map.get(elt, ctypes.c_int64)
    arr = np.ctypeslib.as_array(ctypes.cast(aligned, ctypes.POINTER(ct)), shape=(n,)).copy()
    return arr.reshape(sizes).astype(np_map.get(elt, np.int64))


if __name__ == "__main__":
    main()
