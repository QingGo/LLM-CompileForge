"""Pipeline debug: per-pass intermediate IR + timing.

Each pipeline step reads from a file, runs on a freshly parsed module,
writes result to a new file.  Avoids str(mod) roundtrip bug in PassManager.

Usage:
    python scripts/pipeline_debug.py compiled/opt_125m_fresh
    python scripts/pipeline_debug.py compiled/opt_125m_fresh --timeout 30
"""

import glob
import os
import signal
import sys
import time


def run_pipeline():
    import warnings
    warnings.filterwarnings("ignore")
    import mlir.ir as ir
    import mlir.passmanager as pm

    # ── Config ────────────────────────────────────────────────────────
    artifact_dir = sys.argv[1] if len(sys.argv) > 1 else "compiled/opt_125m_fresh"
    out_dir = "/tmp/pipeline_debug"
    timeout_default = int(next((a for a in sys.argv if a.startswith("--timeout=")), "--timeout=30").split("=")[1])

    os.makedirs(out_dir, exist_ok=True)
    for f in glob.glob(f"{out_dir}/*.mlir") or []:
        os.remove(f)

    mlir_path = os.path.join(artifact_dir, "model.lowered.mlir")
    if not os.path.exists(mlir_path):
        mlir_path = os.path.join(artifact_dir, "model.mlir")
    shutil.copy(mlir_path, f"{out_dir}/00_input.mlir") if 'shutil' in dir() else None

    import shutil
    shutil.copy(mlir_path, f"{out_dir}/00_input.mlir")
    print(f"  [00_input          ] {os.path.getsize(f'{out_dir}/00_input.mlir')/1e6:.2f}MB\n")

    # ── MLIR context ─────────────────────────────────────────────────
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    ctx.load_all_available_dialects()
    try:
        from mlir._mlir_libs import _mlirRegisterEverything
        reg = ir.DialectRegistry()
        _mlirRegisterEverything.register_dialects(reg)
        ctx.append_dialect_registry(reg)
    except Exception:
        pass

    def stage_steps(pipeline_str, in_path, label, is_optional=False):
        """Run pass pipeline on file in_path, write result to label.mlir.
        Returns (out_path, elapsed_seconds, output_text).
        """
        out_path = f"{out_dir}/{label}.mlir"
        mod = ir.Module.parse(open(in_path).read(), ctx)
        t0 = time.time()
        signal.alarm(timeout_default)
        try:
            p = pm.PassManager.parse(pipeline_str, ctx)
            p.run(mod.operation)
            signal.alarm(0)
            elapsed = time.time() - t0
            text = str(mod)
            with open(out_path, "w") as f:
                f.write(text)
            return out_path, elapsed, text
        except Exception as e:
            signal.alarm(0)
            elapsed = time.time() - t0
            if is_optional:
                return in_path, elapsed, open(in_path).read()
            raise

    def stage_custom(fn, label):
        """Run a custom Python function (tiling, vectorize) that modifies mod in-place."""
        out_path = f"{out_dir}/{label}.mlir"
        t0 = time.time()
        result = fn()
        elapsed = time.time() - t0
        return out_path, elapsed, result

    # ── S1: canonicalize + cse ──────────────────────────────────────────
    cur = f"{out_dir}/00_input.mlir"
    cur, e, txt = stage_steps(
        "builtin.module(canonicalize,cse)", cur, "01_canonicalize")
    print(f"  [01_canonicalize   ] {e:6.2f}s  {os.path.getsize(cur)/1e6:.2f}MB")

    # ── S2: linalg-fuse-elementwise-ops (optional) ──────────────────────
    cur, e, txt = stage_steps(
        "builtin.module(linalg-fuse-elementwise-ops,canonicalize,cse)",
        cur, "02_fuse", is_optional=True)

    # ── S3: tile matmul K dim by 64 (per-function transform dialect) ─────
    def _tile():
        script = (
            'module attributes {transform.with_named_sequence} {\n'
            '  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {\n'
            '    %mats = transform.structured.match ops{["linalg.matmul"]} in %arg0\n'
            '      : (!transform.any_op) -> !transform.any_op\n'
            '    transform.structured.tile_using_for %mats tile_sizes [0, 0, 64]\n'
            '      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
            '    transform.yield\n'
            '  }\n'
            '}\n'
        )
        mod = ir.Module.parse(open(cur).read(), ctx)
        block = mod.operation.regions[0].blocks[0]
        for func in list(block):
            if str(func.operation.name) != "func.func":
                continue
            ftxt = str(func)
            if "linalg.matmul" not in ftxt:
                continue
            c = ir.Module.parse(script + "\n" + ftxt, ctx)
            try:
                pm.PassManager.parse(
                    "builtin.module(transform-interpreter)", ctx).run(c.operation)
            except Exception:
                continue
            for op in list(c.operation.regions[0].blocks[0]):
                name = str(op.operation.name)
                src = None
                if name == "func.func":
                    src = op
                elif name == "builtin.module":
                    for inner in op.operation.regions[0].blocks[0]:
                        if str(inner.operation.name) == "func.func":
                            src = inner; break
                if src is not None:
                    cloned = src.operation.clone()
                    func.operation.erase()
                    block.append(cloned)
                    break
        txt = str(mod)
        with open(f"{out_dir}/03_tiled.mlir", "w") as f:
            f.write(txt)
        return txt

    cur, e, txt = stage_custom(_tile, "03_tiled")
    print(f"  [03_tile_matmul    ] {e:6.2f}s  {os.path.getsize(cur)/1e6:.2f}MB"
          f"  for={txt.count('scf.for')} matmul={txt.count('linalg.matmul')} batch={txt.count('linalg.batch_matmul')}")

    # ── S4: canonicalize post-tile ───────────────────────────────────────
    cur, e, txt = stage_steps(
        "builtin.module(canonicalize,cse)", cur, "04_canonicalized")
    print(f"  [04_canonicalize   ] {e:6.2f}s  {os.path.getsize(cur)/1e6:.2f}MB")

    # ── S5: vectorize children of func.func ──────────────────────────────
    def _vectorize():
        vec_script = (
            'module attributes {transform.with_named_sequence} {\n'
            '  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {\n'
            '    %funcs = transform.structured.match ops{["func.func"]} in %arg0\n'
            '      : (!transform.any_op) -> !transform.any_op\n'
            '    transform.structured.vectorize_children_and_apply_patterns %funcs\n'
            '      {create_named_contraction, vectorize_padding}\n'
            '      : (!transform.any_op) -> !transform.any_op\n'
            '    transform.yield\n'
            '  }\n'
            '}\n'
        )
        mod_text = open(cur).read()
        combined = ir.Module.parse(vec_script + "\n" + mod_text, ctx)
        pm.PassManager.parse(
            "builtin.module(transform-interpreter)", ctx).run(combined.operation)
        # Extract func.func from combined back into a fresh module
        m = ir.Module.parse(mod_text, ctx)
        blk = m.operation.regions[0].blocks[0]
        for op in list(blk):
            op.operation.erase()
        for op in list(combined.operation.regions[0].blocks[0]):
            name = str(op.operation.name)
            if name == "func.func":
                blk.append(op.operation.clone())
            elif name == "builtin.module":
                a = list(op.operation.attributes.keys())
                if "transform.with_named_sequence" in a:
                    continue
                for inner in op.operation.regions[0].blocks[0]:
                    if str(inner.operation.name) == "func.func":
                        blk.append(inner.operation.clone())
        txt = str(m)
        with open(f"{out_dir}/05_vectorized.mlir", "w") as f:
            f.write(txt)
        return txt

    cur, e, txt = stage_custom(_vectorize, "05_vectorized")
    print(f"  [05_vectorize      ] {e:6.2f}s  {os.path.getsize(cur)/1e6:.2f}MB"
          f"  contracts={txt.count('vector.contract')} matmul={txt.count('linalg.matmul')} batch={txt.count('linalg.batch_matmul')}")

    # ── S6: one-shot-bufferize ──────────────────────────────────────────
    cur, e, txt = stage_steps(
        "builtin.module(one-shot-bufferize{bufferize-function-boundaries},canonicalize,cse)",
        cur, "06_bufferized")
    print(f"  [06_bufferize      ] {e:6.2f}s  {os.path.getsize(cur)/1e6:.2f}MB"
          f"  tensor.={txt.count('tensor.')} forall={txt.count('scf.forall')}")

    # ── S7: linalg-to-loops ────────────────────────────────────────────
    cur, e, _ = stage_steps(
        "builtin.module(convert-bufferization-to-memref,convert-linalg-to-loops,lower-affine,convert-scf-to-cf)",
        cur, "07_loops")
    print(f"  [07_loops          ] {e:6.2f}s  {os.path.getsize(cur)/1e6:.2f}MB")

    # ── S8: expand + lower-vector-mask + convert-vector-to-scf ──────────
    cur, e, txt = stage_steps(
        "builtin.module(expand-strided-metadata,lower-affine,"
        "func.func(lower-vector-mask),func.func(convert-vector-to-scf),"
        "canonicalize,cse,convert-scf-to-cf,lower-affine)",
        cur, "08_vec_scf")
    print(f"  [08_vec_scf        ] {e:6.2f}s  {os.path.getsize(cur)/1e6:.2f}MB"
          f"  vector.={txt.count('vector.')}")

    # ── S9: cf-to-llvm + finalize-memref-to-llvm ────────────────────────
    cur, e, txt = stage_steps(
        "builtin.module(convert-cf-to-llvm,"
        "finalize-memref-to-llvm{use-generic-functions=false})",
        cur, "09_memref_llvm")
    print(f"  [09_memref_llvm    ] {e:6.2f}s  {os.path.getsize(cur)/1e6:.2f}MB"
          f"  contracts={txt.count('vector.contract')}")

    # ── S10: convert-vector-to-llvm (default strategy=dot) ──────────────
    print(f"  [10_vec_to_llvm    ] ", end="", flush=True)
    signal.alarm(timeout_default)
    t0 = time.time()
    try:
        mod = ir.Module.parse(open(cur).read(), ctx)
        pm.PassManager.parse(
            "builtin.module(convert-vector-to-llvm)", ctx
        ).run(mod.operation)
        signal.alarm(0)
        elapsed = time.time() - t0
        cur = f"{out_dir}/10_vec_to_llvm.mlir"
        txt10 = str(mod)
        with open(cur, "w") as f:
            f.write(txt10)
        print(f"{elapsed:6.2f}s  {os.path.getsize(cur)/1e6:.2f}MB"
              f"  contracts={txt10.count('vector.contract')} reduce={txt10.count('vector.reduce')}")
    except Exception as e:
        signal.alarm(0)
        elapsed = time.time() - t0
        print(f"FAIL after {elapsed:.1f}s: {e}")

    # ── S11: finalize (convert remaining dialects) ───────────────────────
    cur, e, txt = stage_steps(
        "builtin.module(convert-cf-to-llvm,convert-math-to-llvm,"
        "convert-arith-to-llvm,convert-ub-to-llvm,"
        "convert-func-to-llvm,reconcile-unrealized-casts)",
        cur, "11_final")
    print(f"  [11_finalize       ] {e:6.2f}s  {os.path.getsize(cur)/1e6:.2f}MB")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\nFiles in {out_dir}/:")
    for f in sorted(glob.glob(f"{out_dir}/*.mlir")):
        print(f"  {os.path.basename(f)}: {os.path.getsize(f)/1e6:.1f}MB")
    print(f"\nFinal: {txt.count('vector.contract')} contracts, {len(txt):,} chars")


if __name__ == "__main__":
    # signal handler for SIGALRM timeouts
    def _timeout_handler(s, f):
        raise TimeoutError("stage timed out")
    signal.signal(signal.SIGALRM, _timeout_handler)
    run_pipeline()
