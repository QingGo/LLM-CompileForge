"""Lowered IR verification and failure diagnostic tools."""

import logging
import re
import shutil
import traceback
from datetime import datetime
from pathlib import Path


def _verify_lowered_ir(lowered_text: str) -> None:
    """Verify lowered IR contains no illegal ops that would fail re-parse."""
    errors: list[str] = []

    # 1. No bare arith ops on tensors at module level (must be inside linalg.generic)
    bare_arith = re.findall(
        r'%\d+\s+=\s+"(arith\.\w+)"\(',
        lowered_text,
    )
    if bare_arith:
        tensor_arith = re.findall(
            r'"arith\.(mul|add|sub|div)f".*tensor<',
            lowered_text,
        )
        if tensor_arith:
            for op in tensor_arith:
                errors.append(
                    f"Bare arith.{op}f on tensor detected — should be inside linalg.generic"
                )

    # 2. No unresolved sf.* ops (except sf.weight/sf.constant which are handled later)
    sf_ops = set(re.findall(r'"sf\.(\w+)"', lowered_text))
    sf_ignored = {"weight", "constant"}
    unresolved = sf_ops - sf_ignored
    if unresolved:
        errors.append(f"Unresolved sf ops remaining: {sorted(unresolved)}")

    # 3. Must contain at least one linalg op (sanity check)
    if "linalg." not in lowered_text and "scf." not in lowered_text:
        errors.append("No linalg or scf ops found — lowering may have produced nothing")

    # 4. Warn about 0D tensors (tensor<f32>, tensor<i64>, etc. — no dimensions)
    zero_dim_tensors = re.findall(
        r'tensor<(f32|f64|i1|i8|i16|i32|i64)>',
        lowered_text,
    )
    if zero_dim_tensors:
        logging.warning(
            "Lowered IR contains %d zero-dimensional tensor(s) — "
            "types: %s",
            len(zero_dim_tensors),
            sorted(set(zero_dim_tensors)),
        )

    if errors:
        raise ValueError(
            "Lowered IR verification failed:\n  - " + "\n  - ".join(errors)
        )


def _save_failure_context(
    step_num: str,
    pass_name: str,
    compiled_path: Path,
    ir_text: str | None = None,
    copy_source: str | None = None,
) -> Path:
    """Save diagnostic context on pipeline failure and print diagnosis guide."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    failure_dir = Path("outputs/logs/pipeline") / f"failure_{timestamp}"
    failure_dir.mkdir(parents=True, exist_ok=True)

    if ir_text is not None:
        ir_path = failure_dir / "model.snapshot.mlir"
        ir_path.write_text(ir_text)
        print(f"   Saved IR snapshot: {ir_path}")

    if copy_source is not None:
        src = Path(copy_source)
        if src.exists():
            shutil.copy2(str(src), str(failure_dir / "source.mlir"))
            print(f"   Saved source MLIR: {failure_dir / 'source.mlir'}")

    error_path = failure_dir / "error.txt"
    with open(error_path, "w") as f:
        f.write(f"Step: [{step_num}/5] ({pass_name})\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Compiled dir: {compiled_path}\n")
        f.write("-" * 60 + "\n")
        traceback.print_exc(file=f)
    print(f"   Error details saved to: {error_path}")

    print(f"\n❌ Pipeline failed at step [{step_num}/5] ({pass_name})")
    print(f"📍 Diagnostic context saved to: {failure_dir}")
    print("📋 Suggested next steps:")
    print("   1. Check saved IR files for unexpected ops")
    print("   2. Run: python scripts/bisect_pipeline_stages.py --auto")
    print("   3. Or use: python compiler/compile_dylib.py --debug for per-pass snapshots")

    return failure_dir
