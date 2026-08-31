#!/usr/bin/env python3
"""Op-level MLIR reducer within a single function.

Given a compiled function's MLIR and an interestingness test, binary-deletes
operations within the function body to find the minimal reproducing set.

Uses MLIR Python API for module-level parsing (from reduce_mlir.py) and
text-based op splitting for candidate building to avoid MLIR printer timeouts
on IR with dangling SSA references.

Usage:
    python scripts/diagnostics/reduce_function_ops.py model.mlir \\
        --function main_1 --interestingness "python check.py {}" --output reduced.mlir
    python scripts/diagnostics/reduce_function_ops.py model.mlir \\
        --function main_1 --metric ops --interestingness "true" --output reduced.mlir
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path

# Reuse MLIR infrastructure from reduce_mlir (same directory)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reduce_mlir import (  # noqa: E402  # pyright: ignore[reportImplicitRelativeImport]
    GlobalCounter,
    InterestingnessTest,
    PassInterestingness,
    ShellInterestingness,
    _get_func_names,
    _get_mlir_ctx,
    _preprocess_mlir,
)

_log = logging.getLogger("reduce_function_ops")


# =========================================================================
# Extended preprocessing (handles #sf<...> dialect attributes)
# =========================================================================


def _preprocess_mlir_aggressive(mlir_text: str) -> str:
    """Strip all sf dialect attributes including #sf<...> dialect attrs.

    Extends _preprocess_mlir() to also handle sf dialect attributes using the
    ``#sf<...>`` syntax that the MLIR parser cannot handle without the sf
    dialect library loaded.
    """
    stripped = _preprocess_mlir(mlir_text)
    # Strip sf.* = #sf<...> dialect attributes (on func.func and elsewhere)
    stripped = re.sub(r"\bsf\.\w+\s*=\s*#sf<[^>]*>", "", stripped)
    # Clean up artifacts: empty attribute dicts, double semicolons
    stripped = re.sub(r";\s*;", ";", stripped)
    stripped = re.sub(r"\{\s*;\s*", "{", stripped)
    stripped = re.sub(r"\{\s*,\s*\}", "{}", stripped)
    stripped = re.sub(r"attributes\s*\{\s*,\s*\}\s*", "", stripped)
    return stripped


# =========================================================================
# Function extraction
# =========================================================================


def extract_function_mlir(mlir_text: str, func_name: str) -> str:
    """Extract a single function's MLIR and wrap it in a new module.

    Uses the MLIR API to find the function and returns a valid ``module { ... }``
    wrapper containing only that function.

    Args:
        mlir_text: Full module MLIR text.
        func_name: Symbol name of the function to extract.

    Returns:
        MLIR text of a module containing only the named function.

    Raises:
        ValueError: If *func_name* is not found.
    """
    import mlir.ir as ir  # noqa: E402

    ctx = _get_mlir_ctx()
    parseable = _preprocess_mlir_aggressive(mlir_text)

    with ir.Location.unknown(ctx):
        mod = ir.Module.parse(parseable, ctx)

    body = mod.operation.regions[0].blocks[0]
    for op in body.operations:
        if str(op.operation.name) == "func.func":
            attr = op.operation.attributes.get("sym_name")
            if str(attr).strip('"') == func_name:
                return "module {\n" + str(op) + "\n}\n"

    raise ValueError(f"Function '{func_name}' not found in MLIR module")


# =========================================================================
# Op-level text splitting (brace-depth tracking)
# =========================================================================


def _split_body_ops(body_lines: list[str]) -> list[tuple[str, bool]]:
    """Split function body lines into top-level operations.

    Uses brace-depth tracking to handle nested regions (scf.for, linalg.generic
    blocks, etc.) correctly. Returns list of ``(op_text, is_return)`` tuples.

    Args:
        body_lines: Lines of the function body (between header ``{`` and footer ``}``).

    Returns:
        List of ``(op_text, is_return)`` where *is_return* is True for
        ``return`` / ``func.return`` operations.
    """
    ops: list[tuple[str, bool]] = []
    current_lines: list[str] = []
    brace_depth = 0
    base_indent: int | None = None

    for line in body_lines:
        stripped = line.strip()

        # Preserve blank lines within multi-line ops
        if not stripped:
            if current_lines:
                current_lines.append(line)
            continue

        indent = len(line) - len(line.lstrip())
        if base_indent is None:
            base_indent = indent

        # A new top-level operation starts when:
        #   - we have accumulated lines from a previous op
        #   - brace depth is back to 0 (not inside a nested region)
        #   - current line is at the base indentation level
        if current_lines and brace_depth == 0 and indent == base_indent:
            op_text = "\n".join(current_lines)
            is_return = _is_return_op_text(op_text)
            ops.append((op_text, is_return))
            current_lines = [line]
        else:
            current_lines.append(line)

        brace_depth += stripped.count("{") - stripped.count("}")

    if current_lines:
        op_text = "\n".join(current_lines)
        is_return = _is_return_op_text(op_text)
        ops.append((op_text, is_return))

    return ops


def _is_return_op_text(text: str) -> bool:
    """Check if an MLIR operation text is a function return."""
    stripped = text.strip()
    return stripped.startswith("return") or stripped.startswith("func.return")


# =========================================================================
# Candidate building (text-based)
# =========================================================================


def _parse_func_text(func_text: str) -> tuple[str, list[str], str]:
    """Parse a function's MLIR text into header, body op texts, and footer.

    Args:
        func_text: Full MLIR text of a single ``func.func`` operation.

    Returns:
        Tuple of ``(header, op_texts, footer)``.
    """
    lines = func_text.split("\n")
    if len(lines) < 3:
        raise ValueError("Function text too short to parse")

    header_line = lines[0]  # e.g. "  func.func @name(...) -> (...) {"
    body_lines = lines[1:-1]
    footer_line = lines[-1]  # e.g. "  }"

    ops = _split_body_ops(body_lines)
    op_texts = [op[0] for op in ops]

    return header_line, op_texts, footer_line


def _build_func_module(header: str, op_texts: list[str], footer: str) -> str:
    """Build a module containing a single function with the given body ops."""
    body = "\n".join(op_texts)
    func_text = header + "\n" + body + "\n" + footer
    return "module {\n" + func_text + "\n}\n"


def _build_candidate(
    func_text: str, keep_indices: set[int]
) -> str | None:
    """Build a module with only the specified ops kept in the function.

    Args:
        func_text: Full MLIR text of a single function.
        keep_indices: Set of operation indices (0-based) to keep.

    Returns:
        Module MLIR text, or None if building fails.
    """
    try:
        header, op_texts, footer = _parse_func_text(func_text)
        kept = [op_texts[i] for i in sorted(keep_indices) if i < len(op_texts)]
        return _build_func_module(header, kept, footer)
    except Exception:
        return None


# =========================================================================
# Reduction strategies
# =========================================================================


def reduce_ops_in_function(
    mlir_text: str,
    func_name: str,
    interesting: Callable[[str], bool],
    retry: bool = True,
) -> str:
    """One-at-a-time op deletion within a single function.

    Parses the function via MLIR API, splits its body into top-level operations
    using text analysis, and removes operations one at a time. Never deletes
    ``return`` / ``func.return`` operations.

    Args:
        mlir_text: Full module MLIR text.
        func_name: Symbol name of the function to reduce.
        interesting: Callable that returns True if the MLIR reproduces the bug.
        retry: If True, restarts scanning from the beginning after each deletion.

    Returns:
        Reduced MLIR text (single-function module).
    """
    import mlir.ir as ir  # noqa: E402

    ctx = _get_mlir_ctx()
    parseable = _preprocess_mlir_aggressive(mlir_text)

    # Get function text via MLIR API (reliable formatting)
    with ir.Location.unknown(ctx):
        mod = ir.Module.parse(parseable, ctx)

    func_op = None
    body = mod.operation.regions[0].blocks[0]
    for op in body.operations:
        if str(op.operation.name) == "func.func":
            attr = op.operation.attributes.get("sym_name")
            if str(attr).strip('"') == func_name:
                func_op = op
                break

    if func_op is None:
        raise ValueError(f"Function '{func_name}' not found")

    func_text = str(func_op)
    header, op_texts, footer = _parse_func_text(func_text)
    n_original = len(op_texts)

    # Identify return op indices (never delete these)
    return_indices: set[int] = set()
    for i, text in enumerate(op_texts):
        if _is_return_op_text(text):
            return_indices.add(i)

    n_deletable = n_original - len(return_indices)
    _log.info(
        "  Function '%s': %d top-level ops, %d return ops → %d deletable",
        func_name, n_original, len(return_indices), n_deletable,
    )

    # Build the initial module text for candidate construction
    initial_module = _build_func_module(header, op_texts, footer)

    # One-at-a-time removal
    keep_indices: list[int] = list(range(n_original))
    i = 0
    deletions = 0
    iterations = 0

    while i < len(keep_indices):
        iterations += 1
        idx = keep_indices[i]

        # Never delete return ops
        if idx in return_indices:
            i += 1
            continue

        # Build candidate without this op
        candidate_indices = set(keep_indices) - {idx}
        candidate = _build_candidate(func_text, candidate_indices)

        if candidate is not None and interesting(candidate):
            keep_indices = keep_indices[:i] + keep_indices[i + 1 :]
            deletions += 1
            if retry:
                i = 0  # restart from beginning
            if deletions % 10 == 0:
                _log.info(
                    "  ... %d ops deleted (%d remaining)",
                    deletions, len(keep_indices),
                )
        else:
            i += 1

    _log.info(
        "  Op reduction: %d → %d ops (%d deletions in %d iterations)",
        n_original, len(keep_indices), deletions, iterations,
    )

    # Build final result
    final_indices = set(keep_indices)
    result = _build_candidate(func_text, final_indices)
    if result is None:
        _log.warning("  Final build failed, returning original single-function module")
        return initial_module
    return result


def reduce_ops_binary(
    mlir_text: str,
    func_name: str,
    interesting: Callable[[str], bool],
) -> str:
    """Binary search on op ranges to find the minimal reproducing set.

    First finds the minimal prefix [0..pivot] that still reproduces the bug,
    then does one-at-a-time removal within that prefix.

    Args:
        mlir_text: Full module MLIR text.
        func_name: Symbol name of the function to reduce.
        interesting: Callable that returns True if the MLIR reproduces the bug.

    Returns:
        Reduced MLIR text (single-function module).
    """
    import mlir.ir as ir  # noqa: E402

    ctx = _get_mlir_ctx()
    parseable = _preprocess_mlir_aggressive(mlir_text)

    with ir.Location.unknown(ctx):
        mod = ir.Module.parse(parseable, ctx)

    func_op = None
    body = mod.operation.regions[0].blocks[0]
    for op in body.operations:
        if str(op.operation.name) == "func.func":
            attr = op.operation.attributes.get("sym_name")
            if str(attr).strip('"') == func_name:
                func_op = op
                break

    if func_op is None:
        raise ValueError(f"Function '{func_name}' not found")

    func_text = str(func_op)
    header, op_texts, footer = _parse_func_text(func_text)
    n_original = len(op_texts)

    # Identify return op indices
    return_indices: set[int] = set()
    for i, text in enumerate(op_texts):
        if _is_return_op_text(text):
            return_indices.add(i)

    n_deletable = n_original - len(return_indices)

    # Build initial module for fallback
    initial_module = _build_func_module(header, op_texts, footer)

    # Exclude return ops for binary search range
    if n_deletable == 0:
        _log.info("  No deletable ops — nothing to reduce")
        return initial_module

    # Binary search: find the minimal prefix that includes the essential op(s)
    # We binary search on deletable indices only
    deletable_indices = [i for i in range(n_original) if i not in return_indices]
    lo, hi = 0, len(deletable_indices) - 1
    iterations = 0

    while lo < hi:
        iterations += 1
        mid = (lo + hi) // 2
        # Keep ops 0..deletable_indices[mid] + all return ops
        max_op_idx = deletable_indices[mid]
        keep = set(range(max_op_idx + 1)) | return_indices
        candidate = _build_candidate(func_text, keep)

        if candidate is not None and interesting(candidate):
            hi = mid
        else:
            lo = mid + 1

    pivot = deletable_indices[lo]
    _log.info(
        "  Binary search: pivot at op %d/%d (%d deletable indices, %d iterations)",
        pivot, n_original, n_deletable, iterations,
    )

    # Now do one-at-a-time removal within ops [0..pivot] + return ops
    keep = set(range(pivot + 1)) | return_indices
    keep_list = sorted(keep)

    i = 0
    deletions = 0
    while i < len(keep_list):
        idx = keep_list[i]
        if idx in return_indices:
            i += 1
            continue

        candidate = _build_candidate(func_text, keep - {idx})
        if candidate is not None and interesting(candidate):
            keep.discard(idx)
            keep_list = sorted(keep)
            deletions += 1
            i = 0
        else:
            i += 1

    _log.info(
        "  Fine reduction: %d → %d ops (%d deletions)",
        pivot + 1 + len(return_indices), len(keep), deletions,
    )

    result = _build_candidate(func_text, keep)
    if result is None:
        _log.warning("  Final build failed, returning initial module")
        return initial_module
    return result


# =========================================================================
# CLI
# =========================================================================


def build_interestingness(args: argparse.Namespace) -> InterestingnessTest:
    if args.pass_name:
        return PassInterestingness(args.pass_name, args.pass_timeout)
    if args.interestingness:
        return ShellInterestingness(args.interestingness, args.timeout)
    raise ValueError("Must specify either --interestingness or --pass")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Op-level MLIR reducer within a single function",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    _ = parser.add_argument("input", help="Input MLIR file")
    _ = parser.add_argument(
        "--function", "-f", required=True,
        help="Function name to reduce (e.g., main_1)",
    )
    _ = parser.add_argument(
        "--output", "-o", default="",
        help="Output file (default: <input>.<func>.reduced.mlir)",
    )
    _ = parser.add_argument(
        "--interestingness", metavar="CMD",
        help="Shell command, {} = temp file path, exit 0 = bug present",
    )
    _ = parser.add_argument(
        "--pass", dest="pass_name", metavar="PIPELINE",
        help="MLIR pass pipeline; crash/hang = bug present",
    )
    _ = parser.add_argument("--pass-timeout", type=float, default=30.0)
    _ = parser.add_argument("--timeout", type=float, default=60.0)
    _ = parser.add_argument(
        "--metric", choices=["lines", "ops", "bytes"],
        help="Global counter metric (minimize while still interesting)",
    )
    _ = parser.add_argument(
        "--strategy", choices=["ops", "binary"], default="ops",
        help="Reduction strategy: 'ops' = one-at-a-time, 'binary' = binary search + fine (default: ops)",
    )
    _ = parser.add_argument(
        "--no-retry", action="store_true",
        help="Disable retry (single pass only)",
    )
    _ = parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",
    )

    input_path = Path(args.input)
    if not input_path.exists():
        _log.error("Input file not found: %s", input_path)
        return 1

    mlir_text = input_path.read_text()
    original_lines = len(mlir_text.splitlines())
    _log.info("Input: %s (%d lines)", input_path, original_lines)

    # Verify function exists
    import mlir.ir as ir  # noqa: E402

    ctx = _get_mlir_ctx()
    parseable = _preprocess_mlir_aggressive(mlir_text)
    with ir.Location.unknown(ctx):
        mod = ir.Module.parse(parseable, ctx)
    func_names = _get_func_names(mod)
    _log.info("Module has %d functions: %s", len(func_names), ", ".join(func_names[:5]) +
              ("..." if len(func_names) > 5 else ""))

    if args.function not in func_names:
        _log.error("Function '%s' not found in module", args.function)
        _log.info("Available functions: %s", ", ".join(func_names))
        return 1

    # Get op count for the target function
    for op in mod.operation.regions[0].blocks[0].operations:
        if str(op.operation.name) == "func.func":
            attr = op.operation.attributes.get("sym_name")
            if str(attr).strip('"') == args.function:
                func_body = op.operation.regions[0].blocks[0]
                n_ops = len(list(func_body.operations))
                _log.info("Function '%s': %d top-level ops", args.function, n_ops)
                break

    # Build interestingness test
    try:
        interesting = build_interestingness(args)
    except ValueError as e:
        _log.error(str(e))
        return 1

    # Wrap with GlobalCounter if metric specified
    if args.metric:
        import hashlib

        ihash = hashlib.md5(input_path.read_bytes()).hexdigest()[:8]
        interesting = GlobalCounter(interesting, args.metric, ihash)
        _log.info("Global counter enabled: metric=%s", args.metric)

    # Validate initial input is interesting
    # Build single-function module for initial test
    initial_func_mod = extract_function_mlir(mlir_text, args.function)
    _log.info("Validating initial function is interesting...")
    if not interesting.is_interesting(initial_func_mod):
        _log.error(
            "Initial function is NOT interesting — check your interestingness test"
        )
        return 1
    _log.info("  Initial function is interesting ✓")

    # Run reduction
    t0 = time.perf_counter()
    if args.strategy == "binary":
        reduced_text = reduce_ops_binary(
            mlir_text, args.function, interesting.is_interesting,
        )
    else:
        reduced_text = reduce_ops_in_function(
            mlir_text, args.function, interesting.is_interesting,
            retry=not args.no_retry,
        )
    elapsed = time.perf_counter() - t0

    reduced_lines = len(reduced_text.splitlines())
    reduction_pct = (
        (1 - reduced_lines / max(original_lines, 1)) * 100
    )

    # Count ops in reduced output
    reduced_op_count = _count_ops_in_text(reduced_text)

    # Output
    if not args.output:
        stem = input_path.stem
        output_path = str(input_path).replace(
            input_path.name, f"{stem}.{args.function}.reduced.mlir"
        )
    else:
        output_path = args.output
    _ = Path(output_path).write_text(reduced_text)

    _log.info("=" * 60)
    _log.info("Reduction complete in %.1fs", elapsed)
    _log.info("  Module lines: %d → %d (%.1f%% reduction)",
              original_lines, reduced_lines, reduction_pct)
    _log.info("  Function ops: (see above)")
    _log.info("  Reduced output ops: ~%d", reduced_op_count)
    if args.metric and isinstance(interesting, GlobalCounter):
        _log.info("  Best %s:  %d", args.metric, interesting.best_value)
    _log.info("  Output:    %s", output_path)
    return 0


def _count_ops_in_text(mlir_text: str) -> int:
    """Count approximate number of MLIR operations in text."""
    count = 0
    for line in mlir_text.splitlines():
        s = line.strip()
        if s and not s.startswith("//") and not s.startswith("#"):
            if re.search(r"\b[a-zA-Z_]\w*\.[a-zA-Z_]\w*", s):
                count += 1
    return count


if __name__ == "__main__":
    sys.exit(main())
