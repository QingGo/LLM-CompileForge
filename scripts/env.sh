#!/usr/bin/env bash
# env.sh — single self-healing environment for LLM-CompileForge.
#
# Usage:
#   source scripts/env.sh
#
# Eliminates "environment wedge" failures (TCC / PATH / dyld / stale
# Python bindings) by exporting every path the build, compiler, and
# runtime need in one place.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "env.sh must be sourced, not executed: source scripts/env.sh" >&2
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── PATH: project binaries first, then Homebrew LLVM, then system ──
_llvm_bin="$PROJECT_ROOT/llvm-project/build/bin"
_venv_bin="$PROJECT_ROOT/.venv/bin"
for _dir in "$_venv_bin" "$_llvm_bin" "$HOME/.cargo/bin" /usr/local/bin /usr/local/opt/llvm/bin "$HOME/.local/bin"; do
  case ":$PATH:" in
    *":$_dir:"*) ;;
    *) PATH="$_dir:$PATH" ;;
  esac
done
export PATH

# ── dyld: MLIR/SF/torch native libraries ────────────────────────────
_mlir_libs="$PROJECT_ROOT/llvm-project/build/tools/mlir/python_packages/mlir_core/mlir/_mlir_libs"
_sf_libs="$PROJECT_ROOT/sf-dialect/build/python_packages/sf/mlir_sf/_mlir_libs"
_torch_libs="$PROJECT_ROOT/.venv/lib/python3.10/site-packages/torch/lib"
DYLD_LIBRARY_PATH="$_sf_libs:$_torch_libs:$_mlir_libs${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
export DYLD_LIBRARY_PATH

# ── Python/OpenMP runtime quirks ────────────────────────────────────
export KMP_DUPLICATE_LIB_OK=TRUE
unset CONDA_PREFIX

# ── Binding freshness hint ──────────────────────────────────────────
if [[ -d "$PROJECT_ROOT/runtime/target" ]]; then
  _bindings_so="$(find "$PROJECT_ROOT/.venv/lib" -maxdepth 4 -name 'llm_serveforge_runtime*.so' -print -quit 2>/dev/null)"
  if [[ -n "$_bindings_so" ]]; then
    _rust_lib_mtime="$(find "$PROJECT_ROOT/runtime/src" -name '*.rs' -newer "$_bindings_so" -print -quit 2>/dev/null)"
    if [[ -n "$_rust_lib_mtime" ]]; then
      echo "⚠️  Rust Python bindings may be stale vs runtime sources." >&2
      echo "   Rebuild with: make build-rust" >&2
    fi
  else
    echo "ℹ️  Rust Python bindings not installed. Run: make build-rust" >&2
  fi
fi

echo "env.sh: PATH + DYLD_LIBRARY_PATH configured for $(uname -s)"
