"""Compiler backend — IR verification, dylib compilation, LLVM lowering, and fixups."""

# ruff: noqa: F401 — re-exports for backward compatibility

from compiler.backend.compile_utils import (
    _compile_embedded_data,
    _compile_mlir_to_dylib_with_constants,
    _compile_serveforge_free,
    _find_cc,
    _find_llc,
    _find_mlir_tool,
    _find_mlir_translate,
    _has_bindings,
    _patch_transformers_torch,
    _setup_mlir_path,
    _short_shape,
    compile_mlir_to_dylib,
    compile_module_to_dylib,
    emit_llvm_ir_to_file,
    link_dylib,
    llc_compile,
    mlir_module_to_llvm_ir,
)
from compiler.backend.dylib import (
    _check_sf_dialect_freshness,
    _compile_blob_to_o,
    _sfa_relink_dylib,
)
from compiler.backend.fixups import (
    _fixup_arith_tensor_constants_mlir,
    _fixup_unrealized_casts_pass,
    _walk_and_fix_tensor_constants,
)
from compiler.backend.llvm_backend import (
    _register_sf_passes,
    lower_linalg_to_llvm_ir,
    lower_linalg_to_llvm_ir_text,
)
from compiler.backend.verify import _save_failure_context, _verify_lowered_ir
