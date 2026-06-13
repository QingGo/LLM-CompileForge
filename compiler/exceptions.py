"""Compiler error hierarchy — structured exception types for the LLM-CompileForge compiler.

All compiler-specific errors inherit from ``CompileError``, allowing callers
to catch a broad ``CompileError`` for top-level handling, or specific subtypes
for fine-grained recovery.

Inspired by Rust's ``thiserror`` pattern: each variant carries context that
enables structured diagnostics without relying on string parsing.
"""

from __future__ import annotations


class CompileError(Exception):
    """Base exception for all compiler errors."""


class ToolNotFoundError(CompileError):
    """External compiler tool (llc, mlir-translate, cc) not found."""

    def __init__(self, tool: str, hint: str = "") -> None:
        self.tool = tool
        self.hint = hint
        msg = f"{tool} not found"
        if hint:
            msg += f". {hint}"
        super().__init__(msg)


class PipelineStageError(CompileError):
    """A pipeline stage failed or timed out."""

    def __init__(self, stage_name: str, message: str, snapshot_path: str = "") -> None:
        self.stage_name = stage_name
        self.snapshot_path = snapshot_path
        msg = f"Stage '{stage_name}' failed: {message}"
        if snapshot_path:
            msg += f" — IR snapshot: {snapshot_path}"
        super().__init__(msg)


class MLIRTranslateError(CompileError):
    """mlir-translate failed to produce valid LLVM IR."""

    def __init__(self, returncode: int, stderr: str = "") -> None:
        self.returncode = returncode
        self.stderr = stderr
        snippet = stderr[:2000] if stderr else "(no stderr)"
        msg = f"mlir-translate failed (exit {returncode}):\n{snippet}"
        super().__init__(msg)


class LLCError(CompileError):
    """llc compilation failed."""

    def __init__(self, returncode: int, stderr: str = "") -> None:
        msg = f"llc compilation failed (exit {returncode})"
        if stderr:
            msg += f":\n{stderr}"
        super().__init__(msg)


class LinkError(CompileError):
    """Linking object files to .dylib failed."""

    def __init__(self, returncode: int, stderr: str = "", output: str = "") -> None:
        msg = f"link_dylib failed (exit {returncode})"
        if stderr:
            msg += f":\n{stderr}"
        if output:
            msg += f"\nOutput: {output}"
        super().__init__(msg)


class MissingBindingsError(CompileError):
    """MLIR Python bindings are not available."""

    def __init__(self) -> None:
        super().__init__("MLIR Python bindings not available")
