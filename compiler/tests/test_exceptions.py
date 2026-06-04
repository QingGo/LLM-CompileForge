"""Unit tests for compiler exception classes."""

from compiler.exceptions import (
    CompileError,
    LinkError,
    LLCError,
    MissingBindingsError,
    MLIRTranslateError,
    PipelineStageError,
    ToolNotFoundError,
)


class TestCompileError:
    def test_base_compile_error_is_exception(self):
        assert issubclass(CompileError, Exception)

    def test_raise_and_catch_base(self):
        try:
            raise CompileError("test")
        except CompileError as e:
            assert str(e) == "test"


class TestToolNotFoundError:
    def test_message_includes_tool_name(self):
        err = ToolNotFoundError("llc")
        assert "llc" in str(err)

    def test_hint_in_message(self):
        err = ToolNotFoundError("cc", hint="Install Xcode CLI tools")
        assert "Install Xcode CLI tools" in str(err)


class TestPipelineStageError:
    def test_message_includes_stage_name(self):
        err = PipelineStageError("canonicalize", "timeout after 30s")
        assert "canonicalize" in str(err)
        assert "timeout after 30s" in str(err)

    def test_snapshot_path_in_message(self):
        err = PipelineStageError("cse", "verification failed", snapshot_path="/tmp/ir.mlir")
        assert "/tmp/ir.mlir" in str(err)

    def test_no_snapshot_path(self):
        err = PipelineStageError("cse", "error")
        assert "/tmp" not in str(err)


class TestMLIRTranslateError:
    def test_message_includes_exit_code(self):
        err = MLIRTranslateError(1)
        assert "1" in str(err)

    def test_stderr_truncated_in_message(self):
        err = MLIRTranslateError(3, stderr="x" * 3000)
        assert "x" * 2000 in str(err)
        assert len(str(err)) < 2100


class TestLLCError:
    def test_message_includes_exit_code(self):
        err = LLCError(2)
        assert "2" in str(err)


class TestLinkError:
    def test_message_includes_exit_code(self):
        err = LinkError(1)
        assert "1" in str(err)


class TestMissingBindingsError:
    def test_message_is_meaningful(self):
        err = MissingBindingsError()
        assert "bindings" in str(err).lower()

    def test_inherits_from_compile_error(self):
        assert isinstance(MissingBindingsError(), CompileError)
