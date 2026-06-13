"""Seam tests for the compile_mlir() kwargs validation pattern.

The validator in ``compiler/pipeline.py`` lines 68-77 uses this logic:

    apply_lowering = kwargs.pop("apply_lowering", False)
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {kwargs}")

This test models that exact pattern — a function accepting ``**kwargs``
pops known kwargs and raises TypeError for any remaining unknown ones.
The pattern is independent of torch, MLIR, or sf-dialect imports.
"""

import pytest

# ── Model of the kwargs validation pattern in compile_mlir() ──────────


def _compile_mlir_model(**kwargs: object) -> dict[str, object]:
    """Minimal reproduction of the kwarg guard from compiler/pipeline.py."""
    kwargs.pop("apply_lowering", False)
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {kwargs}")
    return kwargs


class TestPipelineKwargsGuard:
    """Verify the kwarg guard pattern used by compile_mlir()."""

    def test_known_kwargs_passed_through(self) -> None:
        """Known kwargs ('apply_lowering') are consumed — no error raised."""
        result = _compile_mlir_model(apply_lowering=True)
        assert result == {}  # all kwargs consumed

    def test_unknown_kwarg_raises_typeerror(self) -> None:
        """Any unknown keyword argument raises TypeError immediately."""
        with pytest.raises(TypeError, match="Unexpected keyword arguments"):
            _compile_mlir_model(nonexistent_kwarg=True)

    def test_unknown_kwarg_after_known_kwarg(self) -> None:
        """Unknown kwargs are caught even when mixed with known ones."""
        with pytest.raises(TypeError, match="Unexpected keyword arguments"):
            _compile_mlir_model(apply_lowering=True, bad_arg="should_fail")

    def test_multiple_unknown_kwargs(self) -> None:
        """Multiple unknown kwargs are all reported in the error message."""
        with pytest.raises(TypeError) as exc_info:
            _compile_mlir_model(foo=1, bar=2, baz=3)
        msg = str(exc_info.value)
        assert "foo" in msg
        assert "bar" in msg
        assert "baz" in msg
