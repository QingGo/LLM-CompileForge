"""Unit tests for compiler/serialize.py — artifact loading."""

import pytest

from compiler.serialize import load_artifact


class TestLoadArtifact:
    def test_raises_on_nonexistent_dir(self):
        with pytest.raises(FileNotFoundError, match="Directory not found"):
            load_artifact("/nonexistent/path/xyz")

    def test_raises_on_file_not_dir(self, tmp_path):
        f = tmp_path / "not_a_dir"
        f.write_text("hello")
        with pytest.raises(FileNotFoundError, match="Directory not found"):
            load_artifact(str(f))

    def test_raises_on_empty_dir(self, tmp_path):
        """Empty dir has no model.mlir — load_artifact should fail gracefully."""
        with pytest.raises(FileNotFoundError):
            load_artifact(str(tmp_path))
