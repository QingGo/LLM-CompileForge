"""Unit tests for compiler/utils/logging.py — init_logging, get_logger, LogSession."""

import logging
import os
from pathlib import Path

from compiler.utils.logging import LogSession, get_logger, init_logging


class TestInitLogging:
    def test_init_logging_is_idempotent(self, monkeypatch):
        monkeypatch.delenv("LLM_SERVEFORGE_LOG", raising=False)
        init_logging()
        init_logging()  # should not crash
        assert logging.getLogger().level == logging.WARNING


class TestGetLogger:
    def test_returns_logger_with_expected_name(self):
        log = get_logger("test.module")
        assert log.name == "test.module"

    def test_returns_different_loggers_for_different_names(self):
        a = get_logger("test.a")
        b = get_logger("test.b")
        assert a is not b


class TestLogSession:
    def test_creates_directory(self, tmp_path):
        base = str(tmp_path)
        LogSession("pytest", "test_model", base_dir=base)
        dirs = list(Path(base).glob("pytest/test_model*"))
        assert len(dirs) > 0

    def test_save_ir_writes_file(self, tmp_path):
        base = str(tmp_path)
        session = LogSession("pytest", "test_model", base_dir=base)
        path = session.save_ir("module { }", "snapshot")
        assert os.path.isfile(path)

    def test_save_report_writes_file(self, tmp_path):
        base = str(tmp_path)
        session = LogSession("pytest", "test_model", base_dir=base)
        path = session.save_report("timing.txt", "step1: 0.5s")
        assert os.path.isfile(path)
