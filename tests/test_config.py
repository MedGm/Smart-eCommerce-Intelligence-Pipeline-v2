"""Tests for shared configuration module."""

import os
from pathlib import Path

from src.config import analytics_dir, data_dir, get_logger, processed_dir, raw_dir


def test_data_dir_default():
    """Default data dir should be 'data'."""
    # Remove DATA_DIR if set
    env_backup = os.environ.pop("DATA_DIR", None)
    try:
        assert data_dir() == Path("data")
    finally:
        if env_backup:
            os.environ["DATA_DIR"] = env_backup


def test_data_dir_override(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    assert data_dir() == tmp_path


def test_subdirs(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    assert raw_dir() == tmp_path / "raw"
    assert processed_dir() == tmp_path / "processed"
    assert analytics_dir() == tmp_path / "analytics"


def test_get_logger():
    logger = get_logger("test_config")
    assert logger.name == "test_config"
    assert len(logger.handlers) > 0


def test_get_logger_no_duplicate_handlers():
    """Multiple calls should not add duplicate handlers."""
    logger1 = get_logger("test_dup")
    n_handlers = len(logger1.handlers)
    logger2 = get_logger("test_dup")
    assert len(logger2.handlers) == n_handlers
