"""
Shared configuration: paths, constants, logging setup.
Single source of truth for data directories and project-wide settings.
"""

import logging
import os
from pathlib import Path


# ── Paths ──────────────────────────────────────────────────────
def data_dir() -> Path:
    """Return the project data root (overridable via DATA_DIR env var)."""
    return Path(os.environ.get("DATA_DIR", "data"))


def raw_dir() -> Path:
    return data_dir() / "raw"


def processed_dir() -> Path:
    return data_dir() / "processed"


def analytics_dir() -> Path:
    return data_dir() / "analytics"


def models_dir() -> Path:
    return data_dir() / "models"


# ── Logging ────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s — %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a named logger with consistent formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ── Scraping ───────────────────────────────────────────────────
SCRAPING_DELAY = float(os.environ.get("SCRAPING_DELAY", "0.5"))
SCRAPING_TIMEOUT = int(os.environ.get("SCRAPING_TIMEOUT", "12"))

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)
