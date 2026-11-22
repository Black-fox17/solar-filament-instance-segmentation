"""Structured console logging helper."""

from __future__ import annotations

import logging
import sys


def get_logger(name: str = "solar_filament", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger without attaching duplicate stream handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger
