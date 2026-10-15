"""
src/utils/logging.py

Structured logging setup for VisionInspect AI.
Uses loguru if available, otherwise falls back gracefully to Python's standard logging module.
"""

import sys
import logging
from pathlib import Path

try:
    from loguru import logger as _loguru_logger
    HAS_LOGURU = True
except ImportError:
    HAS_LOGURU = False


def get_logger(name: str = "visioninspect", log_dir: str = "logs", level: str = "INFO"):
    """
    Configure and return a logger.

    Args:
        name: Logger name (used in log filename).
        log_dir: Directory to write log files.
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    if HAS_LOGURU:
        _loguru_logger.remove()
        _loguru_logger.add(
            sys.stdout,
            level=level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                "<level>{message}</level>"
            ),
            colorize=True,
        )
        _loguru_logger.add(
            Path(log_dir) / f"{name}.log",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="50 MB",
            retention="30 days",
            compression="zip",
        )
        return _loguru_logger
    else:
        logger = logging.getLogger(name)
        if not logger.handlers:
            logger.setLevel(getattr(logging, level.upper(), logging.INFO))
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            sh = logging.StreamHandler(sys.stdout)
            sh.setFormatter(formatter)
            logger.addHandler(sh)

            fh = logging.FileHandler(Path(log_dir) / f"{name}.log", encoding="utf-8")
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        return logger
