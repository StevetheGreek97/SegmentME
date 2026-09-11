"""
Application logging.

Usage:
    from services.logger import get_logger
    logger = get_logger(__name__)

Records carry the module name (e.g. "segmentme.ui.main_window"). Console
shows INFO+ (override with the SEGMENTME_LOG_LEVEL env var, e.g. DEBUG);
a rotating file in the user config dir keeps DEBUG+ with tracebacks so
crashes in the field can be diagnosed after the fact.
"""
import logging
import logging.handlers
import os
import platform
import sys
from pathlib import Path

import psutil

APP_LOGGER_NAME = "segmentme"


def _log_dir() -> Path:
    if platform.system() == "Windows":
        base = Path(os.getenv("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base / "segmentme" / "logs"
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Logs" / "segmentme"
    return Path.home() / ".config" / "segmentme" / "logs"


def _configure() -> logging.Logger:
    app = logging.getLogger(APP_LOGGER_NAME)
    if app.handlers:  # already configured
        return app

    app.setLevel(logging.DEBUG)
    app.propagate = False

    # A windowed (console-less) frozen app on Windows has no stderr at all.
    if sys.stderr is not None:
        console = logging.StreamHandler()
        level_name = os.getenv("SEGMENTME_LOG_LEVEL", "INFO").upper()
        console.setLevel(getattr(logging, level_name, logging.INFO))
        console.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s: %(message)s", "%H:%M:%S"))
        app.addHandler(console)

    try:
        log_dir = _log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "segmentme.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s [%(filename)s:%(lineno)d]: %(message)s"))
        app.addHandler(file_handler)
    except OSError as e:
        app.warning("File logging disabled (cannot write to %s): %s", _log_dir(), e)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    return app


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a module logger nested under the app logger."""
    _configure()
    if not name or name == "__main__":
        return logging.getLogger(APP_LOGGER_NAME)
    return logging.getLogger(f"{APP_LOGGER_NAME}.{name}")


# Back-compat: modules that still do `from services.logger import logger`
logger = _configure()


def log_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logger.debug("Memory usage: %.2f MB RSS", memory_info.rss / (1024 ** 2))


__all__ = ["logger", "get_logger", "log_memory_usage"]
