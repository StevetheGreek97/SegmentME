"""
Application logging.

Usage:
    from services.logger import get_logger
    logger = get_logger(__name__)

Records carry the module name (e.g. "segmentme.ui.main_window"). Console
shows INFO+ (override with the SEGMENTME_LOG_LEVEL env var, e.g. DEBUG);
a rotating file in the user config dir keeps DEBUG+ with tracebacks,
milliseconds, process and thread, so crashes in the field can be
diagnosed after the fact. ui/diagnostics.py adds the rest: a system
report per session, user actions, Qt warnings, freezes, hard crashes.
"""
import logging
import logging.handlers
import os
import platform
import sys
from pathlib import Path

import psutil

APP_LOGGER_NAME = "segmentme"


class _NullStream:
    """A harmless sink standing in for a missing stdout/stderr.

    A windowed (console-less) frozen app on Windows has sys.stdout and
    sys.stderr set to None -- there is no console for them to point at.
    Any third-party code that writes to them directly then crashes, e.g.
    ultralytics.data.split.autosplit()'s tqdm progress bar ("AttributeError:
    'NoneType' object has no attribute 'write'"), which silently aborts
    SegmentME's YOLO export before it writes the autosplit_*.txt files.
    Installing this instead of None keeps that code working without a
    console to print to.
    """

    def write(self, *args, **kwargs):
        pass

    def flush(self, *args, **kwargs):
        pass

    def isatty(self):
        return False


LOG_FILE_NAME = "segmentme.log"
LOG_MAX_BYTES = 10_000_000      # per file; with the backups up to 60 MB of history
LOG_BACKUPS = 5


class _SafeRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Rotation can fail on Windows while another program (an antivirus, a
    text editor, a second SegmentME) has the file open. Then keep writing
    to the current file instead of losing messages."""

    def doRollover(self):
        try:
            super().doRollover()
        except OSError:
            if self.stream is None:
                self.stream = self._open()

    def handleError(self, record):
        pass                      # a full disk must never take the app down


def log_dir() -> Path:
    """Where the log files are (Help > Open Log Folder)."""
    return _log_dir()


def log_file() -> Path:
    return _log_dir() / LOG_FILE_NAME


def _log_dir() -> Path:
    override = os.getenv("SEGMENTME_LOG_DIR")   # e.g. the test suite's own folder
    if override:
        return Path(override)
    if platform.system() == "Windows":
        base = Path(os.getenv("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base / "segmentme" / "logs"
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Logs" / "segmentme"
    return Path.home() / ".config" / "segmentme" / "logs"


HELPER_FLAGS = ("--inference-worker", "--training-worker")


def _is_helper_process() -> bool:
    if len(sys.argv) > 1 and sys.argv[1] in HELPER_FLAGS:
        return True
    try:
        import multiprocessing
        return multiprocessing.current_process().name != "MainProcess"
    except Exception:
        return False


def _configure() -> logging.Logger:
    app = logging.getLogger(APP_LOGGER_NAME)
    if app.handlers:  # already configured
        return app

    app.setLevel(logging.DEBUG)
    app.propagate = False

    # A windowed (console-less) frozen app on Windows has no stdout/stderr at
    # all -- both are None. Give them a harmless sink instead of leaving them
    # None, so third-party code that writes to them directly (print(), a
    # tqdm progress bar, ...) doesn't crash; see _NullStream. Checked before
    # replacing them: our own console handler is still skipped in that case,
    # exactly as before -- only the file handler below logs.
    windowed = sys.stderr is None
    if windowed:
        if sys.stdout is None:
            sys.stdout = _NullStream()
        sys.stderr = _NullStream()
    else:
        console = logging.StreamHandler()
        level_name = os.getenv("SEGMENTME_LOG_LEVEL", "INFO").upper()
        console.setLevel(getattr(logging, level_name, logging.INFO))
        console.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s: %(message)s", "%H:%M:%S"))
        app.addHandler(console)

    if _is_helper_process():
        # Inference/training helpers and their worker processes print to
        # stderr, which the app reads into its own log: two processes
        # writing (and rotating) one file would garble it.
        if not app.handlers:
            helper = logging.StreamHandler()
            helper.setLevel(logging.DEBUG)
            helper.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
            app.addHandler(helper)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        return app

    try:
        log_dir = _log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = _SafeRotatingFileHandler(
            log_dir / LOG_FILE_NAME, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUPS,
            encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s.%(msecs)03d %(levelname)-8s %(process)d/%(threadName)s "
            "%(name)s [%(filename)s:%(lineno)d]: %(message)s", "%Y-%m-%d %H:%M:%S"))
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


# The app's own logger (log_memory_usage below)
logger = _configure()


def log_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logger.debug("Memory usage: %.2f MB RSS", memory_info.rss / (1024 ** 2))


__all__ = ["logger", "get_logger", "log_memory_usage", "log_dir", "log_file"]
