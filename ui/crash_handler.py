"""
A safety net for unexpected errors.

Without it, PyQt6 ends the whole app the moment any button, menu or mouse
handler raises an error it did not expect. With it, the error is written
to the log and the user sees a short message and can keep working:
finished masks are already in the project database (it saves every
change at once).

The message is shown at most once at a time, and the same error repeating
(e.g. on every mouse move) is shown once and then only logged.
Errors on worker threads are logged only; their own code reports them.
"""
import sys
import threading
import time
import traceback

from services.logger import get_logger

logger = get_logger(__name__)

REPEAT_QUIET_SECONDS = 60


class CrashHandler:
    def __init__(self):
        self.showing = False
        self.last_shown = {}          # error "signature" -> time shown
        self.count = 0

    @staticmethod
    def signature(exc_type, tb):
        """Same kind of error at the same place = same signature."""
        frames = traceback.extract_tb(tb)
        where = frames[-1] if frames else None
        return (exc_type.__name__, where.filename if where else "", where.lineno if where else 0)

    def __call__(self, exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        self.count += 1
        details = "".join(traceback.format_exception(exc_type, exc, tb))
        try:
            from ui.diagnostics import recent_actions
            steps = recent_actions()
        except Exception:
            steps = ""
        if steps:
            details += "\nWhat happened just before (most recent last):\n" + steps + "\n"
        logger.critical("Unexpected error:\n%s", details)
        try:
            self.maybe_show(exc_type, exc, tb, details)
        except Exception:                 # the safety net itself must never fail
            logger.exception("Could not show the error message")

    def maybe_show(self, exc_type, exc, tb, details):
        from PyQt6.QtWidgets import QApplication
        if QApplication.instance() is None or self.showing:
            return False
        key = self.signature(exc_type, tb)
        now = time.monotonic()
        if now - self.last_shown.get(key, -1e9) < REPEAT_QUIET_SECONDS:
            return False
        self.last_shown[key] = now
        self.showing = True
        try:
            show_error_message(exc_type, exc, details)
        finally:
            self.showing = False
        return True


def show_error_message(exc_type, exc, details):
    from PyQt6.QtCore import QUrl
    from PyQt6.QtGui import QDesktopServices, QGuiApplication
    from PyQt6.QtWidgets import QApplication, QMessageBox
    from services.logger import _log_dir

    short = str(exc).strip().splitlines()[0][:300] if str(exc).strip() else exc_type.__name__
    box = QMessageBox(QApplication.activeWindow())
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle("Something went wrong")
    box.setText("SegmentME ran into a problem, but you can keep working.\n\n"
                "Masks you have finished are saved. If the last action did not "
                "work, try it again.")
    box.setInformativeText(f"Error: {short}")
    box.setDetailedText(details)
    copy = box.addButton("Copy Details", QMessageBox.ButtonRole.ActionRole)
    logs = box.addButton("Open Log Folder", QMessageBox.ButtonRole.ActionRole)
    box.addButton(QMessageBox.StandardButton.Ok)
    box.setDefaultButton(QMessageBox.StandardButton.Ok)
    box.exec()
    if box.clickedButton() is copy:
        QGuiApplication.clipboard().setText(details)
    elif box.clickedButton() is logs:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(_log_dir())))


def _thread_hook(args):
    if args.exc_type is SystemExit:
        return
    logger.critical("Unexpected error on thread %s:\n%s",
                    getattr(args.thread, "name", "?"),
                    "".join(traceback.format_exception(args.exc_type, args.exc_value,
                                                       args.exc_traceback)))


def install():
    """Call once, after QApplication exists (main.py)."""
    handler = CrashHandler()
    sys.excepthook = handler
    threading.excepthook = _thread_hook
    return handler
