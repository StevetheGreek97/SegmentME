"""
Run a long job on a worker thread behind a progress window with Cancel.

The window stays responsive (it keeps repainting and can be moved); the
caller gets the outcome when the job is over:

    run = run_in_background(parent, "Import Annotations", work)
    if run.cancelled: ...
    elif run.error: ...
    else: use(run.result)

`work(progress, cancelled)` runs on the worker thread: it calls
progress(done, total, text) now and then, and should stop (raising
`cancel_exception`) once cancelled() is True. It must not touch Qt widgets.
"""
import threading
import time

from PyQt6.QtCore import QEventLoop, Qt, QTimer
from PyQt6.QtWidgets import QProgressDialog

from services.logger import get_logger

logger = get_logger(__name__)


class TaskRun:
    def __init__(self):
        self.result = None
        self.cancelled = False
        self.error = None
        self.seconds = 0.0


def run_in_background(parent, title, work, label=None, cancel_exception=(),
                      start_text=None, min_duration_ms=300, cancel_text="Cancelling…"):
    """Run `work` on a worker thread with a progress window; returns a TaskRun
    once it is over. label(done, total, text) -> the window's text (default:
    text, plus "(done of total)"). cancel_exception: exception type(s) that
    mean the job stopped because of Cancel."""
    run = TaskRun()
    shared = {"done": 0, "total": 0, "text": start_text or f"{title}…", "finished": False}
    cancel = threading.Event()

    def progress(done, total, text=None):
        shared["done"], shared["total"] = done, total
        if text:
            shared["text"] = text

    def target():
        try:
            run.result = work(progress, cancel.is_set)
        except cancel_exception:
            run.cancelled = True
        except BaseException as exc:          # reported by the caller
            logger.exception("%s failed", title)
            run.error = exc
        finally:
            shared["finished"] = True

    if label is None:
        def label(done, total, text):
            return f"{text} ({min(done + 1, total):,} of {total:,})" if total else text

    dialog = QProgressDialog(shared["text"], "Cancel", 0, 1000, parent)
    dialog.setWindowTitle(title)
    dialog.setWindowModality(Qt.WindowModality.WindowModal)
    dialog.setMinimumDuration(min_duration_ms)
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)
    dialog.canceled.connect(cancel.set)
    loop = QEventLoop()

    def refresh():
        if shared["finished"]:
            timer.stop()
            loop.quit()
            return
        if dialog.wasCanceled():               # Cancel button, Esc or closing the window
            cancel.set()
        done, total = shared["done"], shared["total"]
        dialog.setValue(min(999, int(1000 * done / total)) if total else 0)
        dialog.setLabelText(cancel_text if cancel.is_set()
                            else label(done, total, shared["text"]))

    timer = QTimer()
    timer.setInterval(100)
    timer.timeout.connect(refresh)

    started = time.monotonic()
    thread = threading.Thread(target=target, name=title.lower().replace(" ", "-"), daemon=True)
    thread.start()
    timer.start()
    if not shared["finished"]:
        loop.exec()
    thread.join()
    timer.stop()
    dialog.close()
    run.seconds = time.monotonic() - started
    logger.debug("%s: %s in %.1f s", title,
                 "cancelled" if run.cancelled else "failed" if run.error else "done", run.seconds)
    return run
