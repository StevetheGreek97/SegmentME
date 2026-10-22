"""
Adding many images (New Project, Import Images) without freezing the window.

The files are copied or linked on a worker thread (services/image_files.py
place_images); this side shows a progress window with Cancel and keeps the
app responsive -- copying hundreds of large scans, maybe from a network or
cloud folder, can take a long time. Before copying, the free disk space is
checked.
"""
import shutil
from pathlib import Path

from PyQt6.QtWidgets import QMessageBox

from services import image_files
from services.logger import get_logger
from ui.dialogs.background_task import run_in_background

logger = get_logger(__name__)

_SPARE_BYTES = 500 * 1024 ** 2   # leave some room for the project database etc.


def _gb(n: int) -> str:
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.1f} GB"
    return f"{n / 1024 ** 2:.1f} MB" if n < 10 * 1024 ** 2 else f"{n / 1024 ** 2:.0f} MB"


def enough_space(parent, sources, dest_dir, mode, title) -> bool:
    """False (after telling the user) when copying would not fit on the disk."""
    needed = image_files.bytes_to_copy(sources, mode)
    if not needed:
        return True
    probe = Path(dest_dir)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if needed + _SPARE_BYTES <= free:
        logger.debug("Disk space OK: %s needed, %s free", _gb(needed), _gb(free))
        return True
    logger.warning("Not copying %d image(s): %s needed, only %s free on %s",
                   len(sources), _gb(needed), _gb(free), probe)
    QMessageBox.warning(
        parent, title,
        f"Copying these {len(sources)} image(s) needs {_gb(needed)}, but only {_gb(free)} "
        f"is free on that disk.\n\nChoose \"Link to the original files\" instead (no copy, "
        f"no disk space), or free up space.")
    return False


class PlacementRun:
    """Result of run_placement: `result` (image_files.BatchResult) when it
    finished, else `cancelled` or `error` is set."""

    def __init__(self):
        self.result = None
        self.cancelled = False
        self.error = None


def run_placement(parent, jobs, mode, title="Adding images", stop_on_error=False) -> PlacementRun:
    """Copy/link `jobs` [(src, dest), ...] on a worker thread while showing
    progress; returns when done. Cancel (or an error that stops the batch)
    removes what was added so far."""
    run = PlacementRun()
    total_bytes = image_files.bytes_to_copy([src for src, _ in jobs], mode)
    copied = {"bytes": 0, "file": 0}
    verb = "Copying" if mode == image_files.COPY else "Linking"

    def count_bytes(n):
        copied["bytes"] += n

    def work(progress, cancelled):
        def report(i, total, name):
            copied["file"] = i
            # the bar follows the bytes when copying, the files when linking
            if total_bytes:
                progress(copied["bytes"], total_bytes, name)
            else:
                progress(i, total, name)
        return image_files.place_images(
            jobs, mode, cancelled=cancelled, progress=report,
            on_bytes=count_bytes, stop_on_error=stop_on_error)

    def label(done, total, name):
        if name == f"{title}…":                    # nothing reported yet
            return name
        n = len(jobs)
        amount = f" — {_gb(copied['bytes'])} of {_gb(total_bytes)}" if total_bytes else ""
        return f"{verb} {name} ({min(copied['file'] + 1, n)} of {n}){amount}"

    task = run_in_background(parent, title, work, label=label,
                             cancel_exception=image_files.PlacementCancelled,
                             min_duration_ms=400,
                             cancel_text="Cancelling… (removing what was added)")
    run.result, run.cancelled, run.error = task.result, task.cancelled, task.error
    logger.info("%s %d image(s): %s in %.1f s", verb, len(jobs),
                "cancelled" if run.cancelled else "failed" if run.error else "done",
                task.seconds)
    return run
