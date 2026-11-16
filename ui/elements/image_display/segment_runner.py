"""
E with SAM or DEXTR: the model runs on a worker thread (seconds on a CPU),
so the window stays responsive; the outline is shown when it is done.
F with SAM: the same, for every object in the box (find_all_job /
compute_all / apply_all, see core/tools/sam_everything.py); the status bar
shows how far it is.

E (or F) again while it runs: one more run afterwards with the newest
prompts (runs never overlap: the model holds one image embedding). A result
for an image or tool that is no longer current, or arriving while the next
image loads, is dropped.
"""
import time
from concurrent.futures import ThreadPoolExecutor

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QApplication

from core.tools.sam_everything import Cancelled
from services.logger import get_logger

logger = get_logger(__name__)


class SegmentRunner:
    POLL_MS = 30

    def __init__(self, view):
        self._view = view
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="segment")
        self._timer = QTimer(view)
        self._timer.setInterval(self.POLL_MS)
        self._timer.timeout.connect(self._poll)
        self._job = None            # (future, image, tool, start time, find_all) while running
        self._again = None          # None, or find_all of the run asked for meanwhile
        self._shown_progress = None

    def busy(self) -> bool:
        return self._job is not None

    def run(self, tool, find_all=False):
        """Segment with the tool's current prompts: the one object they point
        at (mask_job / compute_polygon / apply_polygon), or with find_all
        every object in the box (find_all_job / compute_all / apply_all)."""
        if self._job is not None:
            self._again = find_all
            return
        job = tool.find_all_job() if find_all else tool.mask_job()
        if job is None:
            if find_all:
                self._view.status_message.emit(
                    "Draw a box first (hold Ctrl and drag), then press F", 5000)
            return
        image = self._view.parent.state_manager.current_image
        if image is None:
            return
        self._again = None
        compute = tool.compute_all if find_all else tool.compute_polygon
        future = self._pool.submit(compute, image, *job)
        self._job = (future, image, tool, time.perf_counter(), find_all)
        self._shown_progress = None
        QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
        self._view.status_message.emit(
            f"{tool.label} is finding the objects in the box…" if find_all
            else f"{tool.label} is segmenting…", 0)
        self._timer.start()

    def _poll(self):
        future, image, tool, started, find_all = self._job
        if not future.done():
            if find_all:
                text = tool.progress_text()
                if text != self._shown_progress:
                    self._shown_progress = text
                    self._view.status_message.emit(
                        f"{tool.label} is finding the objects in the box… {text} "
                        "(another image or tool stops it)", 0)
            return
        self._timer.stop()
        self._job = None
        QApplication.restoreOverrideCursor()
        view = self._view
        view.status_message.emit("", 0)
        current_tool = view.parent.tool_manager.current_tool
        try:
            result = future.result()
        except Cancelled:
            logger.info("%s search stopped", tool.label)
        except Exception:
            logger.exception("%s segmentation failed", tool.label)
            view.status_message.emit(f"{tool.label} segmentation failed (see the log)", 5000)
        else:
            current = (image is view.parent.state_manager.current_image and tool is current_tool
                       and not view.loading())
            if result is None:
                view.status_message.emit(f"{tool.label} found nothing here", 4000)
            elif not current:
                logger.info("%s result dropped: the image or tool changed while it ran", tool.label)
            elif find_all:
                view.status_message.emit(tool.apply_all(result), 8000)
                logger.debug("%s found %d object(s) in %.2f s", tool.label, len(result),
                             time.perf_counter() - started)
            else:
                tool.apply_polygon(*result)
                logger.debug("%s result shown after %.2f s", tool.label, time.perf_counter() - started)
        again, self._again = self._again, None
        if again is not None and current_tool is not None and current_tool.segments_on_e:
            self.run(current_tool, find_all=again and current_tool.finds_all)
