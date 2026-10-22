"""
E with SAM or DEXTR: the model runs on a worker thread (seconds on a CPU),
so the window stays responsive; the outline is shown when it is done.

E again while it runs: one more run afterwards with the newest prompts (runs
never overlap: the model holds one image embedding). A result for an image
or tool that is no longer current, or arriving while the next image loads,
is dropped.
"""
import time
from concurrent.futures import ThreadPoolExecutor

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QApplication

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
        self._job = None            # (future, image, tool, start time) while running
        self._again = False

    def busy(self) -> bool:
        return self._job is not None

    def run(self, tool):
        """Segment with the tool's current prompts (mask_job / compute_polygon
        / apply_polygon, see core/tools/base.py)."""
        if self._job is not None:
            self._again = True
            return
        job = tool.mask_job()
        if job is None:
            return
        image = self._view.parent.state_manager.current_image
        if image is None:
            return
        self._again = False
        future = self._pool.submit(tool.compute_polygon, image, *job)
        self._job = (future, image, tool, time.perf_counter())
        QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
        self._view.status_message.emit(f"{tool.label} is segmenting…", 0)
        self._timer.start()

    def _poll(self):
        future, image, tool, started = self._job
        if not future.done():
            return
        self._timer.stop()
        self._job = None
        QApplication.restoreOverrideCursor()
        view = self._view
        view.status_message.emit("", 0)
        current_tool = view.parent.tool_manager.current_tool
        try:
            result = future.result()
        except Exception:
            logger.exception("%s segmentation failed", tool.label)
            view.status_message.emit(f"{tool.label} segmentation failed (see the log)", 5000)
        else:
            if result is None:
                view.status_message.emit(f"{tool.label} found nothing here", 4000)
            elif (image is view.parent.state_manager.current_image and tool is current_tool
                    and not view.loading()):
                tool.apply_polygon(*result)
                logger.debug("%s result shown after %.2f s", tool.label, time.perf_counter() - started)
            else:
                logger.info("%s result dropped: the image or tool changed while it ran", tool.label)
        if self._again and current_tool is not None and current_tool.segments_on_e:
            self.run(current_tool)
