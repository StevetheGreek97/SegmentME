"""
Reading image files without freezing the window.

A scan is ~270 MB. From a local disk it reads in ~0.6 s; from a cloud
folder (e.g. an rclone mount) whose copy isn't on this computer yet it
takes ~5 s, and doing that on the GUI thread froze the window ("not
responding"). So Next / Previous / slider:

- BackgroundImageReader reads the file on a worker thread and hands the
  result back on the GUI thread;
- Prefetcher then reads the neighbouring files in the background, so the
  cloud drive (or the OS) already has them when the user gets there;
- LoadingLabel says "Loading <name>…" over the old image meanwhile.
"""
import os
import threading
import time

import cv2
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QLabel

from services.logger import get_logger
from ui.elements.image_display import mask_render

logger = get_logger(__name__)


def read_rgb(image_path):
    """The image file as an RGB array, or None. Thread-safe."""
    image = cv2.imread(image_path)
    return None if image is None else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def prepare_image(image_path, db_path, colors, masks_shown):
    """Worker thread: read the file and draw its masks into it -- all the
    heavy part of showing an image. Returns a dict, or None if unreadable.
    colors / masks_shown are taken on the GUI thread when the load starts."""
    t0 = time.perf_counter()
    image = read_rgb(image_path)
    if image is None:
        return None
    t1 = time.perf_counter()
    name = os.path.splitext(os.path.basename(image_path))[0]
    masks = mask_render.load_masks(db_path, name)
    picture = mask_render.blend_masks(image, masks, colors) if masks_shown else None
    return {"image": image, "masks": masks, "picture": picture, "colors": colors,
            "visible": masks_shown, "read_s": t1 - t0, "draw_s": time.perf_counter() - t1}


class BackgroundImageReader:
    """One read at a time on a worker thread. `read(path)` runs there;
    `done(path, image, seconds)` is called on the GUI thread afterwards.
    `want` is the path the view is waiting for (None = nothing): the owner
    sets it, and reads the newest one when an older read comes back."""

    POLL_MS = 15

    def __init__(self, parent, read, done):
        self._read = read
        self._done = done
        self.want = None
        self.future = None          # (path, result holder, thread, start time) while reading
        self._timer = QTimer(parent)
        self._timer.setInterval(self.POLL_MS)
        self._timer.timeout.connect(self._poll)

    def start(self, path):
        holder = {}

        def run():
            try:
                holder["image"] = self._read(path)
            except Exception:
                logger.exception("Reading %s failed", path)
                holder["image"] = None

        thread = threading.Thread(target=run, name="image-read", daemon=True)
        self.future = (path, holder, thread, time.perf_counter())
        thread.start()
        self._timer.start()

    def _poll(self):
        path, holder, thread, started = self.future
        if thread.is_alive():
            return
        self._timer.stop()
        self.future = None
        seconds = time.perf_counter() - started
        self._done(path, holder.get("image"), seconds)


class Prefetcher:
    """Reads the files next to the current image (next two, previous one)
    on a background thread and throws the bytes away: what stays is the
    file in the cloud drive's cache / the OS file cache, so opening it later
    takes a local read instead of a download. Stops as soon as the user
    moves on (cancel), between 8 MB pieces."""

    AHEAD = (1, 2, -1)
    CHUNK = 8 * 1024 * 1024
    _REMEMBER = 2000

    def __init__(self):
        self._lock = threading.Condition()
        self._queue = []
        self._generation = 0
        self._done = set()          # paths read already this session
        self._thread = None

    def around(self, paths, index):
        """Read ahead around paths[index]."""
        targets = [paths[index + step] for step in self.AHEAD if 0 <= index + step < len(paths)]
        with self._lock:
            self._generation += 1
            self._queue = [p for p in targets if p not in self._done]
            if self._queue and self._thread is None:
                self._thread = threading.Thread(target=self._run, name="image-prefetch", daemon=True)
                self._thread.start()
            self._lock.notify()

    def cancel(self):
        with self._lock:
            self._generation += 1
            self._queue = []

    def _run(self):
        buffer = bytearray(self.CHUNK)
        while True:
            with self._lock:
                if not self._queue:
                    self._lock.wait(timeout=30)
                if not self._queue:
                    self._thread = None     # idle: end; around() starts a new thread
                    return
                path = self._queue.pop(0)
                generation = self._generation
            self._read_through(path, generation, buffer)

    def _read_through(self, path, generation, buffer):
        started = time.perf_counter()
        total = 0
        try:
            with open(path, "rb", buffering=0) as f:
                while True:
                    if self._generation != generation:
                        logger.debug("Read-ahead of %s stopped (moved on)", os.path.basename(path))
                        return
                    n = f.readinto(buffer)
                    if not n:
                        break
                    total += n
        except OSError as e:
            logger.debug("Read-ahead of %s failed: %s", path, e)
            return
        with self._lock:
            if len(self._done) > self._REMEMBER:
                self._done.clear()
            self._done.add(path)
        logger.debug("Read ahead %s: %.0f MB in %.2f s", os.path.basename(path),
                     total / 1e6, time.perf_counter() - started)


class LoadingLabel(QLabel):
    """'Loading <name>…' at the top of the image view. Appears only if the
    read takes longer than a moment, so fast loads don't flicker."""

    DELAY_MS = 150
    _MARGIN = 14

    def __init__(self, view):
        super().__init__(view.viewport())
        self._view = view
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setStyleSheet("QLabel { background: rgba(0, 0, 0, 150); color: white;"
                           " border-radius: 6px; padding: 6px 12px; font-weight: bold; }")
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._appear)
        self.hide()

    def show_for(self, name):
        self.setText(f"Loading {name}…")
        self.adjustSize()
        if self.isVisible():
            self.place()
        else:
            self._timer.start(self.DELAY_MS)

    def _appear(self):
        self.place()
        self.show()
        self.raise_()

    def hide(self):
        self._timer.stop()
        super().hide()

    def place(self):
        """Top centre of the view (scrolling moves the viewport's children)."""
        width = self._view.viewport().width()
        self.move(max((width - self.width()) // 2, 0), self._MARGIN)
