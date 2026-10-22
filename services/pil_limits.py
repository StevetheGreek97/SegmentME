"""
Lift Pillow's "decompression bomb" size check while reading image headers.

Scans are far larger than Pillow's default limit (~89 megapixels), which only
guards against hostile files. The limit is one process-wide setting, so it is
switched off with a counter: several threads can read headers at once, and
the limit returns only when the last one is done.
"""
import threading
from contextlib import contextmanager

from PIL import Image

_lock = threading.Lock()
_active = 0
_saved = None


@contextmanager
def large_images_allowed():
    global _active, _saved
    with _lock:
        if _active == 0:
            _saved = Image.MAX_IMAGE_PIXELS
            Image.MAX_IMAGE_PIXELS = None   # no limit: no warning or error either
        _active += 1
    try:
        yield
    finally:
        with _lock:
            _active -= 1
            if _active == 0:
                Image.MAX_IMAGE_PIXELS = _saved
