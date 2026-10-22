"""
Reading image files the same way on every OS.

Not cv2.imread: on Windows, importing ultralytics (inference, export)
replaces it for the whole app with a version that raises FileNotFoundError
for a missing file instead of returning None -- a linked image whose
original was moved then crashed the image view instead of showing its
"missing" message. Reading the bytes with numpy and decoding them works
with any path (also non-English folder names on Windows) and never raises
for a missing or broken file.
"""
import cv2
import numpy as np

from services.logger import get_logger

logger = get_logger(__name__)


def read_image(path, flags=cv2.IMREAD_COLOR):
    """The image as OpenCV decodes it (BGR for IMREAD_COLOR), or None when
    the file is missing, unreadable or not an image."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        logger.debug("Cannot read %s: %s", path, exc)
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)
