"""
Masks blended into an image: the picture the image view shows.

Plain numpy / OpenCV (no Qt; its own database connection), so it can run on a worker thread:
the view prepares the next image's picture there while the window stays
responsive (see image_display.show_image).
"""
from collections import defaultdict

import cv2
import numpy as np

from core.database.connection import connect_read_only

ALPHA = 0.4
WHITE = (255, 255, 255)        # colour of a class the project doesn't know


def blend_masks(image, masks, colors, alpha=ALPHA, outline_thickness=1, exclude_ids=None):
    """
    Copy of `image` (RGB) with the masks blended in (one colour per class)
    and outlined in black.

    masks: [(id, polygon Nx2, class name, area), ...] as the database gives them.
    colors: class name -> (r, g, b).
    exclude_ids: mask ids to leave out (e.g. one being brush-edited).

    Blends inside each polygon's bounding box rather than over full-size
    class masks, so the cost follows the masks' area instead of the image's
    (it matters on large scans). The result is the same pixel for pixel.
    """
    output = image.copy()
    class_to_polys = defaultdict(list)
    for mask_id, mask, class_name, _ in masks:
        if exclude_ids and mask_id in exclude_ids:
            continue
        if mask.shape[0] >= 3:
            class_to_polys[class_name].append(mask.astype(np.int32))

    H, W = image.shape[:2]
    class_mask = None  # one raster for all classes; all zero again after each class
    for class_name, polys in class_to_polys.items():
        if class_mask is None:
            class_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(class_mask, polys, 255)
        color_rgb = colors.get(class_name, WHITE)

        for poly in polys:
            x, y, w, h = cv2.boundingRect(poly)
            x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
            if x1 <= x0 or y1 <= y0:
                continue  # polygon entirely outside the image
            roi_mask = class_mask[y0:y1, x0:x1]
            if not cv2.countNonZero(roi_mask):
                continue  # already blended through an overlapping box
            roi_out = output[y0:y1, x0:x1]
            roi_color = np.empty_like(roi_out)
            roi_color[:] = color_rgb
            blended = cv2.addWeighted(roi_out, 1.0 - alpha, roi_color, alpha, 0.0)
            cv2.copyTo(blended, roi_mask, roi_out)
            # Clear the box: pixels shared with an overlapping polygon of
            # this class must be blended once, as with one class-wide mask.
            roi_mask[:] = 0

        cv2.polylines(output, polys, isClosed=True, color=(0, 0, 0), thickness=outline_thickness)

    return output


def load_masks(db_path, image_name):
    """The image's masks, read through a connection of its own (read-only),
    so a worker thread never shares the window's connection."""
    conn = connect_read_only(db_path)
    try:
        rows = conn.execute("SELECT id, mask_data, class_name, surface_area FROM masks "
                            "WHERE image_name = ?", (image_name,)).fetchall()
    finally:
        conn.close()
    return [(mask_id, np.frombuffer(data, dtype=np.float32).reshape(-1, 2), class_name, area)
            for mask_id, data, class_name, area in rows]
