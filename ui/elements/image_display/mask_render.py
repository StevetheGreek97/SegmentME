"""
Masks blended into an image: the picture the image view shows.

Plain numpy / OpenCV (no Qt; its own database connection), so it can run on a worker thread:
the view prepares the next image's picture there while the window stays
responsive (see image_display.show_image).
"""
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from core.database.connection import connect_read_only

ALPHA = 0.4
WHITE = (255, 255, 255)        # colour of a class the project doesn't know
BLACK = (0, 0, 0)


CHANNELS = ("all", "red", "green", "blue")
CLAHE_CLIP = 2.5               # auto-contrast strength (OpenCV CLAHE clip limit)
CLAHE_TILES = (8, 8)
SHARPEN_SIGMA = 2.0            # blur radius the sharpening works against (px)


@dataclass(frozen=True)
class DisplayStyle:
    """How masks and the image are shown (Settings > Display…). View only:
    nothing here changes the saved image or masks, and the drawing tools
    (SAM, DEXTR, Scissors) still work on the original image."""
    fill: float = ALPHA            # opacity of the mask colour, 0..1
    outlines_only: bool = False    # V: no fill, outlines in the class colour
    hidden: frozenset = frozenset()  # class names not drawn (and not clickable)
    brightness: int = 0            # -100..100
    contrast: int = 0              # -75..100
    gamma: int = 100               # x100: 30..300 (100 = unchanged, >100 = brighter mid-tones)
    saturation: int = 0            # -100 (black & white)..100 (twice as colourful)
    auto_contrast: bool = False    # CLAHE on the lightness
    sharpen: int = 0               # 0..100
    invert: bool = False
    channel: str = "all"           # "all", or show only "red" / "green" / "blue"

    @property
    def alpha(self) -> float:
        return 0.0 if self.outlines_only else self.fill

    @property
    def outline_px(self) -> int:
        return 2 if self.outlines_only else 1

    def outline_color(self, class_color):
        return tuple(class_color) if self.outlines_only else BLACK

    @property
    def image_key(self):
        """Everything that changes the image itself (not the masks)."""
        return (self.brightness, self.contrast, self.gamma, self.saturation,
                self.auto_contrast, self.sharpen, self.invert, self.channel)

    @property
    def adjusts_image(self) -> bool:
        return self.image_key != DEFAULT_IMAGE_KEY

    def adjust(self, image):
        """A copy of `image` (RGB uint8) as the view shows it. Order: one
        channel, auto-contrast, brightness / contrast / gamma, saturation,
        sharpening, invert."""
        if not self.adjusts_image:
            return image.copy()
        out = image
        grey = self.channel in CHANNELS[1:]
        if grey:
            band = out[:, :, CHANNELS.index(self.channel) - 1]
            out = cv2.cvtColor(np.ascontiguousarray(band), cv2.COLOR_GRAY2RGB)
        if self.auto_contrast:
            clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILES)
            if grey:
                out = cv2.cvtColor(clahe.apply(out[:, :, 0]), cv2.COLOR_GRAY2RGB)
            else:
                lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        if self.brightness or self.contrast or self.gamma != 100:
            out = cv2.LUT(out, _lut(self.brightness, self.contrast, self.gamma))
        if self.saturation and not grey:
            hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV)
            factor = 1.0 + self.saturation / 100.0
            hsv[:, :, 1] = cv2.convertScaleAbs(hsv[:, :, 1], alpha=factor)
            out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        if self.sharpen:
            amount = self.sharpen / 50.0          # 0..2
            blurred = cv2.GaussianBlur(out, (0, 0), SHARPEN_SIGMA)
            out = cv2.addWeighted(out, 1.0 + amount, blurred, -amount, 0)
        if self.invert:
            out = cv2.bitwise_not(out)
        return out if out is not image else image.copy()


@lru_cache(maxsize=32)
def _lut(brightness, contrast, gamma=100):
    x = np.arange(256, dtype=np.float32)
    factor = 1.0 + contrast / 100.0
    y = np.clip((x - 128.0) * factor + 128.0 + brightness * 1.27, 0, 255)
    if gamma != 100:
        y = 255.0 * (y / 255.0) ** (100.0 / gamma)
    return np.clip(np.round(y), 0, 255).astype(np.uint8)


DEFAULT_IMAGE_KEY = (0, 0, 100, 0, False, 0, False, "all")
DEFAULT_STYLE = DisplayStyle()


def blend_masks(image, masks, colors, alpha=None, outline_thickness=None, exclude_ids=None,
                style=DEFAULT_STYLE, base=None):
    """
    Copy of `image` (RGB) with the masks blended in (one colour per class)
    and outlined -- as `style` says (fill opacity, outlines only, hidden
    classes, brightness / contrast). alpha / outline_thickness override it.

    masks: [(id, polygon Nx2, class name, area), ...] as the database gives them.
    colors: class name -> (r, g, b).
    exclude_ids: mask ids to leave out (e.g. one being brush-edited).

    Blends inside each polygon's bounding box rather than over full-size
    class masks, so the cost follows the masks' area instead of the image's
    (it matters on large scans). The result is the same pixel for pixel.
    """
    alpha = style.alpha if alpha is None else alpha
    thickness = style.outline_px if outline_thickness is None else outline_thickness
    # base: `image` already adjusted by style (the renderer keeps one per image)
    output = base.copy() if base is not None else style.adjust(image)
    class_to_polys = defaultdict(list)
    for mask_id, mask, class_name, _ in masks:
        if exclude_ids and mask_id in exclude_ids:
            continue
        if class_name in style.hidden:
            continue
        if mask.shape[0] >= 3:
            class_to_polys[class_name].append(mask.astype(np.int32))

    H, W = image.shape[:2]
    class_mask = None  # one raster for all classes; all zero again after each class
    for class_name, polys in class_to_polys.items():
        if class_mask is None and alpha > 0:
            class_mask = np.zeros((H, W), dtype=np.uint8)
        if alpha > 0:
            cv2.fillPoly(class_mask, polys, 255)
        color_rgb = colors.get(class_name, WHITE)

        for poly in (polys if alpha > 0 else ()):
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

        cv2.polylines(output, polys, isClosed=True, color=style.outline_color(color_rgb),
                      thickness=thickness)

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
