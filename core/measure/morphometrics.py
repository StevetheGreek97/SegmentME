"""
Shape measurements of one mask outline (pixel coordinates), and converting
them to real units.

Length and width are the long and short side of the tightest rotated
rectangle around the outline -- robust on rough masks; for a strongly bent
animal it is the length of its bounding box, not along its midline.
Orientation is the angle of that long side: 0 deg = horizontal, 90 deg =
vertical, counting counter-clockwise as seen on screen.
The shape descriptors are those of HiReS (hires.models.Annotation), computed
on the pixel outline here so that non-square images are not distorted.
"""
import math
from typing import Dict, Optional

import cv2
import numpy as np

# name -> how it scales with the pixel size: 2 = area, 1 = length, 0 = none
FIELDS = {
    "area": 2,
    "length": 1,
    "width": 1,
    "perimeter": 1,
    "orientation_deg": 0,
    "aspect_ratio": 0,
    "solidity": 0,
    "circularity": 0,
    "centroid_x": 1,
    "centroid_y": 1,
}


def measure(polygon) -> Optional[Dict[str, float]]:
    """Measurements in pixels of an Nx2 outline, or None if it has no area."""
    pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3:
        return None
    x, y = pts[:, 0].astype(np.float64), pts[:, 1].astype(np.float64)
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    signed = 0.5 * float(cross.sum())
    area = abs(signed)
    if area <= 0:
        return None
    perimeter = float(np.hypot(xn - x, yn - y).sum())
    cx = float(((x + xn) * cross).sum() / (6.0 * signed))
    cy = float(((y + yn) * cross).sum() / (6.0 * signed))

    box = cv2.boxPoints(cv2.minAreaRect(pts)).astype(np.float64)
    edges = np.roll(box, -1, axis=0) - box
    sides = np.hypot(edges[:, 0], edges[:, 1])
    long_edge = edges[int(np.argmax(sides))]
    length, width = float(sides.max()), float(sides.min())
    # screen y points down: flip it so angles count counter-clockwise as seen
    orientation = math.degrees(math.atan2(-long_edge[1], long_edge[0])) % 180.0

    hull_area = float(cv2.contourArea(cv2.convexHull(pts)))
    return {
        "area": area,
        "length": length,
        "width": width,
        "perimeter": perimeter,
        "orientation_deg": orientation,
        "aspect_ratio": length / width if width > 0 else float("nan"),
        "solidity": min(area / hull_area, 1.0) if hull_area > 0 else float("nan"),
        "circularity": 4.0 * math.pi * area / perimeter ** 2 if perimeter > 0 else float("nan"),
        "centroid_x": cx,
        "centroid_y": cy,
    }


def to_units(pixels: Dict[str, float], um_per_px: Optional[float], unit_um: float) -> Dict[str, Optional[float]]:
    """Convert pixel measurements to a unit of `unit_um` micrometres (mm: 1000).
    Size measurements become None when there is no scale."""
    out = {}
    for name, power in FIELDS.items():
        value = pixels[name]
        if power == 0:
            out[name] = value
        elif not um_per_px:
            out[name] = None
        else:
            out[name] = value * (um_per_px / unit_um) ** power
    return out


def nice_bar_length(um_per_screen_px: float, target_px: float = 120.0) -> float:
    """A round length in micrometres (1, 2 or 5 x 10^n) whose bar is close to
    `target_px` screen pixels long."""
    if um_per_screen_px <= 0:
        return 0.0
    want = um_per_screen_px * target_px
    exp = math.floor(math.log10(want))
    best = None
    for k in (exp - 1, exp, exp + 1):
        for m in (1, 2, 5):
            value = m * 10 ** k
            if best is None or abs(math.log(value / want)) < abs(math.log(best / want)):
                best = value
    return float(best)


def format_length(um: float) -> str:
    """'1 mm', '500 µm', '2.5 mm'."""
    if um >= 1000:
        mm = um / 1000
        return f"{mm:g} mm"
    return f"{um:g} µm"
