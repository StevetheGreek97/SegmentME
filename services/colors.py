"""
Class colours: a palette of clearly different colours, and how alike two
colours look.

"Alike" is measured as people see colour (distance in CIELAB, "delta E"),
not as RGB numbers: #00ff00 and #00e000 differ by 31 in RGB yet look the
same, while some pairs closer in RGB are easy to tell apart.
"""
import cv2
import numpy as np

# Clearly different, saturated colours that stay visible on light and dark
# scans (masks are drawn at 40 % over the image).
PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff",
    "#9a6324", "#800000", "#aaffc3", "#000075", "#ffffff", "#808000",
]   # (no black: mask outlines are drawn in black)

# Below this delta E two colours are hard to tell apart as mask fills.
SIMILAR_DELTA_E = 20.0


def _lab(hex_color):
    h = hex_color.lstrip("#")
    rgb = np.uint8([[[int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)]]])
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)[0, 0]
    return lab


def color_distance(a, b) -> float:
    """How different two colours ("#rrggbb") look: 0 = the same, about 2 =
    just noticeable, over 50 = very different (CIE76 delta E)."""
    return float(np.linalg.norm(_lab(a) - _lab(b)))


def closest(color, others):
    """(name, distance) of the colour in `others` ({name: "#rrggbb"}) that
    looks most like `color`, or (None, inf) when there are none."""
    best = (None, float("inf"))
    for name, other in others.items():
        try:
            d = color_distance(color, other)
        except (ValueError, IndexError):
            continue
        if d < best[1]:
            best = (name, d)
    return best
