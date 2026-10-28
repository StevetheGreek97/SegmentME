"""
The picture the image view shows: the image with its masks blended in.

A full render draws every mask (mask_render.blend_masks). After an edit only
the area around the masks that changed is redrawn -- the result is the same,
pixel for pixel (tests/test_render_partial.py) -- which takes ~40 ms instead
of ~450 ms on a busy scan. The picture is shown as is by the view's
ArrayImageItem, so a partial redraw only repaints the changed boxes.
"""
from collections import Counter

import cv2
import numpy as np
from PyQt6.QtGui import QColor

from services.logger import get_logger
from ui.elements.image_display import mask_render

logger = get_logger(__name__)


class MaskRenderer:
    def __init__(self, view):
        self._view = view
        self.picture = None            # the image with masks, as shown
        self._render_state = None      # what the picture was drawn from (for partial redraws)
        self.last_redraw = None        # "partial" / "full": for tests and the log
        self._base, self._base_key, self._base_of = None, None, None

    @property
    def _app(self):
        return self._view.parent.state_manager

    def masks_shown(self) -> bool:
        """Masks are drawn (not hidden from the View menu, H not held)."""
        return self._view.masks_visible and not self._view._peeking

    def style(self):
        """The current DisplayStyle (Settings > Display…)."""
        settings = getattr(self._view.parent, "display_settings", None)
        return settings.style() if settings is not None else mask_render.DEFAULT_STYLE

    def base(self, image, style=None):
        """`image` as the view shows it without masks (Settings > Display…:
        brightness, auto-contrast, sharpening...). Worked out once per image
        and look, then reused by every redraw -- sharpening and auto-contrast
        look at neighbouring pixels, so a patch can't be adjusted on its own.
        The image itself when nothing is adjusted (no copy)."""
        style = style or self.style()
        if not style.adjusts_image:
            return image
        key = (id(image), style.image_key)
        if self._base_key != key or self._base_of is not image:
            self._base = style.adjust(image)
            self._base_key, self._base_of = key, image
        return self._base

    def adopt_base(self, image, base, style):
        """A base the worker thread already made (prepare_image)."""
        if base is not None and style.adjusts_image:
            self._base = base
            self._base_key, self._base_of = (id(image), style.image_key), image

    def plain(self, image):
        """The image without masks, as shown (H, View > Show Masks off)."""
        return self.base(image)

    def show(self, array):
        """Show this RGB array in the view (shared, not copied)."""
        self._view.pixmap_item.set_array(array)

    def redraw(self, image, masks):
        """The masks changed: redraw where needed. Returns "partial" or "full"."""
        how = "partial" if self._render_changes(image, masks) else "full"
        if how == "full":
            self.render_full(image, masks)
        self.last_redraw = how
        return how

    def class_colors(self):
        """Class name -> (r, g, b) for drawing the masks."""
        colors = {}
        for _id, name, color in self._app.class_manager.list_classes():
            c = QColor(color)
            colors[name] = (c.red(), c.green(), c.blue())
        return colors

    def prepared_still_valid(self, prepared) -> bool:
        """The worker drew the masks a moment ago: is that picture still right
        (same masks, class colours and visibility)? Edits are off while an
        image loads, but e.g. inference may have added masks meanwhile."""
        if prepared["visible"] != self.masks_shown():
            return False
        if prepared.get("style", mask_render.DEFAULT_STYLE) != self.style():
            return False
        if prepared["visible"] and prepared["colors"] != self.class_colors():
            return False
        count, top, total = self._app.mask_manager.fingerprint(self._app.current_image_name)
        ids = [m[0] for m in prepared["masks"]]
        ok = (count, top, total) == (len(ids), max(ids) if ids else None, float(sum(ids)))
        if not ok:
            logger.debug("Masks changed while the image loaded: drawing them again")
        return ok

    def overlay(self, image, alpha=None, outline_thickness=None, exclude_ids=None, masks=None):
        """
        Copy of `image` with the masks blended in (one colour per class) and outlined.
        exclude_ids: optional set of mask ids to leave unrendered (e.g. while brush-editing one).

        Blends inside each polygon's bounding box rather than over full-size
        class masks, so the cost follows the masks' area instead of the image's
        (it matters on large scans). The result is the same pixel for pixel.
        """
        if not self.masks_shown():
            return self.base(image).copy()
        if masks is None:
            masks = self._app.mask_manager.load_masks(
                self._app.current_image_name
            )
        return mask_render.blend_masks(image, masks, self.class_colors(), alpha=alpha,
                                       outline_thickness=outline_thickness, exclude_ids=exclude_ids,
                                       style=self.style(), base=self.base(image))

    def render_full(self, image, masks=None, picture=None):
        """Blend all masks into a copy of the image and show it.
        picture: that copy, already drawn (by show_image()'s worker)."""
        if picture is None:
            picture = self.overlay(image, masks=masks)
        self.picture = picture
        self.show(picture)
        if self.masks_shown():
            if masks is None:
                masks = self._app.mask_manager.load_masks(
                    self._app.current_image_name)
            self._render_state = self._render_key(image, masks)
        else:
            self._render_state = None

    def _render_key(self, image, masks):
        """What the render depends on: class order and colours, and per class
        the outlines drawn (as a multiset: ids and row order don't matter)."""
        order, polys = [], Counter()
        seen = set()
        style = self.style()
        for _id, mask, class_name, _ in masks:
            if mask.shape[0] < 3 or class_name in style.hidden:
                continue
            if class_name not in seen:
                seen.add(class_name)
                order.append(class_name)
            polys[(class_name, mask.astype(np.int32).tobytes())] += 1
        cm = self._app.class_manager
        colors = tuple(cm.get_class_color(c).rgb() for c in order)
        return {"image": image, "array": self.picture, "style": style,
                "order": tuple(order), "colors": colors, "polys": polys}

    def _render_changes(self, image, masks) -> bool:
        """Redraw only around masks that were added, removed, reshaped or
        re-classed since the last render (the result is the same, pixel for
        pixel, as a full render). False when a full render is needed instead:
        nothing to compare with, class order or colours changed, or too much
        changed for this to be faster."""
        old = self._render_state
        if (old is None or not self.masks_shown()
                or old["image"] is not image or old["array"] is not self.picture):
            return False
        new = self._render_key(image, masks)
        if (new["order"] != old["order"] or new["colors"] != old["colors"]
                or new["style"] != old["style"]):
            return False
        style = new["style"]
        pad = style.outline_px            # thick outlines reach past their boxes
        changed = (old["polys"] - new["polys"]) + (new["polys"] - old["polys"])
        if not changed:
            self._render_state = new | {"array": self.picture}
            return True
        H, W = image.shape[:2]
        rects = []
        for (_cls, raw) in changed:
            x, y, w, h = cv2.boundingRect(np.frombuffer(raw, np.int32).reshape(-1, 2))
            x0, y0 = max(x - pad, 0), max(y - pad, 0)
            x1, y1 = min(x + w + pad, W), min(y + h + pad, H)
            if x1 > x0 and y1 > y0:
                rects.append((x0, y0, x1, y1))
        area = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in rects)
        if len(rects) > 200 or area > 0.25 * H * W:
            return False

        # All drawn outlines with their boxes, in render order
        order_index = {c: i for i, c in enumerate(new["order"])}
        polys, boxes = [], []
        for _id, mask, class_name, _ in masks:
            if mask.shape[0] >= 3 and class_name not in style.hidden:
                poly = mask.astype(np.int32)
                x, y, w, h = cv2.boundingRect(poly)
                polys.append((order_index[class_name], class_name, poly))
                boxes.append((x - pad, y - pad, x + w + pad, y + h + pad))
        boxes = np.array(boxes, dtype=np.int64).reshape(-1, 4)
        out = self.picture
        base = self.base(image, style)
        cm = self._app.class_manager
        alpha = style.alpha

        for rx0, ry0, rx1, ry1 in rects:
            hit = np.nonzero((boxes[:, 0] < rx1) & (boxes[:, 2] > rx0)
                             & (boxes[:, 1] < ry1) & (boxes[:, 3] > ry0))[0]
            # Work area: the region plus every touching outline in full, so
            # outlines are rasterised exactly as in a full render (no clipping
            # except at the image border, as there).
            bx0, by0, bx1, by1 = rx0, ry0, rx1, ry1
            if len(hit):
                bx0 = max(0, min(bx0, int(boxes[hit, 0].min())))
                by0 = max(0, min(by0, int(boxes[hit, 1].min())))
                bx1 = min(W, max(bx1, int(boxes[hit, 2].max())))
                by1 = min(H, max(by1, int(boxes[hit, 3].max())))
            buf = base[by0:by1, bx0:bx1].copy()
            offset = np.array([bx0, by0], dtype=np.int32)
            by_class = {}
            for i in hit:
                k, class_name, poly = polys[i]
                by_class.setdefault(k, (class_name, []))[1].append(poly - offset)
            for k in sorted(by_class):
                class_name, cps = by_class[k]
                c = cm.get_class_color(class_name)
                rgb = (int(c.red()), int(c.green()), int(c.blue()))
                if alpha > 0:
                    mask = np.zeros(buf.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, cps, 255)
                    color = np.empty_like(buf)
                    color[:] = rgb
                    cv2.copyTo(cv2.addWeighted(buf, 1.0 - alpha, color, alpha, 0.0), mask, buf)
                cv2.polylines(buf, cps, isClosed=True, color=style.outline_color(rgb),
                              thickness=style.outline_px)
            out[ry0:ry1, rx0:rx1] = buf[ry0 - by0:ry1 - by0, rx0 - bx0:rx1 - bx0]

        self._patch_pixmap(out, rects)
        self._render_state = new
        return True

    def _patch_pixmap(self, image_rgb, rects):
        """image_rgb was changed in place inside `rects`: repaint just those."""
        item = self._view.pixmap_item
        if item.array() is image_rgb:
            item.changed(rects)
        else:
            item.set_array(image_rgb)
