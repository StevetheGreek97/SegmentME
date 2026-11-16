import threading
import time
import weakref

import numpy as np
from PyQt6.QtGui import QPen, QColor, QPolygonF
from PyQt6.QtWidgets import QGraphicsEllipseItem, QGraphicsPolygonItem, QGraphicsRectItem
from PyQt6.QtCore import QPointF, QRectF, Qt
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
import cv2
from core.tools import sam_everything
from core.tools.base import MaskTool
from core.tools.sam2_loader import load_sam2_model
from services.logger import get_logger

logger = get_logger(__name__)


def load_predictor(device, variant):
    """The SAM2-family model for `variant`, ready to use (seconds; touches
    no Qt objects, so it can run on a worker thread)."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    return SAM2ImagePredictor(load_sam2_model(device, variant))


class SamMasker2(MaskTool):
    """
    A class for creating masks interactively using the SAM 2 model.

    Prompts can be combined freely:
    * left click        — positive (foreground) point
    * right click       — negative (background) point
    * Ctrl + left drag  — bounding box
    * E                 — the one object the prompts point at
    * F                 — every object inside the box (a box is needed;
                          core/tools/sam_everything.py): green points are always included, objects under a red
                          point are left out; right-click a result to drop it,
                          S saves them all

    SAM resizes whatever it is given to a fixed square (1024 x 1024 for SAM2),
    so on a large scan objects shrink ~10x and the masks come out coarse. The
    tool therefore runs SAM on a full-resolution window around the prompts
    instead of the whole image (see generate_mask).
    """
    label = "SAM"
    segments_on_e = True     # E runs SAM on a worker thread (see the image view)
    finds_all = True         # F: every object in the box, also on a worker thread

    # Side of the window SAM looks at, in image pixels: SAM2's input size, so
    # a window this big is segmented at the image's own resolution.
    WINDOW_SIZE = 1024
    # How often the window may double when the mask runs into its edge (an
    # object bigger than the window); after that the result is used as is.
    MAX_WINDOW_GROWTH = 3

    def __init__(self, host, device: str = "cpu", variant=None, predictor=None):
        """
        Initialize the Sam2 class.

        Args:
            host: the app as tools see it (core/tools/base.py).
            device (str): Device to run the model on ('cpu' or 'cuda').
            variant: Optional sam_registry.SamVariant choosing which
                SAM2-family checkpoint to load (default: SAM2 Tiny).
            predictor: the model, already loaded (ToolManager loads it on a
                worker thread, see load_predictor); None loads it here.
        """
        super().__init__(host)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.variant = variant

        # The segmentation model (subclasses override _load_model)
        self.predictor = predictor if predictor is not None else self._load_model()

        # Interactive points and items
        self.foreground_points = []
        self.background_points = []
        self.foreground_items = []
        self.background_items = []

        # Optional bounding-box prompt (drawn with Ctrl + left drag)
        self.box = None  # np.array([x1, y1, x2, y2]) or None
        self.box_item = None  # QGraphicsRectItem shown on the scene
        self.is_drawing_box = False
        self.box_start = None

        self.current_polygon_item = None  # Reference to displayed polygon
        self.mask = None

        # The window SAM currently holds an embedding for, and a weak reference
        # to the image it was cut from: reused while the prompts stay inside it,
        # and never stale, since the reference dies with the image.
        self._embedded_image = None   # weakref.ref to the image array
        self._embedded_window = None  # (x0, y0, x1, y1)
        self._object_window_side = None  # window side the current object needed

        # F: objects found in the box (outlines + their items on the scene)
        self.found = []
        self.found_items = []
        self._stop = threading.Event()   # set to stop a search that is running
        self._progress = None            # (done, total) of the running search

    # -------------------------------------------------------------- input
    def left_click(self, point, ctrl=False):
        if ctrl:                                  # Ctrl + left drag: box prompt
            self.box_start = point
            self.is_drawing_box = True
        else:
            self.add_point(point, 1)

    def right_click(self, point):
        # On a found object: drop it (the red point keeps it out of the next F too)
        self.drop_found_at(point)
        self.add_point(point, 0)

    def mouse_move(self, point):
        if self.is_drawing_box:
            self.update_box_preview(self.box_start, point)

    def left_release(self, point):
        if self.is_drawing_box:
            self.add_box(self.box_start, point)
            self.is_drawing_box = False

    def current_polygon(self):
        return self.mask

    # ------------------------------------------------- F: everything in the box
    def find_all_job(self):
        """The prompts for F as arguments for compute_all; None without a box
        (F only searches inside a box the user drew)."""
        if self.box is None:
            logger.info("Find all: no box drawn; nothing to search")
            return None
        existing = []
        get_existing = getattr(self.host, "existing_polygons", None)
        if get_existing is not None:
            try:
                existing = get_existing()
            except Exception:
                logger.exception("Find all: could not read the saved masks; finding everything")
        self._stop = threading.Event()
        self._progress = None
        min_scale = None                 # how far SAM may shrink the image: from the
        get_sizes = getattr(self.host, "typical_object_sizes", None)   # objects saved so far
        if get_sizes is not None:
            try:
                min_scale = sam_everything.min_scale_for(get_sizes())
            except Exception:
                logger.exception("Find all: could not read the object sizes; using the default")
        return (self.box.copy(), list(self.foreground_points), list(self.background_points),
                existing, self._stop, min_scale)

    def compute_all(self, image, box, positives, negatives, existing, stop, min_scale=None):
        """The model part of F (worker thread, no Qt objects): the outlines found."""
        # The search embeds its own crop: the window held for E is gone.
        self._embedded_image = None
        self._embedded_window = None

        def progress(done, total):
            self._progress = (done, total)

        return sam_everything.find_objects(self.predictor, image, box, positives, negatives,
                                           existing, progress=progress, cancelled=stop.is_set,
                                           min_scale=min_scale)

    def progress_text(self):
        """For the status bar while F runs."""
        if self._progress is None:
            return "looking at the area"
        done, total = self._progress
        return f"{100 * done // max(total, 1)}%"

    def apply_all(self, polygons):
        """Show what F found (GUI thread). Returns the message for the status bar."""
        self.clear_found()
        if self.current_polygon_item:                # E's outline would be saved instead
            self.scene.removeItem(self.current_polygon_item)
            self.current_polygon_item = None
        self.mask = None
        for polygon in polygons:
            self._add_found(np.asarray(polygon, dtype=np.float32))
        logger.info("Find all: showing %d object(s)", len(self.found))
        return self.found_message()

    def found_message(self):
        if not self.found:
            return "SAM found no new objects here"
        return (f"SAM found {len(self.found)} object(s): S saves them all, "
                "right-click one to leave it out")

    def _add_found(self, polygon):
        item = QGraphicsPolygonItem(QPolygonF([QPointF(float(x), float(y)) for x, y in polygon]))
        pen = QPen(QColor(0, 255, 0), 2)
        pen.setCosmetic(True)
        item.setPen(pen)
        item.setBrush(QColor(0, 255, 0, 50))
        item.setZValue(15)
        self.scene.addItem(item)
        self.found.append(polygon)
        self.found_items.append(item)

    def drop_found_at(self, point) -> bool:
        """Leave out the found object(s) under `point`."""
        x, y = float(point[0]), float(point[1])
        dropped = False
        for i in reversed(range(len(self.found))):
            contour = self.found[i].reshape(-1, 1, 2)
            if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                self.scene.removeItem(self.found_items[i])
                del self.found[i], self.found_items[i]
                dropped = True
        if dropped:
            logger.debug("Find all: dropped object at %s, %d left", point, len(self.found))
        return dropped

    def clear_found(self):
        for item in self.found_items:
            self.scene.removeItem(item)
        self.found_items.clear()
        self.found.clear()

    def complete_mask(self):
        """S: after F, save every object found (one class, one step);
        otherwise the outline E made."""
        if not self.found:
            return super().complete_mask()
        class_name = self.host.choose_class()
        if class_name is None:
            return                     # keep the results: the user can pick a class and press S again
        image_name = self.host.image_name()
        polygons = list(self.found)
        save_many = getattr(self.host, "save_masks", None)
        if save_many is not None:
            save_many(polygons, image_name, class_name)
        else:
            for polygon in polygons:
                self.host.save_mask(polygon, image_name, class_name)
        logger.info("Saved %d SAM object(s) for image %s as class %r",
                    len(polygons), image_name, class_name)
        for polygon in polygons:
            self.mask_added.emit(image_name, polygon)
        self.clear_temp_items()

    def _load_model(self):
        """Load the SAM 2 model once."""
        return load_predictor(self.device, self.variant)

    def _add_graphics_point(self, point: tuple, label: int):
        """Helper function to add a point to the scene."""
        color = QColor(0, 255, 0) if label == 1 else QColor(255, 0, 0)
        pen = QPen(color, 2)

        ellipse = QGraphicsEllipseItem(point[0] - 2, point[1] - 2, 4, 4)
        ellipse.setPen(pen)
        ellipse.setBrush(color)
        self.scene.addItem(ellipse)

        return ellipse

    def add_point(self, point: tuple, label: int):
        """
        Add a point to the appropriate list (foreground or background).

        Args:
            point (tuple): The point to add as (x, y).
            label (int): The label for the point (1 for foreground, 0 for background).
        """
        ellipse = self._add_graphics_point(point, label)

        if label == 1:
            self.foreground_points.append(point)
            self.foreground_items.append(ellipse)
        else:
            self.background_points.append(point)
            self.background_items.append(ellipse)

        logger.debug("Added %s point at %s",
                     "foreground" if label == 1 else "background", point)

    def add_box(self, start_point: tuple, end_point: tuple):
        """
        Set the bounding-box prompt and draw it on the scene.

        Args:
            start_point (tuple): One corner of the box (x1, y1).
            end_point (tuple): The opposite corner (x2, y2).
        """
        self._set_box(start_point, end_point, QPen(QColor(0, 0, 255), 2))
        logger.debug("Added bounding box %s", self.box)

    def update_box_preview(self, start_point: tuple, end_point: tuple):
        """Redraw the dashed box preview while the user is still dragging."""
        self._set_box(start_point, end_point,
                      QPen(QColor(0, 0, 255), 2, Qt.PenStyle.DashLine))

    def _set_box(self, start_point: tuple, end_point: tuple, pen: QPen):
        """Replace the current box (data + graphics item) with a new one."""
        self.clear_box()

        x1, y1 = start_point
        x2, y2 = end_point
        self.box = np.array([x1, y1, x2, y2])

        rect = QRectF(x1, y1, x2 - x1, y2 - y1)
        self.box_item = QGraphicsRectItem(rect)
        self.box_item.setPen(pen)
        self.scene.addItem(self.box_item)

    def clear_box(self):
        """Remove any existing bounding box."""
        if self.box_item:
            self.scene.removeItem(self.box_item)
            self.box_item = None
        self.box = None

    def display_polygon(self, polygon_points: list, color=QColor(0, 255, 0), line_width: int = 2):
        """
        Display a polygon on the scene using the provided points.

        Args:
            polygon_points (list): List of (x, y) tuples representing the polygon vertices.
            color (QColor): Color of the polygon outline.
            line_width (int): Width of the polygon lines.
        """
        if self.current_polygon_item:
            self.scene.removeItem(self.current_polygon_item)
            self.current_polygon_item = None

        if not polygon_points:
            logger.warning("No polygon points to display.")
            return

        polygon = QPolygonF([QPointF(x, y) for x, y in polygon_points])
        pen = QPen(color)
        pen.setWidth(line_width)

        polygon_item = QGraphicsPolygonItem(polygon)
        polygon_item.setPen(pen)
        self.scene.addItem(polygon_item)

        self.current_polygon_item = polygon_item
        self.mask = None

    def generate_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Generate a mask using the SAM model based on added points and/or bounding box.

        SAM runs on a window of the image around the prompts, at least
        WINDOW_SIZE pixels a side, at full resolution. If the mask runs into
        the window's edge (inside the image), the window doubles and SAM runs
        again, up to MAX_WINDOW_GROWTH times.

        Runs everything here; the E key uses mask_job() + compute_polygon()
        on a worker thread and apply_polygon() back on the GUI thread instead.

        Returns:
            numpy.ndarray: Generated mask polygon (Nx2, image coordinates).
        """
        job = self.mask_job()
        if job is None:
            return None
        return self.apply_polygon(*self.compute_polygon(image, *job))

    def mask_job(self):
        """The current prompts as (points, labels, box), or None without any."""
        if self.box is None and not self.foreground_points:
            logger.warning("Cannot generate mask: no box or foreground points added")
            return None
        points = np.array(self.foreground_points + self.background_points,
                          dtype=np.float32).reshape(-1, 2)
        labels = np.array([1] * len(self.foreground_points) + [0] * len(self.background_points),
                          dtype=np.int32)
        box = None
        if self.box is not None:  # as dragged; SAM wants top-left, bottom-right
            x1, y1, x2, y2 = (float(v) for v in self.box)
            box = np.array([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], dtype=np.float32)
        return points, labels, box

    def compute_polygon(self, image, points, labels, box):
        """The model part (seconds on a CPU): (polygon, window side used).
        Touches no Qt objects, so it can run on a worker thread -- one call
        at a time, as the predictor holds one embedding."""
        side = self._initial_window_side(points, box)
        for growth in range(self.MAX_WINDOW_GROWTH + 1):
            window = self._window_for(image, points, box, side)
            mask = self._predict_in_window(image, window, points, labels, box)
            whole_image = window == (0, 0, image.shape[1], image.shape[0])
            if (growth == self.MAX_WINDOW_GROWTH or whole_image
                    or not self._cut_by_window(mask, window, image.shape)):
                break
            side *= 2
            logger.debug("SAM mask reaches the window edge; retrying with a %d px window", side)

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE, offset=window[:2])
        # The largest piece is the object; SAM often adds specks around it.
        polygon = max(contours, key=cv2.contourArea).squeeze(axis=1).tolist() if contours else []
        return polygon, side

    def apply_polygon(self, polygon, side):
        """Show the result (GUI thread)."""
        self.clear_found()            # E's one object replaces what F found
        self._object_window_side = side
        self.display_polygon(polygon)
        self.mask = np.array(polygon, dtype=np.float32)
        return self.mask

    # ------------------------------------------------------------ windowing
    @staticmethod
    def _prompt_extent(points, box):
        """(xmin, ymin, xmax, ymax) around all prompt points and the box."""
        xy = points if box is None else np.vstack([points, box.reshape(2, 2)])
        return xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()

    def _initial_window_side(self, points, box):
        """WINDOW_SIZE, more when the prompts span a big area (with room around
        them), and at least what this object needed last time (refining it)."""
        xmin, ymin, xmax, ymax = self._prompt_extent(points, box)
        side = max(self.WINDOW_SIZE, int(max(xmax - xmin, ymax - ymin) * 1.5) + 64)
        return max(side, self._object_window_side or 0)

    def _window_for(self, image, points, box, side):
        """
        Square window of `side` pixels centred on the prompts, shifted to lie
        inside the image (as wide or tall as the image where it is smaller).

        The window SAM already holds is kept instead when it fits: big enough
        (but not over twice the size), with the prompts in its middle half.
        That way refining an object, or the next object close by, needs no new
        embedding.
        """
        H, W = image.shape[:2]
        w, h = min(side, W), min(side, H)
        xmin, ymin, xmax, ymax = self._prompt_extent(points, box)

        held = self._held_window(image)
        if held is not None:
            x0, y0, x1, y1 = held
            mx, my = (x1 - x0) // 4, (y1 - y0) // 4
            size_ok = w <= x1 - x0 <= 2 * w and h <= y1 - y0 <= 2 * h
            inside = ((x0 == 0 or xmin - x0 >= mx) and (y0 == 0 or ymin - y0 >= my)
                      and (x1 == W or x1 - xmax >= mx) and (y1 == H or y1 - ymax >= my))
            if size_ok and inside:
                return held

        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        x0 = int(min(max(round(cx - w / 2), 0), W - w))
        y0 = int(min(max(round(cy - h / 2), 0), H - h))
        return (x0, y0, x0 + w, y0 + h)

    def _held_window(self, image):
        """The window SAM holds an embedding for, if it was cut from `image`."""
        ref = self._embedded_image
        if ref is not None and ref() is image:
            return self._embedded_window
        return None

    def _predict_in_window(self, image, window, points, labels, box):
        """Run SAM on `window`, embedding it first unless it already holds it.
        Returns the window's mask as a bool array."""
        x0, y0, x1, y1 = window
        if self._held_window(image) != window:
            t0 = time.perf_counter()
            self.predictor.set_image(np.ascontiguousarray(image[y0:y1, x0:x1]))
            self._embedded_image = weakref.ref(image)
            self._embedded_window = window
            logger.debug("SAM embedded window x=%d-%d, y=%d-%d (%dx%d px) in %.2f s",
                         x0, x1, y0, y1, x1 - x0, y1 - y0, time.perf_counter() - t0)

        offset = np.array([x0, y0], dtype=np.float32)
        masks, _, _ = self.predictor.predict(
            point_coords=points - offset if len(points) else None,
            point_labels=labels if len(labels) else None,
            box=(box - np.tile(offset, 2))[None, :] if box is not None else None,
            multimask_output=False,
        )
        return masks[0] > 0

    @staticmethod
    def _cut_by_window(mask, window, shape):
        """True when the mask touches a window edge that is not an image edge:
        the object probably continues outside the window."""
        x0, y0, x1, y1 = window
        H, W = shape[:2]
        return bool((x0 > 0 and mask[:, 0].any()) or (y0 > 0 and mask[0, :].any())
                    or (x1 < W and mask[:, -1].any()) or (y1 < H and mask[-1, :].any()))

    def clear_temp_items(self):
        """
        Clear temporary points, lines, and polygons.
        """
        for item in self.foreground_items + self.background_items:
            self.scene.removeItem(item)

        self.foreground_items.clear()
        self.background_items.clear()
        self.foreground_points.clear()
        self.background_points.clear()

        self.clear_box()
        self.is_drawing_box = False
        self.box_start = None

        if self.current_polygon_item:
            self.scene.removeItem(self.current_polygon_item)
            self.current_polygon_item = None

        self.mask = None
        self._object_window_side = None  # the next object starts with a fresh window size
        self._stop.set()                 # a search still running is not wanted any more
        self.clear_found()
        logger.debug("Cleared temporary items")
