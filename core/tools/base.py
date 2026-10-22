"""
What every drawing tool has in common.

A tool (manual polygon, Intelligent Scissors, SAM, DEXTR) only knows its
*host* -- the small set of things it may use from the app:

    host.scene                  where it draws its points and previews
    host.image()                the image on screen (RGB array) or None
    host.image_name()           its name in the database (file name without extension)
    host.choose_class()         the class to save as, or None (the host tells the user why)
    host.save_mask(polygon, image_name, class_name)

and the image view only calls the methods below (left_click, right_click,
mouse_move, left_release, image_changed, complete_mask, clear_temp_items).
So the tools don't depend on the sidebar or the window layout, the view
doesn't need to know which tool is active, and a test can drive a tool with
a fake host.

Saving (S) is the same for all tools and lives here: a tool only says which
outline would be saved (current_polygon).
"""
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from services.logger import get_logger

logger = get_logger(__name__)


class MaskTool(QObject):
    mask_added = pyqtSignal(str, np.ndarray)

    label = "mask"                # shown in messages and the log, e.g. "SAM"
    segments_on_e = False         # E runs a model (compute_polygon on a worker thread)

    def __init__(self, host):
        super().__init__()
        self.host = host

    @property
    def scene(self):
        return self.host.scene

    # ------------------------------------------------- input (from the view)
    def left_click(self, point, ctrl=False):
        """Left click on the image at `point` (x, y) in image pixels."""

    def right_click(self, point):
        """Right click on the image."""

    def mouse_move(self, point):
        """The mouse moved over the image."""

    def left_release(self, point):
        """Left button released."""

    def image_changed(self, image):
        """Another image is shown: unfinished work belongs to the old one."""
        self.clear_temp_items()

    # ------------------------------------------------------------- saving (S)
    def current_polygon(self):
        """The outline S would save (N x 2), or None."""
        return None

    def complete_mask(self):
        """S: save the current outline as a mask of the chosen class."""
        polygon = self.current_polygon()
        if polygon is None or len(polygon) < 3:
            logger.warning("%s: nothing to save yet (need an outline of at least 3 points)",
                           self.label)
            return
        polygon = np.asarray(polygon, dtype=np.float32)
        if polygon.ndim != 2 or polygon.shape[1] != 2:
            logger.error("%s: invalid outline shape %s (expected N x 2); not saved",
                         self.label, polygon.shape)
            return
        class_name = self.host.choose_class()
        if class_name is None:
            self.clear_temp_items()
            return
        image_name = self.host.image_name()
        self.host.save_mask(polygon, image_name, class_name)
        logger.info("Saved %s mask (%d points) for image %s as class %r",
                    self.label, len(polygon), image_name, class_name)
        self.mask_added.emit(image_name, polygon)
        self.clear_temp_items()

    def clear_temp_items(self):
        """Remove the tool's points and previews from the scene."""
