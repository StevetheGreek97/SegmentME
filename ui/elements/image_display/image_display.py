from PyQt6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsPathItem, QTableWidget,
)
from PyQt6.QtCore import Qt, QPointF, QItemSelection, QItemSelectionModel
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPainterPath, QPen, QColor, QPolygonF
from core.tools.manual_mask import ManualMask
from core.tools.sam2_masker import SamMasker2
from core.tools.dextr_mask import DEXTRMasker
from core.tools.intellignent_scissors import IntelligentScissors
from ui.custom_components.mask_context_menu import MaskContextMenu
from ui.elements.image_display.edit_modes import SplitStrokeMode, BrushEditMode
from services.logger import get_logger
from services.image_files import missing_image_message
import cv2
import numpy as np
import time

logger = get_logger(__name__)

from PyQt6.QtWidgets import QApplication


class ImageDisplay(QGraphicsView):
    """
    QGraphicsView that shows the base image+filled masks as one pixmap. Selection
    outlines are a separate vector item on top, so changing the selection never
    touches image-sized buffers; edit modes draw their previews into
    `highlight_item` (see edit_modes.py).
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        self.x, self.y = None, None
        self._zoom_factor = 1.0             # relative to fit-to-window
        self._zoom_min = 1.0
        self._zoom_max = 10.0               # zoom-in always allowed up to this (relative)...
        self._max_pixel_zoom = 20.0         # ...and on until 1 image px = this many screen px
        self._is_panning = False

        self.highlighted_mask_ids = []

        # Scene + items
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        # Base image item
        self.pixmap_item = QGraphicsPixmapItem()
        self.pixmap_item.setZValue(0)
        self.scene.addItem(self.pixmap_item)

        # Edit-mode preview layer (brush edit draws into it; see edit_modes.py)
        self.highlight_item = QGraphicsPixmapItem()
        self.highlight_item.setZValue(10)
        self.scene.addItem(self.highlight_item)

        # Selection outlines: one vector path for all selected masks, so its
        # cost follows the selected outlines, not the image size.
        self.selection_item = QGraphicsPathItem()
        outline_pen = QPen(QColor(255, 255, 255))
        outline_pen.setWidthF(2)
        outline_pen.setCosmetic(True)  # 2 screen pixels at every zoom level
        self.selection_item.setPen(outline_pen)
        self.selection_item.setZValue(11)
        self.scene.addItem(self.selection_item)

        # View settings
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setRenderHints(QPainter.RenderHint.SmoothPixmapTransform | QPainter.RenderHint.Antialiasing)
        self.setMouseTracking(True)

        # Cached data
        self.cached_image_with_masks = None
        self._mask_index = {}               # str(mask_id) -> np.ndarray(int32) polygon
        self._mask_outlines = {}            # str(mask_id) -> QPolygonF, built on first highlight

        # Mask visibility
        self.masks_visible = True
        self._peeking = False               # True while H is held down

        # Active mask edit mode (split stroke / brush) — see edit_modes.py
        self._edit_mode = None

        # Shown instead of the image when its file can't be opened (e.g. the
        # original of a linked image was moved); drawing is off meanwhile.
        self._missing_item = None

    # ---------- Public API ----------

    def display_image(self, image_path, preserve_zoom=False):
        """
        Load and display the image; render filled masks once to base pixmap.
        Prepare a transparent overlay for fast highlights.
        """
        if not image_path:
            logger.warning("display_image called without an image path")
            return

        image = cv2.imread(image_path)
        if image is None:
            logger.error("Failed to load image %s (missing or unreadable)", image_path)
            self._show_missing_image(image_path)
            return
        self._clear_missing_image()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.parent.state_manager.current_image = image

        if not preserve_zoom:
            # A selection belongs to the image it was made on (the table's
            # selection is reset for the new image too).
            self.highlighted_mask_ids = []

        # Build polygon index for this image so highlights don't call load_masks every time
        self._rebuild_mask_index()

        # Render base once
        t0 = time.perf_counter()
        self.cached_image_with_masks = self.overlay_base_masks(image)
        self._set_base_pixmap(self.cached_image_with_masks)
        t1 = time.perf_counter()

        # Clear any edit-mode preview; redraw the selection outlines
        self.highlight_item.setPixmap(QPixmap())
        self.highlight_item.setOffset(0, 0)
        self.highlight_selected_masks(self.highlighted_mask_ids)
        t2 = time.perf_counter()

        # Fit view
        if not preserve_zoom:
            self.cancel_edit_mode()  # navigating away aborts a pending split/brush edit
            self.resetTransform()
            self._zoom_factor = 1.0
            self.scene.setSceneRect(self.pixmap_item.boundingRect())
            self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

        logger.debug("display_image: base_render=%.2f ms, outlines=%.2f ms",
                     (t1 - t0) * 1000, (t2 - t1) * 1000)

    def overlay_base_masks(self, image, alpha=0.4, outline_thickness=1, exclude_ids=None):
        """
        Copy of `image` with the masks blended in (one colour per class) and outlined.
        exclude_ids: optional set of mask ids to leave unrendered (e.g. while brush-editing one).

        Blends inside each polygon's bounding box rather than over full-size
        class masks, so the cost follows the masks' area instead of the image's
        (it matters on large scans). The result is the same pixel for pixel.
        """
        output = image.copy()
        if not self.masks_visible or self._peeking:
            return output
        masks = self.parent.state_manager.mask_manager.load_masks(
            self.parent.state_manager.current_image_name
        )

        # group by class for single blends
        from collections import defaultdict
        class_to_polys = defaultdict(list)
        for mask_id, mask, class_name, _ in masks:
            if exclude_ids and mask_id in exclude_ids:
                continue
            if mask.shape[0] >= 3:
                class_to_polys[class_name].append(mask.astype(np.int32))

        H, W = image.shape[:2]
        class_mask = None  # one raster for all classes; all zero again after each class
        for class_name, polys in class_to_polys.items():
            if not polys:
                continue
            if class_mask is None:
                class_mask = np.zeros((H, W), dtype=np.uint8)
            # rasterize all polygons of the class
            cv2.fillPoly(class_mask, polys, 255)

            color = self.parent.state_manager.class_manager.get_class_color(class_name)
            color_rgb = (int(color.red()), int(color.green()), int(color.blue()))

            for poly in polys:
                x, y, w, h = cv2.boundingRect(poly)
                x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
                if x1 <= x0 or y1 <= y0:
                    continue  # polygon entirely outside the image
                roi_mask = class_mask[y0:y1, x0:x1]
                if not cv2.countNonZero(roi_mask):
                    continue  # already blended through an overlapping box
                roi_out = output[y0:y1, x0:x1]
                roi_color = np.empty_like(roi_out); roi_color[:] = color_rgb
                blended = cv2.addWeighted(roi_out, 1.0 - alpha, roi_color, alpha, 0.0)
                cv2.copyTo(blended, roi_mask, roi_out)
                # Clear the box: pixels shared with an overlapping polygon of
                # this class must be blended once, as with one class-wide mask.
                roi_mask[:] = 0

            cv2.polylines(output, polys, isClosed=True, color=(0, 0, 0), thickness=outline_thickness)

        return output

    def highlight_selected_masks(self, mask_ids):
        """
        Outline the given masks in white (vector item; no base redraw).
        """
        t0 = time.perf_counter()
        path = QPainterPath()
        drawn = 0
        for mid in mask_ids:
            outline = self._mask_outline(mid)
            if outline is None:
                continue
            path.addPolygon(outline)
            path.closeSubpath()
            drawn += 1
        self.selection_item.setPath(path)

        logger.debug("highlight_overlay: %.2f ms total (%d polygons)",
                     (time.perf_counter() - t0) * 1000, drawn)

    def _mask_outline(self, mask_id):
        """QPolygonF of a mask on the current image (built once, then cached), or None."""
        key = str(mask_id)
        outline = self._mask_outlines.get(key)
        if outline is None:
            poly = self._mask_index.get(key)
            if poly is None or poly.shape[0] < 3:
                return None
            # +0.5 puts the outline through the pixel centres, where the masks'
            # own (baked-in) outlines are drawn.
            outline = QPolygonF([QPointF(x + 0.5, y + 0.5) for x, y in poly.tolist()])
            self._mask_outlines[key] = outline
        return outline

    def set_highlighted_masks(self, selected_rows):
        """
        Sync table selection -> overlay highlights (no base upload).
        """
        annotations = getattr(self.parent, "annotations", None)
        if annotations is None or not hasattr(annotations, "table"):
            return

        selected_ids = []
        for item in annotations.table.selectedItems():
            if item and item.column() == 1:
                selected_ids.append(item.text())

        self.highlighted_mask_ids = selected_ids
        self.highlight_selected_masks(self.highlighted_mask_ids)

    def set_masks_visible(self, visible: bool):
        """Persistent toggle — called by the View menu action."""
        self.masks_visible = visible
        self._apply_visibility()

    def set_peeking(self, peeking: bool):
        """Transient hold-to-hide — called by the H key press/release."""
        self._peeking = peeking
        self._apply_visibility()

    def _apply_visibility(self):
        """Switch pixmap without re-rendering masks."""
        if self.parent.state_manager.current_image is None:
            return
        if not self.masks_visible or self._peeking:
            self._set_base_pixmap(self.parent.state_manager.current_image)
        elif self.cached_image_with_masks is not None:
            self._set_base_pixmap(self.cached_image_with_masks)

    def fit_to_view(self):
        self.resetTransform()
        self._zoom_factor = 1.0
        self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def refresh_overlay(self, image_name, mask):
        """A tool saved a mask: redraw the masks on the image already loaded
        (re-reading a large scan from disk would only cost time)."""
        self.refresh_masks()

    def refresh_masks(self):
        """
        Re-render base overlay and keep highlights; overlay is redrawn after base upload.
        """
        if self.parent.state_manager.current_image is None:
            return

        self._rebuild_mask_index()
        self.cached_image_with_masks = self.overlay_base_masks(self.parent.state_manager.current_image)
        self._set_base_pixmap(self.cached_image_with_masks)

        self.highlight_selected_masks(self.highlighted_mask_ids)

    def delete_selected_masks(self):
        if not self.highlighted_mask_ids:
            logger.debug("Delete requested with no masks selected")
            return

        mask_ids_to_delete = [int(mask_id) for mask_id in self.highlighted_mask_ids]
        info = self.parent.state_manager.mask_manager.delete_masks(
            self.parent.state_manager.current_image_name,
            mask_ids_to_delete,
            profile=True
        )

        # Clear selection locally; let the coalesced UI update handle repaint/table/plot
        if hasattr(self.parent, 'annotations'):
            self.parent.annotations.table.clearSelection()
        self.highlighted_mask_ids = []
        self.refresh_masks()

        logger.info("Deleted %d selected mask(s)", info['rows'])

    # ---------- Edit modes (split stroke / brush — see edit_modes.py) ----------

    def start_edit_mode(self, mode):
        """Activate a MaskEditMode; any previous mode is cancelled first."""
        self.cancel_edit_mode()
        self.clear_selection()
        if mode.start():
            self._edit_mode = mode
            self.setCursor(mode.cursor)

    def cancel_edit_mode(self):
        if self._edit_mode is not None:
            mode, self._edit_mode = self._edit_mode, None
            mode.stop()
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def start_split_mode(self, mask_id):
        self.start_edit_mode(SplitStrokeMode(self, mask_id))

    def start_brush_mode(self, mask_id):
        self.start_edit_mode(BrushEditMode(self, mask_id))

    # ---------- Events ----------

    def wheelEvent(self, event):
        if self._edit_mode is not None and self._edit_mode.wheel(event):
            return

        zoom_step = 1.1
        if event.angleDelta().y() > 0:
            if self._can_zoom_in():
                self.scale(zoom_step, zoom_step)
                self._zoom_factor *= zoom_step
        else:
            if self._zoom_factor > self._zoom_min:
                inv = 1 / zoom_step
                self.scale(inv, inv)
                self._zoom_factor *= inv

    def _can_zoom_in(self):
        """Within _zoom_max of the fitted view, or while one image pixel is
        still smaller than _max_pixel_zoom screen pixels. The second limit is
        what matters on large scans: fitted, they are shown at a few percent,
        so a limit relative to the fitted view would stop short of 100%."""
        if self._zoom_factor < self._zoom_max:
            return True
        return self.transform().m11() < self._max_pixel_zoom

    @property
    def image_missing(self) -> bool:
        """True while the current image's file can't be opened."""
        return self._missing_item is not None

    def _show_missing_image(self, image_path):
        """Blank the view and say why, instead of leaving the previous image
        on screen (annotations drawn on it would land on this image)."""
        from PyQt6.QtGui import QPalette

        self.cancel_edit_mode()
        self._clear_missing_image()
        self.parent.state_manager.current_image = None
        self.cached_image_with_masks = None
        self.highlighted_mask_ids = []
        self.pixmap_item.setPixmap(QPixmap())
        self.highlight_item.setPixmap(QPixmap())
        self.highlight_item.setOffset(0, 0)
        self.selection_item.setPath(QPainterPath())
        self._mask_index, self._mask_outlines = {}, {}

        self._missing_item = self.scene.addText(missing_image_message(image_path))
        self._missing_item.setDefaultTextColor(self.palette().color(QPalette.ColorRole.WindowText))
        self._missing_item.setTextWidth(520)
        self._missing_item.setZValue(20)
        self.resetTransform()
        self._zoom_factor = 1.0
        self.scene.setSceneRect(self._missing_item.boundingRect())
        self.centerOn(self._missing_item)

    def _clear_missing_image(self):
        if self._missing_item is not None:
            self.scene.removeItem(self._missing_item)
            self._missing_item = None

    def keyPressEvent(self, event):
        if self.image_missing and event.key() in (Qt.Key.Key_S, Qt.Key.Key_E):
            return  # no image to draw on
        if self._edit_mode is not None and self._edit_mode.key_press(event):
            return

        if event.key() == Qt.Key.Key_S:
            if self.parent.tool_manager.current_tool:
                self.parent.tool_manager.current_tool.complete_mask()
                if hasattr(self.parent, 'annotations') and self.parent.annotations is not None:
                    self.parent.annotations.refresh_table(self.parent.state_manager.current_image_path)
                if hasattr(self.parent, 'statistics') and self.parent.statistics is not None:
                    self.parent.statistics.refresh_plot()

        if event.key() == Qt.Key.Key_E:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                tool = self.parent.tool_manager.current_tool
                if isinstance(tool, (SamMasker2, DEXTRMasker)):
                    tool.generate_mask(self.parent.state_manager.current_image)
            finally:
                QApplication.restoreOverrideCursor()

    def mousePressEvent(self, event):
        if self.image_missing and event.button() in (Qt.MouseButton.LeftButton,
                                                     Qt.MouseButton.RightButton):
            return  # no image to draw on or select masks in
        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.position().toPoint())
            point = (int(scene_pos.x()), int(scene_pos.y()))

            if self._edit_mode is not None and self._edit_mode.mouse_press(
                    point, Qt.MouseButton.LeftButton):
                return

            selected_mask_ids = self.get_clicked_masks(point)

            if selected_mask_ids:
                self._select_masks(selected_mask_ids)
            elif self.highlighted_mask_ids and self.parent.tool_manager.current_tool is None:
                # Click on empty area with no tool active: deselect everything
                self.clear_selection()

            tool = self.parent.tool_manager.current_tool
            if isinstance(tool, ManualMask):
                tool.add_point(point)
            elif isinstance(tool, DEXTRMasker):
                tool.add_point(point)
            elif isinstance(tool, SamMasker2):
                if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                    # Ctrl + left drag draws the bounding-box prompt
                    tool.box_start = point
                    tool.is_drawing_box = True
                else:
                    tool.add_point(point, 1)
            elif isinstance(tool, IntelligentScissors):
                tool.add_seed_point(point)

        elif event.button() == Qt.MouseButton.RightButton:
            if self._edit_mode is not None:
                scene_pos = self.mapToScene(event.position().toPoint())
                point = (int(scene_pos.x()), int(scene_pos.y()))
                if self._edit_mode.mouse_press(point, Qt.MouseButton.RightButton):
                    return

            tool = self.parent.tool_manager.current_tool
            if isinstance(tool, ManualMask):
                tool.pop_last_point()
            elif isinstance(tool, SamMasker2):
                scene_pos = self.mapToScene(event.position().toPoint())
                point = (int(scene_pos.x()), int(scene_pos.y()))
                tool.add_point(point, 0)
            elif tool is None:
                # No tool active: right-click offers bulk class assignment
                self._show_mask_context_menu(event)

        elif event.button() == Qt.MouseButton.MiddleButton:
            self._is_panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._is_panning:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
            event.accept()
        else:
            super().mouseMoveEvent(event)

        scene_pos = self.mapToScene(event.position().toPoint())
        point = (int(scene_pos.x()), int(scene_pos.y()))

        self.x, self.y = point

        if self._edit_mode is not None and self._edit_mode.mouse_move(point):
            return

        tool = self.parent.tool_manager.current_tool
        if isinstance(tool, IntelligentScissors) and getattr(tool, "seed_points", None):
            tool.update_dynamic_path((self.x, self.y))

        if isinstance(tool, SamMasker2) and getattr(tool, "is_drawing_box", False):
            tool.update_box_preview(tool.box_start, point)

    def mouseReleaseEvent(self, event):
        if self._edit_mode is not None and self._edit_mode.mouse_release(event.button()):
            return

        if event.button() == Qt.MouseButton.LeftButton:
            tool = self.parent.tool_manager.current_tool
            if isinstance(tool, SamMasker2) and getattr(tool, "is_drawing_box", False):
                scene_pos = self.mapToScene(event.position().toPoint())
                point = (int(scene_pos.x()), int(scene_pos.y()))
                tool.add_box(tool.box_start, point)
                tool.is_drawing_box = False

        if event.button() == Qt.MouseButton.MiddleButton:
            self._is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    # ---------- Helpers ----------

    def clear_selection(self):
        """Deselect all masks (highlight overlay + annotations table)."""
        self.highlighted_mask_ids = []
        self.highlight_selected_masks([])

        annotations = getattr(self.parent, "annotations", None)
        if annotations is not None:
            annotations.table.clearSelection()

    def _select_masks(self, selected_mask_ids, skip_already_selected=False):
        """
        Add mask ids to the highlight selection and sync the annotations table.

        Matching rows are TOGGLED, as selectRow() does in MultiSelection mode,
        so rows that are already selected flip off. Click-selection relies on
        that toggle to deselect; bulk selection (e.g. "Select all") must pass
        skip_already_selected=True to leave selected rows untouched. All rows
        change in one selection update, so the table and the outlines refresh
        once instead of once per row.
        """
        for mask_id in selected_mask_ids:
            if mask_id not in self.highlighted_mask_ids:
                self.highlighted_mask_ids.append(mask_id)
        self.highlight_selected_masks(self.highlighted_mask_ids)

        # Ensure results dialog is open
        if not hasattr(self.parent, 'annotations') or self.parent.annotations is None:
            self.parent.show_results()

        table = self.parent.annotations.table
        table.setSelectionMode(QTableWidget.SelectionMode.MultiSelection)
        model = table.model()
        sel_model = table.selectionModel()
        wanted = {str(mask_id) for mask_id in selected_mask_ids}
        last_column = max(model.columnCount() - 1, 0)

        toggle = QItemSelection()
        last_toggled_row = None
        selected_rows = []
        for row in range(table.rowCount()):
            table_mask_id = table.item(row, 1).text()
            if table_mask_id in wanted:
                already = sel_model.isSelected(model.index(row, 0))
                if not (skip_already_selected and already):
                    toggle.select(model.index(row, 0), model.index(row, last_column))
                    last_toggled_row = row
                selected_rows.append({
                    "image_name": table.item(row, 0).text(),
                    "mask_id": table_mask_id,
                    "class": table.item(row, 3).text(),
                })

        if last_toggled_row is not None:
            sel_model.select(toggle, QItemSelectionModel.SelectionFlag.Toggle
                             | QItemSelectionModel.SelectionFlag.Rows)
            # Like selectRow(): the last toggled row becomes current, which
            # scrolls the table to it.
            sel_model.setCurrentIndex(model.index(last_toggled_row, 0),
                                      QItemSelectionModel.SelectionFlag.NoUpdate)

        if selected_rows:
            self.parent.annotations.masks_selected.emit(selected_rows)

    def _show_mask_context_menu(self, event):
        """Right-click on the image: bulk class assignment for the selected masks."""
        scene_pos = self.mapToScene(event.position().toPoint())
        point = (int(scene_pos.x()), int(scene_pos.y()))

        # Right-clicking an unselected mask selects it first
        clicked = self.get_clicked_masks(point)
        if clicked and not any(mid in self.highlighted_mask_ids for mid in clicked):
            self._select_masks(clicked)

        image_name = self.parent.state_manager.current_image_name
        rows = [(image_name, int(mid)) for mid in self.highlighted_mask_ids]
        menu = MaskContextMenu(self.parent, rows, parent=self)
        menu.exec(event.globalPosition().toPoint())

    def _rebuild_mask_index(self):
        """Cache mask_id -> polygon(int32) for fast highlight path."""
        self._mask_index.clear()
        self._mask_outlines.clear()
        masks = self.parent.state_manager.mask_manager.load_masks(
            self.parent.state_manager.current_image_name
        )
        for mask_id, mask, *_ in masks:
            if mask.shape[0] >= 3:
                self._mask_index[str(mask_id)] = mask.astype(np.int32)

    def _set_base_pixmap(self, image_rgb):
        """Upload base image once."""
        h, w, ch = image_rgb.shape
        q_image = QImage(image_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.pixmap_item.setPixmap(QPixmap.fromImage(q_image))

    def _update_pixmap(self, image):
        """(kept for compatibility, but base is uploaded via _set_base_pixmap)"""
        self._set_base_pixmap(image)

    def get_clicked_masks(self, click_point):
        """Use cached polygons for hit-test; no DB calls."""
        clicked_masks = []
        for mask_id, poly in self._mask_index.items():
            if cv2.pointPolygonTest(poly, click_point, measureDist=False) >= 0:
                clicked_masks.append(str(mask_id))
        return clicked_masks
