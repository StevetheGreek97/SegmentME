from PyQt6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
)
from PyQt6.QtCore import Qt, QPointF, QTimer, pyqtSignal
from PyQt6.QtGui import QPixmap, QPainter, QPolygonF
from ui.custom_components.mask_context_menu import MaskContextMenu
from ui.elements.image_display.edit_modes import SplitStrokeMode, BrushEditMode
from services.logger import get_logger
from ui import diagnostics
from services.image_files import missing_image_message
from ui.elements.image_display.scale_bar import ScaleBar
from ui.elements.image_display.selection_outlines import SelectionOutlines
from ui.elements.image_display.image_loader import (BackgroundImageReader, LoadingLabel, Prefetcher,
                                                    prepare_image, read_rgb)
from ui.elements.image_display.array_image_item import ArrayImageItem
from ui.elements.image_display.tool_host import ViewToolHost
from ui.elements.image_display.mask_renderer import MaskRenderer
from ui.elements.image_display.segment_runner import SegmentRunner
import os
import cv2
import numpy as np
import time

logger = get_logger(__name__)



class ImageDisplay(QGraphicsView):
    """
    QGraphicsView that shows the base image+filled masks as one pixmap. Selection
    outlines are a separate vector item on top, so changing the selection never
    touches image-sized buffers; edit modes draw their previews into
    `highlight_item` (see edit_modes.py).
    """

    # Requests to the rest of the window (connected in MainApp)
    select_in_table = pyqtSignal(list, bool)    # mask ids, skip rows already selected
    table_selection_cleared = pyqtSignal()
    status_message = pyqtSignal(str, int)       # text ("" clears), ms to show (0 = until replaced)

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
        self.pixmap_item = ArrayImageItem()   # draws the numpy array itself: no QPixmap copy
        self.pixmap_item.setZValue(0)
        self.scene.addItem(self.pixmap_item)

        # Edit-mode preview layer (brush edit draws into it; see edit_modes.py)
        self.highlight_item = QGraphicsPixmapItem()
        self.highlight_item.setZValue(10)
        self.scene.addItem(self.highlight_item)

        # Selection outlines: a vector item drawing only the outlines in view,
        # so its cost follows the visible selected masks, not the image size.
        self.selection_item = SelectionOutlines()   # see selection_outlines.py
        self.selection_item.setZValue(11)
        self.scene.addItem(self.selection_item)

        # View settings
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setRenderHints(QPainter.RenderHint.SmoothPixmapTransform | QPainter.RenderHint.Antialiasing)
        self.setMouseTracking(True)

        # Cached data
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
        self.shown_path = None              # path of the image on screen (None: none / missing)

        # Scale bar (bottom-left), shown when the image has a scale
        self.scale_bar = ScaleBar(self)
        self._bar_um_per_px = None

        # Next / Previous / slider read the file on a worker thread, then the
        # neighbours are read ahead (see image_loader.py)
        self._image_loader = BackgroundImageReader(self, self._prepare_image, self._image_loaded)
        self._pending_render = None          # (db path, class colours, masks shown) for the worker
        self._prefetch = Prefetcher()
        self.renderer = MaskRenderer(self)   # draws the masks into the picture shown
        self._segmenter = SegmentRunner(self)  # E with SAM / DEXTR, on a worker thread
        self._loading_label = LoadingLabel(self)

        # What the drawing tools may use of the app (core/tools/base.py)
        self.tool_host = ViewToolHost(self)

        # Masks or classes changed anywhere (tools, table, inference, import):
        # redraw once per burst of changes, unless already redrawn since
        self._masks_version = 0         # bumped by every change
        self._drawn_version = 0         # the version the picture shows
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.timeout.connect(self._redraw_if_changed)
        state = parent.state_manager
        state.masks_updated.connect(self._masks_changed)
        state.classes_changed.connect(self._masks_changed)

    # ---------- Public API ----------

    # ---------- Loading on a worker thread (Next / Previous / slider / search) ----------
    def show_image(self, image_path):
        """Show `image_path` without freezing the window: the file is read on
        a worker thread (a scan on a cloud drive can take seconds to arrive)
        while the current image stays up with "Loading…" on it. Drawing is
        off meanwhile. Moving on again before it arrives: only the image the
        user ends on is shown. Afterwards the neighbours are read ahead."""
        if not image_path:
            return
        loader = self._image_loader
        loader.want = image_path
        # what belongs to the image being left
        self._tell_tool_image_changed(None)
        self.cancel_edit_mode()
        self.highlighted_mask_ids = []
        self.selection_item.set_polygons([])
        self._prefetch.cancel()
        self._loading_label.show_for(os.path.basename(image_path))
        if loader.future is None:
            self._start_load(image_path)

    def _start_load(self, image_path):
        # What the worker needs to draw the masks, taken here on the GUI thread
        self._pending_render = (self.parent.state_manager.db.db_path, self.renderer.class_colors(),
                                self.renderer.masks_shown())
        self._image_loader.start(image_path)

    def _prepare_image(self, image_path):
        """Worker thread: the file read and its masks drawn (image_loader.prepare)."""
        return prepare_image(image_path, *self._pending_render)

    def loading(self) -> bool:
        """True while an image is being read for show_image()."""
        return self._image_loader.want is not None

    def busy(self) -> bool:
        """True while anything runs in the background for the view."""
        return self.loading() or self.segmenting()

    def _image_loaded(self, path, image, seconds):
        """Worker read finished (on the GUI thread). Show it, unless the user
        moved on: then read the image they are on now."""
        loader = self._image_loader
        state = self.parent.state_manager
        if loader.want is None:                      # shown another way meanwhile
            return
        if path != loader.want:
            self._start_load(loader.want)
            return
        loader.want = None
        self._loading_label.hide()
        if path != state.current_image_path:         # e.g. the project changed
            logger.debug("Dropped %s: no longer the current image", os.path.basename(path))
            return
        if image is None:
            logger.error("Failed to load image %s (missing or unreadable)", path)
            self._show_missing_image(path)
            return
        logger.debug("Background load of %s: read %.0f ms, masks drawn %.0f ms",
                     os.path.basename(path), image["read_s"] * 1000, image["draw_s"] * 1000)
        self.display_image(path, image=image["image"], read_seconds=seconds, prepared=image)
        self._prefetch.around(state.image_paths, state.image_index)

    @staticmethod
    def _read_rgb(image_path):
        return read_rgb(image_path)

    def display_image(self, image_path, preserve_zoom=False, image=None, read_seconds=None,
                      prepared=None):
        """
        Load and display the image; render filled masks once to base pixmap.
        Prepare a transparent overlay for fast highlights.
        image: the file already read by show_image() (RGB), or None to read it here.
        prepared: show_image()'s worker result (masks already drawn), if any.
        """
        if not image_path:
            logger.warning("display_image called without an image path")
            return
        self._image_loader.want = None               # an older background load is dropped
        self._loading_label.hide()

        t_start = time.perf_counter()
        if image is None:
            image = self._read_rgb(image_path)
            t_read = time.perf_counter()
        else:
            t_read = t_start
            t_start -= read_seconds or 0.0           # count the background read in the log
        if image is None:
            logger.error("Failed to load image %s (missing or unreadable)", image_path)
            self._show_missing_image(image_path)
            return
        self._clear_missing_image()

        self.parent.state_manager.current_image = image
        self.shown_path = image_path
        t_convert = time.perf_counter()

        if not preserve_zoom:
            # A selection belongs to the image it was made on (the table's
            # selection is reset for the new image too).
            self.highlighted_mask_ids = []

        self._drawn_version = self._masks_version      # drawn from the database now
        masks, picture = None, None
        if prepared is not None and self.renderer.prepared_still_valid(prepared):
            masks, picture = prepared["masks"], prepared["picture"]

        # Build polygon index for this image so highlights don't call load_masks every time
        self._rebuild_mask_index(masks)

        # Render base once (already done on the worker thread for show_image)
        t0 = time.perf_counter()
        self.renderer.render_full(image, masks, picture=picture)
        t1 = time.perf_counter()

        # Clear any edit-mode preview; redraw the selection outlines
        self.highlight_item.setPixmap(QPixmap())
        self.highlight_item.setOffset(0, 0)
        self.highlight_selected_masks(self.highlighted_mask_ids)
        t2 = time.perf_counter()

        # Fit view
        if not preserve_zoom:
            self._tell_tool_image_changed(image)
            self.cancel_edit_mode()  # navigating away aborts a pending split/brush edit
            self.resetTransform()
            self._zoom_factor = 1.0
            self.scene.setSceneRect(self.pixmap_item.boundingRect())
            self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

        self.update_scale()

        total = time.perf_counter() - t_start
        # INFO when slow, so the reason shows up in the log without debug detail
        log = logger.info if total > 1.0 else logger.debug
        log("display_image %s: %.0f ms total = read file %.0f%s + convert %.0f + masks %.0f "
            "(%d) + selection %.0f + fit/scale %.0f ms; %dx%d px",
            os.path.basename(image_path), total * 1000, (t_read - t_start) * 1000,
            " (in the background)" if read_seconds is not None else "",
            (t_convert - t_read) * 1000, (t1 - t0) * 1000, len(self._mask_index),
            (t2 - t1) * 1000, (time.perf_counter() - t2) * 1000, image.shape[1], image.shape[0])

    def _current_tool(self):
        return self.parent.tool_manager.current_tool

    def _tell_tool_image_changed(self, image):
        """Unfinished points etc. of the active tool belong to the image left."""
        tool = self._current_tool()
        if tool is not None:
            tool.image_changed(image)

    # The picture with masks and how it was last redrawn (kept on the
    # renderer; these names are used by the edit modes and the tests)
    @property
    def cached_image_with_masks(self):
        return self.renderer.picture

    @cached_image_with_masks.setter
    def cached_image_with_masks(self, picture):
        self.renderer.picture = picture

    @property
    def last_redraw(self):
        return self.renderer.last_redraw

    def overlay_base_masks(self, image, alpha=0.4, outline_thickness=1, exclude_ids=None, masks=None):
        """Copy of `image` with the masks drawn in (see MaskRenderer.overlay)."""
        return self.renderer.overlay(image, alpha, outline_thickness, exclude_ids, masks)

    def highlight_selected_masks(self, mask_ids):
        """
        Outline the given masks in white (vector item; no base redraw).
        """
        t0 = time.perf_counter()
        outlines = [o for o in (self._mask_outline(mid) for mid in mask_ids) if o is not None]
        self.selection_item.set_polygons(outlines)
        drawn = len(outlines)

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

        # The table sends its selection along; asking it again cost as much
        # once more (0.4 s for 5,000 selected rows).
        if isinstance(selected_rows, list):
            selected_ids = [str(row["mask_id"]) for row in selected_rows]
        else:
            selected_ids = [item.text() for item in annotations.table.selectedItems()
                            if item and item.column() == 1]

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
        self._refresh_scale_bar()

    # ---------- Scale bar ----------
    def update_scale(self):
        """Look up the current image's scale (Actions > Set Scale…) for the bar."""
        um = None
        main = self.parent
        if (not self.image_missing and getattr(main, "project_scale", None) is not None
                and main.project_scale.show_bar):
            try:
                um, _, _ = main.current_image_scale()
            except Exception:
                logger.exception("Could not read the image's scale")
        self._bar_um_per_px = um
        self._refresh_scale_bar()

    def _refresh_scale_bar(self):
        shown = self._bar_um_per_px if not self.pixmap_item.isNull() else None
        self.scale_bar.set_state(shown, self.transform().m11())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_scale_bar()
        self._loading_label.place()

    def scrollContentsBy(self, dx, dy):
        # Qt scrolls the viewport's child widgets with the image: keep the
        # scale bar in its corner (panning, scroll bars, zoom around the mouse).
        super().scrollContentsBy(dx, dy)
        if self.scale_bar.isVisibleTo(self):
            self.scale_bar.place()
        if self._loading_label.isVisibleTo(self):
            self._loading_label.place()

    def redraw_current(self):
        """Classes or class colours changed: redraw the masks on the image
        already in memory, keeping the zoom. (Reading the scan again cost
        seconds on a cloud drive.) An image still loading gets drawn with the
        new classes when it arrives."""
        state = self.parent.state_manager
        if self.loading():
            return
        if state.current_image is None:               # the file was missing: try again
            if state.current_image_path:
                self.display_image(state.current_image_path, preserve_zoom=True)
            return
        self.refresh_masks()

    def _masks_changed(self):
        self._masks_version += 1
        self._redraw_timer.start(0)

    def _redraw_if_changed(self):
        if self._drawn_version != self._masks_version:
            self.redraw_current()

    def refresh_masks(self):
        """
        Re-render base overlay and keep highlights; overlay is redrawn after base upload.
        """
        if self.parent.state_manager.current_image is None:
            return
        self._drawn_version = self._masks_version

        self._rebuild_mask_index()
        image = self.parent.state_manager.current_image
        masks = self.parent.state_manager.mask_manager.load_masks(
            self.parent.state_manager.current_image_name)
        t0 = time.perf_counter()
        how = self.renderer.redraw(image, masks)
        logger.debug("refresh_masks: %s redraw in %.1f ms", how, (time.perf_counter() - t0) * 1000)

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

        # Clear the selection here and in the table; the table and statistics
        # refresh themselves (masks_updated)
        self.table_selection_cleared.emit()
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

    # Mouse wheel, as in Inkscape: Ctrl = move up / down, Shift = left /
    # right; a plain wheel zooms (Inkscape scrolls there; zoom is this app's
    # habit). A sideways touchpad swipe moves left / right.
    PAN_PX_PER_NOTCH = 60

    def wheelEvent(self, event):
        if self._edit_mode is not None and self._edit_mode.wheel(event):
            return

        mods = event.modifiers()
        ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        delta = event.angleDelta()
        if ctrl != shift or (delta.y() == 0 and delta.x() != 0):
            self._pan_by_wheel(event, horizontal=shift or (not ctrl and delta.y() == 0))
            return
        self._zoom_by_wheel(event)

    def _pan_by_wheel(self, event, horizontal):
        """Scroll the image by the wheel. Some systems deliver Shift+wheel
        as a sideways scroll already, so either direction of the event counts."""
        pixels = event.pixelDelta()             # touchpads: exact pixels
        amount = (pixels.x() or pixels.y()) if horizontal else (pixels.y() or pixels.x())
        if not amount:
            delta = event.angleDelta()          # mouse wheels: 120 per notch
            notches = ((delta.x() or delta.y()) if horizontal else (delta.y() or delta.x())) / 120
            amount = round(notches * self.PAN_PX_PER_NOTCH)
        bar = self.horizontalScrollBar() if horizontal else self.verticalScrollBar()
        bar.setValue(bar.value() - amount)

    def _zoom_by_wheel(self, event):
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
        self._refresh_scale_bar()

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
        self.shown_path = None
        self.cached_image_with_masks = None
        self.highlighted_mask_ids = []
        self.pixmap_item.set_array(None)
        self.highlight_item.setPixmap(QPixmap())
        self.highlight_item.setOffset(0, 0)
        self.selection_item.set_polygons([])
        self._mask_index, self._mask_outlines = {}, {}

        self._missing_item = self.scene.addText(missing_image_message(image_path))
        self._missing_item.setDefaultTextColor(self.palette().color(QPalette.ColorRole.WindowText))
        self._missing_item.setTextWidth(520)
        self._missing_item.setZValue(20)
        self.resetTransform()
        self._zoom_factor = 1.0
        self.scene.setSceneRect(self._missing_item.boundingRect())
        self.centerOn(self._missing_item)
        self._bar_um_per_px = None
        self._refresh_scale_bar()

    def _clear_missing_image(self):
        if self._missing_item is not None:
            self.scene.removeItem(self._missing_item)
            self._missing_item = None

    def keyPressEvent(self, event):
        if (self.image_missing or self.loading()) and event.key() in (Qt.Key.Key_S, Qt.Key.Key_E):
            return  # no image to draw on (yet)
        if self._edit_mode is not None and self._edit_mode.key_press(event):
            return

        if event.key() == Qt.Key.Key_S:
            if self._current_tool() is not None:
                self._current_tool().complete_mask()   # table + statistics follow masks_updated

        if event.key() == Qt.Key.Key_E:
            tool = self._current_tool()
            if tool is not None and tool.segments_on_e:
                self.segment_in_background(tool)   # seconds on a CPU: keep the window alive

    # ---------- SAM / DEXTR on a worker thread (E): see segment_runner.py ----------
    def segment_in_background(self, tool):
        self._segmenter.run(tool)

    def segmenting(self) -> bool:
        """True while SAM or DEXTR runs in the background."""
        return self._segmenter.busy()

    def mousePressEvent(self, event):
        if (self.image_missing or self.loading()) and event.button() in (Qt.MouseButton.LeftButton,
                                                                         Qt.MouseButton.RightButton):
            return  # no image to draw on or select masks in (yet)
        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.position().toPoint())
            point = (int(scene_pos.x()), int(scene_pos.y()))

            if self._edit_mode is not None and self._edit_mode.mouse_press(
                    point, Qt.MouseButton.LeftButton):
                diagnostics.action("Image left-click at %s in %s", point,
                                   type(self._edit_mode).__name__, level=10)
                return

            selected_mask_ids = self.get_clicked_masks(point)
            tool = self._current_tool()
            diagnostics.action("Image left-click at %s on %s: tool %s, masks hit %s",
                               point, self.parent.state_manager.current_image_name,
                               type(tool).__name__ if tool else "none", selected_mask_ids,
                               level=10)

            if selected_mask_ids:
                self._select_masks(selected_mask_ids)
            elif self.highlighted_mask_ids and self._current_tool() is None:
                # Click on empty area with no tool active: deselect everything
                self.clear_selection()

            tool = self._current_tool()
            if tool is not None:
                ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
                tool.left_click(point, ctrl)

        elif event.button() == Qt.MouseButton.RightButton:
            tool = self._current_tool()
            diagnostics.action("Image right-click on %s: tool %s, edit mode %s",
                               self.parent.state_manager.current_image_name,
                               type(tool).__name__ if tool else "none",
                               type(self._edit_mode).__name__ if self._edit_mode else "none",
                               level=10)
            if self._edit_mode is not None:
                scene_pos = self.mapToScene(event.position().toPoint())
                point = (int(scene_pos.x()), int(scene_pos.y()))
                if self._edit_mode.mouse_press(point, Qt.MouseButton.RightButton):
                    return

            tool = self._current_tool()
            if tool is not None:
                scene_pos = self.mapToScene(event.position().toPoint())
                tool.right_click((int(scene_pos.x()), int(scene_pos.y())))
            else:
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

        tool = self._current_tool()
        if tool is not None:
            tool.mouse_move(point)

    def mouseReleaseEvent(self, event):
        if self._edit_mode is not None and self._edit_mode.mouse_release(event.button()):
            return

        if event.button() == Qt.MouseButton.LeftButton:
            tool = self._current_tool()
            if tool is not None:
                scene_pos = self.mapToScene(event.position().toPoint())
                tool.left_release((int(scene_pos.x()), int(scene_pos.y())))

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
        self.table_selection_cleared.emit()

    def _select_masks(self, selected_mask_ids, skip_already_selected=False):
        """
        Add mask ids to the highlight selection and select them in the
        annotations table (see MaskResultsDock.select_mask_ids).

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

        # The window opens the annotations table if needed and selects the
        # rows there; the table then reports its whole selection back
        # (set_highlighted_masks), which sets the outlines.
        self.select_in_table.emit([str(m) for m in selected_mask_ids], skip_already_selected)

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

    def _rebuild_mask_index(self, masks=None):
        """Cache mask_id -> polygon(int32) for fast highlight path."""
        self._mask_index.clear()
        self._mask_outlines.clear()
        if masks is None:
            masks = self.parent.state_manager.mask_manager.load_masks(
                self.parent.state_manager.current_image_name
            )
        for mask_id, mask, *_ in masks:
            if mask.shape[0] >= 3:
                self._mask_index[str(mask_id)] = mask.astype(np.int32)

    def _set_base_pixmap(self, image_rgb):
        """Show this RGB array (shared with the view, not copied)."""
        self.pixmap_item.set_array(image_rgb)

    def get_clicked_masks(self, click_point):
        """Use cached polygons for hit-test; no DB calls."""
        clicked_masks = []
        for mask_id, poly in self._mask_index.items():
            if cv2.pointPolygonTest(poly, click_point, measureDist=False) >= 0:
                clicked_masks.append(str(mask_id))
        return clicked_masks
