import os

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QApplication, QVBoxLayout, QMessageBox, QDialog,
    QLineEdit, QAbstractSpinBox, QTextEdit, QPlainTextEdit, QComboBox, QDockWidget,
)
from PyQt6.QtCore import Qt, QEvent, pyqtSignal, QTimer
from PyQt6.QtGui import QIcon

from core.managers.tool_manager import ToolManager
from core.measure.scale import ProjectScale
from core.state import StateManager
from services.file_handlers import get_resource_path
from services.logger import get_logger, log_memory_usage
from ui.dialogs.instance_count import MaskStatisticsDock
from ui.dialogs.results_dialog import SaveResultsDialog, run_results_export
from ui.dialogs.scale_dialog import ScaleDialog
from ui.dialogs.table import MaskResultsDock
from ui.elements.image_display.image_display import ImageDisplay
from ui.elements.menubar import MenuBar
from ui.elements.sidebar import Sidebar
from ui.elements.slider import ImageSlider
from ui.flows import data_io
from ui.flows.runs import InferenceRun, TrainingRun

logger = get_logger(__name__)


class MainApp(QMainWindow):
    image_changed = pyqtSignal(str, object)

    def __init__(self, db_path="masks.db"):
        super().__init__()
        self.setWindowTitle("SegmentME")
        self.resize(1000, 600)
        self.db_path = db_path
        icon_path = get_resource_path("resources/icons/desktop.png")
        self.setWindowIcon(QIcon(icon_path))

        # Core state / managers
        self.state_manager = StateManager(db_path=db_path)
        self.tool_manager = ToolManager(self)
        self.inference = InferenceRun(self)     # Run Inference (ui/flows/runs.py)
        self.training = TrainingRun(self)       # Train Custom Model

        # UI widgets
        self.image_display = ImageDisplay(self)
        self.sidebar = Sidebar(self)
        self.menu_bar = MenuBar(self)
        self.setMenuBar(self.menu_bar)
        self.slider = ImageSlider(self)

        # Layout (sidebar | [slider above image])
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        image_layout = QVBoxLayout()
        image_layout.addWidget(self.slider)
        image_layout.addWidget(self.image_display)

        main_layout.addWidget(self.sidebar, stretch=1)
        main_layout.addLayout(image_layout, stretch=4)

        # Docks are created lazily
        self.annotations = None
        self.statistics = None

        # Connect image change -> slider sync
        self.state_manager.image_changed.connect(self.slider.update_slider)

        # Requests from the image view to the rest of the window
        self.image_display.select_in_table.connect(self._select_in_table)
        self.image_display.table_selection_cleared.connect(self._clear_table_selection)
        self.image_display.status_message.connect(self._show_status)

        # Throttled UI refresher for heavy widgets (table + stats)
        self._ui_update_timer = QTimer(self)
        self._ui_update_timer.setSingleShot(True)
        self._ui_update_timer.timeout.connect(self._refresh_results_and_stats)

        # Any time masks change, schedule a coalesced refresh
        self.state_manager.masks_updated.connect(self._schedule_results_refresh)
        self.state_manager.classes_changed.connect(self._schedule_results_refresh)

        # Global key handling
        QApplication.instance().installEventFilter(self)

    # ---------- Import / export (see ui/flows/data_io.py) ----------
    def import_images(self):
        data_io.import_images(self)

    def import_annotations(self):
        data_io.import_annotations(self)

    def _refresh_after_annotation_import(self):
        data_io.refresh_after_annotation_import(self)

    def _export_results(self):
        data_io.export_annotations_dataset(self)

    def go_to_image(self, index):
        """Show the image at `index` (slider, search box): read in the
        background; the table and statistics follow image_changed."""
        state = self.state_manager
        if not 0 <= index < len(state.image_paths):
            return
        state.image_index = index
        path = state.current_image_path
        logger.debug("Showing image %d/%d %s", index + 1, len(state.image_paths), os.path.basename(path))
        self.image_display.show_image(path)
        state.image_changed.emit(path)

    def next_image(self):
        path_ = self.state_manager.next_image()
        if path_:
            self.image_display.show_image(path_)   # read in the background
            logger.debug("Showing image %d/%d", self.state_manager.current_image_index + 1,
                         len(self.state_manager.image_paths))
            log_memory_usage()
        else:
            logger.debug("Already at the last image")

    def previous_image(self):
        path_ = self.state_manager.previous_image()
        if path_:
            self.image_display.show_image(path_)   # read in the background
            logger.debug("Showing image %d/%d", self.state_manager.current_image_index + 1,
                         len(self.state_manager.image_paths))
        else:
            logger.debug("Already at the first image")

    # ---------- Inference / training (see ui/flows/runs.py) ----------
    def popup_inference_dialog(self, display_text):
        self.inference.start(display_text)

    def popup_training_dialog(self):
        self.training.start()

    # ---------- Docks / Results ----------
    def show_results(self):
        # Create docks lazily
        if self.annotations is None:
            self.annotations = MaskResultsDock(self)
            self.annotations.masks_selected.connect(self.image_display.set_highlighted_masks)
            # (The table refreshes itself on image/mask changes, merged by its timer.)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.annotations)

        if self.statistics is None:
            self.statistics = MaskStatisticsDock(self)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.statistics)
            # One delayed refresh per burst of changes (masks_updated is already
            # routed through _schedule_results_refresh); was also connected
            # directly, which recomputed the plots twice per change.
            self.state_manager.image_changed.connect(lambda _path: self._schedule_results_refresh())

        # Show docks first so the initial refresh's isVisible() checks pass
        self.annotations.show()
        self.statistics.show()
        self._refresh_results_and_stats()

    def _select_in_table(self, mask_ids, skip_already_selected):
        """Masks clicked on the image: select their rows (opening the table)."""
        if self.annotations is None:
            self.show_results()
        self.annotations.select_mask_ids(mask_ids, skip_already_selected)

    def _clear_table_selection(self):
        if self.annotations is not None:
            self.annotations.table.clearSelection()

    def _show_status(self, text, ms):
        if text:
            self.statusBar().showMessage(text, ms)
        else:
            self.statusBar().clearMessage()

    def _schedule_results_refresh(self, delay_ms: int = 50):
        """Coalesce many mask updates into a single UI refresh."""
        # If a refresh is already pending, restarting the timer continues coalescing
        self._ui_update_timer.start(delay_ms)

    def _refresh_results_and_stats(self):
        """Refresh heavy widgets only if they exist and are visible."""
        if self.annotations and self.annotations.isVisible():
            self.annotations.refresh_table()
        if self.statistics and self.statistics.isVisible():
            self.statistics.refresh_plot()

    # ---------- Events ----------
    # Widgets that need Left/Right/Delete/H themselves (moving the text
    # cursor, deleting a character, typing "h").
    _TEXT_INPUTS = (QLineEdit, QAbstractSpinBox, QTextEdit, QPlainTextEdit)

    def _shortcut_target(self, source) -> bool:
        """True if a key press sent to `source` should act as an app shortcut:
        it is in this window (or one of its docks, even floating) and not in
        a text or number box. Dialogs and popups keep their keys."""
        widget = source if isinstance(source, QWidget) else QApplication.focusWidget()
        if widget is None:
            return False
        window = widget.window()
        if window is not self and not (isinstance(window, QDockWidget) and window.parent() is self):
            return False
        if isinstance(widget, self._TEXT_INPUTS):
            return False
        if isinstance(widget, QComboBox) and widget.isEditable():
            return False
        return True

    def eventFilter(self, source, event):
        if event.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease) and not self._shortcut_target(source):
            if event.type() == QEvent.Type.KeyRelease and event.key() == Qt.Key.Key_H:
                self.image_display.set_peeking(False)   # never leave masks hidden
            return super().eventFilter(source, event)
        if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():
            if event.key() == Qt.Key.Key_Delete:
                self.image_display.delete_selected_masks()
                self._schedule_results_refresh()
                return True
            elif event.key() == Qt.Key.Key_Right:
                self.next_image()
                return True
            elif event.key() == Qt.Key.Key_Left:
                self.previous_image()
                return True
            elif event.key() == Qt.Key.Key_H:
                self.image_display.set_peeking(True)
                return True
        elif event.type() == QEvent.Type.KeyRelease and not event.isAutoRepeat():
            if event.key() == Qt.Key.Key_H:
                self.image_display.set_peeking(False)
                return True
        return super().eventFilter(source, event)

    def closeEvent(self, event):
        self.inference.stop()
        event.accept()

    # ---------- Scale (real-world units) ----------
    @property
    def project_scale(self) -> ProjectScale:
        """The project's scale settings (core/measure/scale.py), loaded once."""
        if getattr(self, "_project_scale", None) is None:
            root = getattr(self, "project_root", None) or self.state_manager.project_root
            self._project_scale = ProjectScale.load(root) if root else ProjectScale()
        return self._project_scale

    def set_scale(self):
        """Actions > Set Scale…

        The window is opened without exec(): "Draw on the image…" hides it
        while the user drags the ruler, and hiding a dialog inside exec() ends
        exec() as if Cancel was pressed. The scale is applied on OK instead.
        """
        dialog = getattr(self, "_scale_dialog", None)
        if dialog is not None:
            dialog.show()
            dialog.raise_()
            return dialog
        dialog = ScaleDialog(self, self.project_scale)
        dialog.accepted.connect(lambda: self._apply_scale(dialog))
        dialog.finished.connect(lambda _result: setattr(self, "_scale_dialog", None))
        self._scale_dialog = dialog
        dialog.open()   # modal to this window, but does not block
        return dialog

    def _apply_scale(self, dialog):
        dialog.scale(self.project_scale)
        self._save_scale()
        logger.info("Scale set: %s", self.project_scale.to_dict())
        self._scale_changed()

    def set_scale_bar_visible(self, visible):
        logger.debug("Scale bar %s", "shown" if visible else "hidden")
        if self.project_scale.show_bar != bool(visible):
            self.project_scale.show_bar = bool(visible)
            self._save_scale()
        self.image_display.update_scale()

    def _save_scale(self):
        root = getattr(self, "project_root", None) or self.state_manager.project_root
        try:
            self.project_scale.save(root)
        except OSError as exc:
            QMessageBox.warning(self, "Set Scale", f"Could not save the scale in the project file:\n{exc}")

    def _scale_changed(self):
        """Show the new scale everywhere: scale bar, table, statistics."""
        action = self.menu_bar.scale_bar_action
        action.blockSignals(True)
        action.setChecked(self.project_scale.show_bar)
        action.blockSignals(False)
        self.image_display.update_scale()
        self.state_manager.masks_updated.emit()

    def current_image_scale(self):
        """(micrometres per pixel or None, unit name, micrometres per unit)
        for the image on screen."""
        scale = self.project_scale
        um, _ = scale.for_image(self.state_manager.current_image_path)
        return um, scale.unit, scale.unit_um

    def save_results_csv(self):
        """File > Save Results…: measurements of every mask as CSV."""
        if not self.state_manager.mask_manager.count():
            QMessageBox.information(self, "Save Results", "There are no masks to save yet.")
            return
        classes = self.state_manager.class_manager.get_all_class_names()
        dialog = SaveResultsDialog(self, self.project_scale, classes,
                                   self.state_manager.current_image_name)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        options = dialog.options()
        logger.info("Save Results: %s", {k: v for k, v in options.items() if k != "image_names"}
                    | {"images": options["image_names"] or "all"})
        run_results_export(self, options)
