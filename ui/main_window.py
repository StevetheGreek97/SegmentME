from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QFileDialog, QHBoxLayout,QProgressDialog, QApplication, QVBoxLayout, QMessageBox
)
from PyQt6.QtCore import Qt, QEvent, pyqtSignal, QTimer, QThread
from PyQt6.QtGui import QIcon

import os
import shutil
from pathlib import Path

from ui.elements.image_display.image_display import ImageDisplay
from ui.elements.menubar import MenuBar
from ui.elements.sidebar import Sidebar
from ui.elements.slider import ImageSlider

from ui.dialogs.inference_dialog import InferenceDialog
from ui.dialogs.table import MaskResultsDock
from ui.dialogs.instance_count import MaskStatisticsDock
from ui.dialogs.export_dialog import ExportDialog
from ui.dialogs.training_dialog import TrainingDialog
from ui.dialogs.training_monitor import TrainingMonitor

from core.state import StateManager
from core.inference_manager import InferenceManager
from core.exporters.yolo_exporter import YOLOExporter
from core.exporters.coco_exporter import COCOExporter
from core.managers.tool_manager import ToolManager
from core.trainer.worker import TrainingWorker

from services.file_handlers import loader, get_resource_path
from services import model_store
from services.logger import get_logger, log_memory_usage

logger = get_logger(__name__)

from datetime import datetime
from core.save_res import SaveResultsDBWorker
from services.seproj import update_images as seproj_update_images

class MainApp(QMainWindow):
    image_changed = pyqtSignal(str, object)

    def __init__(self, db_path="masks.db"):
        super().__init__()
        self.setWindowTitle("SegmentME")
        self.resize(1000, 600)
        self.db_path = db_path
        icon_path = get_resource_path("resources/icons/desktop.png")
        self.setWindowIcon(QIcon(icon_path))

        self.current_model_path = None

        # Core state / managers
        self.state_manager = StateManager(db_path=db_path)
        self.tool_manager = ToolManager(self)
        self.inference_manager = None

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

        # Throttled UI refresher for heavy widgets (table + stats)
        self._ui_update_timer = QTimer(self)
        self._ui_update_timer.setSingleShot(True)
        self._ui_update_timer.timeout.connect(self._refresh_results_and_stats)

        # Any time masks change, schedule a coalesced refresh
        self.state_manager.masks_updated.connect(self._schedule_results_refresh)

        # Global key handling
        QApplication.instance().installEventFilter(self)

    # ---------- Images ----------
    def import_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Images to Import",
            filter="Images (*.png *.jpg *.jpeg *.bmp *.tif)"
        )
        if not files:
            return

        # NOTE: assumes self.config is set in the services.recent_projects.initialize_project 
        # I know it should be cleaner, but quick workaround for now.
        images_dir = self.config.get_images_dir()
        os.makedirs(images_dir, exist_ok=True)

        imported_count = 0
        for file in files:
            fname = os.path.basename(file)
            dest = os.path.join(images_dir, fname)
            if not os.path.exists(dest):
                shutil.copy(file, dest)
                imported_count += 1

        image_paths = loader(images_dir)
        self.state_manager.set_image_paths(image_paths)
        self.slider.set_image_count(len(image_paths))
        self.image_display.display_image(self.state_manager.current_image_path)

        # Keep .SEproj in sync with the folder
        project_root = getattr(self, "project_root", None)
        if project_root:
            seproj_update_images(project_root, [os.path.basename(p) for p in image_paths])

        QMessageBox.information(
            self, "Import Complete",
            f"✅ Imported {imported_count} images into your project."
        )

    def next_image(self):
        path_ = self.state_manager.next_image()
        if path_:
            if self.tool_manager.current_tool:
                self.tool_manager.current_tool.clear_temp_items()
            self.image_display.display_image(path_)
            logger.debug("Showing image %d/%d", self.state_manager.current_image_index + 1,
                         len(self.state_manager.image_paths))
            log_memory_usage()
        else:
            logger.debug("Already at the last image")

    def previous_image(self):
        path_ = self.state_manager.previous_image()
        if path_:
            if self.tool_manager.current_tool:
                self.tool_manager.current_tool.clear_temp_items()
            self.image_display.display_image(path_)
            logger.debug("Showing image %d/%d", self.state_manager.current_image_index + 1,
                         len(self.state_manager.image_paths))
        else:
            logger.debug("Already at the first image")

    # ---------- Inference ----------
    def popup_inference_dialog(self, display_text):
        if not self.state_manager.image_paths:
            logger.warning("Inference requested with no images loaded")
            return

        # YOLO weights on offer: the user's models/yolo folder plus best.pt
        # of every training run in this project.
        project_root = getattr(self.state_manager, "project_root", None) or getattr(self, "project_root", None)
        model_paths = model_store.trained_models(project_root) + model_store.installed_yolo_models()
        dialog = InferenceDialog(model_paths, self)
        if dialog.exec():
            mode = dialog.get_mode()
            threshold = dialog.get_threshold()
            img_dim = dialog.get_image_dimensions()
            sam_variant_key = None

            if mode == "sam":
                sam_variant_key = dialog.get_sam_variant_key()
                # A user-browsed custom checkpoint travels as model_path,
                # just like a custom YOLO model.
                model_path = dialog.get_sam_custom_path()
                if sam_variant_key is None and model_path is None:
                    QMessageBox.warning(
                        self, "Run Inference",
                        "No SAM model selected. Download one in Settings -> Models "
                        "or browse for a custom SAM2 checkpoint (.pt)."
                    )
                    return
            else:
                model_path = dialog.get_selected_model()
                if not model_path:
                    QMessageBox.warning(
                        self, "Run Inference",
                        "No YOLO model selected. Train one (Actions -> Train Custom Model), "
                        "put .pt files in the models/yolo folder (Settings -> Models -> Open folder), "
                        "or browse for a .pt file."
                    )
                    return
                self.current_model_path = model_path

            # Stop any ongoing inference
            if self.inference_manager:
                self.inference_manager.stop_inference()

            # Start fresh inference
            self.inference_manager = InferenceManager(self, mode, display_text)
            # (Optional) light debounce on results refresh while inferring
            self._schedule_results_refresh(delay_ms=100)
            self.inference_manager.run_inference(
                model_path,
                self.state_manager.image_paths,
                threshold,
                img_dim,
                sam_variant_key=sam_variant_key,
            )

    # ---------- Docks / Results ----------
    def show_results(self):
        # Create docks lazily
        if self.annotations is None:
            self.annotations = MaskResultsDock(self)
            self.annotations.masks_selected.connect(self.image_display.set_highlighted_masks)
            self.state_manager.image_changed.connect(self.annotations.refresh_table)
            self.state_manager.masks_updated.connect(self.annotations.refresh_table)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.annotations)

        if self.statistics is None:
            self.statistics = MaskStatisticsDock(self)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.statistics)
            self.state_manager.image_changed.connect(self.statistics.refresh_plot)
            self.state_manager.masks_updated.connect(self.statistics.refresh_plot)

        # Show docks first so the initial refresh's isVisible() checks pass
        self.annotations.show()
        self.statistics.show()
        self._refresh_results_and_stats()

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

    # ---------- Export ----------
    def _export_results(self):
        dialog = ExportDialog(self)
        if dialog.exec():
            settings = dialog.get_settings()
            format_ = settings["format"]
            train = settings["train"]
            val = settings["val"]
            test = settings["test"]

            if format_ == "yolo":
                exporter = YOLOExporter(self)
                exporter.export_all_annotations(train, val, test)
            elif format_ == "coco":
                exporter = COCOExporter(self)
                exporter.export_all_annotations(train, val, test)
            else:
                QMessageBox.warning(self, "Unsupported Format", f"Export format '{format_}' not supported.")

    # ---------- Training ----------
    def popup_training_dialog(self):
        dialog = TrainingDialog(self)
        if dialog.exec():
            settings = dialog.get_settings()

            self.training_thread = QThread(self)
            self.training_worker = TrainingWorker(settings)
            self.training_worker.moveToThread(self.training_thread)

            results_csv_path = Path(settings.output_dir) / "results.csv"

            self.training_monitor = TrainingMonitor()
            self.training_monitor.results_csv = str(results_csv_path)
            self.training_monitor.set_stop_callback(self.training_worker.stop)
            self.training_monitor.set_thread(self.training_thread)

            self.training_worker.log_signal.connect(self.training_monitor.append_log)

            def on_training_finished():
                self.training_monitor.append_log("✅ Training finished.")
                self.training_monitor.stop_btn.setText("✅ Close")
                self.training_monitor.stop_btn.clicked.disconnect()
                self.training_monitor.stop_btn.clicked.connect(self.training_monitor.accept)
                self.training_monitor.stop_btn.setEnabled(True)

            self.training_worker.finished_signal.connect(on_training_finished)
            self.training_worker.finished_signal.connect(self.training_worker.deleteLater)
            self.training_thread.finished.connect(self.training_thread.deleteLater)

            def cleanup_monitor():
                if self.training_thread and self.training_thread.isRunning():
                    logger.debug("Stopping training thread before cleanup")
                    self.training_thread.quit()
                    self.training_thread.wait()
                if self.training_monitor.isVisible():
                    self.training_monitor.hide()
                self.training_monitor.deleteLater()
                self.training_worker = None
                self.training_thread = None
                self.training_monitor = None

            self.training_thread.finished.connect(cleanup_monitor)
            self.training_thread.started.connect(self.training_worker.run)

            self.training_thread.start()
            self.training_monitor.exec()

    # ---------- Events ----------
    def eventFilter(self, source, event):
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
        # Stop export thread if present
        if hasattr(self, 'export_thread') and self.export_thread.isRunning():
            logger.debug("Stopping export thread")
            self.export_thread.stop()
            self.export_thread.wait()
        # Stop inference cleanly
        if self.inference_manager:
            self.inference_manager.stop_inference()
        event.accept()

    def save_results_csv(self):
            # 1) Count rows (for progress maximum)
            db = self.state_manager.mask_manager.db
            try:
                total = db.fetch_one("SELECT COUNT(*) FROM masks")[0]
            except Exception as e:
                QMessageBox.critical(self, "Save Results", f"Could not count rows: {e}")
                return

            if total == 0:
                QMessageBox.information(self, "Save Results", "No masks found in the database.")
                return

            # 2) Build output path in project root
            project_root = self.state_manager.project_root
            os.makedirs(project_root, exist_ok=True)
            out_path = os.path.join(project_root, f"annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

            # 3) Setup progress dialog
            progress = QProgressDialog("Saving annotations from database...", "Cancel", 0, total, self)
            progress.setWindowTitle("Saving")
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setAutoClose(False)
            progress.setAutoReset(False)
            progress.setMinimumDuration(0)
            progress.show()

            # 4) Thread + Worker
            self._save_db_thread = QThread(self)
            self._save_db_worker = SaveResultsDBWorker(
                db_path=self.db_path,
                out_path=out_path,
                where_sql="",      # put a WHERE clause here if you want to filter
                params=(),
                chunk_size=2000,   # tweak chunk size if you like
            )
            self._save_db_worker.moveToThread(self._save_db_thread)

            self._save_db_thread.started.connect(self._save_db_worker.run)
            self._save_db_worker.progress.connect(progress.setValue)

            def _cleanup():
                worker = getattr(self, "_save_db_worker", None)
                thread = getattr(self, "_save_db_thread", None)
                if worker:
                    worker.deleteLater()
                if thread and thread.isRunning():
                    thread.quit()
                    thread.wait()
                if thread:
                    thread.deleteLater()
                self._save_db_worker = None
                self._save_db_thread = None

            def on_finished(path):
                progress.setValue(total)
                progress.close()
                QMessageBox.information(self, "Save Results", f"✅ Saved to:\n{path}")
                _cleanup()

            def on_error(msg):
                progress.close()
                QMessageBox.critical(self, "Save Results", f"❌ Error: {msg}")
                _cleanup()

            def on_canceled():
                progress.close()
                QMessageBox.information(self, "Save Results", "✋ Save canceled.")
                _cleanup()

            self._save_db_worker.finished.connect(on_finished)
            self._save_db_worker.error.connect(on_error)
            self._save_db_worker.canceled.connect(on_canceled)

            progress.canceled.connect(self._save_db_worker.cancel)
            self._save_db_thread.finished.connect(_cleanup)

            # 5) Go
            self._save_db_thread.start()