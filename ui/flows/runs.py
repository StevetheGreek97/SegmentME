"""
Run Inference and Train Custom Model. Each is a small object owned by the
main window that keeps what belongs to its run (the inference process
manager; the training thread, worker and monitor window) instead of
storing it on the window.
"""
from pathlib import Path

from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import QMessageBox

from core.inference_manager import InferenceManager
from core.trainer.worker import TrainingWorker
from services import model_store
from services.logger import get_logger
from ui.dialogs.inference_dialog import InferenceDialog
from ui.dialogs.training_dialog import TrainingDialog
from ui.dialogs.training_monitor import TrainingMonitor

logger = get_logger(__name__)


class InferenceRun:
    def __init__(self, window):
        self.window = window
        self.manager = None             # InferenceManager of the current / last run

    def start(self, display_text):
        """Ask for the model and settings, then run on all project images
        (in a separate process; masks arrive as each image is done)."""
        window = self.window
        state = window.state_manager
        if not state.image_paths:
            logger.warning("Inference requested with no images loaded")
            return
        if self.running:
            from ui.dialogs.confirm import ask_to_stop
            if not ask_to_stop(window, "Inference Running",
                               f"An inference is already running ({self.manager.progress_text()}).\n\n"
                               "Stop it and start a new one?", stop_label="Stop and Start New"):
                return

        # YOLO weights on offer: the user's models/yolo folder plus best.pt
        # of every training run in this project.
        model_paths = model_store.trained_models(state.project_root) + model_store.installed_yolo_models()
        dialog = InferenceDialog(model_paths, window)
        if not dialog.exec():
            return
        mode = dialog.get_mode()
        threshold = dialog.get_threshold()
        img_dim = dialog.get_image_dimensions()
        sam_variant_key = None
        # HiReS = the selected YOLO model run tile by tile (see the worker).
        hires_params = dialog.get_hires_params() if mode == "hires" else None

        if mode == "sam":
            sam_variant_key = dialog.get_sam_variant_key()
            # A user-browsed custom checkpoint travels as model_path,
            # just like a custom YOLO model.
            model_path = dialog.get_sam_custom_path()
            if sam_variant_key is None and model_path is None:
                QMessageBox.warning(
                    window, "Run Inference",
                    "No SAM model selected. Download one in Settings -> Models "
                    "or browse for a custom SAM2 checkpoint (.pt)."
                )
                return
        else:
            model_path = dialog.get_selected_model()
            if not model_path:
                QMessageBox.warning(
                    window, "Run Inference",
                    "No YOLO model selected. Train one (Actions -> Train Custom Model), "
                    "put .pt files in the models/yolo folder (Settings -> Models -> Open folder), "
                    "or browse for a .pt file."
                )
                return

        self.stop()                                   # one run at a time
        self.manager = InferenceManager(window, mode, display_text)
        self.manager.run_inference(
            model_path,
            state.image_paths,
            threshold,
            img_dim,
            sam_variant_key=sam_variant_key,
            hires_params=hires_params,
        )

    @property
    def running(self) -> bool:
        return self.manager is not None and self.manager.running

    def stop(self):
        if self.manager:
            self.manager.stop_inference()


class TrainingRun:
    def __init__(self, window):
        self.window = window
        self.thread = None
        self.worker = None
        self.monitor = None

    def start(self):
        """Ask for the training settings, then train in a helper process
        (TrainingWorker on a thread) while the monitor window shows progress."""
        dialog = TrainingDialog(self.window)
        if not dialog.exec():
            return
        settings = dialog.get_settings()

        self.thread = QThread(self.window)
        self.worker = TrainingWorker(settings)
        self.worker.moveToThread(self.thread)

        self.monitor = TrainingMonitor()
        self.monitor.results_csv = str(Path(settings.output_dir) / "results.csv")
        self.monitor.set_stop_callback(self.worker.stop)
        self.monitor.set_thread(self.thread)

        self.worker.log_signal.connect(self.monitor.append_log)
        self.worker.finished_signal.connect(self._finished)
        # The thread only runs the worker: end it as soon as the worker is
        # done. (It used to stay running after "Close"; a QThread still
        # running when the app exits aborts it.)
        self.worker.finished_signal.connect(self.thread.quit)
        self.worker.finished_signal.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self._cleanup)
        self.thread.started.connect(self.worker.run)

        self.thread.start()
        self.monitor.exec()

    @property
    def running(self) -> bool:
        try:
            return self.thread is not None and self.thread.isRunning()
        except RuntimeError:              # the thread object is already deleted
            return False

    def stop(self):
        """The app is closing: stop training and wait for it."""
        if self.running and self.worker is not None:
            try:
                self.worker.stop()
            except RuntimeError:
                pass
            self.thread.quit()
            self.thread.wait(15000)

    def _finished(self):
        monitor = self.monitor
        if monitor is None:
            return
        monitor.append_log("✅ Training finished.")
        monitor.stop_btn.setText("✅ Close")
        monitor.stop_btn.clicked.disconnect()
        monitor.stop_btn.clicked.connect(monitor.accept)
        monitor.stop_btn.setEnabled(True)

    def _cleanup(self):
        if self.monitor is not None:
            if self.monitor.isVisible():
                self.monitor.hide()
            self.monitor.timer.stop()
            self.monitor.deleteLater()
        self.worker = None
        self.thread = None
        self.monitor = None
