import json
import os
import random

import numpy as np
import psutil
from PyQt6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QMessageBox

from ui.dialogs.progress import ProgressDialogManager
from services.app_process import self_command
from services.logger import get_logger

logger = get_logger(__name__)


def _worker_command():
    """(program, arguments) to re-launch this app as an inference worker;
    works from source and from a frozen exe (see services.app_process)."""
    return self_command("--inference-worker")


def _worker_process_environment():
    """Cap the worker's CPU thread pools so it can't starve the GUI process.

    torch/numpy's BLAS backends default to one thread per core; on
    low-core machines that leaves the parent process (and its event loop,
    including the Cancel button) with no CPU time to run while a batch of
    images is being processed. Leave at least one core free."""
    threads = max(1, (os.cpu_count() or 2) - 1)
    env = QProcessEnvironment.systemEnvironment()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS"):
        env.insert(var, str(threads))
    return env


class InferenceManager(QObject):
    def __init__(self, parent, mode, display_text):
        super().__init__()
        self.parent = parent
        self.mode = mode
        self.display_text = display_text
        self.progress_manager = None
        self.process = None

        self._stdout_buffer = ""
        self._processed = 0
        self._total = 0

        # Coalesce UI updates coming from many images
        self._ui_refresh_timer = QTimer(self)
        self._ui_refresh_timer.setSingleShot(True)
        self._ui_refresh_timer.timeout.connect(self._do_ui_refresh)

        self._need_overlay_refresh = False
        self._need_stats_refresh = False
        self._need_class_dropdown_refresh = False

    def run_inference(self, model_path, image_files, conf, dims, sam_variant_key=None):
        if not image_files:
            logger.warning("Inference requested with no image files; ignoring")
            return

        if self.process is not None:
            self.stop_inference()

        request = {
            "mode": self.mode,
            "model_path": model_path,
            "image_paths": image_files,
            "conf": conf,
            "dims": list(dims),
            "sam_variant_key": sam_variant_key,
        }

        self._stdout_buffer = ""
        self._processed = 0
        self._total = len(image_files)

        program, arguments = _worker_command()
        self.process = QProcess(self)
        self.process.setProgram(program)
        self.process.setArguments(arguments)
        self.process.setProcessEnvironment(_worker_process_environment())
        self.process.readyReadStandardOutput.connect(self._on_stdout)
        self.process.finished.connect(self._on_finished)
        self.process.errorOccurred.connect(self._on_process_error)
        self.process.started.connect(self._on_process_started)

        self.progress_manager = ProgressDialogManager(
            self.parent,
            total_items=self._total,
            cancel_callback=self.stop_inference,
            display_text=self.display_text,
        )

        logger.info("Starting %s inference subprocess: model=%s, %d image(s), conf=%.2f, dims=%s, sam_variant=%s",
                    self.mode, model_path, self._total, conf, dims, sam_variant_key)

        self.process.start()
        # QProcess buffers writes until the process is actually running.
        self.process.write((json.dumps(request) + "\n").encode("utf-8"))
        self.process.closeWriteChannel()

    def _on_process_started(self):
        """Best-effort: lower the worker's OS scheduling priority so heavy
        CPU-bound inference (SAM's automatic mask generator especially)
        can never starve the GUI process of CPU time on low-core machines."""
        if self.process is None:
            return
        try:
            proc = psutil.Process(self.process.processId())
            if hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS"):
                proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            else:
                proc.nice(10)
        except (psutil.Error, OSError) as exc:
            logger.debug("Could not lower inference worker priority: %s", exc)

    def stop_inference(self):
        """Kill the worker outright. Unlike the old thread.wait() approach,
        this never blocks the GUI thread waiting on an in-flight
        model.predict() call -- a killed process can't ignore SIGKILL, so
        the bounded wait below is just long enough for the OS to reap it,
        not for the current image to finish."""
        if self.process is not None:
            process, self.process = self.process, None
            process.readyReadStandardOutput.disconnect(self._on_stdout)
            process.finished.disconnect(self._on_finished)
            process.errorOccurred.disconnect(self._on_process_error)
            if process.state() != QProcess.ProcessState.NotRunning:
                process.kill()
                process.waitForFinished(1000)
            process.deleteLater()
        if self.progress_manager:
            self.progress_manager.close()
            self.progress_manager = None

    def _on_process_error(self, error):
        logger.error("Inference worker process error: %s", error)

    def _on_stdout(self):
        if self.process is None:
            return
        chunk = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._stdout_buffer += chunk
        while "\n" in self._stdout_buffer:
            line, self._stdout_buffer = self._stdout_buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Inference worker emitted a non-JSON line: %r", line)
                continue
            self._handle_message(message)

    def _handle_message(self, message):
        if message.get("fatal"):
            logger.error("Inference worker failed: %s", message["fatal"])
            QMessageBox.critical(self.parent, "Inference Failed", message["fatal"])
            self.stop_inference()
            return

        if message.get("done"):
            return

        status = message.get("status")
        if status == "loading_model":
            if self.progress_manager:
                self.progress_manager.progress_dialog.setLabelText(
                    f"{self.display_text}\nLoading {message.get('variant', 'model')}..."
                )
            return
        if status == "processing":
            # Emitted when the worker *starts* an image -- automatic SAM
            # mask generation can take a minute or more per image on CPU,
            # so without this the dialog looks frozen between results.
            if self.progress_manager:
                name = os.path.basename(message.get("image", ""))
                self.progress_manager.progress_dialog.setLabelText(
                    f"{self.display_text}\nImage {self._processed + 1}/{self._total}: {name}"
                )
            return

        image_path = message.get("image")
        if image_path is None:
            return

        if "error" in message:
            logger.warning("Inference failed for %s: %s", image_path, message["error"])
        else:
            masks_xy = [np.array(m, dtype=np.float32) for m in message.get("masks", [])]
            class_names = message.get("classes", [])
            self._save_result(image_path, masks_xy, class_names)

        self._processed += 1
        if self.progress_manager:
            self.progress_manager.update_progress(self._processed)

    def _save_result(self, image_path, masks, class_names):
        """Called once per image from worker stdout. Save to DB, flag what
        needs UI refresh, then schedule one coalesced UI pass."""
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        for mask, class_name in zip(masks, class_names):
            if mask is None or getattr(mask, "size", 0) < 2:
                continue
            self.parent.state_manager.mask_manager.save_mask(mask, image_name, class_name)

            if class_name not in self.parent.state_manager.class_manager.get_all_class_names():
                rand = QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                self.parent.state_manager.class_manager.add_class(class_name, rand)
                self._need_class_dropdown_refresh = True

        self._need_overlay_refresh = True
        self._need_stats_refresh = True
        self._ui_refresh_timer.start(60)

    def _do_ui_refresh(self):
        if self._need_class_dropdown_refresh and hasattr(self.parent, "sidebar"):
            self.parent.sidebar.populate_class_dropdown()
        if self._need_overlay_refresh and hasattr(self.parent, "image_display"):
            self.parent.image_display.refresh_masks()
        if self._need_stats_refresh and hasattr(self.parent, "statistics") and self.parent.statistics.isVisible():
            self.parent.statistics.refresh_plot()

        if hasattr(self.parent, "annotations") and self.parent.annotations.isVisible():
            self.parent.annotations.refresh_table(delay_ms=80)

        self._need_class_dropdown_refresh = False
        self._need_overlay_refresh = False
        self._need_stats_refresh = False

    def _on_finished(self, exit_code, exit_status):
        if exit_code != 0:
            logger.warning("Inference worker exited with code %d (status=%s)", exit_code, exit_status)
        logger.info("Inference finished")
        if self.progress_manager:
            self.progress_manager.close()
            self.progress_manager = None
        if self.process is not None:
            self.process.deleteLater()
            self.process = None
