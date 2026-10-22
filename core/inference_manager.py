import codecs
import json
import os
import random
import re

import numpy as np
import psutil
from PyQt6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QMessageBox

from ui.dialogs.progress import ProgressDialogManager
from services.app_process import self_command
from services.logger import get_logger

logger = get_logger(__name__)

# Worker log lines end in \n; progress bars of third-party libraries redraw with \r.
_LINE_BREAK = re.compile(r"\r\n|\r|\n")

# After reporting a fatal error the worker exits on its own; if it has not
# after this long, it is stopped.
_FATAL_EXIT_GRACE_MS = 5000


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


def _remove_worker_temp_dirs(pid):
    """Delete tile folders a killed HiReS worker had no chance to clean up
    (named segmentme-hires-<pid>-*, see core.inference_worker)."""
    if not pid:
        return
    import glob
    import shutil
    import tempfile

    for path in glob.glob(os.path.join(tempfile.gettempdir(), f"segmentme-hires-{pid}-*")):
        shutil.rmtree(path, ignore_errors=True)


class InferenceManager(QObject):
    def __init__(self, parent, mode, display_text):
        super().__init__()
        self.parent = parent
        self.mode = mode
        self.display_text = display_text
        self.progress_manager = None
        self.process = None

        self._stdout_buffer = ""
        self._stderr_buffer = ""
        self._stderr_decoder = None
        self._processed = 0
        self._total = 0
        self._errors = []
        self._fatal = None
        self._worker_pid = None

        # Coalesce UI updates coming from many images


    def run_inference(self, model_path, image_files, conf, dims, sam_variant_key=None,
                      hires_params=None):
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
        if hires_params is not None:
            request["hires"] = hires_params

        self._stdout_buffer = ""
        self._stderr_buffer = ""
        # Incremental: a multi-byte UTF-8 character may be split across reads.
        self._stderr_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._processed = 0
        self._total = len(image_files)
        self._errors = []
        self._fatal = None

        program, arguments = _worker_command()
        self.process = QProcess(self)
        self.process.setProgram(program)
        self.process.setArguments(arguments)
        self.process.setProcessEnvironment(_worker_process_environment())
        self.process.readyReadStandardOutput.connect(self._on_stdout)
        # The worker's stderr is its human-readable log ("Show log").
        self.process.readyReadStandardError.connect(self._on_stderr)
        self.process.finished.connect(self._on_finished)
        self.process.errorOccurred.connect(self._on_process_error)
        self.process.started.connect(self._on_process_started)

        self.progress_manager = ProgressDialogManager(
            self.parent,
            total_items=self._total,
            cancel_callback=self.stop_inference,
            display_text=self.display_text,
        )

        logger.info("Starting %s inference subprocess: model=%s, %d image(s), conf=%.2f, dims=%s, "
                    "sam_variant=%s, hires=%s",
                    self.mode, model_path, self._total, conf, dims, sam_variant_key, hires_params)

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
        # Remembered for cleanup: processId() is 0 once the process has ended.
        self._worker_pid = self.process.processId()
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
            logger.info("Inference cancelled")
            process, self.process = self.process, None
            process.readyReadStandardOutput.disconnect(self._on_stdout)
            process.readyReadStandardError.disconnect(self._on_stderr)
            process.finished.disconnect(self._on_finished)
            process.errorOccurred.disconnect(self._on_process_error)
            pid = process.processId()
            if process.state() != QProcess.ProcessState.NotRunning:
                process.kill()
                process.waitForFinished(1000)
            process.deleteLater()
            _remove_worker_temp_dirs(pid)
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
                self._append_log(line)  # stray output, not protocol: it belongs in the log
                continue
            self._handle_message(message)

    def _on_stderr(self):
        if self.process is None:
            return
        self._stderr_buffer += self._stderr_decoder.decode(bytes(self.process.readAllStandardError()))
        *lines, self._stderr_buffer = _LINE_BREAK.split(self._stderr_buffer)
        for line in lines:
            self._append_log(line)

    def _append_log(self, line):
        line = line.rstrip()
        if line:
            logger.debug("worker: %s", line)  # also kept in the app's log file
        if self.progress_manager:
            self.progress_manager.append_log(line)

    def _flush_log(self):
        """Last, unterminated piece of the worker's log (if any)."""
        if self._stderr_decoder is not None:
            self._stderr_buffer += self._stderr_decoder.decode(b"", final=True)
        if self._stderr_buffer.strip():
            self._append_log(self._stderr_buffer)
        self._stderr_buffer = ""

    def _handle_message(self, message):
        if message.get("fatal"):
            logger.error("Inference worker failed: %s", message["fatal"])
            # The worker exits right after this; _on_finished then reports it,
            # by which time its log (with the traceback) has fully arrived.
            self._fatal = message["fatal"]
            process = self.process
            QTimer.singleShot(_FATAL_EXIT_GRACE_MS, lambda: self._stop_if_still_running(process))
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
                text = f"{self.display_text}\nImage {self._processed + 1}/{self._total}: {name}"
                if message.get("detail"):  # e.g. HiReS "Predicting tile 37/104"
                    text += f"\n{message['detail']}"
                self.progress_manager.progress_dialog.setLabelText(text)
                fraction = message.get("fraction")  # progress within this image
                if isinstance(fraction, (int, float)):
                    self.progress_manager.update_progress(
                        self._processed + min(max(float(fraction), 0.0), 1.0))
            return

        image_path = message.get("image")
        if image_path is None:
            return

        if "error" in message:
            logger.warning("Inference failed for %s: %s", image_path, message["error"])
            self._errors.append((os.path.basename(image_path), message["error"]))
        else:
            masks_xy = [np.array(m, dtype=np.float32) for m in message.get("masks", [])]
            class_names = message.get("classes", [])
            self._save_result(image_path, masks_xy, class_names)

        self._processed += 1
        if self.progress_manager:
            self.progress_manager.update_progress(self._processed)

    def _save_result(self, image_path, masks, class_names):
        """Called once per image from worker stdout: save to the database."""
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        for mask, class_name in zip(masks, class_names):
            if mask is None or getattr(mask, "size", 0) < 2:
                continue
            self.parent.state_manager.mask_manager.save_mask(mask, image_name, class_name)

            if class_name not in self.parent.state_manager.class_manager.get_all_class_names():
                rand = QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                self.parent.state_manager.class_manager.add_class(class_name, rand)
        # The image view, class list, table and statistics listen to the
        # state's masks_updated / classes_changed and refresh themselves.

    def _stop_if_still_running(self, process):
        """The worker reported a fatal error but did not exit: stop it.
        Its finished signal then ends the run as usual."""
        if self.process is process and process.state() != QProcess.ProcessState.NotRunning:
            logger.warning("Inference worker did not exit after a fatal error; stopping it")
            process.kill()

    def _on_finished(self, exit_code, exit_status):
        process = self.process
        if process is None:
            return
        # Qt delivers all remaining output before `finished`; only an
        # unterminated last log line may still be pending.
        self._flush_log()
        self.process = None
        process.deleteLater()
        # A worker that crashed or was killed leaves its HiReS tiles behind.
        _remove_worker_temp_dirs(self._worker_pid)

        killed = exit_status == QProcess.ExitStatus.CrashExit  # exit_code is meaningless then
        crashed = killed or exit_code != 0
        if crashed:
            logger.warning("Inference worker exited with code %d (status=%s)", exit_code, exit_status)
        logger.info("Inference finished")

        failed = len(self._errors)
        done = self._processed - failed
        if self._fatal:
            first_line = self._fatal.strip().splitlines()[0] if self._fatal.strip() else self._fatal
            summary = f"Inference failed: {first_line[:200]}"
        elif crashed:
            how = ("crashed or was killed (for example because the computer ran out of memory)"
                   if killed else f"exited with error code {exit_code}")
            summary = (f"The inference process {how} after {done} of {self._total} image(s).")
        elif failed:
            summary = f"Inference finished: {done} of {self._total} image(s) processed, {failed} failed."
        else:
            summary = f"Inference finished: {done} of {self._total} image(s) processed."

        if self.progress_manager:
            # On any problem keep the window up, so its log can show why.
            self.progress_manager.finish(summary, keep_open=bool(self._fatal or crashed or failed))
            self.progress_manager = None

        if self._fatal:
            QMessageBox.critical(self.parent, "Inference Failed", self._fatal)
        elif crashed:
            QMessageBox.warning(
                self.parent, "Inference",
                f"{summary}\n\nThe masks of the images finished before that were saved. "
                "'Show log' in the progress window shows the worker's last output.")
        self._report_errors()

    def _report_errors(self):
        """Tell the user which images failed (they are otherwise only logged)."""
        if not self._errors:
            return
        shown = "\n".join(f"• {name}: {error}" for name, error in self._errors[:10])
        if len(self._errors) > 10:
            shown += f"\n… and {len(self._errors) - 10} more (see the log file)"
        QMessageBox.warning(
            self.parent, "Inference",
            f"{len(self._errors)} of {self._total} image(s) could not be processed:\n\n{shown}",
        )
        self._errors = []
