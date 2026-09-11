"""Runs YOLO training in a helper process and streams its output to the GUI.

The helper is the app itself started with --training-worker (see
services.app_process and core.trainer.training_worker), so this works from
a source checkout and from the packaged executable alike. Shelling out to
the `yolo` console script -- the previous approach -- only worked inside a
Python environment that had ultralytics installed, which a PyInstaller
bundle never is.
"""
import codecs
import json
import subprocess
import time

import psutil
from PyQt6.QtCore import QObject, pyqtSignal

from services.app_process import popen_kwargs, self_command
from services.logger import get_logger

logger = get_logger(__name__)


class TrainingWorker(QObject):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    # Minimum time between forwarded progress-bar updates (tqdm redraws its
    # bar with '\r' many times a second; the log view only needs a sample).
    _PROGRESS_INTERVAL = 0.25

    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.process = None
        self._stopping = False

    def run(self):
        try:
            overrides = self.settings.to_train_overrides()
            program, arguments = self_command("--training-worker")

            self.log_signal.emit("🔧 Equivalent command: " + self._describe(overrides))
            logger.info("Launching training: model=%s, data=%s, epochs=%s, device=%s",
                        overrides.get("model"), overrides.get("data"),
                        overrides.get("epochs"), overrides.get("device"))

            self.process = subprocess.Popen(
                [program, *arguments],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                **popen_kwargs(),
            )
            self.process.stdin.write((json.dumps(overrides) + "\n").encode("utf-8"))
            self.process.stdin.close()

            self._pump_output()
            code = self.process.wait()
            logger.info("Training subprocess exited with code %s", code)
            if code != 0 and not self._stopping:
                self.log_signal.emit(f"❌ Training process exited with code {code}; see the log above.")

        except Exception as e:
            logger.exception("Training subprocess failed")
            self.log_signal.emit(f"❌ Error: {e}")

        finally:
            self.finished_signal.emit()

    @staticmethod
    def _describe(overrides):
        return "yolo segment train " + " ".join(f"{k}={v}" for k, v in overrides.items())

    def _pump_output(self):
        """Forward the worker's output to the log as it arrives.

        Lines end with '\\n', except tqdm progress bars which redraw with
        '\\r'; both are forwarded, the latter throttled.
        """
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        buffer = ""
        last_progress = 0.0
        stream = self.process.stdout

        while True:
            chunk = stream.read1(4096)
            if not chunk:
                break
            buffer += decoder.decode(chunk)
            while True:
                nl, cr = buffer.find("\n"), buffer.find("\r")
                if nl == -1 and cr == -1:
                    break
                idx = min(i for i in (nl, cr) if i != -1)
                line, sep, buffer = buffer[:idx], buffer[idx], buffer[idx + 1:]
                if sep == "\r" and buffer.startswith("\n"):  # '\r\n' is one terminator
                    buffer, sep = buffer[1:], "\n"
                line = line.strip()
                if not line:
                    continue
                if sep == "\r":
                    now = time.monotonic()
                    if now - last_progress < self._PROGRESS_INTERVAL:
                        continue
                    last_progress = now
                self.log_signal.emit(line)

        tail = (buffer + decoder.decode(b"", final=True)).strip()
        if tail:
            self.log_signal.emit(tail)

    def stop(self):
        if not (self.process and self.process.poll() is None):
            return
        self._stopping = True
        self.log_signal.emit("🛑 Terminating training process...")

        # Remember the DataLoader workers etc. so none survive as orphans.
        try:
            children = psutil.Process(self.process.pid).children(recursive=True)
        except psutil.Error:
            children = []

        self.process.terminate()
        try:
            self.process.wait(timeout=10)
            self.log_signal.emit("✅ Training terminated.")
        except subprocess.TimeoutExpired:
            self.log_signal.emit("⛔ Force killing training process...")
            self.process.kill()

        for child in children:
            try:
                child.kill()
            except psutil.Error:
                pass
        # run() notices the closed pipe and emits finished_signal itself.
