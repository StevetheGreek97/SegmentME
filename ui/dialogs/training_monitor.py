"""
Training progress window.

Top: what is training, a status chip (Starting / Training / Finished /
Stopped / Failed), the epoch progress, elapsed time, time left and the best
score so far. Middle: the loss and score curves from results.csv (YOLO
writes one row per epoch; the file is re-read only when it changed).
Bottom: the training log (folded away until wanted, opened by itself on a
failure) and Stop, which becomes Close plus "Open results folder" when the
training ends.
"""
import os
import time
from pathlib import Path

import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer, QUrl
from PyQt6.QtGui import QColor, QDesktopServices, QFont, QPalette
from PyQt6.QtWidgets import (
    QDialog, QFrame, QHBoxLayout, QLabel, QPlainTextEdit, QProgressBar, QPushButton,
    QSplitter, QToolButton, QVBoxLayout, QWidget,
)

from services.logger import get_logger

logger = get_logger(__name__)

POLL_MS = 1000              # clock and file check; the file is re-read only when it changed
MAX_LOG_LINES = 5000

# Series colours, in a fixed order (the app's validated categorical palette).
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
LOSSES = [("train/seg_loss", "mask", BLUE), ("train/box_loss", "box", ORANGE),
          ("train/cls_loss", "class", AQUA), ("train/dfl_loss", "edges (dfl)", YELLOW)]
VAL_LOSS = ("val/seg_loss", "mask (validation)", BLUE)
SCORES = [("mAP50-95", "mAP50-95", BLUE), ("mAP50", "mAP50", ORANGE),
          ("precision", "precision", AQUA), ("recall", "recall", YELLOW)]

STATES = {   # status chip: text, colour
    "starting": ("Starting…", "#52514e"),
    "training": ("Training", "#2a78d6"),
    "finished": ("Finished", "#0ca30c"),
    "stopped": ("Stopped", "#c98500"),
    "failed": ("Failed", "#d03b3b"),
}

STYLE = """
QLabel#title { font-size: 16px; font-weight: 600; }
QLabel#chip { border-radius: 9px; padding: 2px 10px; color: white; font-weight: 600; }
QProgressBar { border: 1px solid palette(mid); border-radius: 5px; text-align: center;
               min-height: 18px; background: palette(base); }
QProgressBar::chunk { border-radius: 4px; background: palette(highlight); }
QFrame#stat { border: 1px solid palette(mid); border-radius: 8px; background: palette(base); }
QLabel[role="value"] { font-size: 17px; font-weight: 600; }
QToolButton#logToggle { border: none; font-weight: 600; padding: 2px 0; }
"""


def _duration(seconds) -> str:
    seconds = int(max(0, seconds))
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def score_column(df, name):
    """Mask (M) score when YOLO wrote one, else the box (B) score."""
    for suffix in ("(M)", "(B)"):
        column = f"metrics/{name}{suffix}"
        if column in df.columns:
            return column
    return None


class _Stat(QFrame):
    """A small tile: caption above a value."""

    def __init__(self, caption):
        super().__init__(objectName="stat")
        box = QVBoxLayout(self)
        box.setContentsMargins(12, 8, 12, 8)
        box.setSpacing(0)
        cap = QLabel(caption)
        cap.setForegroundRole(QPalette.ColorRole.PlaceholderText)
        self.value = QLabel("–")
        self.value.setProperty("role", "value")
        self.detail = QLabel("")
        self.detail.setForegroundRole(QPalette.ColorRole.PlaceholderText)
        for w in (cap, self.value, self.detail):
            box.addWidget(w)

    def set(self, value, detail=""):
        self.value.setText(value)
        self.detail.setText(detail)


class TrainingMonitor(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Training Monitor")
        self.setStyleSheet(STYLE)
        self.setMinimumSize(760, 560)
        self.resize(1080, 760)
        self.thread = None
        self.results_csv = None
        self.output_dir = None
        self.total_epochs = None
        self.stop_callback = None
        self._stop_requested = False
        self._state = None
        self._started = time.monotonic()
        self._csv_mtime = None
        self._epochs_done = 0
        self._train_seconds = None
        self._curves = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 12)
        layout.setSpacing(10)

        # --- header: name + status chip, progress
        top = QHBoxLayout()
        self.title = QLabel("Training", objectName="title")
        top.addWidget(self.title)
        top.addStretch(1)
        self.chip = QLabel(objectName="chip")
        top.addWidget(self.chip)
        layout.addLayout(top)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Waiting for the first epoch…")
        layout.addWidget(self.progress)

        tiles = QHBoxLayout()
        tiles.setSpacing(10)
        self.elapsed_tile = _Stat("Elapsed")
        self.left_tile = _Stat("Time left")
        self.best_tile = _Stat("Best mask score (mAP50-95)")
        self.loss_tile = _Stat("Mask loss")
        for tile in (self.elapsed_tile, self.left_tile, self.best_tile, self.loss_tile):
            tiles.addWidget(tile, 1)
        layout.addLayout(tiles)

        # --- plots and log
        # Legends where the curves are not: losses fall (top right is free), scores rise.
        self.metrics_plot = self._plot("Loss (lower is better)", "loss", (-8, 8))
        self.trainval_plot = self._plot("Scores on the validation images (higher is better)",
                                        "score", (-8, -8))
        self.trainval_plot.setYRange(0, 1)
        for column, label, colour in LOSSES:
            self._curves[column] = self.metrics_plot.plot([], [], name=label,
                                                          pen=pg.mkPen(colour, width=2))
        column, label, colour = VAL_LOSS
        self._curves[column] = self.metrics_plot.plot(
            [], [], name=label, pen=pg.mkPen(colour, width=2, style=Qt.PenStyle.DashLine))
        for key, label, colour in SCORES:
            self._curves["score:" + key] = self.trainval_plot.plot(
                [], [], name=label, pen=pg.mkPen(colour, width=2),
                symbol="o", symbolSize=5, symbolBrush=colour, symbolPen=None)

        plots = QWidget()
        row = QHBoxLayout(plots)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        row.addWidget(self.metrics_plot, 1)
        row.addWidget(self.trainval_plot, 1)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(MAX_LOG_LINES)
        self.log_view.setPlaceholderText("Training messages appear here…")
        mono = QFont("monospace")
        mono.setStyleHint(QFont.StyleHint.TypeWriter)
        mono.setPointSizeF(max(8.0, self.font().pointSizeF() - 1))
        self.log_view.setFont(mono)
        self.log_view.setVisible(False)

        log_box = QWidget()
        log_layout = QVBoxLayout(log_box)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(4)
        self.log_toggle = QToolButton(objectName="logToggle")
        self.log_toggle.setText("Training log")
        self.log_toggle.setCheckable(True)
        self.log_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.log_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.log_toggle.toggled.connect(self.show_log)
        self.last_line = QLabel("")
        self.last_line.setForegroundRole(QPalette.ColorRole.PlaceholderText)
        self.last_line.setTextFormat(Qt.TextFormat.PlainText)
        head = QHBoxLayout()
        head.addWidget(self.log_toggle)
        head.addWidget(self.last_line, 1)
        log_layout.addLayout(head)
        log_layout.addWidget(self.log_view, 1)

        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.addWidget(plots)
        self.splitter.addWidget(log_box)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setChildrenCollapsible(False)
        layout.addWidget(self.splitter, 1)

        # --- buttons
        buttons = QHBoxLayout()
        self.open_btn = QPushButton("Open results folder")
        self.open_btn.setAutoDefault(False)
        self.open_btn.clicked.connect(self.open_results_folder)
        self.open_btn.setVisible(False)
        buttons.addWidget(self.open_btn)
        buttons.addStretch(1)
        self.stop_btn = QPushButton("Stop Training")
        self.stop_btn.setAutoDefault(False)   # Enter must not stop hours of training
        self.stop_btn.clicked.connect(self._ask_stop)
        buttons.addWidget(self.stop_btn)
        layout.addLayout(buttons)

        self._set_state("starting")

        self.timer = QTimer(self)
        self.timer.setInterval(POLL_MS)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    # ------------------------------------------------------------- setup
    def _plot(self, title, left, legend_offset):
        pal = self.palette()
        base = pal.color(QPalette.ColorRole.Base)
        text = pal.color(QPalette.ColorRole.Text)
        muted = pal.color(QPalette.ColorRole.PlaceholderText)
        plot = pg.PlotWidget(background=base)
        plot.setTitle(title, color=text.name(), size="10pt")
        plot.setLabel("left", left, color=muted.name())
        plot.setLabel("bottom", "epoch", color=muted.name())
        plot.showGrid(x=True, y=True, alpha=0.15)
        for axis in ("left", "bottom"):
            ax = plot.getAxis(axis)
            ax.setPen(pg.mkPen(muted))
            ax.setTextPen(pg.mkPen(muted))
        legend = plot.addLegend(offset=legend_offset, labelTextColor=text,
                                brush=pg.mkBrush(QColor(base.red(), base.green(), base.blue(), 210)))
        legend.setLabelTextSize("9pt")
        plot.setMenuEnabled(False)
        return plot

    def set_run(self, settings):
        """What is being trained: name, number of epochs, output folder."""
        self.output_dir = getattr(settings, "output_dir", None)
        if self.output_dir:
            self.results_csv = str(Path(self.output_dir) / "results.csv")
            out = Path(self.output_dir)
            self.title.setText(f"Training {out.parent.name}/{out.name}")
        epochs = getattr(settings, "epochs", None)
        self.total_epochs = int(epochs) if isinstance(epochs, (int, float)) and epochs > 0 else None
        self._update_progress()

    def set_thread(self, thread):
        self.thread = thread

    def set_stop_callback(self, callback):
        self.stop_callback = callback

    # ------------------------------------------------------------- state
    def _set_state(self, state):
        self._state = state
        text, colour = STATES[state]
        self.chip.setText(text)
        self.chip.setStyleSheet(f"background: {colour};")

    def set_finished(self, outcome="finished"):
        """Training ended: finished, stopped or failed. Stop becomes Close."""
        outcome = outcome if outcome in ("finished", "stopped", "failed") else "finished"
        if self._stop_requested and outcome == "finished":
            outcome = "stopped"
        self._read_results(force=True)
        self._set_state(outcome)
        self.timer.stop()
        self._update_clock()
        self.left_tile.set("Done" if outcome == "finished" else "–", "")
        done, total = self._epochs_done, self.total_epochs
        ending = STATES[outcome][0]
        if outcome == "finished":
            self.progress.setRange(0, 100)
            self.progress.setValue(100)
            if total and 0 < done < total:      # patience or the time limit ended it
                ending = "Finished early: the score stopped improving (or the time limit)"
            self.append_log("✅ Training finished.")
        elif outcome == "failed":
            self.append_log("❌ Training failed: see the messages above.")
            self.log_toggle.setChecked(True)
        self.progress.setFormat(self.progress.format() + f"  ·  {ending}")
        try:
            self.stop_btn.clicked.disconnect()
        except TypeError:
            pass
        self.stop_btn.setText("Close")
        self.stop_btn.setEnabled(True)
        self.stop_btn.clicked.connect(self.accept)
        self.stop_btn.setDefault(True)
        self.open_btn.setVisible(bool(self.output_dir) and Path(self.output_dir).is_dir())
        logger.info("Training monitor: %s after %d epoch(s)", outcome, self._epochs_done)

    def training(self) -> bool:
        try:
            return (not self._stop_requested and self.thread is not None
                    and self.thread.isRunning())
        except RuntimeError:              # thread already deleted
            return False

    # ------------------------------------------------------------ updates
    def _tick(self):
        self._update_clock()
        self._read_results()

    def _update_clock(self):
        self.elapsed_tile.set(_duration(time.monotonic() - self._started))

    def _read_results(self, force=False):
        path = self.results_csv
        if not path or not os.path.exists(path):
            return
        try:
            mtime = os.path.getmtime(path)
            if not force and mtime == self._csv_mtime:
                return
            self._csv_mtime = mtime
            df = pd.read_csv(path)
        except Exception as e:
            logger.warning("Failed to read training results %s: %s", path, e)
            return
        df.columns = [c.strip() for c in df.columns]
        if df.empty or "epoch" not in df.columns:
            return
        logger.debug("Read %d epoch row(s) from %s", len(df), path)
        self.show_results(df)

    def show_results(self, df):
        """Curves, progress and tiles from YOLO's results table."""
        epochs = pd.to_numeric(df["epoch"], errors="coerce")
        if self._state == "starting":
            self._set_state("training")
        for column in [c for c, _l, _col in LOSSES] + [VAL_LOSS[0]]:
            self._set_curve(column, epochs, df.get(column))
        best_column = None
        for key, _label, _colour in SCORES:
            column = score_column(df, key)
            self._set_curve("score:" + key, epochs, df[column] if column else None)
            if key == "mAP50-95":
                best_column = column

        self._epochs_done = int(epochs.max()) if epochs.notna().any() else len(df)
        if "time" in df.columns:
            seconds = pd.to_numeric(df["time"], errors="coerce").dropna()
            self._train_seconds = float(seconds.iloc[-1]) if len(seconds) else None
        self._update_progress()

        if best_column:
            scores = pd.to_numeric(df[best_column], errors="coerce")
            if scores.notna().any():
                i = scores.idxmax()
                self.best_tile.set(f"{scores[i]:.3f}", f"epoch {int(epochs[i])}")
        if "train/seg_loss" in df.columns:
            losses = pd.to_numeric(df["train/seg_loss"], errors="coerce").dropna()
            if len(losses):
                trend = ""
                if len(losses) > 1:
                    trend = "↓ falling" if losses.iloc[-1] < losses.iloc[-2] else "↑ rising"
                self.loss_tile.set(f"{losses.iloc[-1]:.3f}", trend)

    def _set_curve(self, key, x, y):
        curve = self._curves[key]
        if y is None:
            curve.setData([], [])
            return
        y = pd.to_numeric(y, errors="coerce")
        ok = x.notna() & y.notna()
        curve.setData(x[ok].to_numpy(), y[ok].to_numpy())

    def _update_progress(self):
        done, total = self._epochs_done, self.total_epochs
        if not total:                          # number of epochs unknown
            if done:
                self.progress.setFormat(f"Epoch {done}")
            return
        self.progress.setRange(0, 100)
        self.progress.setValue(int(100 * min(done, total) / total))
        self.progress.setFormat(f"Epoch {done} of {total}" if done else "Waiting for the first epoch…")
        if done and self._state in ("starting", "training"):
            seconds = self._train_seconds or (time.monotonic() - self._started)
            per_epoch = seconds / done
            self.left_tile.set(f"about {_duration(per_epoch * (total - done))}",
                               f"{_duration(per_epoch)} per epoch")

    # ---------------------------------------------------------------- log
    def append_log(self, message: str):
        self.log_view.appendPlainText(message)
        last = message.strip().splitlines()[-1] if message.strip() else ""
        self.last_line.setText(last[:160])

    def show_log(self, on=True):
        self.log_view.setVisible(on)
        self.log_toggle.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)
        if on:
            bar = self.log_view.verticalScrollBar()
            bar.setValue(bar.maximum())

    def open_results_folder(self):
        if self.output_dir:
            logger.info("Opening training folder %s", self.output_dir)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.output_dir)))

    # --------------------------------------------------------------- stop
    def confirm_stop(self) -> bool:
        from ui.dialogs.confirm import ask_to_stop
        return ask_to_stop(
            self, "Stop Training?",
            "Training is still running. Stop it now?\n\nThe best model so far is kept "
            "(in the project's trainings folder).", stop_label="Stop Training")

    def _ask_stop(self):
        if self.training() and not self.confirm_stop():
            return
        self._on_stop()

    def reject(self):
        """Esc: same as the close button (it used to hide the window while
        training went on unseen)."""
        self.close()

    def _on_stop(self):
        self._stop_requested = True
        self.stop_btn.setText("Stopping…")
        self.stop_btn.setEnabled(False)

        if self.stop_callback:
            try:
                self.stop_callback()
            except RuntimeError as e:
                logger.debug("Stop callback skipped (worker already deleted): %s", e)

        self.append_log("🛑 Training stopped manually.")

    def closeEvent(self, event):
        if self.training() and not self.confirm_stop():
            event.ignore()
            return
        if not self._stop_requested and self._state in ("starting", "training"):
            self._on_stop()

        if self.thread is not None:
            try:
                if self.thread.isRunning():
                    logger.debug("Waiting for training thread to stop")
                    self.thread.quit()
                    self.thread.wait()
            except RuntimeError:          # thread already deleted
                pass

        self.timer.stop()
        self.hide()
        event.ignore()
