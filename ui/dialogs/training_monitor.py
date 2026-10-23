"""Training progress window: loss and score plots (from results.csv,
checked every 10 s) and the training log, with a Stop button."""
import os

import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QPushButton, QTextEdit

from services.logger import get_logger

logger = get_logger(__name__)


class TrainingMonitor(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("📉 Training Monitor")
        self.setMinimumSize(1000, 700)
        self.thread = None
        self.results_csv = None
        self._stop_requested = False
        self.stop_callback = None

        # Metrics plot
        self.metrics_plot = pg.PlotWidget(title="📊 Loss Metrics (box, seg, cls, dfl)")
        self.metrics_plot.setLabel('left', 'Loss')
        self.metrics_plot.setLabel('bottom', 'Epoch')
        self.metrics_plot.addLegend()
        self.metrics_plot.showGrid(x=True, y=True)

        # Train/Val plot
        self.trainval_plot = pg.PlotWidget(title="📈 Train/Validation Scores")
        self.trainval_plot.setLabel('left', 'Score')
        self.trainval_plot.setLabel('bottom', 'Epoch')
        self.trainval_plot.addLegend()
        self.trainval_plot.showGrid(x=True, y=True)

        # Log view
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Training logs will appear here...")

        # Stop button
        self.stop_btn = QPushButton("🛑 Stop Training")
        self.stop_btn.setAutoDefault(False)   # Enter must not stop hours of training
        self.stop_btn.clicked.connect(self._ask_stop)

        # Layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.metrics_plot)
        layout.addWidget(self.trainval_plot)
        layout.addWidget(self.log_view)
        layout.addWidget(self.stop_btn)

        # Timer to monitor results.csv
        self.timer = QTimer()
        self.timer.setInterval(10000)  # Check every 10 sec
        self.timer.timeout.connect(self._update_from_csv)
        self.timer.start()

    def _update_from_csv(self):
        if not self.results_csv or not os.path.exists(self.results_csv):
            logger.debug("Waiting for training results file %s", self.results_csv)
            return

        try:
            df = pd.read_csv(self.results_csv)
            if df.empty:
                return

            logger.debug("Read %d epoch row(s) from %s", len(df), self.results_csv)

            epochs = pd.to_numeric(df["epoch"], errors="coerce")

            self.metrics_plot.clear()
            self.trainval_plot.clear()

            # Loss Metrics
            self._plot_metric(self.metrics_plot, epochs, df.get("train/box_loss"), 'box_loss', 'r')
            self._plot_metric(self.metrics_plot, epochs, df.get("train/seg_loss"), 'seg_loss', 'g')
            self._plot_metric(self.metrics_plot, epochs, df.get("train/cls_loss"), 'cls_loss', 'b')
            self._plot_metric(self.metrics_plot, epochs, df.get("train/dfl_loss"), 'dfl_loss', 'y')

            # Eval Metrics
            self._plot_metric(self.trainval_plot, epochs, df.get("metrics/precision(B)"), 'precision', 'c')
            self._plot_metric(self.trainval_plot, epochs, df.get("metrics/recall(B)"), 'recall', 'm')
            self._plot_metric(self.trainval_plot, epochs, df.get("metrics/mAP50(B)"), 'mAP50', 'w')
            self._plot_metric(self.trainval_plot, epochs, df.get("metrics/mAP50-95(B)"), 'mAP50-95', 'orange')

        except Exception as e:
            logger.warning("Failed to read training results %s: %s", self.results_csv, e)

    def _plot_metric(self, widget, x, y, label, color):
        if y is not None and not y.isna().all() and len(x) > 0:
            for item in widget.listDataItems():
                if item.name() == label:
                    widget.removeItem(item)

            plot_item = pg.PlotDataItem(x, y, pen=pg.mkPen(color=color, width=2), name=label)
            widget.addItem(plot_item)
            widget.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)

    def append_log(self, message: str):
        self.log_view.append(message)
        self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

    def set_thread(self, thread):
        self.thread = thread

    def set_stop_callback(self, callback):
        self.stop_callback = callback

    def training(self) -> bool:
        try:
            return (not self._stop_requested and self.thread is not None
                    and self.thread.isRunning())
        except RuntimeError:              # thread already deleted
            return False

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
        self.timer.stop()

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
        if not self._stop_requested:
            self._on_stop()

        if self.thread and self.thread.isRunning():
            logger.debug("Waiting for training thread to stop")
            self.thread.quit()
            self.thread.wait()

        self.timer.stop()
        self.hide()
        event.ignore()
