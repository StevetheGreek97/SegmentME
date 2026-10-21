"""Progress window for batch inference: status text, progress bar, Cancel,
and a collapsible log showing the inference worker's output ("Show log")."""
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFontDatabase
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QPlainTextEdit, QProgressBar, QPushButton, QVBoxLayout,
)


class InferenceProgressDialog(QDialog):
    """Non-modal progress window for one inference run.

    While the run is going, Cancel, the window's close button and Esc all
    emit `canceled`. When the run ends, finish() closes the window -- unless
    the log is open or the caller asks to keep it, in which case it stays up
    with a Close button so the log can still be read.
    """

    canceled = pyqtSignal()

    _STEPS_PER_ITEM = 1000  # progress bar resolution within one image

    def __init__(self, parent, total_items, label_text):
        super().__init__(parent)
        self.setWindowTitle("Inference")
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._running = True
        self._total = max(1, int(total_items))

        self.label = QLabel(label_text)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setWordWrap(True)

        self.bar = QProgressBar()
        self.bar.setRange(0, self._total * self._STEPS_PER_ITEM)
        self.bar.setValue(0)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.log_view.setMaximumBlockCount(20_000)  # keeps memory bounded on huge runs
        self.log_view.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.log_view.setPlaceholderText("Waiting for output...")
        self.log_view.setMinimumSize(700, 320)
        self.log_view.hide()

        self.log_button = QPushButton("Show log")
        self.log_button.setCheckable(True)
        self.log_button.toggled.connect(self._set_log_visible)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.close)  # closeEvent emits `canceled`
        for button in (self.log_button, self.cancel_button):
            button.setAutoDefault(False)  # Enter must not cancel a long run

        buttons = QHBoxLayout()
        buttons.addWidget(self.log_button)
        buttons.addStretch(1)
        buttons.addWidget(self.cancel_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addWidget(self.bar)
        layout.addWidget(self.log_view, 1)
        layout.addLayout(buttons)
        self.setMinimumWidth(400)

    # ------------------------------------------------------------- run state
    def setLabelText(self, text):
        self.label.setText(text)

    def set_progress(self, items_done):
        """Images done so far; fractions show progress within an image."""
        value = min(max(float(items_done), 0.0), float(self._total))
        self.bar.setValue(int(round(value * self._STEPS_PER_ITEM)))

    def append_log(self, line):
        scroll = self.log_view.verticalScrollBar()
        follow = scroll.value() >= scroll.maximum() - 2  # unless the user scrolled up
        self.log_view.appendPlainText(line)
        if follow:
            scroll.setValue(scroll.maximum())

    def finish(self, text, keep_open=False):
        """The run is over: close, or stay up (Close button) when the log is
        open or `keep_open` -- e.g. something failed and the log says why."""
        self._running = False
        if not (keep_open or self.log_button.isChecked()):
            self.close()
            return
        self.label.setText(text)
        self.cancel_button.setText("Close")

    def close_quietly(self):
        """Close without emitting `canceled` (the run is already stopped)."""
        self._running = False
        self.close()

    # ------------------------------------------------------------- internals
    def _set_log_visible(self, visible):
        self.log_button.setText("Hide log" if visible else "Show log")
        self.log_view.setVisible(visible)
        if visible:
            scroll = self.log_view.verticalScrollBar()
            scroll.setValue(scroll.maximum())
        self.adjustSize()

    def closeEvent(self, event):
        # Deliberately not QDialog.closeEvent (that routes through reject()).
        if self._running:
            self._running = False
            self.canceled.emit()
        event.accept()

    def reject(self):  # Esc
        self.close()


class ProgressDialogManager:
    """Owns the progress window of one inference run (see InferenceManager)."""

    def __init__(self, parent, total_items, cancel_callback, display_text="Running Inference..."):
        self.progress_dialog = InferenceProgressDialog(parent, total_items, display_text)
        self.progress_dialog.canceled.connect(cancel_callback)
        self.progress_dialog.show()

    def update_progress(self, current_value):
        self.progress_dialog.set_progress(current_value)

    def append_log(self, line):
        self.progress_dialog.append_log(line)

    def finish(self, text, keep_open=False):
        self.progress_dialog.finish(text, keep_open=keep_open)

    def close(self):
        self.progress_dialog.close_quietly()
